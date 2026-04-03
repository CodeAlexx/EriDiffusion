//! SDXL block harness: runs N forward passes through a single transformer block.
//! Configuration is controlled via environment variables (see docs/SDXL handoff).

use std::{
    env,
    path::Path,
    sync::Arc,
    time::{Duration, Instant},
};

use anyhow::{anyhow, bail, ensure, Context, Result};
use eridiffusion_common_weights::strict_loader::StrictMmapLoader;
use eridiffusion_core::Device as CoreDevice;
use eridiffusion_training::{
    init::init_global_device,
    sdxl::{
        registry::{ConditioningBundle, SdxlLayerRegistry},
        runtime::{AttnChunkConfig, ExecutableBlock, SdxlBlockRuntime},
        weights::SdxlWeightProvider,
        RuntimeMode,
    },
};
use flame_core::{device::Device as FlameDevice, rng, CudaDevice, DType, Shape, Tensor};

const DRIVER_DIM: usize = 1280;
const TIME_PROJ_DIM: usize = 1536;

#[derive(Debug)]
struct HarnessConfig {
    steps: usize,
    batch: usize,
    height: usize,
    width: usize,
    ctx_dim: usize,
    tokens: usize,
    hidden: usize,
    device: String,
    use_synth: bool,
    seed: u64,
    block_index: usize,
    weight_path: Option<String>,
    attn_chunks: AttnChunkConfig,
    attn_reuse: bool,
    attn_max_ws_mb: usize,
    attn_min_free_mb: usize,
}

struct HarnessState {
    block: Arc<SdxlBlockRuntime>,
    ctx: Tensor,
    cond: ConditioningBundle,
    cuda: Arc<CudaDevice>,
    batch: usize,
    h_lat: usize,
    w_lat: usize,
    hidden: usize,
    ctx_dim: usize,
    tokens: usize,
    attn_chunks: AttnChunkConfig,
    attn_reuse: bool,
    attn_max_ws_mb: usize,
    attn_min_free_mb: usize,
    source_desc: String,
}

fn main() -> Result<()> {
    let cfg = HarnessConfig::from_env()?;
    ensure!(cfg.steps > 0, "STEPS must be >= 1");
    ensure!(cfg.batch > 0, "BATCH must be >= 1");
    ensure!(cfg.tokens > 0, "TOKENS must be >= 1");
    ensure!(cfg.ctx_dim > 0, "CTX_DIM must be >= 1");
    ensure!(cfg.hidden > 0 && cfg.hidden % 64 == 0, "HIDDEN must be a positive multiple of 64");
    ensure!(cfg.height % 8 == 0 && cfg.width % 8 == 0, "H and W must be divisible by 8");

    let _ = eridiffusion_core::device::initialize_devices();
    let device = init_global_device(&cfg.device)?;
    let flame_device = match device {
        CoreDevice::Cuda(ord) => FlameDevice::cuda(ord)?,
        CoreDevice::Cpu => bail!("SDXL harness requires CUDA; got cpu device"),
    };

    let _ = rng::set_seed(cfg.seed);

    let harness = if cfg.use_synth {
        build_synthetic_state(&cfg, &flame_device)?
    } else {
        build_real_state(&cfg, &flame_device)?
    };

    println!(
        "sdxl_10steps | steps={} batch={} latent={}x{} hidden={} ctx={} tokens={} source={}",
        cfg.steps,
        cfg.batch,
        harness.h_lat,
        harness.w_lat,
        harness.hidden,
        harness.ctx_dim,
        harness.tokens,
        harness.source_desc
    );
    println!(
        "attention chunking: q_chunk={} kv_chunk={} reuse={} max_ws_mb={} min_free_mb={}",
        harness.attn_chunks.q_chunk,
        harness.attn_chunks.kv_chunk,
        harness.attn_reuse,
        harness.attn_max_ws_mb,
        harness.attn_min_free_mb
    );

    run_steps(&cfg, &harness)?;
    Ok(())
}

fn run_steps(cfg: &HarnessConfig, harness: &HarnessState) -> Result<()> {
    let mut total = Duration::default();
    for step in 0..cfg.steps {
        let sample = randn_bf16(
            &harness.cuda,
            &[harness.batch, harness.h_lat, harness.w_lat, harness.hidden],
            1.0,
        )?;
        let step_start = Instant::now();
        let output = harness.block.forward_with_cond(
            &sample,
            &harness.ctx,
            &harness.cond.driver_1280,
            Some(&harness.cond.time_proj_1536),
        )?;
        let elapsed = step_start.elapsed();
        total += elapsed;
        let elapsed_ms = elapsed.as_secs_f64() * 1_000.0;
        let stats = output.to_dtype(DType::F32)?;
        let mean = stats.mean_all()?.to_scalar::<f32>()?;
        let max_abs = stats.abs()?.max_all()?;
        println!(
            "step {}/{}: {:>8.2} ms | mean {:>8.5} | max|x| {:>8.5}",
            step + 1,
            cfg.steps,
            elapsed_ms,
            mean,
            max_abs
        );
    }
    let avg_ms = (total.as_secs_f64() * 1_000.0) / cfg.steps as f64;
    println!("avg step time: {:.2} ms", avg_ms);
    Ok(())
}

fn build_synthetic_state(cfg: &HarnessConfig, flame_device: &FlameDevice) -> Result<HarnessState> {
    let cuda = flame_device.cuda_device_arc();
    let context = cfg.ctx_dim;
    let hidden = cfg.hidden;
    let ff_dim = hidden * 4;
    let mut tensors = Vec::with_capacity(26);

    tensors.push(ones_bf16(&cuda, &[hidden])?); // block_norm.weight
    tensors.push(zeros_bf16(&cuda, &[hidden])?); // block_norm.bias
    tensors.push(randn_bf16(&cuda, &[hidden, hidden], 0.02)?); // proj_in.weight
    tensors.push(zeros_bf16(&cuda, &[hidden])?); // proj_in.bias

    tensors.push(ones_bf16(&cuda, &[hidden])?); // norm1.weight
    tensors.push(zeros_bf16(&cuda, &[hidden])?); // norm1.bias
    tensors.push(randn_bf16(&cuda, &[hidden, hidden], 0.02)?); // attn1.q.weight
    tensors.push(randn_bf16(&cuda, &[hidden, hidden], 0.02)?); // attn1.k.weight
    tensors.push(randn_bf16(&cuda, &[hidden, hidden], 0.02)?); // attn1.v.weight
    tensors.push(randn_bf16(&cuda, &[hidden, hidden], 0.02)?); // attn1.out.weight
    tensors.push(zeros_bf16(&cuda, &[hidden])?); // attn1.out.bias

    tensors.push(ones_bf16(&cuda, &[hidden])?); // norm2.weight
    tensors.push(zeros_bf16(&cuda, &[hidden])?); // norm2.bias
    tensors.push(randn_bf16(&cuda, &[hidden, hidden], 0.02)?); // attn2.q.weight
    tensors.push(randn_bf16(&cuda, &[hidden, context], 0.02)?); // attn2.k.weight
    tensors.push(randn_bf16(&cuda, &[hidden, context], 0.02)?); // attn2.v.weight
    tensors.push(randn_bf16(&cuda, &[hidden, hidden], 0.02)?); // attn2.out.weight
    tensors.push(zeros_bf16(&cuda, &[hidden])?); // attn2.out.bias

    tensors.push(ones_bf16(&cuda, &[hidden])?); // norm3.weight
    tensors.push(zeros_bf16(&cuda, &[hidden])?); // norm3.bias
    tensors.push(randn_bf16(&cuda, &[ff_dim * 2, hidden], 0.02)?); // ff.proj.weight
    tensors.push(zeros_bf16(&cuda, &[ff_dim * 2])?); // ff.proj.bias
    tensors.push(randn_bf16(&cuda, &[hidden, ff_dim], 0.02)?); // ff.out.weight
    tensors.push(zeros_bf16(&cuda, &[hidden])?); // ff.out.bias

    tensors.push(randn_bf16(&cuda, &[hidden, hidden], 0.02)?); // proj_out.weight
    tensors.push(zeros_bf16(&cuda, &[hidden])?); // proj_out.bias

    let mut block = SdxlBlockRuntime::from_mmap(cfg.block_index, tensors)?;
    block.mod1.weight = randn_bf16(&cuda, &[block.mod1.cond_dim, 3 * hidden], 0.02)?;
    block.mod1.bias = randn_bf16(&cuda, &[3 * hidden], 0.01)?;
    block.mod2.weight = randn_bf16(&cuda, &[block.mod2.cond_dim, 3 * hidden], 0.02)?;
    block.mod2.bias = randn_bf16(&cuda, &[3 * hidden], 0.01)?;
    block.mod3.weight = randn_bf16(&cuda, &[block.mod3.cond_dim, 3 * hidden], 0.02)?;
    block.mod3.bias = randn_bf16(&cuda, &[3 * hidden], 0.01)?;

    let attn_chunks = cfg.attn_chunks;
    let cond = ConditioningBundle {
        driver_1280: randn_bf16(&cuda, &[cfg.batch, DRIVER_DIM], 0.02)?,
        time_proj_1536: randn_bf16(&cuda, &[cfg.batch, TIME_PROJ_DIM], 0.02)?,
    };
    let ctx = randn_bf16(&cuda, &[cfg.batch, cfg.tokens, cfg.ctx_dim.max(cfg.hidden)], 0.02)?;

    Ok(HarnessState {
        block: Arc::new(block),
        ctx,
        cond,
        cuda,
        batch: cfg.batch,
        h_lat: cfg.height / 8,
        w_lat: cfg.width / 8,
        hidden,
        ctx_dim: cfg.ctx_dim,
        tokens: cfg.tokens,
        attn_chunks,
        attn_reuse: cfg.attn_reuse,
        attn_max_ws_mb: cfg.attn_max_ws_mb,
        attn_min_free_mb: cfg.attn_min_free_mb,
        source_desc: "synthetic".to_string(),
    })
}

fn build_real_state(cfg: &HarnessConfig, flame_device: &FlameDevice) -> Result<HarnessState> {
    let path = cfg
        .weight_path
        .as_ref()
        .ok_or_else(|| anyhow!("SDXL_MMAP_PATH must be set when SYNTHETIC=0"))?;
    let mmap = Arc::new(StrictMmapLoader::open(Path::new(path))?);
    let provider = Arc::new(SdxlWeightProvider::new(mmap, flame_device.clone()));
    let registry = SdxlLayerRegistry::build(provider.clone(), RuntimeMode::Resident)?;
    let block = registry.block(cfg.block_index)?;
    let cuda = flame_device.cuda_device_arc();
    let cond = ConditioningBundle {
        driver_1280: randn_bf16(&cuda, &[cfg.batch, DRIVER_DIM], 0.02)?,
        time_proj_1536: randn_bf16(&cuda, &[cfg.batch, TIME_PROJ_DIM], 0.02)?,
    };
    let ctx = randn_bf16(&cuda, &[cfg.batch, cfg.tokens, cfg.ctx_dim.max(cfg.hidden)], 0.02)?;

    Ok(HarnessState {
        block,
        ctx,
        cond,
        cuda,
        batch: cfg.batch,
        h_lat: cfg.height / 8,
        w_lat: cfg.width / 8,
        hidden: cfg.hidden,
        ctx_dim: cfg.ctx_dim,
        tokens: cfg.tokens,
        attn_chunks: cfg.attn_chunks,
        attn_reuse: cfg.attn_reuse,
        attn_max_ws_mb: cfg.attn_max_ws_mb,
        attn_min_free_mb: cfg.attn_min_free_mb,
        source_desc: format!("mmap:{}", path),
    })
}

fn randn_bf16(cuda: &Arc<CudaDevice>, shape: &[usize], stddev: f32) -> Result<Tensor> {
    let tensor = Tensor::randn(Shape::from_dims(shape), 0.0, stddev, cuda.clone())
        .map_err(|e| anyhow!("randn_bf16 failed: {e}"))?;
    Ok(tensor.to_dtype(DType::BF16)?)
}

fn zeros_bf16(cuda: &Arc<CudaDevice>, shape: &[usize]) -> Result<Tensor> {
    Tensor::zeros_dtype(Shape::from_dims(shape), DType::BF16, cuda.clone())
        .map_err(|e| anyhow!("zeros_bf16 failed: {e}"))
}

fn ones_bf16(cuda: &Arc<CudaDevice>, shape: &[usize]) -> Result<Tensor> {
    Tensor::zeros_dtype(Shape::from_dims(shape), DType::BF16, cuda.clone())
        .and_then(|t| t.add_scalar(1.0))
        .map_err(|e| anyhow!("ones_bf16 failed: {e}"))
}

impl HarnessConfig {
    fn from_env() -> Result<Self> {
        Ok(Self {
            steps: env_u64("STEPS", 10)? as usize,
            batch: env_u64("BATCH", 1)? as usize,
            height: env_u64("HEIGHT", 1024)? as usize,
            width: env_u64("WIDTH", 1024)? as usize,
            ctx_dim: env_u64("CTX_DIM", 4096)? as usize,
            tokens: env_u64("TOKENS", 256)? as usize,
            hidden: env_u64("HIDDEN", 3072)? as usize,
            device: env::var("DEVICE").unwrap_or_else(|_| "cuda:0".to_string()),
            use_synth: env_bool("SYNTHETIC", true)?,
            seed: env_u64("SEED", 1337)?,
            block_index: env_u64("BLOCK_INDEX", 0)? as usize,
            weight_path: env::var("SDXL_MMAP_PATH").ok(),
            attn_chunks: AttnChunkConfig {
                q_chunk: env_u64("ATTN_Q_CHUNK", 256)? as usize,
                kv_chunk: env_u64("ATTN_KV_CHUNK", 256)? as usize,
            },
            attn_reuse: env_bool("ATTN_REUSE", true)?,
            attn_max_ws_mb: env_u64("ATTN_MAX_WS_MB", 2048)? as usize,
            attn_min_free_mb: env_u64("ATTN_MIN_FREE_MB", 512)? as usize,
        })
    }
}

fn env_u64(name: &str, default: u64) -> Result<u64> {
    if let Ok(value) = env::var(name) {
        return value.parse::<u64>().with_context(|| format!("parsing {name}={value}"));
    }
    Ok(default)
}

fn env_bool(name: &str, default: bool) -> Result<bool> {
    if let Ok(value) = env::var(name) {
        return match value.to_ascii_lowercase().as_str() {
            "1" | "true" | "on" => Ok(true),
            "0" | "false" | "off" => Ok(false),
            other => Err(anyhow!("invalid bool for {name}: {other}")),
        };
    }
    Ok(default)
}
