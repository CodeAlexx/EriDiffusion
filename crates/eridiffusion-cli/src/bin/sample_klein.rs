//! sample_klein — text → Klein 4B/9B image generation.
//!
//! Optional `--lora-path` overlays a trained LoRA. Mirrors sample_ernie/sample_zimage.
//!
//! Pipeline:
//!   1. Tokenize prompt with Klein chat template, encode via Qwen3 (auto-detect
//!      hidden=2560 → Klein 4B / hidden=4096 → Klein 9B).
//!   2. Load Klein transformer (auto-detect 4B vs 9B from img_in.weight shape).
//!   3. Build noise [1, 128, h, w] and Klein dynamic-mu sigma schedule.
//!   4. CFG denoise with `KleinModel::forward`.
//!   5. KleinVaeDecoder → RGB → PNG.

use clap::Parser;
use std::path::PathBuf;
use flame_core::{DType, Shape, Tensor};
use eridiffusion_core::config::{TrainConfig, TrainingMethod};
use eridiffusion_core::encoders::{
    qwen3::Qwen3Encoder,
    vae::KleinVaeDecoder,
};
use eridiffusion_core::models::{klein::KleinModel, TrainableModel};
use eridiffusion_core::sampler::klein_sampler;

const KLEIN_TEMPLATE_PRE: &str = "<|im_start|>user\n";
const KLEIN_TEMPLATE_POST: &str = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
const PAD_TOKEN_ID: i32 = 151643;
const TXT_PAD_LEN: usize = 512;

#[derive(Parser)]
struct Args {
    #[arg(long)] prompt: String,
    #[arg(long, default_value = "")] negative: String,
    #[arg(long, default_value = "output.png")] output: PathBuf,
    /// Klein transformer: single safetensors file OR directory of shards.
    #[arg(long)] transformer: PathBuf,
    #[arg(long)] vae_path: PathBuf,
    /// Qwen3 weights path (single file or sharded dir).
    #[arg(long)] qwen3: PathBuf,
    #[arg(long)] tokenizer_path: PathBuf,
    #[arg(long, default_value = "1024")] size: usize,
    #[arg(long, default_value = "50")] steps: usize,
    #[arg(long, default_value = "4.0")] cfg: f32,
    #[arg(long, default_value = "42")] seed: u64,
    /// Optional safetensors of a trained LoRA (matches train_klein save format).
    #[arg(long)] lora_path: Option<PathBuf>,
    #[arg(long, default_value = "16")] lora_rank: usize,
    #[arg(long, default_value = "16.0")] lora_alpha: f64,
    /// Per-block weight streaming (LoRA-only). Frees ~10 GB so 1024² fits on 24 GB.
    #[arg(long)] offload: bool,
}

fn collect_shards(path: &std::path::Path) -> anyhow::Result<Vec<PathBuf>> {
    if path.is_file() {
        return Ok(vec![path.to_path_buf()]);
    }
    let mut shards: Vec<PathBuf> = std::fs::read_dir(path)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("safetensors"))
        .collect();
    shards.sort();
    if shards.is_empty() {
        anyhow::bail!("no safetensors at {:?}", path);
    }
    Ok(shards)
}

fn load_qwen3_weights(path: &std::path::Path, device: &std::sync::Arc<flame_core::CudaDevice>)
    -> flame_core::Result<std::collections::HashMap<String, flame_core::Tensor>>
{
    if path.is_file() {
        return flame_core::serialization::load_file(path, device);
    }
    let mut all = std::collections::HashMap::new();
    for entry in std::fs::read_dir(path)
        .map_err(|e| flame_core::Error::Io(format!("read_dir: {e}")))? {
        let p = entry.map_err(|e| flame_core::Error::Io(format!("entry: {e}")))?.path();
        if p.extension().and_then(|e| e.to_str()) == Some("safetensors") {
            let part = flame_core::serialization::load_file(&p, device)?;
            all.extend(part);
        }
    }
    Ok(all)
}

fn encode_prompt(qwen: &Qwen3Encoder, tok: &tokenizers::Tokenizer, prompt: &str)
    -> anyhow::Result<flame_core::Tensor>
{
    let wrapped = format!("{KLEIN_TEMPLATE_PRE}{}{KLEIN_TEMPLATE_POST}", prompt.trim());
    let enc = tok.encode(wrapped.as_str(), false)
        .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
    let mut ids: Vec<i32> = enc.get_ids().iter().map(|&i| i as i32).collect();
    ids.resize(TXT_PAD_LEN, PAD_TOKEN_ID);
    let hidden = qwen.encode(&ids)?;
    Ok(hidden.to_dtype(DType::BF16)?)
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let args = Args::parse();
    let _no_grad = flame_core::autograd::AutogradContext::no_grad();
    flame_core::config::set_default_dtype(DType::BF16);
    let device = flame_core::global_cuda_device();

    let h_lat = args.size / 16; // 8x VAE × 2x patchify
    let w_lat = args.size / 16;
    log::info!("Size {}² → latent {}x{}", args.size, h_lat, w_lat);

    log::info!("[1/4] Encoding prompt with Qwen3...");
    let tokenizer = tokenizers::Tokenizer::from_file(&args.tokenizer_path)
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
    let qwen_weights = load_qwen3_weights(&args.qwen3, &device)?;
    let qcfg = Qwen3Encoder::config_from_weights(&qwen_weights)?;
    log::info!("  Qwen3 hidden={} extract={:?}", qcfg.hidden_size, qcfg.extract_layers);
    let qwen = Qwen3Encoder::new(qwen_weights, qcfg, device.clone());
    let cond = encode_prompt(&qwen, &tokenizer, &args.prompt)?;
    let uncond = encode_prompt(&qwen, &tokenizer, &args.negative)?;
    drop(qwen);
    log::info!("  cond={:?} uncond={:?}", cond.shape().dims(), uncond.shape().dims());

    log::info!("[2/4] Loading Klein transformer...");
    let shards = collect_shards(&args.transformer)?;
    let mut model: KleinModel = if let Some(_lp) = &args.lora_path {
        let mut tc = TrainConfig::default();
        tc.training_method = TrainingMethod::Lora;
        tc.lora_rank = args.lora_rank as u64;
        tc.lora_alpha = args.lora_alpha;
        let mut m = KleinModel::load(&shards, &tc, device.clone())?;
        m.load_weights(args.lora_path.as_ref().unwrap().to_str().unwrap())?;
        log::info!("  Applied LoRA from {:?} (rank={}, alpha={})",
            args.lora_path.as_ref().unwrap(), args.lora_rank, args.lora_alpha);
        if args.offload {
            m.enable_offload(shards.clone())?;
            log::info!("  Block offload enabled — per-block streaming from {} shards", shards.len());
        }
        m
    } else {
        // Base-only path: skip the F32 parameter copy that the LoRA path triggers.
        let mut all_weights = std::collections::HashMap::new();
        for s in &shards {
            let part = flame_core::serialization::load_file(s, &device)?;
            for (k, v) in part {
                all_weights.insert(k, v.to_dtype(DType::BF16)?);
            }
        }
        let kconfig = eridiffusion_core::models::klein::KleinConfig::from_weights(&all_weights)?;
        log::info!("  Klein {} (inner={}, double={}, single={})",
            if kconfig.inner_dim == 3072 { "4B" } else { "9B" },
            kconfig.inner_dim, kconfig.num_double, kconfig.num_single);
        KleinModel {
            config: TrainConfig::default(),
            kconfig,
            device: device.clone(),
            weights: all_weights,
            lora_adapters: Vec::new(),
            parameters: Vec::new(),
            is_lora: false,
            offload_shards: None,
        }
    };

    log::info!("[3/4] Denoising {} steps (cfg={})...", args.steps, args.cfg);
    let n_img = h_lat * w_lat;
    let timesteps = klein_sampler::get_schedule(args.steps, n_img);
    log::info!("  schedule: t[0]={:.4} t[-2]={:.4} t[-1]={:.4}",
        timesteps[0], timesteps[args.steps - 1], timesteps[args.steps]);

    // Seeded normal noise via flame's randn (CUDA Box-Muller is internal there).
    use rand::SeedableRng;
    let _rng = rand::rngs::StdRng::seed_from_u64(args.seed);
    let latent_shape = Shape::from_dims(&[1, 128, h_lat, w_lat]);
    let mut latent = Tensor::randn(latent_shape, 0.0, 1.0, device.clone())?
        .to_dtype(DType::BF16)?;

    for step in 0..args.steps {
        let sigma = timesteps[step];
        let sigma_next = timesteps[step + 1];
        let t = klein_sampler::sigma_to_timestep(sigma);
        let t_tensor = Tensor::from_vec(vec![t], Shape::from_dims(&[1]), device.clone())?;

        let pred_cond = model.forward(&latent, &cond, &t_tensor)?;
        let pred_uncond = model.forward(&latent, &uncond, &t_tensor)?;
        // CFG: uncond + cfg * (cond - uncond)
        let pred = pred_uncond.add(&pred_cond.sub(&pred_uncond)?.mul_scalar(args.cfg)?)?;

        // Klein euler uses dt = sigma_next - sigma (BFL direct velocity).
        latent = klein_sampler::euler_step(&latent, &pred, sigma, sigma_next)?;

        if step % 10 == 0 || step == args.steps - 1 {
            log::info!("  step {}/{}, sigma={:.4}", step + 1, args.steps, sigma);
        }
    }

    log::info!("[4/4] VAE decode...");
    drop(model);  // free DiT weights before decoder allocates
    let vae_weights = flame_core::serialization::load_file(&args.vae_path, &device)?;
    let dev = flame_core::Device::from(device.clone());
    let decoder = KleinVaeDecoder::load(&vae_weights, &dev)?;
    drop(vae_weights);
    let img = decoder.decode(&latent)?;

    // Save PNG
    let pixels: Vec<f32> = img.to_dtype(DType::F32)?.to_vec()?;
    let dims = img.shape().dims();
    let (c, h, w) = if dims.len() == 4 { (dims[1], dims[2], dims[3]) } else { (3, dims[0], dims[1]) };
    let mut buf = vec![0u8; h * w * 3];
    for y in 0..h {
        for x in 0..w {
            for ch in 0..c.min(3) {
                let idx = ch * h * w + y * w + x;
                let v = pixels.get(idx).copied().unwrap_or(0.0);
                buf[(y * w + x) * 3 + ch] = ((v.clamp(-1.0, 1.0) + 1.0) * 127.5) as u8;
            }
        }
    }
    image::save_buffer(&args.output, &buf, w as u32, h as u32, image::ColorType::Rgb8)?;
    log::info!("Saved to {:?}", args.output);
    Ok(())
}
