use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use clap::{Parser, ValueEnum};
use eridiffusion::inference::mmdit_streaming::{
    build_streaming_config, BlockArenaProfile, StreamingMMDiT,
};
use eridiffusion::loaders::lazy_safetensors::LazySafetensorsLoader;
use eridiffusion::loaders::mmdit_weights::{MmditLoadReport, MmditStructure};
use eridiffusion::loaders::{
    dry_run_mmdit_weights, infer_mmdit_structure, load_mmdit_weights_with_report, WeightLoader,
};
use eridiffusion::models::mmdit_blocks::{MMDiT, MMDiTConfig};
use flame_core::device::Device;
use flame_core::memory_pool::MEMORY_POOL;
use flame_core::{DType, Shape, Tensor};
use log::{info, warn};

#[derive(ValueEnum, Clone, Debug)]
enum Variant {
    Auto,
    Medium,
    Large,
}

#[derive(ValueEnum, Clone, Debug, Eq, PartialEq)]
enum InferenceBackend {
    Eager,
    Streaming,
}

impl Variant {
    fn apply(&self, config: &mut MMDiTConfig, inferred: &MmditStructure) {
        match self {
            Variant::Auto => {
                config.hidden_size = inferred.hidden_size;
                config.num_heads = inferred.num_heads;
                config.depth = inferred.depth;
            }
            Variant::Medium => {
                config.hidden_size = 1536;
                config.num_heads = 24;
                config.depth = 24;
                config.mlp_ratio = 4.0;
            }
            Variant::Large => {
                config.hidden_size = 2432;
                config.num_heads = 38;
                config.depth = 38;
                config.mlp_ratio = 4.0;
            }
        }
    }

    fn default_tokens(&self, inferred: &MmditStructure) -> usize {
        match self {
            Variant::Medium => 96,
            Variant::Large => 128,
            Variant::Auto => {
                if inferred.hidden_size <= 1024 {
                    96
                } else {
                    128
                }
            }
        }
    }
}

#[derive(Parser, Debug)]
#[command(name = "sd35-mmdit-validate", about = "SD3.5 MMDiT loader validation harness")]
struct Args {
    /// Path to the SD3.5 MMDiT safetensors checkpoint
    #[arg(long)]
    mmdit: PathBuf,

    /// Model variant (affects hidden size / head count)
    #[arg(long, value_enum, default_value_t = Variant::Auto)]
    variant: Variant,

    /// Optional JSON output path for the telemetry report
    #[arg(long)]
    report_json: Option<PathBuf>,

    /// Run only the dry-run (no weight copies)
    #[arg(long)]
    dry_run_only: bool,

    /// Skip the random smoke forward pass
    #[arg(long)]
    skip_forward: bool,

    /// Height of latent grid (post VAE-scaling, e.g. 128 for 1024x)
    #[arg(long, default_value_t = 128)]
    latent_height: usize,

    /// Width of latent grid (post VAE-scaling, e.g. 128 for 1024x)
    #[arg(long, default_value_t = 128)]
    latent_width: usize,

    /// Sequence length to use for the context smoke test
    #[arg(long)]
    context_tokens: Option<usize>,

    /// Inference backend for the smoke forward
    #[arg(long, value_enum, default_value_t = InferenceBackend::Eager)]
    inference_backend: InferenceBackend,

    /// Collect per-block arena statistics during streaming forward
    #[arg(long)]
    profile_arena: bool,

    /// Optional path override for the arena profile JSON output
    #[arg(long)]
    arena_profile_json: Option<PathBuf>,
}

fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();

    validate_path("mmdit", &args.mmdit)?;

    let device = Device::cuda(0).context("failed to acquire CUDA device 0")?;

    if args.inference_backend == InferenceBackend::Streaming {
        run_streaming(&args, &device)?;
        return Ok(());
    }

    let loader =
        WeightLoader::from_safetensors_with_dtype(&args.mmdit, device.clone(), DType::BF16)
            .with_context(|| format!("failed to load {:?}", args.mmdit))?;

    let meta = loader.infer_mmdit_metadata();
    info!(
        "checkpoint metadata: qk_norm={:?}, x_self_attn_layers={:?}, hidden_size={:?}, num_heads={:?}, depth={:?}, mlp_ratio={:?}",
        meta.qk_norm,
        meta.x_self_attn_layers,
        meta.hidden_size,
        meta.num_heads,
        meta.depth,
        meta.mlp_ratio,
    );

    let inferred = infer_mmdit_structure(&loader)?;
    info!(
        "inferred structure: hidden_size={} num_heads={} depth={}",
        inferred.hidden_size, inferred.num_heads, inferred.depth
    );

    let mut config = MMDiTConfig::default();
    config.hidden_size = meta.hidden_size.unwrap_or(inferred.hidden_size);
    config.num_heads = meta.num_heads.unwrap_or(inferred.num_heads);
    config.depth = meta.depth.unwrap_or(inferred.depth);
    if let Some(ratio) = meta.mlp_ratio {
        config.mlp_ratio = ratio;
    }
    config.qk_norm = meta.qk_norm;
    config.x_self_attn_layers = meta.x_self_attn_layers;
    args.variant.apply(&mut config, &inferred);

    let mut mmdit = MMDiT::new(config.clone(), &device)?;

    let dry_report =
        dry_run_mmdit_weights(&mut mmdit, &loader).context("dry-run inspection failed")?;
    dry_report.log_summary("mmdit.dry_run");

    if let Some(path) = &args.report_json {
        write_report(path, "dry_run", &dry_report)?;
    }

    if args.dry_run_only {
        return Ok(());
    }

    if args.inference_backend == InferenceBackend::Eager {
        let mut eager_mmdit = MMDiT::new(config.clone(), &device)?;
        let load_report = load_mmdit_weights_with_report(&mut eager_mmdit, &loader)
            .context("weight copy failed")?;
        load_report.log_summary("mmdit.copy");

        if let Some(path) = &args.report_json {
            write_report(path, "copy", &load_report)?;
        }

        if !args.skip_forward {
            let default_tokens = args.variant.default_tokens(&inferred);
            smoke_forward(&eager_mmdit, &device, &args, default_tokens)?;
        }
    }

    Ok(())
}

fn run_streaming(args: &Args, device: &Device) -> Result<()> {
    let (mut config, mut inferred) = {
        let lazy_loader = LazySafetensorsLoader::new(&args.mmdit)
            .with_context(|| format!("failed to memory-map {:?}", args.mmdit))?;
        let meta =
            WeightLoader::infer_mmdit_metadata_from_keys(lazy_loader.keys().map(|k| k.as_str()));
        let mut cfg = build_streaming_config(&meta, &lazy_loader, device)
            .context("failed to infer streaming config")?;
        cfg.context_dim = 4096;
        cfg.pooled_dim = Some(2048);
        let inferred = MmditStructure {
            hidden_size: cfg.hidden_size,
            num_heads: cfg.num_heads,
            depth: cfg.depth,
        };
        (cfg, inferred)
    };

    args.variant.apply(&mut config, &inferred);
    inferred.hidden_size = config.hidden_size;
    inferred.num_heads = config.num_heads;
    inferred.depth = config.depth;

    info!(
        "streaming config: hidden_size={} num_heads={} depth={}",
        config.hidden_size, config.num_heads, config.depth
    );

    if args.dry_run_only {
        info!("streaming backend: dry-run telemetry skipped (metadata only)");
        return Ok(());
    }

    if args.skip_forward {
        info!("streaming backend: forward pass skipped by flag");
        return Ok(());
    }

    MEMORY_POOL.clear_all_caches();

    let streaming_model =
        StreamingMMDiT::from_checkpoint(config.clone(), args.mmdit.as_path(), device.clone())
            .context("failed to initialise StreamingMMDiT")?;

    let default_tokens = args.variant.default_tokens(&inferred);
    let profile = smoke_forward_streaming(&streaming_model, device, args, default_tokens)?;

    if args.profile_arena {
        if let Some(blocks) = profile {
            let path =
                args.arena_profile_json.clone().unwrap_or_else(|| default_arena_profile_path(args));
            write_arena_profile(&path, args, &blocks)?;
            info!("arena profile written to {:?}", path);
        } else {
            warn!("profile_arena was requested but no samples were recorded");
        }
    }

    Ok(())
}

fn write_report(path: &Path, stage: &str, report: &MmditLoadReport) -> Result<()> {
    let mut map = if path.exists() {
        let data = fs::read_to_string(path)
            .with_context(|| format!("failed to read existing report at {path:?}"))?;
        serde_json::from_str::<serde_json::Value>(&data)
            .context("failed to parse existing JSON report")?
    } else {
        serde_json::json!({})
    };

    if let Some(obj) = map.as_object_mut() {
        obj.insert(stage.to_string(), serde_json::to_value(report)?);
        fs::create_dir_all(path.parent().unwrap_or_else(|| Path::new(".")))?;
        fs::write(path, serde_json::to_string_pretty(&map)?)
            .with_context(|| format!("failed to write {path:?}"))?;
    } else {
        bail!("expected top-level JSON object in report file");
    }

    Ok(())
}

fn default_arena_profile_path(args: &Args) -> PathBuf {
    let stem = args.mmdit.file_stem().and_then(|s| s.to_str()).unwrap_or("mmdit");
    let variant = format!("{:?}", args.variant).to_lowercase();
    Path::new("artifacts")
        .join("mmdit_validate")
        .join("streaming")
        .join(stem)
        .join(format!("{}-arena_profile.json", variant))
}

fn write_arena_profile(path: &Path, args: &Args, blocks: &[BlockArenaProfile]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create arena profile directory {:?}", parent))?;
    }
    let payload = serde_json::json!({
        "checkpoint": args.mmdit.display().to_string(),
        "variant": format!("{:?}", args.variant),
        "latent_height": args.latent_height,
        "latent_width": args.latent_width,
        "blocks": blocks,
    });
    fs::write(path, serde_json::to_string_pretty(&payload)?)
        .with_context(|| format!("failed to write arena profile to {path:?}"))?;
    Ok(())
}

fn smoke_forward(mmdit: &MMDiT, device: &Device, args: &Args, default_tokens: usize) -> Result<()> {
    let batch = 2usize;
    let height = args.latent_height;
    let width = args.latent_width;
    let context_tokens = args.context_tokens.unwrap_or(default_tokens);

    let in_channels = mmdit.config.in_channels;
    let latents = Tensor::randn(
        Shape::from_dims(&[batch, in_channels, height, width]),
        0.0,
        1.0,
        device.cuda_device_arc(),
    )?
    .to_dtype(DType::BF16)?;
    info!(
        "smoke_forward: latents shape {:?}, in_channels={}, patch weight shape {:?}",
        latents.shape().dims(),
        in_channels,
        mmdit.patch_embed().weight_shape()
    );

    let timesteps = Tensor::randn(Shape::from_dims(&[batch]), 0.0, 1.0, device.cuda_device_arc())?
        .to_dtype(DType::BF16)?;

    let context = Tensor::randn(
        Shape::from_dims(&[batch, context_tokens, mmdit.config.context_dim]),
        0.0,
        1.0,
        device.cuda_device_arc(),
    )?
    .to_dtype(DType::BF16)?;

    let pooled_dim = mmdit.config.pooled_dim.unwrap_or(2048);
    let pooled =
        Tensor::randn(Shape::from_dims(&[batch, pooled_dim]), 0.0, 1.0, device.cuda_device_arc())?
            .to_dtype(DType::BF16)?;

    let output = mmdit
        .forward(&latents, &timesteps, &context, Some(&pooled))
        .context("smoke forward failed")?;

    info!("smoke forward completed → output shape {:?}", output.shape().dims());
    Ok(())
}

fn smoke_forward_streaming(
    mmdit: &StreamingMMDiT,
    device: &Device,
    args: &Args,
    default_tokens: usize,
) -> Result<Option<Vec<BlockArenaProfile>>> {
    // Streaming smoke test keeps batch=1 to reduce peak VRAM while validating the path.
    let batch = 1usize;
    let height = args.latent_height;
    let width = args.latent_width;
    let context_tokens = args.context_tokens.unwrap_or(default_tokens);

    let in_channels = mmdit.config.in_channels;
    let latents = Tensor::randn(
        Shape::from_dims(&[batch, in_channels, height, width]),
        0.0,
        1.0,
        device.cuda_device_arc(),
    )?
    .to_dtype(DType::BF16)?;
    info!(
        "smoke_forward_streaming: latents shape {:?}, in_channels={}",
        latents.shape().dims(),
        in_channels
    );

    let timesteps = Tensor::randn(Shape::from_dims(&[batch]), 0.0, 1.0, device.cuda_device_arc())?
        .to_dtype(DType::BF16)?;

    let context = Tensor::randn(
        Shape::from_dims(&[batch, context_tokens, mmdit.config.context_dim]),
        0.0,
        1.0,
        device.cuda_device_arc(),
    )?
    .to_dtype(DType::BF16)?;

    let pooled_dim = mmdit.config.pooled_dim.unwrap_or(2048);
    let pooled =
        Tensor::randn(Shape::from_dims(&[batch, pooled_dim]), 0.0, 1.0, device.cuda_device_arc())?
            .to_dtype(DType::BF16)?;

    let (output, profile) = if args.profile_arena {
        let (out, profile) = mmdit
            .forward_profiled(&latents, &timesteps, &context, Some(&pooled))
            .context("streaming smoke forward (profiled) failed")?;
        (out, Some(profile))
    } else {
        let out = mmdit
            .forward(&latents, &timesteps, &context, Some(&pooled))
            .context("streaming smoke forward failed")?;
        (out, None)
    };

    info!("streaming smoke forward completed → output shape {:?}", output.shape().dims());
    Ok(profile)
}

fn validate_path(label: &str, path: &Path) -> Result<()> {
    if !path.exists() {
        bail!("{label} path {:?} does not exist", path);
    }
    if !path.is_file() {
        bail!("{label} path {:?} is not a file", path);
    }
    Ok(())
}
