//! Flux trainer smoke: runs a handful of optimizer steps against synthetic or
//! precomputed latents to validate the runtime and LoRA plumbing.

use std::{fs, path::PathBuf, time::Instant};

use anyhow::{anyhow, Context, Result};
use clap::{ArgAction, Parser};
use tokio::runtime::Builder as TokioBuilder;

use eridiffusion_core::{Device, FluxVariant};
use eridiffusion_training::{
    flux::{
        data::FluxDataConfig,
        data_loader::{FluxDataLoader, FluxPrecomputedCfg},
    },
    flux_trainer::{create_flux_trainer, FluxTrainingConfig},
    init::init_global_device,
};

#[derive(Debug, Parser)]
#[command(name = "flux_10steps", about = "Flux 10-step trainer smoke", version)]
struct Args {
    /// Flux base weights (.safetensors)
    #[arg(long = "model", value_name = "PATH")]
    model_path: PathBuf,

    /// Flux VAE weights (.safetensors)
    #[arg(long = "vae", value_name = "PATH")]
    vae_path: PathBuf,

    /// Flux T5 encoder weights (.safetensors)
    #[arg(long = "t5", value_name = "PATH")]
    t5_path: PathBuf,

    /// Flux CLIP weights (.safetensors)
    #[arg(long = "clip", value_name = "PATH")]
    clip_path: PathBuf,

    /// Flux T5 tokenizer (tokenizer.json)
    #[arg(long = "t5-tokenizer", value_name = "PATH")]
    t5_tokenizer: PathBuf,

    /// Flux CLIP tokenizer (tokenizer.json)
    #[arg(long = "clip-tokenizer", value_name = "PATH")]
    clip_tokenizer: PathBuf,

    /// Optional manifest of cached latents/text to replay instead of synthetic data.
    #[arg(long = "manifest", value_name = "PATH")]
    manifest_path: Option<PathBuf>,

    /// Root directory for manifest-relative paths.
    #[arg(long = "manifest-root", value_name = "DIR")]
    manifest_root: Option<PathBuf>,

    /// Optional cache index path for manifest replay.
    #[arg(long = "manifest-index", value_name = "PATH")]
    manifest_index: Option<PathBuf>,

    /// Validate cached tensors when loading the manifest.
    #[arg(long = "manifest-validate", action = ArgAction::SetTrue)]
    manifest_validate: bool,

    /// Maximum epochs to iterate the manifest (omit for infinite loop).
    #[arg(long = "manifest-repeat")]
    manifest_repeat: Option<usize>,

    /// Disable manifest shuffling (deterministic order).
    #[arg(long = "manifest-no-shuffle", action = ArgAction::SetTrue)]
    manifest_no_shuffle: bool,

    /// Force synthetic batches even if a manifest is provided.
    #[arg(long = "synthetic", action = ArgAction::SetTrue)]
    synthetic: bool,

    /// CUDA device string (e.g. cuda:0)
    #[arg(long = "device", default_value = "cuda:0")]
    device: String,

    /// Number of training steps to run.
    #[arg(long = "steps", default_value_t = 10)]
    steps: usize,

    /// Batch size for synthetic batches or manifest replay.
    #[arg(long = "batch", default_value_t = 1)]
    batch: i32,

    /// Gradient accumulation steps.
    #[arg(long = "grad-accum", default_value_t = 1)]
    grad_accum: usize,

    /// Training height (px) for synthetic batches.
    #[arg(long = "height", default_value_t = 256)]
    height: i32,

    /// Training width (px) for synthetic batches.
    #[arg(long = "width", default_value_t = 256)]
    width: i32,

    /// Synthetic token count (defaults to 77).
    #[arg(long = "tokens")]
    tokens: Option<i32>,

    /// Random seed for Flux training.
    #[arg(long = "seed", default_value_t = 1337)]
    seed: u64,

    /// Learning rate for the AdamW optimizer.
    #[arg(long = "lr")]
    lr: Option<f64>,

    /// LoRA rank override.
    #[arg(long = "lora-rank")]
    lora_rank: Option<usize>,

    /// LoRA alpha override.
    #[arg(long = "lora-alpha")]
    lora_alpha: Option<f32>,
}

fn main() -> Result<()> {
    env_logger::builder().format_timestamp_secs().init();
    let args = Args::parse();
    ensure_paths_exist(&args)?;
    anyhow::ensure!(args.steps > 0, "--steps must be >= 1");
    anyhow::ensure!(args.batch > 0, "--batch must be >= 1");
    anyhow::ensure!(args.grad_accum > 0, "--grad-accum must be >= 1");
    anyhow::ensure!(
        args.height % 8 == 0 && args.width % 8 == 0,
        "height/width must be divisible by 8"
    );

    let _ = eridiffusion_core::device::initialize_devices();
    let device = init_global_device(&args.device)?;

    let mut train_cfg = FluxTrainingConfig::default();
    train_cfg.max_steps = args.steps;
    train_cfg.gradient_accumulation_steps = args.grad_accum;
    train_cfg.warmup_steps = 0;
    train_cfg.mixed_precision = false;
    train_cfg.seed = args.seed;
    if let Some(lr) = args.lr {
        train_cfg.learning_rate = lr;
    }
    if let Some(rank) = args.lora_rank {
        train_cfg.lora_rank = rank;
    }
    if let Some(alpha) = args.lora_alpha {
        train_cfg.lora_alpha = alpha;
    }

    let rt = TokioBuilder::new_current_thread().enable_all().build()?;
    let mut trainer = rt.block_on(create_flux_trainer(
        &args.model_path,
        &args.vae_path,
        &args.t5_path,
        &args.clip_path,
        &args.t5_tokenizer,
        &args.clip_tokenizer,
        FluxVariant::Dev,
        train_cfg.clone(),
        device.clone(),
    ))?;

    let mut loader = build_loader(&args, device.clone())?;
    println!(
        "[flux_10steps] steps={} batch={} grad_accum={} synthetic={} device={}",
        args.steps,
        args.batch,
        args.grad_accum,
        args.synthetic || args.manifest_path.is_none(),
        args.device
    );

    let mut completed = 0usize;
    for step in 1..=args.steps {
        let Some(batch) = loader.next()? else {
            println!("[flux_10steps] loader exhausted after {} steps", completed);
            break;
        };
        let t0 = Instant::now();
        let loss = trainer.train_step_precomputed(&batch)?;
        completed += 1;
        let elapsed = t0.elapsed().as_secs_f32();
        println!(
            "[flux_10steps] step {:02}/{:02} loss={:.6} grad_norm={:.6} time={:.3}s",
            step,
            args.steps,
            loss,
            trainer.grad_norm(),
            elapsed
        );
    }

    println!(
        "[flux_10steps] completed {} step(s) | lr={:.2e} lora_rank={} alpha={:.2}",
        completed, train_cfg.learning_rate, train_cfg.lora_rank, train_cfg.lora_alpha
    );
    Ok(())
}

fn ensure_paths_exist(args: &Args) -> Result<()> {
    for (label, path) in [
        ("Flux model", &args.model_path),
        ("Flux VAE", &args.vae_path),
        ("Flux T5", &args.t5_path),
        ("Flux CLIP", &args.clip_path),
        ("T5 tokenizer", &args.t5_tokenizer),
        ("CLIP tokenizer", &args.clip_tokenizer),
    ] {
        if !path.exists() {
            return Err(anyhow!("{label} not found at {}", path.display()));
        }
        if fs::metadata(path).map(|m| m.is_dir()).unwrap_or(false) {
            return Err(anyhow!("{label} at {} is a directory", path.display()));
        }
    }
    if let Some(manifest) = &args.manifest_path {
        if !manifest.exists() {
            return Err(anyhow!("Manifest not found at {}", manifest.display()));
        }
    }
    Ok(())
}

fn build_loader(args: &Args, device: Device) -> Result<FluxDataLoader> {
    if args.synthetic || args.manifest_path.is_none() {
        let mut data_cfg = FluxDataConfig::default();
        data_cfg.batch_size = args.batch;
        data_cfg.height = args.height;
        data_cfg.width = args.width;
        if let Some(tokens) = args.tokens {
            data_cfg.fixed_tokens = Some(tokens);
        }
        Ok(FluxDataLoader::new(data_cfg, device))
    } else {
        let manifest = args.manifest_path.as_ref().unwrap();
        let cfg = FluxPrecomputedCfg {
            manifest_path: manifest.clone(),
            root_dir: args.manifest_root.clone(),
            batch_size: args.batch as usize,
            shuffle: !args.manifest_no_shuffle,
            max_epochs: args.manifest_repeat,
            enforce_bf16: true,
            validate_on_load: args.manifest_validate,
            cache_index: args.manifest_index.clone(),
            device,
        };
        FluxDataLoader::from_precomputed(cfg)
    }
}
