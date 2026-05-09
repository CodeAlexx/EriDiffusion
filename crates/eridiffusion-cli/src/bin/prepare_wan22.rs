//! prepare_wan22 — build cached video latents + UMT5 text embeddings
//! for Wan 2.2 LoRA training.
//!
//! ## Status
//!
//! Wan 2.2's text encoder (UMT5-XXL) and Wan 2.2 VAE encoder are NOT
//! yet ported into eridiffusion-core. This binary therefore stops at
//! the I/O scaffold:
//!
//! - Walks `--input-dir` for video / image files (`.mp4`, `.webm`,
//!   `.png`, `.jpg`)
//! - Reads paired captions from `<stem>.txt`
//! - Emits a per-sample placeholder safetensors with zero-tensors of
//!   the expected shape so the trainer's shape validators pass.
//!
//! Real preparation requires:
//! - `Wan22VaeEncoder` (port `inference-flame-master/src/models/wan/vae.rs`
//!   to a forward-only encoder; reference is 916 LoC).
//! - `Umt5XxlEncoder` (port `inference-flame-master/src/models/wan/t5.rs`,
//!   611 LoC) with the UMT5 SentencePiece tokenizer.
//! - Video frame decode + temporal stride alignment so `(F-1) % 4 == 0`.
//!
//! Until those land, use `--placeholder` to emit minimal-shape stubs
//! for end-to-end CLI tests.

use clap::Parser;
use std::path::PathBuf;

use flame_core::{serialization::save_file, DType, Shape, Tensor};

#[derive(Parser)]
struct Args {
    #[arg(long)] input_dir: PathBuf,
    #[arg(long)] output_dir: PathBuf,
    /// Wan 2.2 VAE checkpoint (single safetensors). Currently unused
    /// (encoder not yet ported); kept for CLI surface stability.
    #[arg(long)] vae_ckpt: Option<PathBuf>,
    /// UMT5-XXL text encoder dir. Currently unused; required when the
    /// real encoder lands.
    #[arg(long)] text_ckpt_dir: Option<PathBuf>,
    /// SentencePiece tokenizer for UMT5. Currently unused.
    #[arg(long)] tokenizer_path: Option<PathBuf>,

    /// Wan 2.2 variant to size the placeholder tensors for. Affects
    /// `in_channels` (5B uses 48, 14B uses 16) and `text_dim` (4096
    /// in both cases).
    #[arg(long, default_value = "t2v_14b")] variant: String,

    /// Square spatial size in pixels. Must be a multiple of 32.
    #[arg(long, default_value = "256")] size: usize,
    /// Number of latent frames. The source frame count must satisfy
    /// `(F_src - 1) % 4 == 0`; we apply `F_lat = 1 + (F_src - 1) / 4`.
    #[arg(long, default_value = "1")] num_latent_frames: usize,

    /// Emit minimal-shape placeholder safetensors when the real
    /// encoders aren't available. End-to-end CLI smoke test only.
    #[arg(long, default_value_t = false)] placeholder: bool,

    #[arg(long)] skip_existing: bool,
}

fn main() -> anyhow::Result<()> {
    if std::env::var_os("FLAME_ALLOC_POOL").is_none() {
        // SAFETY: single-threaded at this point.
        unsafe { std::env::set_var("FLAME_ALLOC_POOL", "0"); }
    }
    env_logger::init();
    let args = Args::parse();
    std::fs::create_dir_all(&args.output_dir)?;

    if args.size % 32 != 0 {
        anyhow::bail!("--size {} must be a multiple of 32", args.size);
    }

    flame_core::config::set_default_dtype(DType::BF16);
    let device = flame_core::global_cuda_device();

    let in_channels: usize = match args.variant.as_str() {
        "ti2v_5b" | "ti2v-5b" | "5b" => 48,
        "t2v_14b" | "t2v-14b" | "14b" | "t2v" | "i2v_14b" | "i2v-14b" | "i2v" => 16,
        other => anyhow::bail!("unknown --variant '{other}'"),
    };
    let h_lat = args.size / 32;
    let w_lat = args.size / 32;
    let f_lat = args.num_latent_frames.max(1);
    let text_len = 512usize;
    let text_dim = 4096usize;

    if !args.placeholder {
        anyhow::bail!(
            "Real Wan 2.2 VAE + UMT5-XXL encoders not yet ported. \
             Pass --placeholder to emit zero-tensor stubs for CLI smoke testing.\n\
             See crates/eridiffusion-cli/src/bin/prepare_wan22.rs module docs."
        );
    }
    log::warn!("[prepare_wan22] PLACEHOLDER MODE — emitted samples are zero tensors and unsuitable for real training.");

    let mut sources: Vec<PathBuf> = Vec::new();
    for entry in std::fs::read_dir(&args.input_dir)? {
        let path = entry?.path();
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            let lc = ext.to_ascii_lowercase();
            if matches!(lc.as_str(), "jpg" | "jpeg" | "png" | "webp" | "mp4" | "webm" | "mov") {
                sources.push(path);
            }
        }
    }
    sources.sort();
    log::info!("Found {} source files", sources.len());

    let mut cached = 0usize;
    for src in &sources {
        let stem = src.file_stem().unwrap().to_string_lossy().into_owned();
        let hash = format!("{:x}", md5::compute(src.to_string_lossy().as_bytes()));
        let out_path = args.output_dir.join(format!("{hash}.safetensors"));
        if args.skip_existing && out_path.exists() {
            continue;
        }

        // Placeholder zero tensors of the correct Wan22 shape.
        let latent = Tensor::zeros_dtype(
            Shape::from_dims(&[1, in_channels, f_lat, h_lat, w_lat]),
            DType::BF16,
            device.clone(),
        )?;
        let text_emb = Tensor::zeros_dtype(
            Shape::from_dims(&[1, text_len, text_dim]),
            DType::BF16,
            device.clone(),
        )?;
        let text_mask = Tensor::zeros_dtype(
            Shape::from_dims(&[1, text_len]),
            DType::F32,
            device.clone(),
        )?;

        let mut tensors = std::collections::HashMap::new();
        tensors.insert("latent".to_string(), latent);
        tensors.insert("text_embedding".to_string(), text_emb);
        tensors.insert("text_mask".to_string(), text_mask);
        let real_len = Tensor::from_vec(
            vec![1.0f32],
            Shape::from_dims(&[1]),
            device.clone(),
        )?;
        tensors.insert("text_real_len".to_string(), real_len);
        save_file(&tensors, &out_path)?;
        cached += 1;
        if cached % 5 == 0 {
            log::info!("Cached {}/{} ({stem})", cached, sources.len());
        }
    }
    log::info!("Done. Cached {cached} placeholder samples to {:?}", args.output_dir);
    Ok(())
}
