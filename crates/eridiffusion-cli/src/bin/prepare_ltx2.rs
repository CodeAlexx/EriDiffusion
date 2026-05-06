//! prepare_ltx2 — image+caption → cached frame-latents+text-embeds
//! for LTX-2 T2V LoRA training (image-as-frame bootstrap).
//!
//! ## Bootstrap mode
//!
//! For the first iteration, this binary treats every input image as a
//! single-frame video latent. This sidesteps video I/O complexity while
//! exercising the full pipeline end-to-end. Multi-frame video reading
//! comes in a follow-up (TODO).
//!
//! ## Cache layout (one safetensors per sample)
//! - `latent`: `[1, 128, 1, H/32, W/32]` BF16 — image-as-frame video latent
//!   produced by Ltx2Vae::encode_image_as_frame **(STUB encoder for now —
//!   replace before real training; see ltx2_vae.rs)**.
//! - `text_embedding`: `[1, 1024, 3840]` BF16 — Gemma3-12B per-layer
//!   hidden states **(STUB Gemma3 encoder for now — replace before real
//!   training; see gemma3.rs)**.
//! - `text_real_len`: `[1]` F32 — real (non-padded) token count.
//! - `caption`: dropped (lives next to the image as `<stem>.txt`).
//!
//! ## CHW transpose
//! Per `prepare_ernie.rs:80-87` (and HANDOFF_2026-05-05_PREPARE_CHW_BUG_FIXED),
//! we do an explicit `enumerate_pixels` + per-channel write loop. Never
//! do `img.pixels().flat_map(...).collect()` straight into `[1, 3, H, W]`.

use clap::Parser;
use std::path::PathBuf;

use eridiffusion_core::encoders::gemma3::{Gemma3Encoder, GEMMA3_PROMPT_LEN};
use eridiffusion_core::encoders::ltx2_vae::{Ltx2Vae, LTX2_BUCKET_DIVISIBILITY};
use flame_core::{serialization::save_file, DType, Shape, Tensor};

#[derive(Parser)]
struct Args {
    #[arg(long)] input_dir: PathBuf,
    #[arg(long)] output_dir: PathBuf,
    /// LTX-2 video VAE checkpoint (single safetensors). Stub if missing.
    #[arg(long)] vae_ckpt: PathBuf,
    /// Gemma-3 text encoder directory. Stub if Gemma3Encoder isn't ported yet.
    #[arg(long)] text_ckpt_dir: PathBuf,
    /// Tokenizer.json path for Gemma-3.
    #[arg(long)] tokenizer_path: PathBuf,
    #[arg(long, default_value = "256")] size: usize,
    #[arg(long)] skip_existing: bool,
}

fn main() -> anyhow::Result<()> {
    // Per HANDOFF_2026-05-05_PREPARE_CHW_BUG_FIXED + memory rule:
    // disable flame_core CUDA alloc pool — dataset prep is one-pass and
    // the pool grows host RSS under text-encoder forward.
    if std::env::var_os("FLAME_ALLOC_POOL").is_none() {
        // SAFETY: single-threaded at this point.
        unsafe { std::env::set_var("FLAME_ALLOC_POOL", "0"); }
    }
    env_logger::init();
    let args = Args::parse();
    std::fs::create_dir_all(&args.output_dir)?;

    if args.size % LTX2_BUCKET_DIVISIBILITY != 0 {
        anyhow::bail!(
            "--size {} must be divisible by {} (LTX-2 bucket divisibility)",
            args.size, LTX2_BUCKET_DIVISIBILITY
        );
    }

    flame_core::config::set_default_dtype(DType::BF16);
    let device = flame_core::global_cuda_device();

    log::info!("Loading LTX-2 VAE (stub-aware)...");
    let vae = Ltx2Vae::load(&args.vae_ckpt, device.clone())
        .map_err(|e| anyhow::anyhow!("Ltx2Vae::load: {e}"))?;
    if !vae.stats_loaded {
        log::warn!(
            "VAE per-channel stats NOT loaded — cached latents will use \
             (mean=0, std=1). Real training requires the real stats!"
        );
    }

    log::info!("Loading Gemma-3 text encoder (stub-aware)...");
    let te = Gemma3Encoder::load(&args.text_ckpt_dir, device.clone())
        .map_err(|e| anyhow::anyhow!("Gemma3Encoder::load: {e}"))?;

    let tokenizer = tokenizers::Tokenizer::from_file(&args.tokenizer_path)
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;

    let mut pairs = Vec::new();
    for entry in std::fs::read_dir(&args.input_dir)? {
        let path = entry?.path();
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            if matches!(ext.to_lowercase().as_str(), "jpg" | "jpeg" | "png" | "webp") {
                let stem = path.file_stem().unwrap().to_str().unwrap();
                pairs.push((path.to_path_buf(), args.input_dir.join(format!("{stem}.txt"))));
            }
        }
    }
    log::info!("Found {} image-text pairs", pairs.len());

    let mut cached = 0usize;
    for (img_path, txt_path) in &pairs {
        let hash = format!("{:x}", md5::compute(img_path.to_string_lossy().as_bytes()));
        let out_path = args.output_dir.join(format!("{hash}.safetensors"));
        if args.skip_existing && out_path.exists() { continue; }

        // Load + center-crop to square + resize.
        let img = image::open(img_path)?
            .resize_exact(args.size as u32, args.size as u32, image::imageops::FilterType::Lanczos3)
            .to_rgb32f();
        let (w, h) = img.dimensions();
        let (wu, hu) = (w as usize, h as usize);
        // CHW transpose (canonical fix — see prepare_ernie.rs:80-87).
        let mut pixels = vec![0f32; 3 * hu * wu];
        for (x, y, p) in img.enumerate_pixels() {
            let (xu, yu) = (x as usize, y as usize);
            for c in 0..3 {
                pixels[c * hu * wu + yu * wu + xu] = p.0[c] * 2.0 - 1.0;
            }
        }
        let img_t = Tensor::from_vec(pixels, Shape::from_dims(&[1, 3, hu, wu]), device.clone())?
            .to_dtype(DType::BF16)?;

        // Image → 1-frame video latent. STUB encoder; real training requires
        // a real Ltx2Vae::encode_image_as_frame implementation.
        let latent = vae
            .encode_image_as_frame(&img_t)
            .map_err(|e| anyhow::anyhow!("encode_image_as_frame: {e}"))?;

        // Caption → text tokens. Gemma-3 requires fixed 1024-token left-padded
        // input (see encoders/gemma3.rs).
        let caption = std::fs::read_to_string(txt_path).unwrap_or_default();
        let encoding = tokenizer.encode(caption.trim(), true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {e}"))?;
        let mut ids: Vec<i32> = encoding.get_ids().iter().map(|&x| x as i32).collect();
        let real_len = ids.len().min(GEMMA3_PROMPT_LEN);
        if ids.len() > GEMMA3_PROMPT_LEN { ids.truncate(GEMMA3_PROMPT_LEN); }
        // Left-pad with 0 (Gemma3 pad id). The caller (Gemma3Encoder::encode)
        // is responsible for converting this to the correct hidden states.
        let pad_n = GEMMA3_PROMPT_LEN - ids.len();
        let mut padded_ids = vec![0i32; pad_n];
        padded_ids.extend_from_slice(&ids);

        let text_emb = te.encode(&padded_ids)
            .map_err(|e| anyhow::anyhow!("Gemma3 encode: {e}"))?;

        let mut tensors = std::collections::HashMap::new();
        tensors.insert("latent".to_string(), latent.to_dtype(DType::BF16)?);
        tensors.insert("text_embedding".to_string(), text_emb.to_dtype(DType::BF16)?);
        let real_len_t = Tensor::from_vec(
            vec![real_len as f32],
            Shape::from_dims(&[1]),
            device.clone(),
        )?;
        tensors.insert("text_real_len".to_string(), real_len_t);
        save_file(&tensors, &out_path)?;
        cached += 1;

        if cached % 5 == 0 {
            log::info!("Cached {cached}/{}", pairs.len());
        }
    }
    log::info!("Done. Cached {cached} samples to {:?}", args.output_dir);
    Ok(())
}
