//! prepare_chroma — image+caption → cached latents+T5 embeddings for Chroma LoRA training.
//!
//! Chroma is a FLUX.1-derived DiT (Lodestone Rock's Chroma1-HD). It uses the
//! same Flux VAE (16-ch LDM, scale=0.3611, shift=0.1159) and T5-XXL text
//! encoder, but has NO CLIP branch — the modulation comes from a small
//! "approximator" MLP fed the timestep, not pooled CLIP. Cache shape:
//!
//!   - `latent`:    BF16 [1, 16, H/8, W/8] — RAW Flux VAE posterior
//!   - `t5_embed`:  BF16 [1, T5_MAX_LEN, 4096] — T5-XXL hidden states
//!
//! Latent convention matches `prepare_flux` (audit fix FLUX_VERIFY §H3): cache
//! stores RAW posterior. `(latent - SHIFT) * SCALE` happens at training time.
//!
//! Differences vs `prepare_flux`:
//!   - No CLIP encoder, no `clip_pool` field.
//!   - T5 padded to 512 tokens (matches `sample_chroma.rs` and the archive
//!     trainer's `T5_SEQ_LEN`).

use clap::Parser;
use flame_core::{serialization::save_file, DType, Shape, Tensor};
use eridiffusion_core::encoders::{
    flux_vae::{FluxVaeEncoder, LATENT_CHANNELS},
    t5_xxl::T5Encoder,
};
use std::collections::HashMap;
use std::path::PathBuf;

const T5_MAX_LEN: usize = 512;

#[derive(Parser)]
struct Args {
    #[arg(long)] input_dir: PathBuf,
    #[arg(long)] output_dir: PathBuf,
    /// Flux VAE safetensors (Chroma uses the same VAE — `ae.safetensors` or
    /// the diffusers-format VAE shipped with Chroma1-HD).
    #[arg(long)] vae_ckpt: PathBuf,
    /// T5-XXL weights (single file or directory of shards).
    #[arg(long)] t5_ckpt: PathBuf,
    /// T5 tokenizer.json.
    #[arg(long)] t5_tokenizer: PathBuf,
    /// Default 1024 (chroma1-HD's native training resolution). Must be a
    /// multiple of 16 (patch=2 × VAE 8 = 16).
    #[arg(long, default_value = "1024")] resolution: u32,
    #[arg(long, default_value_t = true)] skip_existing: bool,
    #[arg(long, default_value_t = 0)] max_samples: usize,
    /// Image augmentations at prep time. All default-off → byte-identical
    /// caches. Set `--aug-flip` for 50% horizontal flip; `--aug-brightness`
    /// and `--aug-contrast` jitter pixel values uniformly. `--aug-seed`
    /// seeds the per-sample RNG.
    #[arg(long, default_value_t = false)] aug_flip: bool,
    #[arg(long, default_value_t = 0.0)] aug_brightness: f32,
    #[arg(long, default_value_t = 0.0)] aug_contrast: f32,
    #[arg(long, default_value_t = 0)] aug_seed: u64,
}

fn main() -> anyhow::Result<()> {
    // Disable flame_core CUDA alloc pool — see prepare_klein.rs for full
    // rationale. Dataset prep is one-pass; the pool retains slabs and
    // grows host RSS by ~1 GB per sample at 512² with text-encoder forward,
    // OOM-killing the box around sample 75 on 62 GB. Pool off → flat RSS.
    if std::env::var_os("FLAME_ALLOC_POOL").is_none() {
        // SAFETY: single-threaded at this point.
        unsafe { std::env::set_var("FLAME_ALLOC_POOL", "0"); }
    }
    env_logger::init();
    let args = Args::parse();
    if args.resolution % 16 != 0 {
        anyhow::bail!("--resolution must be a multiple of 16 (patch=2 × VAE 8), got {}", args.resolution);
    }
    std::fs::create_dir_all(&args.output_dir)?;
    flame_core::config::set_default_dtype(DType::BF16);
    let _no_grad = flame_core::autograd::AutogradContext::no_grad();
    let device = flame_core::global_cuda_device();

    log::info!("[1/3] Loading Flux VAE encoder (16-ch LDM)...");
    let vae = FluxVaeEncoder::from_safetensors(
        args.vae_ckpt.to_str().unwrap(),
        LATENT_CHANNELS,
        &device,
    )?;

    log::info!("[2/3] Loading T5-XXL (4096-d, 24L)...");
    let mut t5 = T5Encoder::load(args.t5_ckpt.to_str().unwrap(), &device)?;

    let t5_tok = tokenizers::Tokenizer::from_file(&args.t5_tokenizer)
        .map_err(|e| anyhow::anyhow!("T5 tokenizer: {e}"))?;

    log::info!("[3/3] Encoding samples at {}²...", args.resolution);
    let mut pairs = Vec::new();
    for entry in std::fs::read_dir(&args.input_dir)? {
        let p = entry?.path();
        if let Some(ext) = p.extension().and_then(|e| e.to_str()) {
            if matches!(ext.to_lowercase().as_str(), "jpg" | "jpeg" | "png" | "webp" | "bmp") {
                let stem = p.file_stem().unwrap().to_str().unwrap();
                pairs.push((p.clone(), args.input_dir.join(format!("{stem}.txt"))));
            }
        }
    }
    log::info!("Found {} image-caption pairs", pairs.len());
    if args.max_samples > 0 && pairs.len() > args.max_samples {
        pairs.truncate(args.max_samples);
    }

    let aug_cfg = eridiffusion_core::training::features::image_aug::AugConfig {
        flip: args.aug_flip,
        brightness: args.aug_brightness,
        contrast: args.aug_contrast,
    };
    if aug_cfg.is_active() {
        log::info!(
            "[image-aug] flip={} brightness={} contrast={} seed={}",
            aug_cfg.flip, aug_cfg.brightness, aug_cfg.contrast, args.aug_seed
        );
    }

    // Cache version sentinel. Mirrors prepare_anima — Chroma cache v1 is the
    // initial published format (raw VAE posterior + T5 hidden, no CLIP).
    const CACHE_VERSION: u32 = 1;
    let meta = format!(
        r#"{{"version": {}, "format": "chroma-v1", "fields": ["latent", "t5_embed"]}}"#,
        CACHE_VERSION
    );
    let _ = std::fs::write(args.output_dir.join("_meta.json"), &meta);

    let mut cached = 0usize;
    for (idx, (img_path, txt_path)) in pairs.iter().enumerate() {
        let hash = format!("{:x}", md5::compute(img_path.to_string_lossy().as_bytes()));
        let out_path = args.output_dir.join(format!("{hash}.safetensors"));
        if args.skip_existing && out_path.exists() { continue; }

        // ── Image → VAE latent (RAW, no shift/scale, no patchify) ──
        let img = image::open(img_path)?
            .resize_exact(args.resolution, args.resolution, image::imageops::FilterType::Lanczos3)
            .to_rgb32f();
        let mut img = img;
        if aug_cfg.is_active() {
            use rand::SeedableRng;
            let mut aug_rng = rand::rngs::StdRng::seed_from_u64(args.aug_seed ^ idx as u64);
            eridiffusion_core::training::features::image_aug::apply_augs(
                &mut img,
                None,
                &aug_cfg,
                &mut aug_rng,
            );
        }
        let (w, h) = img.dimensions();
        // CHW transpose — see prepare_klein.rs for full bug writeup. Without
        // this, image::pixels() (HWC interleaved) reshaped to [1, 3, H, W]
        // (CHW) scrambles channels and the VAE silently encodes garbage.
        let (wu, hu) = (w as usize, h as usize);
        let mut pixels = vec![0f32; 3 * hu * wu];
        for (x, y, p) in img.enumerate_pixels() {
            let (xu, yu) = (x as usize, y as usize);
            for c in 0..3 { pixels[c * hu * wu + yu * wu + xu] = p.0[c] * 2.0 - 1.0; }
        }
        let img_t = Tensor::from_vec(pixels, Shape::from_dims(&[1, 3, hu, wu]), device.clone())?
            .to_dtype(DType::BF16)?;
        // RAW posterior: trainer applies `(latent - SHIFT) * SCALE` per step
        // (matches FLUX_VERIFY §H3 / SKEPTIC §H3). Archive's prepare_dataset
        // pre-scaled latents; we deliberately diverge so all FLUX-family
        // caches (Klein/Flux/Chroma) share one contract.
        let latent_raw = vae.encode(&img_t)?; // [1, 16, H/8, W/8]

        // ── Caption → T5 ──
        let caption = std::fs::read_to_string(txt_path).unwrap_or_default();
        let caption = caption.trim();

        let t5_enc = t5_tok.encode(caption, true)
            .map_err(|e| anyhow::anyhow!("T5 tokenize: {e}"))?;
        let mut t5_ids: Vec<i32> = t5_enc.get_ids().iter().map(|&x| x as i32).collect();
        if t5_ids.len() > T5_MAX_LEN { t5_ids.truncate(T5_MAX_LEN); }
        // T5Encoder::encode pads to max_seq_len (512) internally with id=0.
        let t5_embed = t5.encode(&t5_ids)?;

        let mut tensors = HashMap::new();
        tensors.insert("latent".to_string(), latent_raw.to_dtype(DType::BF16)?);
        tensors.insert("t5_embed".to_string(), t5_embed.to_dtype(DType::BF16)?);
        save_file(&tensors, &out_path)?;
        cached += 1;

        if cached % 5 == 0 || cached == pairs.len() {
            log::info!("Cached {}/{}", cached, pairs.len());
        }
    }

    log::info!("Done. {} samples cached to {:?}", cached, args.output_dir);
    Ok(())
}
