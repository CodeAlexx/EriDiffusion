//! prepare_flux — image+caption → cached latents+embeddings for FLUX.1 (Dev/Schnell) LoRA training.
//!
//! Mirrors prepare_ernie / prepare_klein structure. Per sample (one safetensors in `--output-dir`):
//!   - `latent`:    BF16 [1, 16, H/8, W/8] — RAW Flux VAE posterior (NO shift/scale, NO patchify)
//!   - `t5_embed`:  BF16 [1, 512, 4096]    — T5-XXL @ 512 tokens (BFL convention)
//!   - `clip_pool`: BF16 [1, 768]          — CLIP-L pooled (used as `vector` input to DiT)
//!
//! T5 tokenizer:   `tokenizer_2/spiece.model`-derived JSON (HF T5TokenizerFast)
//! CLIP tokenizer: `tokenizer/tokenizer.json` (HF CLIPTokenizer)
//!
//! Audit fix FLUX_VERIFY §H3 / SKEPTIC §H3: the cache stores RAW VAE latent
//! (`posterior.mode()`), no shift/scale, no patchify, no img/txt position IDs.
//! `(latent - SHIFT) * SCALE` and `pack_latents` happen at training time
//! (matches upstream Python contract — `BaseFluxSetup.py:235`).

use clap::Parser;
use flame_core::{serialization::save_file, DType, Shape, Tensor};
use eridiffusion_core::encoders::{
    clip_l::{ClipConfig, ClipEncoder},
    flux_vae::{FluxVaeEncoder, LATENT_CHANNELS},
    t5_xxl::T5Encoder,
};
use std::collections::HashMap;
use std::path::PathBuf;

const T5_MAX_LEN: usize = 512;
const CLIP_MAX_LEN: usize = 77;
const CLIP_PAD_ID: i32 = 49407; // CLIPTokenizer pads with eos_token_id

#[derive(Parser)]
struct Args {
    #[arg(long)] input_dir: PathBuf,
    #[arg(long)] output_dir: PathBuf,
    /// Flux VAE safetensors (FLUX.1-dev/ae.safetensors).
    #[arg(long)] vae_ckpt: PathBuf,
    /// T5-XXL weights (single file or directory of shards).
    #[arg(long)] t5_ckpt: PathBuf,
    /// CLIP-L weights (single file).
    #[arg(long)] clip_ckpt: PathBuf,
    /// T5 tokenizer.json.
    #[arg(long)] t5_tokenizer: PathBuf,
    /// CLIP-L tokenizer.json.
    #[arg(long)] clip_tokenizer: PathBuf,
    /// OT preset default 768 ("#flux LoRA.json").
    #[arg(long, default_value = "768")] resolution: u32,
    #[arg(long, default_value_t = true)] skip_existing: bool,
    #[arg(long, default_value_t = 0)] max_samples: usize,
}

fn load_shards(path: &std::path::Path, device: &std::sync::Arc<flame_core::CudaDevice>)
    -> flame_core::Result<HashMap<String, Tensor>>
{
    if path.is_file() {
        return flame_core::serialization::load_file(path, device);
    }
    let mut all = HashMap::new();
    let mut entries: Vec<PathBuf> = std::fs::read_dir(path)
        .map_err(|e| flame_core::Error::Io(format!("read_dir: {e}")))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("safetensors"))
        .collect();
    entries.sort();
    for p in entries {
        let part = flame_core::serialization::load_file(&p, device)?;
        all.extend(part);
    }
    Ok(all)
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
    std::fs::create_dir_all(&args.output_dir)?;
    flame_core::config::set_default_dtype(DType::BF16);
    let _no_grad = flame_core::autograd::AutogradContext::no_grad();
    let device = flame_core::global_cuda_device();

    log::info!("[1/4] Loading Flux VAE encoder (16-ch LDM)...");
    let vae = FluxVaeEncoder::from_safetensors(
        args.vae_ckpt.to_str().unwrap(),
        LATENT_CHANNELS,
        &device,
    )?;

    log::info!("[2/4] Loading T5-XXL (4096-d, 24L)...");
    let mut t5 = T5Encoder::load(args.t5_ckpt.to_str().unwrap(), &device)?;

    log::info!("[3/4] Loading CLIP-L (768-d, 12L)...");
    let clip_weights = load_shards(&args.clip_ckpt, &device)?;
    let clip = ClipEncoder::new(clip_weights, ClipConfig::default(), device.clone());

    let t5_tok = tokenizers::Tokenizer::from_file(&args.t5_tokenizer)
        .map_err(|e| anyhow::anyhow!("T5 tokenizer: {e}"))?;
    let clip_tok = tokenizers::Tokenizer::from_file(&args.clip_tokenizer)
        .map_err(|e| anyhow::anyhow!("CLIP tokenizer: {e}"))?;

    log::info!("[4/4] Encoding samples at {}²...", args.resolution);
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

    let mut cached = 0usize;
    for (img_path, txt_path) in &pairs {
        let hash = format!("{:x}", md5::compute(img_path.to_string_lossy().as_bytes()));
        let out_path = args.output_dir.join(format!("{hash}.safetensors"));
        if args.skip_existing && out_path.exists() { continue; }

        // ── Image → VAE latent → pack ──
        let img = image::open(img_path)?
            .resize_exact(args.resolution, args.resolution, image::imageops::FilterType::Lanczos3)
            .to_rgb32f();
        let (w, h) = img.dimensions();
        // [-1, 1] normalized like Klein/Ernie prepare.
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
        // Audit fix FLUX_VERIFY §H2 + §H3 / SKEPTIC §H2 + §H3: store RAW
        // posterior latent (`[1, 16, H/8, W/8]`). Shift/scale and patchify
        // happen in the trainer at flow-matching step time. The pre-fix path
        // applied `(latent - SHIFT) / SCALE` here — that's the *decode*
        // direction (BFL `autoencoder.py:308-315`). Encode is `(raw - shift) *
        // scale` (multiply, not divide). Pre-fix latents were ~7.7× the
        // variance the BFL DiT was trained on; the symmetric inversion in the
        // sampler made the bug silent (cancels for round-trip but kills LoRA
        // identity transfer because the DiT is fed garbage-magnitude latents).
        let latent_raw = vae.encode(&img_t)?; // [1, 16, H/8, W/8]

        // ── Caption → T5 + CLIP ──
        let caption = std::fs::read_to_string(txt_path).unwrap_or_default();
        let caption = caption.trim();

        let t5_enc = t5_tok.encode(caption, true)
            .map_err(|e| anyhow::anyhow!("T5 tokenize: {e}"))?;
        let mut t5_ids: Vec<i32> = t5_enc.get_ids().iter().map(|&x| x as i32).collect();
        if t5_ids.len() > T5_MAX_LEN { t5_ids.truncate(T5_MAX_LEN); }
        // T5Encoder::encode pads to max_seq_len (512) internally with id=0.
        let t5_embed = t5.encode(&t5_ids)?;

        let clip_enc = clip_tok.encode(caption, true)
            .map_err(|e| anyhow::anyhow!("CLIP tokenize: {e}"))?;
        let mut clip_ids: Vec<i32> = clip_enc.get_ids().iter().map(|&x| x as i32).collect();
        if clip_ids.len() > CLIP_MAX_LEN { clip_ids.truncate(CLIP_MAX_LEN); }
        // ClipEncoder::encode pads with eos_token_id internally.
        let _ = CLIP_PAD_ID; // documented constant; encoder pads itself
        let (_clip_hidden, clip_pool) = clip.encode(&clip_ids)?;

        let mut tensors = HashMap::new();
        tensors.insert("latent".to_string(), latent_raw.to_dtype(DType::BF16)?);
        tensors.insert("t5_embed".to_string(), t5_embed.to_dtype(DType::BF16)?);
        tensors.insert("clip_pool".to_string(), clip_pool.to_dtype(DType::BF16)?);
        save_file(&tensors, &out_path)?;
        cached += 1;

        if cached % 5 == 0 || cached == pairs.len() {
            log::info!("Cached {}/{}", cached, pairs.len());
        }
    }

    log::info!("Done. {} samples cached to {:?}", cached, args.output_dir);
    Ok(())
}
