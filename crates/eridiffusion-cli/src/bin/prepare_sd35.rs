//! prepare_sd35 — image+caption → cached SD3.5 training samples.
//!
//! Per OT preset `#sd 3 LoRA.json` and upstream Python `BaseStableDiffusion3Setup`:
//!   - Resolution 1024 (preset default).
//!   - 16-channel SD3 VAE (LdmVAEEncoder), `(z - shift) * scale` with
//!     `scale=1.5305, shift=0.0609` (matches inference-flame `sd3_lora_infer`).
//!   - Triple text encoder: CLIP-L (768), CLIP-G (1280), T5-XXL (4096).
//!     SD3 forms `combined_context` and `combined_pooled` per
//!     `StableDiffusion3Model.combine_text_encoder_output`:
//!         clip_l_h = penultimate hidden [77, 768]
//!         clip_g_h = penultimate hidden [77, 1280]
//!         clip_lg  = cat([clip_l_h, clip_g_h], -1)         → [77, 2048]
//!         clip_lg_padded = pad_last_dim(clip_lg, 4096)     → [77, 4096]
//!         t5_h     = T5 final hidden                       → [N, 4096]   (N=77 typical)
//!         context  = cat([clip_lg_padded, t5_h], dim=1)    → [77+N, 4096]
//!         pooled   = cat([clip_l_pool, clip_g_pool], -1)   → [2048]
//!
//! Cache layout (per-sample safetensors):
//!   - `latent`         : [1, 16, H/8, W/8] BF16
//!   - `text_embedding` : [1, seq, 4096]    BF16   (combined CLIP+T5 context)
//!   - `pooled`         : [1, 2048]         BF16   (concat CLIP-L pool + CLIP-G pool)
//!
//! Hard constraints (project memory):
//!   - `FLAME_ALLOC_POOL=0` set in-binary so dataset prep doesn't OOM the box.
//!   - CHW transpose via `enumerate_pixels` (HWC → CHW); plain `.flat_map`
//!     scrambles channels (memory: prepare_klein.rs:108-126 fix).

use clap::Parser;
use flame_core::{serialization::save_file, DType, Shape, Tensor};
use eridiffusion_core::encoders::{
    clip_g::ClipGEncoder,
    clip_l::{ClipConfig, ClipEncoder},
    ldm_vae::LdmVAEEncoder,
    t5_xxl::T5Encoder,
};
use std::collections::HashMap;
use std::path::PathBuf;

const CLIP_MAX_LEN: usize = 77;
// Per-encoder pad ids — see SDXL audit H1 / prepare_sdxl.rs:36-38.
const CLIP_L_PAD_ID: i32 = 49407;
const CLIP_G_PAD_ID: i32 = 0;

const VAE_LATENT_CHANNELS: usize = 16;
const VAE_SCALE: f32 = 1.5305;   // SD3 VAE scaling_factor
const VAE_SHIFT: f32 = 0.0609;   // SD3 VAE shift_factor

#[derive(Parser)]
struct Args {
    #[arg(long)] input_dir: PathBuf,
    #[arg(long)] output_dir: PathBuf,

    /// SD3.5 base safetensors (carries the 16ch VAE under `first_stage_model.`)
    /// or a separate VAE-only safetensors.
    #[arg(long)] vae_ckpt: PathBuf,

    /// CLIP-L weights (HF `text_encoder/`).
    #[arg(long)] clip_l_ckpt: PathBuf,
    /// CLIP-G weights (HF `text_encoder_2/`).
    #[arg(long)] clip_g_ckpt: PathBuf,
    /// T5-XXL weights (HF `text_encoder_3/`).
    #[arg(long)] t5_ckpt: PathBuf,

    #[arg(long)] clip_l_tokenizer: PathBuf,
    #[arg(long)] clip_g_tokenizer: PathBuf,
    #[arg(long)] t5_tokenizer: PathBuf,

    #[arg(long, default_value = "1024")] resolution: u32,
    /// T5 max sequence length. SD3 reference uses 256 (lower than the
    /// 512 T5-XXL trains at) for VRAM. 256 matches diffusers SD3 default.
    #[arg(long, default_value = "256")] t5_max_len: usize,
    #[arg(long, default_value_t = true)] skip_existing: bool,
    #[arg(long, default_value_t = 0)] max_samples: usize,
}

fn load_one_or_dir(
    path: &std::path::Path,
    device: &std::sync::Arc<flame_core::CudaDevice>,
) -> flame_core::Result<HashMap<String, Tensor>> {
    if path.is_file() { return flame_core::serialization::load_file(path, device); }
    let mut all = HashMap::new();
    let mut entries: Vec<PathBuf> = std::fs::read_dir(path)
        .map_err(|e| flame_core::Error::Io(format!("read_dir: {e}")))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("safetensors"))
        .collect();
    entries.sort();
    for p in entries {
        all.extend(flame_core::serialization::load_file(&p, device)?);
    }
    Ok(all)
}

fn tokenize_clip(tok: &tokenizers::Tokenizer, text: &str, pad_id: i32) -> anyhow::Result<Vec<i32>> {
    let enc = tok.encode(text, true).map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
    let mut ids: Vec<i32> = enc.get_ids().iter().map(|&x| x as i32).collect();
    if ids.len() > CLIP_MAX_LEN { ids.truncate(CLIP_MAX_LEN); }
    while ids.len() < CLIP_MAX_LEN { ids.push(pad_id); }
    Ok(ids)
}

fn tokenize_t5(
    tok: &tokenizers::Tokenizer, text: &str, max_len: usize,
) -> anyhow::Result<Vec<i32>> {
    let enc = tok.encode(text, true).map_err(|e| anyhow::anyhow!("t5 tokenize: {e}"))?;
    let mut ids: Vec<i32> = enc.get_ids().iter().map(|&x| x as i32).collect();
    if ids.len() > max_len { ids.truncate(max_len); }
    while ids.len() < max_len { ids.push(0); } // T5 pad id = 0
    Ok(ids)
}

fn main() -> anyhow::Result<()> {
    // Disable flame_core's CUDA alloc pool to keep host RSS flat across the
    // dataset (memory: feedback_prepare_bins_pool_off / prepare_klein.rs).
    if std::env::var_os("FLAME_ALLOC_POOL").is_none() {
        // SAFETY: single-threaded at startup.
        unsafe { std::env::set_var("FLAME_ALLOC_POOL", "0"); }
    }
    env_logger::init();
    let args = Args::parse();
    std::fs::create_dir_all(&args.output_dir)?;
    flame_core::config::set_default_dtype(DType::BF16);
    let _no_grad = flame_core::autograd::AutogradContext::no_grad();
    let device = flame_core::global_cuda_device();

    log::info!("[1/5] Loading SD3.5 VAE encoder (16-ch latent, scale={}, shift={})",
        VAE_SCALE, VAE_SHIFT);
    let vae = LdmVAEEncoder::from_safetensors(
        args.vae_ckpt.to_str().ok_or_else(|| anyhow::anyhow!("vae path utf8"))?,
        VAE_LATENT_CHANNELS,
        &device,
    )?;

    log::info!("[2/5] Loading CLIP-L (768-d, 12L, quick_gelu)...");
    let clip_l_w = load_one_or_dir(&args.clip_l_ckpt, &device)?;
    let clip_l = ClipEncoder::new(clip_l_w, ClipConfig::default(), device.clone());

    log::info!("[3/5] Loading CLIP-G (1280-d, 32L, gelu)...");
    let clip_g_w = load_one_or_dir(&args.clip_g_ckpt, &device)?;
    let clip_g = ClipGEncoder::new(clip_g_w, device.clone());

    log::info!("[4/5] Loading T5-XXL (4096-d, 24L)...");
    let mut t5 = T5Encoder::load(
        args.t5_ckpt.to_str().ok_or_else(|| anyhow::anyhow!("t5 path utf8"))?,
        &device,
    )?;

    let tok_l = tokenizers::Tokenizer::from_file(&args.clip_l_tokenizer)
        .map_err(|e| anyhow::anyhow!("clip_l tokenizer: {e}"))?;
    let tok_g = tokenizers::Tokenizer::from_file(&args.clip_g_tokenizer)
        .map_err(|e| anyhow::anyhow!("clip_g tokenizer: {e}"))?;
    let tok_t5 = tokenizers::Tokenizer::from_file(&args.t5_tokenizer)
        .map_err(|e| anyhow::anyhow!("t5 tokenizer: {e}"))?;

    log::info!("[5/5] Encoding samples at {}²...", args.resolution);
    let mut pairs: Vec<(PathBuf, PathBuf)> = Vec::new();
    for entry in std::fs::read_dir(&args.input_dir)? {
        let p = entry?.path();
        if let Some(ext) = p.extension().and_then(|e| e.to_str()) {
            if matches!(ext.to_lowercase().as_str(), "jpg" | "jpeg" | "png" | "webp") {
                let stem = p.file_stem().unwrap().to_str().unwrap().to_string();
                pairs.push((p.clone(), args.input_dir.join(format!("{stem}.txt"))));
            }
        }
    }
    pairs.sort();
    if args.max_samples > 0 { pairs.truncate(args.max_samples); }
    log::info!("Found {} image-caption pairs", pairs.len());

    let mut cached = 0usize;
    for (img_path, txt_path) in &pairs {
        let hash = format!("{:x}", md5::compute(img_path.to_string_lossy().as_bytes()));
        let out_path = args.output_dir.join(format!("{hash}.safetensors"));
        if args.skip_existing && out_path.exists() { continue; }

        // -- Image → VAE latent --
        let img = image::open(img_path)?
            .resize_exact(args.resolution, args.resolution, image::imageops::FilterType::Lanczos3)
            .to_rgb32f();
        let (w, h) = img.dimensions();
        // CHW transpose. enumerate_pixels gives HWC interleaved; manual
        // index math is the only reliable conversion to [1, 3, H, W] CHW.
        // Pixel range -1..1.
        let (wu, hu) = (w as usize, h as usize);
        let mut pixels = vec![0f32; 3 * hu * wu];
        for (x, y, p) in img.enumerate_pixels() {
            let (xu, yu) = (x as usize, y as usize);
            for c in 0..3 { pixels[c * hu * wu + yu * wu + xu] = p.0[c] * 2.0 - 1.0; }
        }
        let img_t = Tensor::from_vec(pixels, Shape::from_dims(&[1, 3, hu, wu]), device.clone())?
            .to_dtype(DType::BF16)?;
        // SD3 cache convention: store the SCALED latent (post `(z-shift)*scale`)
        // so the trainer can use it directly without re-applying the
        // scaling each step. Matches upstream Python `scaled_latent_image` in the
        // training loop.
        let latent = vae.encode_scaled(&img_t, VAE_SCALE, VAE_SHIFT)?;

        // -- Caption → triple-encoder context + pooled --
        let caption = std::fs::read_to_string(txt_path).unwrap_or_default();
        let caption = caption.trim();

        // CLIP-L: penultimate hidden + pooled (encode_sd3 returns both).
        let ids_l = tokenize_clip(&tok_l, caption, CLIP_L_PAD_ID)?;
        let (clip_l_h, clip_l_pool) = clip_l.encode_sd3(&ids_l)?;
        // CLIP-G: penultimate hidden + projected pooled.
        let ids_g = tokenize_clip(&tok_g, caption, CLIP_G_PAD_ID)?;
        let (clip_g_h, clip_g_pool) = clip_g.encode_sdxl(&ids_g)?;
        // T5: full hidden, no pooled.
        let ids_t5 = tokenize_t5(&tok_t5, caption, args.t5_max_len)?;
        // Encode_with_len honors --t5-max-len; plain encode() previously
        // re-padded to cfg.max_seq_len=512 ignoring the CLI value, producing
        // [1, 589, 4096] cache instead of upstream Python's [1, 154, 4096].
        let t5_h = t5.encode_with_len(&ids_t5, args.t5_max_len)?;

        // Build SD3 combined context per
        // `StableDiffusion3Model.combine_text_encoder_output`:
        //   clip_lg = cat([clip_l_h, clip_g_h], -1)               # [B, 77, 2048]
        //   clip_lg_pad = F.pad(clip_lg, (0, 4096 - 2048))         # [B, 77, 4096]
        //   context = cat([clip_lg_pad, t5_h], dim=1)              # [B, 77+t5_max_len, 4096]
        let clip_lg = Tensor::cat(&[&clip_l_h, &clip_g_h], 2)?;
        // Pad last dim 2048 → 4096 with zeros.
        let pad_zeros = Tensor::zeros_dtype(
            Shape::from_dims(&[
                clip_lg.dims()[0], clip_lg.dims()[1], 4096 - 2048,
            ]),
            DType::BF16,
            device.clone(),
        )?;
        let clip_lg_padded = Tensor::cat(
            &[&clip_lg.to_dtype(DType::BF16)?, &pad_zeros], 2,
        )?;
        let context = Tensor::cat(
            &[&clip_lg_padded, &t5_h.to_dtype(DType::BF16)?], 1,
        )?.to_dtype(DType::BF16)?;

        let pooled = Tensor::cat(&[&clip_l_pool, &clip_g_pool], 1)?
            .to_dtype(DType::BF16)?; // [1, 2048]

        let mut out = HashMap::new();
        out.insert("latent".to_string(), latent.to_dtype(DType::BF16)?);
        out.insert("text_embedding".to_string(), context);
        out.insert("pooled".to_string(), pooled);
        save_file(&out, &out_path)?;
        cached += 1;
        if cached == 1 || cached % 5 == 0 {
            log::info!("  cached {cached}/{}", pairs.len());
        }
    }

    log::info!("Done. Cached {cached} samples to {:?}", args.output_dir);
    Ok(())
}
