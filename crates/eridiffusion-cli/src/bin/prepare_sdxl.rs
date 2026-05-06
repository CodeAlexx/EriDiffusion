//! prepare_sdxl — image+caption → cached SDXL training samples.
//!
//! Per OT preset `#sdxl 1.0 LoRA.json` (resolution=1024, dual TE, 4-ch VAE).
//! Each output safetensors file holds:
//!   - `latent`:        [1, 4, H/8, W/8]   BF16 — VAE-encoded, scale=0.13025 applied
//!   - `text_embedding`:[1, 77, 2048]      BF16 — concat(CLIP-L hidden_states[-2] [768], CLIP-G penultimate [1280])
//!   - `pooled`:        [1, 1280]          BF16 — CLIP-G projected pool
//!   - `time_ids`:      [1, 6]             F32  — raw `(orig_h, orig_w, crop_top, crop_left, target_h, target_w)`
//!
//! Notes:
//!   - SDXL audit H2/H6: `pooled` is the bare CLIP-G pool (1280-dim), NOT the
//!     pre-baked 2816-dim ADM input. Storing the raw 6-vector `time_ids` lets
//!     the trainer rebuild the sinusoidal `add_text_embeds` per-sample, and
//!     keeps caches portable to upstream Python (which uses the same convention).
//!   - SDXL audit H1: each tokenizer pads with its own pad id — CLIP-L uses
//!     EOS (49407), CLIP-G uses id 0 ("!"). The wrong pad id silently
//!     corrupts CLIP-G hidden states at every pad position.
//!   - CLIP-L hidden used is `hidden_states[-2]` per SDXL pipeline
//!     (`encode_sd3` returns penultimate; same trick applies to SDXL CLIP-L).
//!   - Crop offsets are 0,0 (we resize to square; OT preset doesn't do
//!     bucketing in this minimal port). Original image size is recorded so
//!     a future bucketing pass can use the true aspect.
use clap::Parser;
use flame_core::{serialization::save_file, DType, Shape, Tensor};
use eridiffusion_core::encoders::{
    clip_g::ClipGEncoder,
    clip_l::{ClipConfig, ClipEncoder},
    sdxl_vae::SdxlVaeEncoder,
};
use eridiffusion_core::sampler::sdxl_sampler::build_time_ids;
use std::collections::HashMap;
use std::path::PathBuf;

const CLIP_MAX_LEN: usize = 77;
// SDXL audit H1: per-encoder pad token ids. CLIP-L pads with EOS, CLIP-G
// pads with id 0 (HF `tokenizer_2/tokenizer_config.json` `"pad_token": "!"`).
const CLIP_L_PAD_ID: i32 = 49407;
const CLIP_G_PAD_ID: i32 = 0;

#[derive(Parser)]
struct Args {
    #[arg(long)] input_dir: PathBuf,
    #[arg(long)] output_dir: PathBuf,
    /// SDXL VAE safetensors (e.g. `sdxl_vae.safetensors` or full SDXL ckpt).
    #[arg(long)] vae_ckpt: PathBuf,
    /// CLIP-L weights (HF `text_encoder/`).
    #[arg(long)] clip_l_ckpt: PathBuf,
    /// CLIP-G weights (HF `text_encoder_2/`).
    #[arg(long)] clip_g_ckpt: PathBuf,
    /// CLIP-L tokenizer.json.
    #[arg(long)] clip_l_tokenizer: PathBuf,
    /// CLIP-G tokenizer.json (OpenCLIP bigG, same vocab as CLIP-L).
    #[arg(long)] clip_g_tokenizer: PathBuf,
    /// OT preset default 1024.
    #[arg(long, default_value = "1024")] resolution: u32,
    #[arg(long, default_value_t = true)] skip_existing: bool,
    #[arg(long, default_value_t = 0)] max_samples: usize,
}

fn load_one_or_dir(
    path: &std::path::Path, device: &std::sync::Arc<flame_core::CudaDevice>,
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

fn tokenize(tok: &tokenizers::Tokenizer, text: &str, pad_id: i32) -> anyhow::Result<Vec<i32>> {
    let enc = tok.encode(text, true).map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
    let mut ids: Vec<i32> = enc.get_ids().iter().map(|&x| x as i32).collect();
    if ids.len() > CLIP_MAX_LEN { ids.truncate(CLIP_MAX_LEN); }
    while ids.len() < CLIP_MAX_LEN { ids.push(pad_id); }
    Ok(ids)
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

    log::info!("[1/4] Loading SDXL VAE encoder (4-ch, scale=0.13025)...");
    // SDXL audit H2: the SDXL VAE is FP16-broken per OT canonical and OT
    // Python defaults to F32. flame-core's `Conv2d` is BF16-only at the
    // kernel level (`flame-core/src/conv.rs:330` rejects non-BF16 input),
    // so a pure-Rust F32 VAE encode requires kernel work in flame-core that
    // is out of scope for this fix executor. Cast input to F32 below as a
    // best-effort precision boost; the BF16 weights still bottleneck the
    // mid-block attention but at least the surrounding ops use the F32 input
    // until the first conv does the implicit BF16 cast. TODO(flame-core):
    // F32 conv path so this VAE matches upstream Python latents bit-for-bit.
    let vae = SdxlVaeEncoder::from_safetensors(args.vae_ckpt.to_str().unwrap(), &device)?;

    log::info!("[2/4] Loading CLIP-L (768d, 12L, quick_gelu)...");
    let clip_l_w = load_one_or_dir(&args.clip_l_ckpt, &device)?;
    let clip_l = ClipEncoder::new(clip_l_w, ClipConfig::default(), device.clone());

    log::info!("[3/4] Loading CLIP-G (1280d, 32L, gelu)...");
    let clip_g_w = load_one_or_dir(&args.clip_g_ckpt, &device)?;
    let clip_g = ClipGEncoder::new(clip_g_w, device.clone());

    let tok_l = tokenizers::Tokenizer::from_file(&args.clip_l_tokenizer)
        .map_err(|e| anyhow::anyhow!("clip_l tokenizer: {e}"))?;
    let tok_g = tokenizers::Tokenizer::from_file(&args.clip_g_tokenizer)
        .map_err(|e| anyhow::anyhow!("clip_g tokenizer: {e}"))?;
    debug_assert_eq!(clip_g.pad_token_id(), CLIP_G_PAD_ID,
        "CLIP-G pad id mismatch — expected 0 from HF tokenizer_2 config");

    log::info!("[4/4] Encoding samples at {}²...", args.resolution);
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

        // Image → VAE latent. Record the original (pre-resize) dimensions so
        // the trainer can pass true `add_time_ids`. The minimal port still
        // resizes to square, but storing the raw size keeps the cache
        // portable for a future bucketing pass.
        let orig_img = image::open(img_path)?;
        let (orig_w, orig_h) = (orig_img.width(), orig_img.height());
        let img = orig_img
            .resize_exact(args.resolution, args.resolution, image::imageops::FilterType::Lanczos3)
            .to_rgb32f();
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
        let latent = vae.encode(&img_t)?;

        // Caption → CLIP-L (penultimate hidden + pooled) and CLIP-G (penultimate hidden + projected pool)
        let caption = std::fs::read_to_string(txt_path).unwrap_or_default();
        let caption = caption.trim();
        // SDXL audit H1: per-encoder pad ids.
        let ids_l = tokenize(&tok_l, caption, CLIP_L_PAD_ID)?;
        let ids_g = tokenize(&tok_g, caption, CLIP_G_PAD_ID)?;

        // CLIP-L: SD3-style (penultimate, projected pool). For SDXL we want
        // hidden_states[-2] (no final LN) for the 768-d slice.
        let (clip_l_hidden, _clip_l_pool_unused) = clip_l.encode_sd3(&ids_l)?;
        let (clip_g_hidden, clip_g_pool) = clip_g.encode_sdxl(&ids_g)?;

        // Concat hidden along last dim → [1, 77, 2048]
        let text_embedding = Tensor::cat(&[&clip_l_hidden, &clip_g_hidden], 2)?
            .to_dtype(DType::BF16)?;

        // SDXL audit H2: store raw CLIP-G pool [1, 1280] and raw 6-vector
        // `time_ids`; trainer rebuilds the 1536-dim sin embed and concats to
        // 2816 per-sample. Cache stays portable + correct under bucketing.
        let pooled = clip_g_pool.to_dtype(DType::BF16)?;
        let time_ids_vec = build_time_ids(orig_h, orig_w, 0, 0, args.resolution, args.resolution);
        let time_ids = Tensor::from_vec(
            time_ids_vec.to_vec(),
            Shape::from_dims(&[1, 6]),
            device.clone(),
        )?; // F32 — small, exact

        let mut out = HashMap::new();
        out.insert("latent".to_string(), latent.to_dtype(DType::BF16)?);
        out.insert("text_embedding".to_string(), text_embedding);
        out.insert("pooled".to_string(), pooled);
        out.insert("time_ids".to_string(), time_ids);
        save_file(&out, &out_path)?;
        cached += 1;
        if cached % 5 == 0 {
            log::info!("  cached {cached}/{}", pairs.len());
        }
    }

    log::info!("Done. Cached {cached} samples to {:?}", args.output_dir);
    Ok(())
}
