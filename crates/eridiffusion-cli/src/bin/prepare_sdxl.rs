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
    /// Image augmentations at prep time. All default-off → byte-identical
    /// caches. Set `--aug-flip` for 50% horizontal flip; `--aug-brightness`
    /// and `--aug-contrast` jitter pixel values uniformly. `--aug-seed`
    /// seeds the per-sample RNG.
    #[arg(long, default_value_t = false)] aug_flip: bool,
    #[arg(long, default_value_t = 0.0)] aug_brightness: f32,
    #[arg(long, default_value_t = 0.0)] aug_contrast: f32,
    #[arg(long, default_value_t = 0)] aug_seed: u64,
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

// CLIP EOS token id. Both CLIP-L and CLIP-G tokenizers use 49407 as EOS;
// only the pad-id differs (CLIP-L pads with EOS, CLIP-G pads with 0).
const CLIP_EOS_ID: i32 = 49407;

fn tokenize(tok: &tokenizers::Tokenizer, text: &str, pad_id: i32) -> anyhow::Result<Vec<i32>> {
    let enc = tok.encode(text, true).map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
    let mut ids: Vec<i32> = enc.get_ids().iter().map(|&x| x as i32).collect();
    // SDXL audit CRIT-2: HF CLIPTokenizer with `truncation=True, max_length=77`
    // guarantees `[BOS, ...75 content tokens..., EOS]`. The raw `tokenizers`
    // crate doesn't auto-truncate, so a >77-token sequence has its trailing
    // EOS sliced off by `truncate(77)`. Downstream `text_projection` finds
    // EOS via `position(== 49407)`; a missing EOS makes it gather pooled
    // output at a pad slot (or the last token), corrupting CLIP-G pooled.
    // Fix: preserve EOS at slot 76 when truncating long captions.
    if ids.len() > CLIP_MAX_LEN {
        ids.truncate(CLIP_MAX_LEN - 1);
        ids.push(CLIP_EOS_ID);
    }
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
    // SDXL audit HIGH-2 / MED-4: OT preset pins `vae.weight_dtype = FLOAT_32`
    // because the SDXL VAE is FP16-broken in mid-block attention
    // (huggingface/diffusers#3994). EDv2 currently runs the VAE encode in
    // BF16 throughout — flame-core's `Conv2d` is BF16-only at the kernel
    // level (`flame-core/src/conv.rs:330` rejects non-BF16 input), and
    // bumping it to F32 requires NHWC F32 conv kernels we don't yet have.
    //
    // Effect: cached latents diverge from OT-encoded latents at the ~0.5–1%
    // level. Trainer + sampler use the same biased VAE so LoRAs converge
    // fine internally, but cached latents are NOT byte-portable to/from
    // OneTrainer or Kohya pipelines. Color-sensitive datasets may see
    // visible drift in dark regions.
    //
    // TODO(flame-core): F32 conv path → drop the BF16 cast at line 200,
    // load VAE weights at F32, match OT bit-for-bit.
    log::warn!("[VAE] running encode in BF16 (flame-core Conv2d limitation). \
        Latents will diverge from OT F32 reference at ~0.5-1%. See \
        prepare_sdxl.rs source for the TODO.");
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

    let mut cached = 0usize;
    for (idx, (img_path, txt_path)) in pairs.iter().enumerate() {
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
