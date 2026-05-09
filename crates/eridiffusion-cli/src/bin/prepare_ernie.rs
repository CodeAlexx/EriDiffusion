//! prepare_ernie — image+caption → cached latents+embeddings for Ernie LoRA training.
use clap::Parser;
use flame_core::{serialization::save_file, DType, Shape, Tensor};
use eridiffusion_core::encoders::{vae::KleinVaeEncoder, mistral3b::Mistral3bEncoder};
use std::path::PathBuf;

#[derive(Parser)]
struct Args {
    #[arg(long)] input_dir: PathBuf,
    #[arg(long)] output_dir: PathBuf,
    #[arg(long)] vae_ckpt: PathBuf,
    #[arg(long)] text_ckpt: PathBuf,
    #[arg(long)] tokenizer_path: PathBuf,
    #[arg(long, default_value = "128")] size: usize,
    #[arg(long)] skip_existing: bool,
    /// Image augmentations at prep time. All default-off → byte-identical
    /// caches. Set `--aug-flip` for 50% horizontal flip; `--aug-brightness`
    /// and `--aug-contrast` jitter pixel values uniformly. `--aug-seed`
    /// seeds the per-sample RNG.
    #[arg(long, default_value_t = false)] aug_flip: bool,
    #[arg(long, default_value_t = 0.0)] aug_brightness: f32,
    #[arg(long, default_value_t = 0.0)] aug_contrast: f32,
    #[arg(long, default_value_t = 0)] aug_seed: u64,
}

/// ERNIE text-encoder pad token id. Matches:
///   /home/alex/models/ERNIE-Image/text_encoder/config.json:32 ("pad_token_id": 11)
/// upstream Python relies on `tokenizer.pad_token_id` (= 11 for Mistral-3) via
/// HF tokenizer.padding('max_length'). The repo's earlier `</s>` (id=2) was
/// wrong: pads land on the EOS embedding row instead of the dedicated PAD row.
const ERNIE_PAD_ID: i32 = 11;

/// upstream Python ErnieModel.PROMPT_MAX_LENGTH (model/ErnieModel.py:5).
/// Used at BOTH train cache time (ErnieBaseDataLoader.py:34) and sample time.
const ERNIE_MAX_LEN: usize = 512;

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
    let device = flame_core::global_cuda_device();

    let dev = flame_core::Device::from(device.clone());
    let vae_weights = flame_core::serialization::load_file(&args.vae_ckpt, &device)?;
    let vae = KleinVaeEncoder::load(&vae_weights, &dev)?;
    drop(vae_weights);

    let te = Mistral3bEncoder::load(args.text_ckpt.to_str().unwrap(), &device)?;

    // Load tokenizer. Pad id = 11 per ERNIE config (NOT </s> = id 2).
    let tokenizer = tokenizers::Tokenizer::from_file(&args.tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;
    let pad_id = ERNIE_PAD_ID;
    log::info!("Tokenizer loaded, pad_id={pad_id} (ERNIE config), max_len={ERNIE_MAX_LEN}");

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

    let mut cached = 0;
    for (idx, (img_path, txt_path)) in pairs.iter().enumerate() {
        let hash = format!("{:x}", md5::compute(img_path.to_string_lossy().as_bytes()));
        let out_path = args.output_dir.join(format!("{hash}.safetensors"));
        if args.skip_existing && out_path.exists() { continue; }

        // Image → VAE latent
        let img = image::open(img_path)?
            .resize_exact(args.size as u32, args.size as u32, image::imageops::FilterType::Lanczos3)
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
        // KleinVaeEncoder::encode() ALREADY applies patchify (32→128 ch) and
        // BN normalization using bn.running_mean/var from the VAE checkpoint
        // (encoders/vae.rs:835-868). Cache shape: [B, 128, H/16, W/16] BF16
        // — matches upstream Python's post-`patchify_latents` + `scale_latents` form.
        let latent = vae.encode(&img_t)?;

        // Caption → text tokens → Mistral3B embedding
        let caption = std::fs::read_to_string(txt_path).unwrap_or_default();
        // CRITICAL: add_special_tokens=true to prepend BOS (id=1).
        // Mistral-3 has "attention sink" behavior — the first token's hidden
        // state grows to large magnitudes (~1000s) as a register for the model.
        // With BOS at position 0, BOS absorbs the spike (a content-free anchor,
        // safe to corrupt). Without BOS, the FIRST CONTENT TOKEN (e.g. "box")
        // becomes the sink and gets a 996-magnitude spike on dim 0, corrupting
        // identity-bearing conditioning. sample_ernie uses true; this MUST match.
        let encoding = tokenizer.encode(caption.trim(), true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {e}"))?;
        let mut ids: Vec<i32> = encoding.get_ids().iter().map(|&x| x as i32).collect();

        // Pad/truncate to upstream Python PROMPT_MAX_LENGTH = 512.
        if ids.len() > ERNIE_MAX_LEN { ids.truncate(ERNIE_MAX_LEN); }
        let real_len = ids.len();  // post-truncate, pre-pad: real token count
        while ids.len() < ERNIE_MAX_LEN { ids.push(pad_id); }

        // Encode with explicit pad_id so padded positions get the proper PAD
        // embedding row (and the encoder's causal mask zeroes them out).
        let txt_emb = te.encode_with_pad(&ids, ERNIE_MAX_LEN, pad_id)?;

        // Save. `real_len` lets the trainer trim padded text positions before
        // feeding the DiT (matches upstream Python text_lengths.max() trim,
        // ErnieModel.py:153-154 — DiT only sees real tokens).
        let mut tensors = std::collections::HashMap::new();
        tensors.insert("latent".to_string(), latent.to_dtype(DType::BF16)?);
        tensors.insert("text_embedding".to_string(), txt_emb.to_dtype(DType::BF16)?);
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
