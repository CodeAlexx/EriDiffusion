//! prepare_anima — image+caption → cached latents+embeddings for Anima LoRA training.
//!
//! Reference: kohya `library/strategy_anima.py` + `library/anima_utils.load_qwen3_text_encoder`.
//! Mirrors `prepare_zimage.rs` but with Anima conventions:
//!   - VAE: same `qwen_image_vae.safetensors` (16-ch latents, 8x spatial down).
//!     Reuses `LdmVAEEncoder` with `latent_channels = 16`.
//!   - Text encoder: Qwen3-0.6B (hidden=1024, num_layers=28). Returns
//!     `last_hidden_state` (final layer output, NOT layer-26 like ZImage).
//!     Uses `extract_layers = [num_layers - 1]` to match `outputs.last_hidden_state`.
//!   - LLM Adapter (Anima preview's `use_llm_adapter=True`): caches T5 token IDs
//!     too. T5 tokens are NEVER fed through T5; they go straight into the
//!     LLM Adapter's embedding table (`anima_models.LLMAdapter.embed`). So we
//!     only need a T5 *tokenizer* file, no T5 model weights.
//!
//! Output per sample (one safetensors file in `--output-dir`, name = md5 of
//! image path so partial runs are resumable):
//!   - latent:        BF16 [1, 16, H/8, W/8]   (raw VAE encode output)
//!   - text_embedding: BF16 [1, qwen3_max_len, 1024]   Qwen3-0.6B last_hidden_state
//!   - text_mask:      F32  [1, qwen3_max_len]         1.0 at valid Qwen3 tokens
//!   - t5_input_ids:   I32  [1, t5_max_len]            T5 token IDs for LLM Adapter
//!   - t5_attn_mask:   F32  [1, t5_max_len]            1.0 at valid T5 tokens

use clap::Parser;
use flame_core::{serialization::save_file, DType, Shape, Tensor};
use eridiffusion_core::encoders::{
    qwen3::Qwen3Encoder,
    wan21_encoder::Wan21VaeEncoder,
};
use std::collections::HashMap;
use std::path::PathBuf;

const QWEN3_PAD_ID: i32 = 151643; // Qwen3 default pad/eos token id
const QWEN3_MAX_LEN_DEFAULT: usize = 512;
const T5_MAX_LEN_DEFAULT: usize = 512;
const T5_PAD_ID: i32 = 0; // T5 pad token id

#[derive(Parser)]
struct Args {
    #[arg(long)] input_dir: PathBuf,
    #[arg(long)] output_dir: PathBuf,
    /// `qwen_image_vae.safetensors` from circlestone-labs/Anima split_files/vae/.
    #[arg(long)] vae_ckpt: PathBuf,
    /// `qwen_3_06b_base.safetensors` from circlestone-labs/Anima split_files/text_encoders/.
    /// Either a single safetensors file or a HF directory.
    #[arg(long)] qwen3: PathBuf,
    /// HF Qwen3-0.6B `tokenizer.json` (matches the qwen_3_06b_base shipped weights).
    #[arg(long)] tokenizer_path: PathBuf,
    /// HF T5 `tokenizer.json` (e.g. google/t5-v1_1-xxl). Required for Anima's
    /// LLM Adapter target tokens.
    #[arg(long)] t5_tokenizer_path: PathBuf,
    #[arg(long, default_value = "512")] resolution: u32,
    #[arg(long, default_value_t = QWEN3_MAX_LEN_DEFAULT)] qwen3_max_len: usize,
    #[arg(long, default_value_t = T5_MAX_LEN_DEFAULT)] t5_max_len: usize,
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
    std::fs::create_dir_all(&args.output_dir)?;
    flame_core::config::set_default_dtype(DType::BF16);
    let _no_grad = flame_core::autograd::AutogradContext::no_grad();
    let device = flame_core::global_cuda_device();

    log::info!("[1/4] Loading Wan21 VAE encoder (16-ch latents, normalized for Anima)...");
    // qwen_image_vae.safetensors is Wan-VAE in wan21 internal-key format
    // (zero-pad CausalConv3d, QwenImage convention). Anima trains against
    // pre-normalized latents — encoder applies canonical per-channel
    // (z - MEAN[c]) / STD[c] internally.
    let vae = Wan21VaeEncoder::from_safetensors(
        args.vae_ckpt.to_str().unwrap(), &device,
    )?;

    log::info!("[2/4] Loading Qwen3-0.6B text encoder (last-layer extract)...");
    let qwen_weights = load_qwen3_weights(&args.qwen3, &device)?;
    let mut qcfg = Qwen3Encoder::config_from_weights(&qwen_weights)?;
    // Anima uses outputs.last_hidden_state — final transformer layer.
    // For Qwen3-0.6B: num_layers=28 → extract_layers = [27].
    qcfg.extract_layers = vec![qcfg.num_layers - 1];
    log::info!(
        "  config: hidden={} layers={} heads={} extract={:?}",
        qcfg.hidden_size, qcfg.num_layers, qcfg.num_heads, qcfg.extract_layers,
    );
    if qcfg.hidden_size != eridiffusion_core::models::anima::CROSSATTN_EMB_CHANNELS {
        log::warn!(
            "Qwen3 hidden_size {} != Anima CROSSATTN_EMB_CHANNELS {} — DiT cross-attn shapes will mismatch",
            qcfg.hidden_size, eridiffusion_core::models::anima::CROSSATTN_EMB_CHANNELS,
        );
    }
    let qwen3 = Qwen3Encoder::new(qwen_weights, qcfg, device.clone());

    log::info!("[3/4] Loading tokenizers (Qwen3 + T5)...");
    let qwen_tok = tokenizers::Tokenizer::from_file(&args.tokenizer_path)
        .map_err(|e| anyhow::anyhow!("qwen3 tokenizer: {e}"))?;
    let t5_tok = tokenizers::Tokenizer::from_file(&args.t5_tokenizer_path)
        .map_err(|e| anyhow::anyhow!("t5 tokenizer: {e}"))?;

    log::info!("[4/4] Encoding samples...");
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
    log::info!("Found {} (image, caption) pairs", pairs.len());

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

    // Cache format sentinel. Version 2 = T5 pad rows zeroed in `text_embedding`
    // post-Qwen3 (matches Anima-Standalone-Trainer reference). Version 1 (no
    // sentinel) = pre-fix caches that silently train on non-zero pad row
    // activations through all 28 cross-attn layers. Trainer bails on legacy.
    const CACHE_VERSION: u32 = 2;
    let meta = format!(
        r#"{{"version": {}, "mask_zeroed": true, "format": "anima-v2"}}"#,
        CACHE_VERSION
    );
    let _ = std::fs::write(args.output_dir.join("_meta.json"), &meta);

    let mut written = 0usize;
    let mut skipped = 0usize;
    let t_start = std::time::Instant::now();
    for (idx, (img_path, txt_path)) in pairs.iter().enumerate() {
        if args.max_samples > 0 && written + skipped >= args.max_samples { break; }
        let hash = format!("{:x}", md5::compute(img_path.to_string_lossy().as_bytes()));
        let out_path = args.output_dir.join(format!("{hash}.safetensors"));
        if args.skip_existing && out_path.exists() { skipped += 1; continue; }

        // ── Image → VAE latent ────────────────────────────────────────────
        let img = match image::open(img_path) {
            Ok(i) => i.resize_exact(args.resolution, args.resolution, image::imageops::FilterType::Lanczos3).to_rgb32f(),
            Err(e) => { log::warn!("[{idx}] skipping {}: {e}", img_path.display()); continue; }
        };
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
        let img_t = Tensor::from_vec(
            pixels, Shape::from_dims(&[1, 3, hu, wu]), device.clone(),
        )?.to_dtype(DType::BF16)?;
        // [1, 16, H/8, W/8] with per-channel norm applied inside the encoder.
        let latent = vae.encode_image_normalized(&img_t)?;

        // ── Caption → Qwen3 hidden state + masks + T5 ids ─────────────────
        let caption = std::fs::read_to_string(txt_path).unwrap_or_default();
        let caption_str = caption.trim();

        // Qwen3 tokens (full caption, no chat template — Anima feeds raw
        // captions per `strategy_anima.AnimaTokenizeStrategy.tokenize`).
        let qwen_enc = qwen_tok.encode(caption_str, true)
            .map_err(|e| anyhow::anyhow!("qwen3 tokenize: {e}"))?;
        let mut qwen_ids: Vec<i32> = qwen_enc.get_ids().iter().map(|&i| i as i32).collect();
        let qwen_valid_len = qwen_ids.len().min(args.qwen3_max_len);
        if qwen_ids.len() > args.qwen3_max_len { qwen_ids.truncate(args.qwen3_max_len); }
        qwen_ids.resize(args.qwen3_max_len, QWEN3_PAD_ID);
        let qwen_hidden = qwen3.encode(&qwen_ids)?; // [1, qwen3_max_len, 1024]

        let mut qwen_mask = vec![0.0f32; args.qwen3_max_len];
        for slot in qwen_mask.iter_mut().take(qwen_valid_len) { *slot = 1.0; }
        let qwen_mask_t = Tensor::from_vec(
            qwen_mask, Shape::from_dims(&[1, args.qwen3_max_len]), device.clone(),
        )?;

        // Zero pad positions in text_embedding. Reference:
        // `library/strategy_anima.py:166-169` — pad rows of last_hidden_state
        // are set to 0 before the embedding leaves the strategy. Without
        // this, junk Qwen3 outputs at pad positions feed Anima's cross-attn
        // (which has no mask path) and corrupt the caption signal at every
        // training step.
        let qwen_hidden_dims = qwen_hidden.shape().dims().to_vec();
        let mask_3d = qwen_mask_t
            .reshape(&[1, args.qwen3_max_len, 1])?
            .to_dtype(qwen_hidden.dtype())?
            .broadcast_to(&Shape::from_dims(&qwen_hidden_dims))?;
        let qwen_hidden = qwen_hidden.mul(&mask_3d)?;

        // T5 tokens — used as input IDs to LLM Adapter's embedding table only.
        let t5_enc = t5_tok.encode(caption_str, true)
            .map_err(|e| anyhow::anyhow!("t5 tokenize: {e}"))?;
        let mut t5_ids: Vec<i32> = t5_enc.get_ids().iter().map(|&i| i as i32).collect();
        let t5_valid_len = t5_ids.len().min(args.t5_max_len);
        if t5_ids.len() > args.t5_max_len { t5_ids.truncate(args.t5_max_len); }
        t5_ids.resize(args.t5_max_len, T5_PAD_ID);

        let mut t5_mask = vec![0.0f32; args.t5_max_len];
        for slot in t5_mask.iter_mut().take(t5_valid_len) { *slot = 1.0; }
        // Save T5 ids as F32. Ideally these would be I32 (vocab=32128 fits
        // comfortably in i32; F32 mantissa is unsafe past 2^24, and the
        // file size is 2× larger than necessary), but the flame_core
        // safetensors loader explicitly skips I32 dtype tensors at read
        // time (see `flame-core/src/serialization.rs:520-522` —
        // `if !matches!(dtype_str, "F32" | "BF16" | "F16" | "F8_E4M3")`).
        // Switching the cache to I32 would silently drop the t5_input_ids
        // tensor when the trainer reloads the cache. Keep F32 until the
        // safetensors path supports I32 round-trip.
        // FIXME: vocab > 2^24 unsafe — current T5 vocab=32128 is bit-stable.
        let t5_ids_f32: Vec<f32> = t5_ids.iter().map(|&i| i as f32).collect();
        let t5_ids_t = Tensor::from_vec(
            t5_ids_f32, Shape::from_dims(&[1, args.t5_max_len]), device.clone(),
        )?;
        let t5_mask_t = Tensor::from_vec(
            t5_mask, Shape::from_dims(&[1, args.t5_max_len]), device.clone(),
        )?;

        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        tensors.insert("latent".into(), latent.to_dtype(DType::BF16)?);
        tensors.insert("text_embedding".into(), qwen_hidden.to_dtype(DType::BF16)?);
        tensors.insert("text_mask".into(), qwen_mask_t);
        tensors.insert("t5_input_ids".into(), t5_ids_t);
        tensors.insert("t5_attn_mask".into(), t5_mask_t);
        save_file(&tensors, &out_path)?;
        written += 1;

        if written % 10 == 0 || written == 1 {
            let elapsed = t_start.elapsed().as_secs_f32();
            log::info!("  cached {written} (skipped {skipped}) — {:.2}/s",
                written as f32 / elapsed.max(1e-3));
        }
    }

    log::info!("Done: wrote {written}, skipped {skipped}, total {} in {:.1}s",
        pairs.len(), t_start.elapsed().as_secs_f32());
    Ok(())
}

/// Qwen3 may be one .safetensors file or a sharded directory.
fn load_qwen3_weights(path: &std::path::Path, device: &std::sync::Arc<flame_core::CudaDevice>)
    -> flame_core::Result<HashMap<String, Tensor>>
{
    if path.is_file() {
        return flame_core::serialization::load_file(path, device);
    }
    let mut all = HashMap::new();
    for entry in std::fs::read_dir(path).map_err(|e| flame_core::Error::Io(format!("read_dir {}: {e}", path.display())))? {
        let p = entry.map_err(|e| flame_core::Error::Io(format!("entry: {e}")))?.path();
        if p.extension().and_then(|e| e.to_str()) == Some("safetensors") {
            let part = flame_core::serialization::load_file(&p, device)?;
            all.extend(part);
        }
    }
    Ok(all)
}
