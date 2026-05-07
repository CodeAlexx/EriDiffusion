//! prepare_qwenimage — image+caption → cached latents+embeddings for
//! Qwen-Image-2512 LoRA training.
//!
//! Patterned after `prepare_zimage.rs` and `prepare_anima.rs`.
//!
//! Output per sample (one safetensors file in `--output-dir`,
//! filename = md5 of image path so partial runs are resumable):
//!   - latent:         BF16 [1, 16, H/8, W/8]   raw VAE z (no shift/scale —
//!                     Qwen-Image trains on raw VAE output per
//!                     `qwen_image_train_network.py:scale_shift_latents` which
//!                     is identity).
//!   - text_embedding: BF16 [1, T_seq, 3584]    Qwen2.5-VL last hidden state.
//!
//! Reference for the cache schema:
//!   `qwenimage-trainer/src/dataset.rs::QwenImageCachedDataset`.

use clap::Parser;
use flame_core::{serialization::save_file, DType, Shape, Tensor};
use eridiffusion_core::encoders::{
    qwen25vl::{Qwen25VLConfig, Qwen25VLEncoder},
    wan21_encoder::Wan21VaeEncoder,
};
use std::collections::HashMap;
use std::path::PathBuf;

const QWEN_PAD_ID: i32 = 151643;
const TXT_PAD_LEN_DEFAULT: usize = 512;

/// Qwen-Image canonical prompt template — matches `pipeline_qwenimage.py:
/// PROMPT_TEMPLATE_ENCODE`. The DiT was trained against text embeddings
/// produced by this exact template, so caching plain captions produces
/// out-of-distribution conditioning and wrecks both training and
/// inference. Keep verbatim.
const PROMPT_PREFIX: &str =
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, \
     texture, quantity, text, spatial relationships of the objects and background:\
     <|im_end|>\n<|im_start|>user\n";
const PROMPT_SUFFIX: &str = "<|im_end|>\n<|im_start|>assistant\n";

/// Number of leading tokens to drop from the encoded hidden state — the
/// system-prompt portion. Matches Python `PROMPT_TEMPLATE_ENCODE_START_IDX`.
const DROP_IDX: usize = 34;

#[derive(Parser)]
struct Args {
    #[arg(long)] input_dir: PathBuf,
    #[arg(long)] output_dir: PathBuf,
    /// `qwen_image_vae.safetensors` — wan21 internal-key format.
    #[arg(long)] vae_ckpt: PathBuf,
    /// Directory of Qwen2.5-VL text encoder safetensors shards (the
    /// `text_encoder/` subdir of `qwen-image-2512`), or a single combined file.
    #[arg(long)] text_encoder: PathBuf,
    /// Tokenizer.json at `qwen-image-2512/tokenizer/tokenizer.json`.
    #[arg(long)] tokenizer_path: PathBuf,
    #[arg(long, default_value = "512")] resolution: u32,
    #[arg(long, default_value_t = TXT_PAD_LEN_DEFAULT)] max_text_len: usize,
    #[arg(long, default_value_t = true)] skip_existing: bool,
    #[arg(long, default_value_t = 0)] max_samples: usize,
}

fn main() -> anyhow::Result<()> {
    // Disable flame_core CUDA alloc pool — see prepare_klein.rs writeup. The
    // pool retains slabs and grows host RSS by ~1 GB per sample at 512²
    // with text-encoder forward, OOM-killing a 62 GB box around sample 75.
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

    log::info!("[1/3] Loading Wan21 VAE encoder (qwen_image_vae)...");
    // Qwen-Image trains on RAW VAE z (musubi `scale_shift_latents` is identity).
    let vae = Wan21VaeEncoder::from_safetensors(
        args.vae_ckpt.to_str().unwrap(), &device,
    )?;

    log::info!("[2/3] Loading Qwen2.5-VL-7B text encoder...");
    let te_weights = load_text_encoder_weights(&args.text_encoder, &device)?;
    let te_cfg = Qwen25VLEncoder::config_from_weights(&te_weights)?;
    log::info!(
        "  config: hidden={} layers={} heads={} kv_heads={} head_dim={} max_seq_len={}",
        te_cfg.hidden_size, te_cfg.num_layers, te_cfg.num_heads,
        te_cfg.num_kv_heads, te_cfg.head_dim, te_cfg.max_seq_len,
    );
    let te = Qwen25VLEncoder::new(te_weights, te_cfg, device.clone());

    let tokenizer = tokenizers::Tokenizer::from_file(&args.tokenizer_path)
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;

    log::info!("[3/3] Encoding samples...");
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

    let mut written = 0usize;
    let mut skipped = 0usize;
    let t_start = std::time::Instant::now();

    for (idx, (img_path, txt_path)) in pairs.iter().enumerate() {
        if args.max_samples > 0 && written + skipped >= args.max_samples { break; }
        let hash = format!("{:x}", md5::compute(img_path.to_string_lossy().as_bytes()));
        let out_path = args.output_dir.join(format!("{hash}.safetensors"));
        if args.skip_existing && out_path.exists() { skipped += 1; continue; }

        // ── Image → VAE latent (raw) ──────────────────────────────────────
        let img = match image::open(img_path) {
            Ok(i) => i.resize_exact(args.resolution, args.resolution, image::imageops::FilterType::Lanczos3).to_rgb32f(),
            Err(e) => { log::warn!("[{idx}] skipping {}: {e}", img_path.display()); continue; }
        };
        let (w, h) = img.dimensions();
        // CHW transpose — see prepare_klein.rs for full bug writeup. image::pixels
        // (HWC interleaved) reshaped to [1, 3, H, W] (CHW) scrambles channels and
        // the VAE silently encodes garbage without this manual transpose.
        let (wu, hu) = (w as usize, h as usize);
        let mut pixels = vec![0f32; 3 * hu * wu];
        for (x, y, p) in img.enumerate_pixels() {
            let (xu, yu) = (x as usize, y as usize);
            for c in 0..3 { pixels[c * hu * wu + yu * wu + xu] = p.0[c] * 2.0 - 1.0; }
        }
        let img_t = Tensor::from_vec(
            pixels, Shape::from_dims(&[1, 3, hu, wu]), device.clone(),
        )?.to_dtype(DType::BF16)?;
        // [1, 16, H/8, W/8] **normalized** — diffusers QwenImage DiT trains
        // and predicts in `(z - MEAN) / STD` space. Cache must be in this
        // space too so the trainer's targets match the DiT's native output
        // distribution. (Sampler's `decode_image_normalized` un-normalizes
        // before running VAE convs.)
        let latent = vae.encode_image_normalized(&img_t)?;

        // ── Caption → Qwen2.5-VL hidden state with PROPER template ───────
        // Wrap the caption in the qwen-image PROMPT_TEMPLATE_ENCODE, encode,
        // then drop the system-prompt prefix (DROP_IDX tokens). The DiT was
        // trained against this exact slice — feeding it raw or full
        // template embeddings produces out-of-distribution conditioning.
        let caption = std::fs::read_to_string(txt_path).unwrap_or_default();
        let caption = caption.trim();
        let wrapped = format!("{PROMPT_PREFIX}{caption}{PROMPT_SUFFIX}");
        let enc = tokenizer.encode(wrapped, false)
            .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
        let raw_ids: Vec<i32> = enc.get_ids().iter().map(|&i| i as i32).collect();
        // Pad/truncate to (max_text_len + DROP_IDX) so post-drop length is
        // exactly max_text_len.
        let work_len = args.max_text_len + DROP_IDX;
        let mut ids: Vec<i32> = raw_ids.iter().take(work_len).copied().collect();
        ids.resize(work_len, QWEN_PAD_ID);
        // Encode: [1, work_len, 3584]
        let full_hidden = te.encode(&ids)?.to_dtype(DType::BF16)?;
        // Drop the system-prompt prefix → [1, max_text_len, 3584]
        let text_hidden = full_hidden.narrow(1, DROP_IDX, args.max_text_len)?;

        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        tensors.insert("latent".into(), latent.to_dtype(DType::BF16)?);
        tensors.insert("text_embedding".into(), text_hidden.to_dtype(DType::BF16)?);
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
    let _ = Qwen25VLConfig::default();
    Ok(())
}

/// Qwen2.5-VL ships sharded across 4 .safetensors. Load all .safetensors in
/// the directory; works for a single-file path too.
fn load_text_encoder_weights(
    path: &std::path::Path,
    device: &std::sync::Arc<flame_core::CudaDevice>,
) -> flame_core::Result<HashMap<String, Tensor>> {
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
