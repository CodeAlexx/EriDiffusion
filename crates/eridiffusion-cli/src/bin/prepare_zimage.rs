//! prepare_zimage — image+caption → cached latents+embeddings for Z-Image LoRA training.
//!
//! Mirrors flame-diffusion/zimage-trainer/src/bin/prepare_dataset.rs but uses
//! the ED-v2 vendored encoders (`eridiffusion_core::encoders::{ldm_vae, qwen3}`).
//!
//! Output per sample (one safetensors file in `--output-dir`, name = md5 of
//! image path so partial runs are resumable):
//!   - latent:         BF16 [1, 16, H/8, W/8]   — raw VAE posterior.mode (no scale/shift)
//!   - text_embedding: BF16 [1, 512, 2560]      — Qwen3-4B layer 26 hidden state
//!   - text_mask:      F32  [1, 512]            — 1 at valid token positions, 0 at PADs

use clap::Parser;
use flame_core::{serialization::save_file, DType, Shape, Tensor};
use eridiffusion_core::encoders::{
    qwen3::Qwen3Encoder,
    wan21_encoder::Wan21VaeEncoder,
};
use std::collections::HashMap;
use std::path::PathBuf;

// CRITICAL: This is the template Qwen3 tokenizer's `apply_chat_template`
// produces with `enable_thinking=True` (upstream Python's setting). The previous
// version included `<think>\n\n</think>\n\n` — that's what `enable_thinking=False`
// emits. They are inverted. Adding the think block injected 4 spurious tokens
// (151667 / 271 / 151668 / 271) into every conditioning sample, which OT
// Python and musubi do NOT emit. Verified empirically against the released
// Z-Image tokenizer.
const ZIMAGE_TEMPLATE_PRE: &str = "<|im_start|>user\n";
const ZIMAGE_TEMPLATE_POST: &str = "<|im_end|>\n<|im_start|>assistant\n";
const PAD_TOKEN_ID: i32 = 151643;
const TXT_PAD_LEN: usize = 512;

#[derive(Parser)]
struct Args {
    #[arg(long)] input_dir: PathBuf,
    #[arg(long)] output_dir: PathBuf,
    #[arg(long)] vae_ckpt: PathBuf,
    #[arg(long)] qwen3: PathBuf,
    #[arg(long)] tokenizer_path: PathBuf,
    #[arg(long, default_value = "512")] resolution: u32,
    #[arg(long, default_value_t = true)] skip_existing: bool,
    #[arg(long, default_value_t = 0)] max_samples: usize,
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

    log::info!("[1/3] Loading Wan21 VAE encoder (16-ch latents, raw output for Z-Image)...");
    // Z-Image trainer applies its own scalar shift+scale at predict time
    // (`(raw - 0.1159) * 0.3611`), so we save **raw** VAE z here. Same
    // semantics as the previous LdmVAEEncoder path, but on a backend that
    // can actually load the on-disk Wan-VAE keys (the qwen_image_vae.safetensors
    // shipped with Z-Image is in wan21 internal-key format, NOT diffusers).
    let vae = Wan21VaeEncoder::from_safetensors(
        args.vae_ckpt.to_str().unwrap(), &device,
    )?;

    log::info!("[2/3] Loading Qwen3 encoder (single-layer extract @26)...");
    let qwen_weights = load_qwen3_weights(&args.qwen3, &device)?;
    let mut qcfg = Qwen3Encoder::config_from_weights(&qwen_weights)?;
    // CRITICAL: Qwen3-4B has 36 hidden layers. `hidden_states[-2]` (the layer
    // upstream Python / musubi / official Z-Image pipeline all use as caption
    // conditioning) = layer index 34, NOT 26. Layer 26 was wrongly inherited
    // from the ERNIE Mistral-3 (26-layer) port. Z-Image's pretrained
    // `cap_embedder` was fit on layer-34 distributions; feeding layer-26
    // conditioning means the DiT sees an out-of-distribution embedding and
    // the LoRA tries to fit garbage.
    qcfg.extract_layers = vec![34];
    log::info!(
        "  config: hidden={} layers={} heads={} extract={:?}",
        qcfg.hidden_size, qcfg.num_layers, qcfg.num_heads, qcfg.extract_layers
    );
    let qwen3 = Qwen3Encoder::new(qwen_weights, qcfg, device.clone());

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

        let img = match image::open(img_path) {
            Ok(i) => i.resize_exact(args.resolution, args.resolution, image::imageops::FilterType::Lanczos3).to_rgb32f(),
            Err(e) => { log::warn!("[{idx}] skipping {}: {e}", img_path.display()); continue; }
        };
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
        let latent = vae.encode_image_raw(&img_t)?;

        let caption = std::fs::read_to_string(txt_path).unwrap_or_default();
        let prompt = format!("{ZIMAGE_TEMPLATE_PRE}{}{ZIMAGE_TEMPLATE_POST}", caption.trim());
        let enc = tokenizer.encode(prompt.as_str(), false)
            .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
        let mut ids: Vec<i32> = enc.get_ids().iter().map(|&i| i as i32).collect();
        let valid_len = ids.len().min(TXT_PAD_LEN);
        ids.resize(TXT_PAD_LEN, PAD_TOKEN_ID);
        let text_hidden = qwen3.encode(&ids)?; // [1, TXT_PAD_LEN, 2560]

        let mut mask_data = vec![0.0f32; TXT_PAD_LEN];
        for slot in mask_data.iter_mut().take(valid_len) { *slot = 1.0; }
        let text_mask = Tensor::from_vec(
            mask_data, Shape::from_dims(&[1, TXT_PAD_LEN]), device.clone(),
        )?;

        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        tensors.insert("latent".into(), latent.to_dtype(DType::BF16)?);
        tensors.insert("text_embedding".into(), text_hidden.to_dtype(DType::BF16)?);
        tensors.insert("text_mask".into(), text_mask);
        save_file(&tensors, &out_path)?;
        written += 1;

        if written % 10 == 0 || written == 1 {
            let elapsed = t_start.elapsed().as_secs_f32();
            log::info!("  cached {written} (skipped {skipped}) — {:.2}/s", written as f32 / elapsed.max(1e-3));
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
