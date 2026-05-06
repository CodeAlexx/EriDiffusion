//! prepare_klein — image+caption → cached latents+embeddings for Klein 4B/9B LoRA training.
//!
//! Klein uses:
//!   - `KleinVaeEncoder` (Flux-2 16ch posterior + 4× patchify → 128ch packed latents)
//!   - Qwen3 text encoder, `KLEIN_EXTRACT_LAYERS = [8, 17, 26]` stacked along hidden:
//!       Klein 4B: hidden=2560 → text dim 7680
//!       Klein 9B: hidden=4096 → text dim 12288
//!     Auto-detects from the loaded Qwen3 weights' embed_tokens shape.
//!
//! Output per sample (one safetensors file in `--output-dir`):
//!   - latent:         BF16 [1, 128, H/16, W/16]   — KleinVaeEncoder.encode (BN-normalised, packed)
//!   - text_embedding: BF16 [1, 512, joint_dim]    — Qwen3 stacked extract layers
//!   - text_mask:      F32  [1, 512]
//!
//! Mirrors prepare_zimage.rs / prepare_ernie.rs structure.

use clap::Parser;
use flame_core::{serialization::save_file, DType, Shape, Tensor};
use eridiffusion_core::encoders::{
    qwen3::Qwen3Encoder,
    vae::KleinVaeEncoder,
};
use std::collections::HashMap;
use std::path::PathBuf;

const KLEIN_TEMPLATE_PRE: &str = "<|im_start|>user\n";
const KLEIN_TEMPLATE_POST: &str = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
const PAD_TOKEN_ID: i32 = 151643;
const TXT_PAD_LEN: usize = 512;

#[derive(Parser)]
struct Args {
    #[arg(long)] input_dir: PathBuf,
    #[arg(long)] output_dir: PathBuf,
    /// Klein VAE safetensors (e.g. flux2-vae.safetensors). Same VAE for 4B and 9B.
    #[arg(long)] vae_ckpt: PathBuf,
    /// Qwen3 weights path (single file or sharded dir). qwen_3_4b for 4B, larger for 9B.
    #[arg(long)] qwen3: PathBuf,
    #[arg(long)] tokenizer_path: PathBuf,
    #[arg(long, default_value = "512")] resolution: u32,
    #[arg(long, default_value_t = true)] skip_existing: bool,
    #[arg(long, default_value_t = 0)] max_samples: usize,
}

fn main() -> anyhow::Result<()> {
    // Disable the flame_core CUDA alloc pool. Dataset prep is one-pass
    // (no shape recurrence to amortize), and on the trainer-bench profile we
    // observed +1.13 GB host RSS PER SAMPLE with the pool enabled — slabs
    // returned by Tensor::drop weren't reaching `clear_cache()` because of
    // Arc-storage refcount patterns, and at sample ~75 the process was
    // OOM-killed at 62 GB resident, freezing the box. With the pool off
    // every drop calls cudaFree directly and RSS stays flat at ~0.8 GB.
    // Must be set before any flame_core call (OnceLock-cached on first read).
    if std::env::var_os("FLAME_ALLOC_POOL").is_none() {
        // SAFETY: single-threaded at this point (before main's first action).
        unsafe { std::env::set_var("FLAME_ALLOC_POOL", "0"); }
    }
    env_logger::init();
    let args = Args::parse();
    std::fs::create_dir_all(&args.output_dir)?;
    flame_core::config::set_default_dtype(DType::BF16);
    let _no_grad = flame_core::autograd::AutogradContext::no_grad();
    let device = flame_core::global_cuda_device();

    log::info!("[1/3] Loading Klein VAE encoder (128-ch packed latents)...");
    let dev = flame_core::Device::from(device.clone());
    let vae_weights = flame_core::serialization::load_file(&args.vae_ckpt, &device)?;
    let vae = KleinVaeEncoder::load(&vae_weights, &dev)?;
    drop(vae_weights);

    log::info!("[2/3] Loading Qwen3 text encoder (Klein extract layers [8,17,26])...");
    let qwen_weights = load_qwen3_weights(&args.qwen3, &device)?;
    // Auto-detect config from embed shape; default extract is KLEIN_EXTRACT_LAYERS already.
    let qcfg = Qwen3Encoder::config_from_weights(&qwen_weights)?;
    let joint_dim = qcfg.extract_layers.len() * qcfg.hidden_size;
    log::info!(
        "  Qwen3 hidden={} layers={} extract={:?} → text dim {}",
        qcfg.hidden_size, qcfg.num_layers, qcfg.extract_layers, joint_dim,
    );
    let qwen3 = Qwen3Encoder::new(qwen_weights, qcfg, device.clone());

    let tokenizer = tokenizers::Tokenizer::from_file(&args.tokenizer_path)
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;

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
        // CHW transpose: image::pixels() yields HWC interleaved [R,G,B,R,G,B,...]
        // but Tensor::from_vec(_, [1, 3, H, W]) interprets as CHW. Without
        // transposing, channels are scrambled — the VAE silently encodes
        // garbage and training looks "lower-loss" because targets are bogus.
        // (Bisect 2026-05-05: same image direct-encode std=0.96, prepare-cache
        // std=0.85; fix collapses the gap to <0.1%.)
        let (wu, hu) = (w as usize, h as usize);
        let mut pixels = vec![0f32; 3 * hu * wu];
        for (x, y, p) in img.enumerate_pixels() {
            let (xu, yu) = (x as usize, y as usize);
            for c in 0..3 {
                pixels[c * hu * wu + yu * wu + xu] = p.0[c] * 2.0 - 1.0;
            }
        }
        let img_t = Tensor::from_vec(
            pixels, Shape::from_dims(&[1, 3, hu, wu]), device.clone(),
        )?.to_dtype(DType::BF16)?;
        // KleinVaeEncoder.encode handles posterior.mode + patchify + BN → [B, 128, H/16, W/16].
        let latent = vae.encode(&img_t)?;

        let caption = std::fs::read_to_string(txt_path).unwrap_or_default();
        let prompt = format!("{KLEIN_TEMPLATE_PRE}{}{KLEIN_TEMPLATE_POST}", caption.trim());
        let enc = tokenizer.encode(prompt.as_str(), false)
            .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
        let mut ids: Vec<i32> = enc.get_ids().iter().map(|&i| i as i32).collect();
        let valid_len = ids.len().min(TXT_PAD_LEN);
        ids.resize(TXT_PAD_LEN, PAD_TOKEN_ID);
        let text_hidden = qwen3.encode(&ids)?; // [1, TXT_PAD_LEN, joint_dim]

        let mut mask_data = vec![0.0f32; TXT_PAD_LEN];
        for slot in mask_data.iter_mut().take(valid_len) { *slot = 1.0; }
        let text_mask = Tensor::from_vec(
            mask_data, Shape::from_dims(&[1, TXT_PAD_LEN]), device.clone(),
        )?;

        // Both `latent` and `text_hidden` are already BF16 — the previous
        // `to_dtype(BF16)` calls were no-op clones that doubled GPU
        // allocation per sample without changing the saved bytes.
        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        tensors.insert("latent".into(), latent);
        tensors.insert("text_embedding".into(), text_hidden);
        tensors.insert("text_mask".into(), text_mask);
        save_file(&tensors, &out_path)?;

        // Explicit drops aren't strictly needed (Rust would drop these at end
        // of the loop body anyway), but they're cheap and document intent.
        drop(tensors);
        drop(img_t);

        written += 1;

        if written % 10 == 0 || written == 1 {
            let elapsed = t_start.elapsed().as_secs_f32();
            // Read /proc/self/status VmRSS so the user can spot regressions.
            let rss_kb: usize = std::fs::read_to_string("/proc/self/status")
                .ok()
                .and_then(|s| s.lines()
                    .find(|l| l.starts_with("VmRSS:"))
                    .and_then(|l| l.split_whitespace().nth(1))
                    .and_then(|n| n.parse().ok()))
                .unwrap_or(0);
            log::info!("  cached {written} (skipped {skipped}) — {:.2}/s — RSS {:.1} GB",
                written as f32 / elapsed.max(1e-3),
                rss_kb as f32 / 1024.0 / 1024.0);
        }
    }

    log::info!("Done: wrote {written}, skipped {skipped}, total {} in {:.1}s",
        pairs.len(), t_start.elapsed().as_secs_f32());
    Ok(())
}

fn load_qwen3_weights(path: &std::path::Path, device: &std::sync::Arc<flame_core::CudaDevice>)
    -> flame_core::Result<HashMap<String, Tensor>>
{
    if path.is_file() {
        return flame_core::serialization::load_file(path, device);
    }
    let mut all = HashMap::new();
    for entry in std::fs::read_dir(path)
        .map_err(|e| flame_core::Error::Io(format!("read_dir {}: {e}", path.display())))?
    {
        let p = entry.map_err(|e| flame_core::Error::Io(format!("entry: {e}")))?.path();
        if p.extension().and_then(|e| e.to_str()) == Some("safetensors") {
            let part = flame_core::serialization::load_file(&p, device)?;
            all.extend(part);
        }
    }
    Ok(all)
}
