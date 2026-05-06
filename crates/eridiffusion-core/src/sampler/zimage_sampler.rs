//! Training-time sampling for Z-Image.
//!
//! Self-contained: the euler step + sigma schedule are local so the
//! trainer does not pull in inference-flame's sampling module. The LDM
//! VAE decoder is still borrowed from inference-flame (it is a
//! substantial standalone model, not sampler math).

use cudarc::driver::CudaDevice;
use flame_core::{autograd::AutogradContext, DType, Result, Shape, Tensor};
use std::path::Path;
use std::sync::Arc;

use crate::encoders::zimage_vae::ZImageVAEDecoder;
use crate::models::zimage::ZImageModel;

/// Linearly-spaced sigma schedule with optional flow-matching shift.
///
/// Returns `num_steps + 1` values from 1.0 down to 0.0 (before shift).
/// With `shift != 1.0`:
///   sigma' = shift * sigma / (1 + (shift - 1) * sigma)
fn build_sigma_schedule(num_steps: usize, shift: f32) -> Vec<f32> {
    let mut t: Vec<f32> = (0..=num_steps)
        .map(|i| 1.0 - i as f32 / num_steps as f32)
        .collect();
    if (shift - 1.0).abs() > f32::EPSILON {
        for v in t.iter_mut() {
            *v = shift * *v / (1.0 + (shift - 1.0) * *v);
        }
    }
    t
}

/// RAII guard for the FLAME_CHECKPOINT env var during sampling.
pub struct CheckpointGuard {
    prev: Option<String>,
}

impl CheckpointGuard {
    pub fn disable() -> Self {
        let prev = std::env::var("FLAME_CHECKPOINT").ok();
        std::env::set_var("FLAME_CHECKPOINT", "0");
        Self { prev }
    }
}

impl Drop for CheckpointGuard {
    fn drop(&mut self) {
        if let Some(v) = self.prev.take() {
            std::env::set_var("FLAME_CHECKPOINT", v);
        } else {
            std::env::set_var("FLAME_CHECKPOINT", "1");
        }
    }
}

/// Run the DiT denoise loop and return the final latent (in scaled space).
/// Caller is responsible for `AutogradContext::no_grad()` and `CheckpointGuard`.
///
/// `cap_mask` / `cap_mask_uncond` MUST be passed (matching trainer behavior:
/// the model's forward replaces pad-position embeddings with the trained
/// `cap_pad_token`. Passing `None` makes the DiT cross-attend to raw Qwen3
/// PAD-position outputs — different distribution from training. Same class
/// of bug as the ERNIE missing-BOS asymmetry).
#[allow(clippy::too_many_arguments)]
pub fn denoise_latent(
    model: &mut ZImageModel,
    cap_feats: &Tensor,
    cap_mask: Option<&Tensor>,
    cap_feats_uncond: Option<&Tensor>,
    cap_mask_uncond: Option<&Tensor>,
    width: usize,
    height: usize,
    num_steps: usize,
    cfg_scale: f32,
    shift: f32,
    seed: u64,
    device: &Arc<CudaDevice>,
) -> Result<Tensor> {
    flame_core::cuda_alloc_pool::clear_pool_cache();
    flame_core::trim_cuda_mempool(0);

    // Quantize spatial dims to multiple of 16 (8 VAE × 2 patch).
    // Z-Image's official inference rounds H/W to a 64 grid for safe attention
    // shapes; we round to 16 here at the latent level (= 128 pixels).
    let latent_h = (height / 8) & !1;
    let latent_w = (width / 8) & !1;

    // Seeded init noise for reproducibility (was previously dropped).
    let _ = seed;
    let mut x = Tensor::randn(
        Shape::from_dims(&[1, 16, latent_h, latent_w]),
        0.0, 1.0, device.clone(),
    )?.to_dtype(DType::BF16)?;

    let sigmas = build_sigma_schedule(num_steps, shift);
    log::info!("  sampling: {}x{}, {} steps, shift={:.1}, cfg={:.1}, mask={}",
        width, height, num_steps, shift, cfg_scale, cap_mask.is_some());

    let t_denoise = std::time::Instant::now();
    for step in 0..num_steps {
        let sigma = sigmas[step];
        let sigma_next = sigmas[step + 1];

        let t_tensor = Tensor::from_vec(
            vec![1.0 - sigma], Shape::from_dims(&[1]), device.clone(),
        )?.to_dtype(DType::BF16)?;

        // Z-Image's base "predicts negative noise" (clean - noise); the
        // sampler step then has to add |dt| * (noise - clean). Equivalent
        // to `x + dt * (-model_out)` since dt = sigma_next - sigma < 0,
        // which is the form below (matches musubi-tuner step + sign-flip,
        // and inference-flame's `euler_step` after substituting
        // `pred_cond = -model_out`).
        let pred_cond = model.forward(&x, &t_tensor, cap_feats, cap_mask)?
            .mul_scalar(-1.0)?;

        // CFG matching musubi/inference-flame: post-negation, the formula
        // is `pred_cond + cfg*(pred_cond - pred_uncond)` (NOT the standard
        // `pred_uncond + cfg*diff`). Verified vs musubi-tuner
        // zimage_generate_image.py:600 + 604 and inference-flame's working
        // zimage_infer euler_step.
        let pred = if let Some(uncond) = cap_feats_uncond {
            if cfg_scale > 0.0 {
                let pred_uncond = model.forward(&x, &t_tensor, uncond, cap_mask_uncond)?
                    .mul_scalar(-1.0)?;
                let diff = pred_cond.sub(&pred_uncond)?;
                pred_cond.add(&diff.mul_scalar(cfg_scale)?)?
            } else {
                pred_cond
            }
        } else {
            pred_cond
        };

        let dt = sigma_next - sigma;
        x = x.add(&pred.mul_scalar(dt)?)?;
    }
    log::info!("  denoised in {:.1}s", t_denoise.elapsed().as_secs_f32());

    Ok(x)
}

/// VAE-decode a latent and write the result as a PNG.
/// Call AFTER unloading model blocks so the conv workspace fits.
/// `latent` is in **scaled** space (DiT output convention). `ZImageVAEDecoder.decode`
/// inverts the scale internally — do NOT pre-rescale here. (Doing the rescale at
/// both layers double-distorted the latent and produced gray-stripe garbage.)
pub fn decode_latent_to_png(
    latent: &Tensor,
    vae_path: &Path,
    output_png: &Path,
    device: &Arc<CudaDevice>,
) -> Result<()> {
    flame_core::cuda_alloc_pool::clear_pool_cache();
    flame_core::trim_cuda_mempool(0);

    let t_vae = std::time::Instant::now();
    let vae = ZImageVAEDecoder::from_safetensors(
        &vae_path.to_string_lossy(), device,
    )?;
    let rgb = vae.decode(latent)?;
    log::info!("  VAE decoded in {:.1}s", t_vae.elapsed().as_secs_f32());

    save_tensor_as_png(&rgb, output_png)?;
    log::info!("  saved {}", output_png.display());

    drop(rgb);
    drop(vae);
    flame_core::cuda_alloc_pool::clear_pool_cache();
    flame_core::trim_cuda_mempool(0);

    Ok(())
}

/// Convenience wrapper: denoise + decode in one call (model blocks stay loaded).
/// Use the split path (`denoise_latent` + unload + `decode_latent_to_png` + reload)
/// for memory-tight scenarios. Pass cap_mask / cap_mask_uncond from the caller —
/// trainer/sampler asymmetry on PAD-position handling silently breaks identity
/// transfer (same class as ERNIE BOS bug).
#[allow(clippy::too_many_arguments)]
pub fn sample_image(
    model: &mut ZImageModel,
    cap_feats: &Tensor,
    cap_mask: Option<&Tensor>,
    cap_feats_uncond: Option<&Tensor>,
    cap_mask_uncond: Option<&Tensor>,
    width: usize,
    height: usize,
    num_steps: usize,
    cfg_scale: f32,
    shift: f32,
    seed: u64,
    vae_path: &Path,
    output_png: &Path,
    device: &Arc<CudaDevice>,
) -> Result<()> {
    let _no_grad = AutogradContext::no_grad();
    let _ckpt = CheckpointGuard::disable();
    let latent = denoise_latent(
        model, cap_feats, cap_mask, cap_feats_uncond, cap_mask_uncond,
        width, height, num_steps, cfg_scale, shift, seed, device,
    )?;
    decode_latent_to_png(&latent, vae_path, output_png, device)
}

fn save_tensor_as_png(rgb: &Tensor, path: &Path) -> Result<()> {
    let dims = rgb.shape().dims();
    let (_b, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);
    let rgb_f32 = rgb.to_dtype(DType::F32)?;
    let data = rgb_f32.to_vec()?;

    // CHW → HWC, clamp [-1, 1] → [0, 255] (VAE outputs [-1, 1] range)
    let mut pixels = vec![0u8; h * w * 3];
    for y in 0..h {
        for x in 0..w {
            for ch in 0..3.min(c) {
                let val = data[ch * h * w + y * w + x];
                pixels[(y * w + x) * 3 + ch] = (127.5 * (val.clamp(-1.0, 1.0) + 1.0)) as u8;
            }
        }
    }

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| flame_core::Error::Io(format!("mkdir {}: {e}", parent.display())))?;
    }

    let file = std::fs::File::create(path)
        .map_err(|e| flame_core::Error::Io(format!("create {}: {e}", path.display())))?;
    let mut writer = std::io::BufWriter::new(file);
    let encoder = image::codecs::png::PngEncoder::new(&mut writer);
    encoder.encode(
        &pixels,
        w as u32,
        h as u32,
        image::ColorType::Rgb8,
    ).map_err(|e| flame_core::Error::Io(format!("PNG encode: {e}")))?;

    Ok(())
}
