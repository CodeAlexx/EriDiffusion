//! Training-time sampling for Chroma.
//!
//! Ported from the working `chroma_gen` binary — noise generation, FLUX-style
//! Euler CFG denoise loop, LDM VAE decode, and PNG save. Differences from
//! `chroma_gen`:
//!
//!   * Uses `ChromaTrainingModel::forward(latent, txt, timesteps)` which takes
//!     raw NCHW latents (internal patchify/unpatchify). No pack_latent or
//!     unpack_latent, no FlameSwap.
//!   * Two separate forwards per step (cond + uncond) instead of batched B=2,
//!     because the training model may not support batch>1 with LoRA.
//!   * Wrapped in `AutogradContext::no_grad` + `FLAME_CHECKPOINT=0` scope guard
//!     so it can be called from inside the training loop.
//!
//! Do not "clean up" this file. It matches chroma_gen.rs logic by design so
//! sampled images are identical to what the working inference binary produces
//! for the same model + text + seed.

use cudarc::driver::CudaDevice;
use flame_core::{autograd::AutogradContext, DType, Error, Result, Shape, Tensor};
use crate::sampler::flux_sampler;
use crate::encoders::flux_vae_decoder::LdmVAEDecoder;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::{path::Path, sync::Arc};

use crate::models::ChromaTrainingModel;

// FLUX VAE constants — Chroma uses the same VAE.
const AE_IN_CHANNELS: usize = 16;
const AE_SCALE_FACTOR: f32 = 0.3611;
const AE_SHIFT_FACTOR: f32 = 0.1159;

/// Sample one image from the LIVE training model. Ported from `chroma_gen`,
/// adapted for `ChromaTrainingModel::forward` (NCHW, no pack/unpack).
#[allow(clippy::too_many_arguments)]
pub fn sample_image(
    model: &ChromaTrainingModel,
    text_embed: &Tensor,     // [1, N, 4096] T5 cond
    neg_text_embed: &Tensor, // [1, N, 4096] T5 uncond
    width: usize,
    height: usize,
    num_steps: usize,
    cfg_scale: f32,
    seed: u64,
    vae_path: &Path,
    output_path: &Path,
    device: &Arc<CudaDevice>,
) -> Result<()> {
    if width % 16 != 0 || height % 16 != 0 {
        return Err(Error::InvalidInput(format!(
            "width and height must be divisible by 16, got {width}x{height}"
        )));
    }

    let _no_grad = AutogradContext::no_grad();
    let prev_ckpt = std::env::var("FLAME_CHECKPOINT").ok();
    std::env::set_var("FLAME_CHECKPOINT", "0");

    let result = sample_inner(
        model,
        text_embed,
        neg_text_embed,
        width,
        height,
        num_steps,
        cfg_scale,
        seed,
        vae_path,
        output_path,
        device,
    );

    if let Some(v) = prev_ckpt {
        std::env::set_var("FLAME_CHECKPOINT", v);
    } else {
        std::env::set_var("FLAME_CHECKPOINT", "1");
    }
    result
}

#[allow(clippy::too_many_arguments)]
fn sample_inner(
    model: &ChromaTrainingModel,
    text_embed: &Tensor,
    neg_text_embed: &Tensor,
    width: usize,
    height: usize,
    num_steps: usize,
    cfg_scale: f32,
    seed: u64,
    vae_path: &Path,
    output_path: &Path,
    device: &Arc<CudaDevice>,
) -> Result<()> {
    // ------------------------------------------------------------------
    // Stage 1: Build noise [1, 16, H_lat, W_lat]
    // ------------------------------------------------------------------
    // FLUX-style latent geometry: VAE 8x + patchify 2x = 16x effective.
    // latent_h/w are the VAE-level spatial dims (before patchify).
    let latent_h = 2 * ((height + 15) / 16);
    let latent_w = 2 * ((width + 15) / 16);
    // n_img = patchified token count (used for schedule shift).
    let n_img = (latent_h / 2) * (latent_w / 2);

    log::info!(
        "[chroma sample] {}x{} → latent [{}, {}, {}, {}], n_img={}, steps={}, cfg={}",
        width,
        height,
        1,
        AE_IN_CHANNELS,
        latent_h,
        latent_w,
        n_img,
        num_steps,
        cfg_scale,
    );

    let numel = AE_IN_CHANNELS * latent_h * latent_w;
    let noise_data: Vec<f32> = {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut v = Vec::with_capacity(numel);
        while v.len() < numel {
            let u1: f32 = rng.r#gen::<f32>().max(1e-10);
            let u2: f32 = rng.r#gen::<f32>();
            let mag = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f32::consts::PI * u2;
            v.push(mag * theta.cos());
            if v.len() < numel {
                v.push(mag * theta.sin());
            }
        }
        v
    };
    let mut x = Tensor::from_f32_to_bf16(
        noise_data,
        Shape::from_dims(&[1, AE_IN_CHANNELS, latent_h, latent_w]),
        device.clone(),
    )?;

    // ------------------------------------------------------------------
    // Stage 2: FLUX-style flow-match Euler CFG denoise
    // ------------------------------------------------------------------
    // Chroma is NOT distilled — real CFG with 2 forwards per step.
    let timesteps = flux_sampler::schedule(num_steps, width, height);
    let _ = n_img; // EDv2's schedule takes w/h directly; n_img stays informational.

    for step in 0..num_steps {
        let t_curr = timesteps[step];
        let t_next = timesteps[step + 1];
        let dt = t_next - t_curr;

        let next_x = {
            let t_vec = Tensor::from_f32_to_bf16(
                vec![t_curr],
                Shape::from_dims(&[1]),
                device.clone(),
            )?;

            // Cond forward
            let pred_cond = model.forward(&x, text_embed, &t_vec)?;
            // Uncond forward
            let pred_uncond = model.forward(&x, neg_text_embed, &t_vec)?;

            // CFG: pred = uncond + cfg_scale * (cond - uncond)
            let diff = pred_cond.sub(&pred_uncond)?;
            let scaled = diff.mul_scalar(cfg_scale)?;
            let pred = pred_uncond.add(&scaled)?;

            // Euler step: x_next = x + dt * pred
            let step_delta = pred.mul_scalar(dt)?;
            x.add(&step_delta)?
        };
        x = next_x;

        if (step + 1) % 5 == 0 || step == 0 || step + 1 == num_steps {
            log::info!(
                "[chroma sample] step {}/{} t={:.4}",
                step + 1,
                num_steps,
                t_curr,
            );
        }
    }

    // ------------------------------------------------------------------
    // Stage 3: VAE decode
    // ------------------------------------------------------------------
    log::info!("[chroma sample] VAE decode...");
    let vae_path_str = vae_path
        .to_str()
        .ok_or_else(|| Error::InvalidInput("vae_path not valid UTF-8".into()))?;
    let vae = LdmVAEDecoder::from_safetensors(
        vae_path_str,
        AE_IN_CHANNELS,
        AE_SCALE_FACTOR,
        AE_SHIFT_FACTOR,
        device,
    )?;
    let rgb = vae.decode(&x)?;
    drop(vae);
    drop(x);

    // ------------------------------------------------------------------
    // Stage 4: Save PNG
    // ------------------------------------------------------------------
    save_rgb_png(&rgb, output_path)?;
    log::info!("[chroma sample] saved {}", output_path.display());
    Ok(())
}

fn save_rgb_png(rgb: &Tensor, path: &Path) -> Result<()> {
    let rgb_f32 = rgb.to_dtype(DType::F32)?;
    let data = rgb_f32.to_vec()?;
    let dims = rgb_f32.shape().dims().to_vec();
    if dims.len() != 4 || dims[1] != 3 {
        return Err(Error::InvalidInput(format!(
            "expected [B,3,H,W], got {dims:?}"
        )));
    }
    let (out_h, out_w) = (dims[2], dims[3]);
    let mut pixels = vec![0u8; out_h * out_w * 3];
    for y in 0..out_h {
        for x in 0..out_w {
            for c in 0..3 {
                let idx = c * out_h * out_w + y * out_w + x;
                let val = (127.5 * (data[idx].clamp(-1.0, 1.0) + 1.0)) as u8;
                pixels[(y * out_w + x) * 3 + c] = val;
            }
        }
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| Error::Io(format!("create dir {}: {e}", parent.display())))?;
    }
    image::RgbImage::from_raw(out_w as u32, out_h as u32, pixels)
        .ok_or_else(|| Error::InvalidInput("RgbImage::from_raw failed".into()))?
        .save(path)
        .map_err(|e| Error::Io(format!("save png {}: {e}", path.display())))?;
    Ok(())
}
