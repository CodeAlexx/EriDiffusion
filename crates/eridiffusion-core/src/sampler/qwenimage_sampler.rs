//! Training-time sampling for QwenImage-2512.
//!
//! Uses an Euler flow-matching denoise loop with the live training model
//! (dual-stream MMDiT), then decodes with inference-flame's LDM VAE
//! decoder (16-channel, same family as Z-Image but different scale/shift).
//! Saves the output as a PNG preview.

use cudarc::driver::CudaDevice;
use flame_core::{autograd::AutogradContext, DType, Result, Shape, Tensor};
use crate::encoders::wan21_decoder::Wan21VaeDecoder;
use std::path::Path;
use std::sync::Arc;

use crate::models::qwenimage::{self as qwen_model, QwenImageTrainingModel, OUT_CHANNELS};

/// Sample one image from the live QwenImage training model.
///
/// `txt_embed`: [1, txt_seq, 3584] BF16 — pre-encoded Qwen2.5-VL text embedding.
/// `txt_embed_uncond`: optional negative embedding for CFG.
///
/// Latents are 16-channel, packed 2x2 to 64-channel sequence format for
/// the transformer, then unpacked back to [B, 16, H, W] for VAE decode.
#[allow(clippy::too_many_arguments)]
pub fn sample_image(
    model: &mut QwenImageTrainingModel,
    txt_embed: &Tensor,
    txt_embed_uncond: Option<&Tensor>,
    width: usize,
    height: usize,
    num_steps: usize,
    cfg_scale: f32,
    seed: u64,
    vae_path: &Path,
    output_png: &Path,
    device: &Arc<CudaDevice>,
) -> Result<()> {
    let _no_grad = AutogradContext::no_grad();
    let prev_ckpt = std::env::var("FLAME_CHECKPOINT").ok();
    std::env::set_var("FLAME_CHECKPOINT", "0");

    let result = sample_inner(
        model,
        txt_embed,
        txt_embed_uncond,
        width,
        height,
        num_steps,
        cfg_scale,
        seed,
        vae_path,
        output_png,
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
    model: &mut QwenImageTrainingModel,
    txt_embed: &Tensor,
    txt_embed_uncond: Option<&Tensor>,
    width: usize,
    height: usize,
    num_steps: usize,
    cfg_scale: f32,
    _seed: u64,
    vae_path: &Path,
    output_png: &Path,
    device: &Arc<CudaDevice>,
) -> Result<()> {
    let latent_h = height / 8;
    let latent_w = width / 8;
    let h_patched = latent_h / 2; // after 2x2 packing
    let w_patched = latent_w / 2;
    let seq_len = h_patched * w_patched;
    // Generate noise in unpacked space [1, 16, H_lat, W_lat], then pack
    let noise_unpacked = Tensor::randn(
        Shape::from_dims(&[1, OUT_CHANNELS, latent_h, latent_w]),
        0.0,
        1.0,
        device.clone(),
    )?
    .to_dtype(DType::BF16)?;

    // Pack: [1, 16, H, W] → [1, (H/2)*(W/2), 64]
    let mut x = qwen_model::pack_latents(&noise_unpacked)?;

    // Sigma schedule: dynamic exponential shift (same as qwenimage_gen.rs)
    // Reference: pipeline_qwenimage.py:634-649 + scheduling_flow_match_euler_discrete.py
    let base_shift: f32 = 0.5;
    let max_shift: f32 = 0.9;
    let base_seq_len: f32 = 256.0;
    let max_seq_len_shift: f32 = 8192.0;
    let shift_terminal: f32 = 0.02;

    let m = (max_shift - base_shift) / (max_seq_len_shift - base_seq_len);
    let bb = base_shift - m * base_seq_len;
    let mu = (seq_len as f32) * m + bb;
    let exp_mu = mu.exp();

    // 1. Linear sigmas in descending order
    let mut sigmas: Vec<f32> = (0..num_steps)
        .map(|i| {
            let t = i as f32 / (num_steps - 1).max(1) as f32;
            1.0 - t * (1.0 - 1.0 / num_steps as f32)
        })
        .collect();
    // 2. Exponential time shift
    for s in sigmas.iter_mut() {
        let denom = exp_mu + (1.0 / *s - 1.0);
        *s = exp_mu / denom;
    }
    // 3. Stretch to terminal
    let last = *sigmas.last().unwrap();
    let one_minus_last = 1.0 - last;
    if one_minus_last.abs() > 1e-12 {
        let scale = one_minus_last / (1.0 - shift_terminal);
        for s in sigmas.iter_mut() {
            let o = 1.0 - *s;
            *s = 1.0 - o / scale;
        }
    }
    // 4. Append terminal sigma = 0
    sigmas.push(0.0);

    log::info!(
        "  sampling: {}x{} (lat {}x{}, seq={}), {} steps, cfg={:.1}",
        width,
        height,
        latent_h,
        latent_w,
        seq_len,
        num_steps,
        cfg_scale
    );
    log::info!(
        "  sigmas[0]={:.4}  sigmas[-2]={:.4}  mu={:.4}",
        sigmas[0],
        sigmas[num_steps - 1],
        mu
    );

    // Euler denoise loop
    let t_denoise = std::time::Instant::now();
    for step in 0..num_steps {
        let sigma_curr = sigmas[step];
        let sigma_next = sigmas[step + 1];
        let dt = sigma_next - sigma_curr;

        let t_vec = Tensor::from_vec(
            vec![sigma_curr],
            Shape::from_dims(&[1]),
            device.clone(),
        )?
        .to_dtype(DType::BF16)?;

        // Conditional prediction
        let cond_pred = model.forward(&x, &t_vec, txt_embed, latent_h, latent_w)?;

        // CFG if enabled
        let noise_pred = if cfg_scale > 1.0 {
            if let Some(uncond) = txt_embed_uncond {
                let uncond_pred = model.forward(&x, &t_vec, uncond, latent_h, latent_w)?;
                let diff = cond_pred.sub(&uncond_pred)?;
                let scaled = diff.mul_scalar(cfg_scale)?;
                uncond_pred.add(&scaled)?
            } else {
                cond_pred
            }
        } else {
            cond_pred
        };

        // Euler step: x = x + dt * noise_pred
        x = x.add(&noise_pred.mul_scalar(dt)?)?;

        if (step + 1) % 5 == 0 || step == 0 || step + 1 == num_steps {
            log::info!(
                "  step {}/{} sigma={:.4} ({:.1}s)",
                step + 1,
                num_steps,
                sigma_curr,
                t_denoise.elapsed().as_secs_f32()
            );
        }
    }
    log::info!("  denoised in {:.1}s", t_denoise.elapsed().as_secs_f32());

    // Unpack: [1, seq, 64] → [1, 16, H_lat, W_lat]
    let latent = qwen_model::unpack_latents(&x, latent_h, latent_w)?;

    // VAE decode. qwen_image_vae.safetensors is wan21 internal-key format
    // (verified per docs/MIGRATION_AUDIT_2026-05-05.md). Qwen-Image trains
    // on RAW VAE z (musubi `scale_shift_latents` is identity), so decode
    // through the raw path — no scale/shift, no per-channel normalization.
    // Wan21VaeDecoder clamps RGB output to [-1, 1].
    let t_vae = std::time::Instant::now();
    let vae = Wan21VaeDecoder::from_safetensors(
        &vae_path.to_string_lossy(),
        device,
    )?;
    let rgb = vae.decode_image_raw(&latent)?;
    log::info!("  VAE decoded in {:.1}s", t_vae.elapsed().as_secs_f32());

    // Save PNG
    save_tensor_as_png(&rgb, output_png)?;
    log::info!("  saved {}", output_png.display());

    Ok(())
}

fn save_tensor_as_png(rgb: &Tensor, path: &Path) -> Result<()> {
    let dims = rgb.shape().dims();
    let (_b, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);
    let rgb_f32 = rgb.to_dtype(DType::F32)?;
    let data = rgb_f32.to_vec()?;

    // CHW → HWC. Wan21VaeDecoder outputs [-1, 1]; convert to [0, 255].
    let mut pixels = vec![0u8; h * w * 3];
    for y in 0..h {
        for x in 0..w {
            for ch in 0..3.min(c) {
                let val = data[ch * h * w + y * w + x];
                pixels[(y * w + x) * 3 + ch] =
                    ((val.clamp(-1.0, 1.0) + 1.0) * 127.5) as u8;
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
    encoder
        .encode(&pixels, w as u32, h as u32, image::ColorType::Rgb8)
        .map_err(|e| flame_core::Error::Io(format!("PNG encode: {e}")))?;

    Ok(())
}
