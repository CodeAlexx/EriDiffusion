use anyhow::Result;
use eridiffusion_core::Device;
use eridiffusion_models::devtensor::{randn_on, shape4};
// Note: AutoencoderKL is provided by flame-core's VAE module in this setup.
// If your flame-core exposes a different path, adjust the import accordingly.
use flame_core::vae::AutoencoderKL;
use flame_core::{DType, Tensor};

/// Decode-only sampler: makes random latents and decodes via VAE.
/// Useful for periodic eval hooks (no denoising required).
pub fn sample_decode_only(
    vae: &AutoencoderKL,
    n: usize,
    h: usize,
    w: usize,
    device: &Device,
) -> Result<Tensor> {
    let hh = h / 8;
    let ww = w / 8;
    // Latents ~ N(0,1), NHWC: [N, H/8, W/8, 4]
    let z = randn_on(shape4(n as i64, hh as i64, ww as i64, 4), device, DType::F32, None)
        .map_err(eridiffusion_core::Error::from)?;
    let x = vae.decode(&z)?; // Expect [N, H, W, 3] NHWC
    Ok(x)
}
