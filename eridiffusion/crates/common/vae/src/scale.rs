use anyhow::Result;
use flame_core::Tensor;
use crate::spec::VaeSpec;

/// Fetch the latent scaling factor encoded in the VAE spec.
/// Defaults to SDXL/Flux convention (~0.13025) if spec leaves it at zero.
pub fn read_vae_scaling(spec: &VaeSpec) -> f32 {
    let scale = spec.latent_scale;
    if scale > 0.0 { scale } else { 0.13025 }
}

/// Apply encode-time scaling (multiply) to latents/image tensors.
pub fn apply_encode_scale(t: &Tensor, scale: f32) -> Result<Tensor> {
    if (scale - 1.0).abs() < f32::EPSILON { Ok(t.clone()) } else { t.mul_scalar(scale).map_err(|e| e.into()) }
}

/// Apply decode-time scaling (divide) to tensors.
pub fn apply_decode_scale(t: &Tensor, scale: f32) -> Result<Tensor> {
    if (scale - 1.0).abs() < f32::EPSILON { Ok(t.clone()) } else { t.div_scalar(scale).map_err(|e| e.into()) }
}
