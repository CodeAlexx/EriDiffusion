//! Diffusers-style sinusoidal time embeddings (no learned weights).
//! Exposed for SDXL conditioning glue; returns `[N, dim]` tensors on the same device as `timesteps`.

use eridiffusion_core::{Error, Result};
use eridiffusion_models::devtensor::{tensor_from_slice_on, F32_};
use flame_core::{DType, Shape, Tensor};

/// Build sinusoidal embeddings for `timesteps: [N]` → `[N, dim]`.
/// - `dim` must be even (sin|cos halves)
/// - `max_period` typically 10000.0
/// - `flip_sin_to_cos`: if true, returns [cos, sin] for parity with some repos
/// - `scale`: optional multiplier applied to timesteps before sin/cos
pub fn timestep_embedding(
    timesteps: &Tensor,
    dim: usize,
    max_period: f32,
    flip_sin_to_cos: bool,
    scale: Option<f32>,
) -> Result<Tensor> {
    if dim % 2 != 0 {
        return Err(Error::InvalidInput(format!("timestep_embedding: dim {} must be even", dim)));
    }
    if timesteps.dtype() != DType::F32 {
        return Err(Error::InvalidInput("timestep_embedding: timesteps must be F32".into()));
    }

    let device = timesteps.device().clone();

    let shape = timesteps.shape();
    if shape.rank() != 1 {
        return Err(Error::InvalidInput(format!(
            "timestep_embedding: expected [N], got shape {:?}",
            shape.dims()
        )));
    }
    let n = shape.dims()[0];
    let half = dim / 2;

    // Build frequency exponents: exp(-log(max_period) * i / (half-1))
    let denom = (half.saturating_sub(1)).max(1) as f32;
    let exps: Vec<f32> = (0..half)
        .map(|i| {
            let ratio = i as f32 / denom;
            (-max_period.ln() * ratio).exp()
        })
        .collect();
    let freqs =
        Tensor::from_vec_dtype(exps, Shape::from_dims(&[half]), device.clone(), DType::F32)?;

    // Broadcast to [N, half]
    let s = scale.unwrap_or(1.0);
    let t_scaled = timesteps.mul_scalar(s)?; // [N]
    let t2d = t_scaled.reshape(&[n, 1])?;
    let freqs2d = freqs.reshape(&[1, half])?;
    let arg = t2d.matmul(&freqs2d)?; // [N, half]

    let sin = arg.sin()?; // [N, half]
    let cos = arg.cos()?; // [N, half]

    if flip_sin_to_cos {
        Tensor::cat(&[&cos, &sin], 1).map_err(Error::from)
    } else {
        Tensor::cat(&[&sin, &cos], 1).map_err(Error::from)
    }
}

/// Convenience: turn a scalar sigma into a `[batch]` F32 tensor (already clamped elsewhere).
pub fn sigma_to_timestep_vec(
    batch: usize,
    sigma: f32,
    device: &eridiffusion_core::Device,
) -> Result<Tensor> {
    let values = vec![sigma; batch];
    Ok(tensor_from_slice_on(&values, Shape::from_dims(&[batch]), device, F32_)?)
}
