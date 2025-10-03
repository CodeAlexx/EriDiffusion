//! Utilities for preparing raw timesteps before building sinusoidal embeddings.

use eridiffusion_core::{Error, Result};
use flame_core::{DType, Tensor};

/// Cast integer timesteps [N] (I32/I64) to F32 tensor on the same device.
pub fn cast_to_f32(int_ts: &Tensor) -> Result<Tensor> {
    match int_ts.dtype() {
        DType::I32 | DType::I64 => int_ts.to_dtype(DType::F32).map_err(Error::from),
        DType::F32 => Ok(int_ts.clone()),
        other => Err(Error::InvalidInput(format!("cast_to_f32: unsupported dtype {:?}", other))),
    }
}

/// Clamp a F32 timestep tensor into `[min_val, max_val]`.
pub fn clamp_f32(ts: &Tensor, min_val: f32, max_val: f32) -> Result<Tensor> {
    if ts.dtype() != DType::F32 {
        return Err(Error::InvalidInput("clamp_f32: tensor must be F32".into()));
    }
    let clamped_min = ts.maximum_scalar(min_val).map_err(Error::from)?;
    clamped_min.minimum_scalar(max_val).map_err(Error::from)
}

/// Convenience: preprocess integer or float timesteps into F32 and clamp range in one call.
pub fn preprocess_timesteps(ts: &Tensor, min_val: f32, max_val: f32) -> Result<Tensor> {
    let f32ts = cast_to_f32(ts)?;
    clamp_f32(&f32ts, min_val, max_val)
}
