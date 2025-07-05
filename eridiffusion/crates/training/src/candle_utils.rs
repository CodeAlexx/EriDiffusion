//! Utilities for Candle operations specific to training

use eridiffusion_core::{Result, Error};
use candle_core::{Tensor, Device, DType};

/// Helper to create random integer tensor
pub fn randint(
    low: i64,
    high: i64,
    shape: &[usize],
    device: &Device,
) -> Result<Tensor> {
    let range = (high - low) as f64;
    let rand_tensor = Tensor::rand(0.0, 1.0, shape, device)?;
    let scaled = rand_tensor.affine(range, low as f64)?;
    scaled.to_dtype(DType::I64)
        .map_err(|e| Error::TensorOp(e))
}