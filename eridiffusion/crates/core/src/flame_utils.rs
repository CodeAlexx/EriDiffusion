//! Utilities and extensions for FLAME functionality

use crate::Error;
use flame_core::{Tensor, Shape};
use flame_core::device::Device;

// FLAME doesn't have Parameter type yet, using Tensor directly
// These extension traits will need to be adapted when FLAME adds variable support

// FLAME doesn't have VarMap yet, will need to adapt when added

/// Helper to create random integer tensor
pub fn randint(
    _low: i64,
    _high: i64,
    _shape: &[usize],
    _device: &Device,
) -> anyhow::Result<Tensor> {
    // FLAME doesn't have rand or integer tensor support yet
    Err(Error::TensorOp("randint not yet implemented for FLAME".into()).into())
}

/// Convert vector to Shape
pub fn vec_to_shape(dims: &[usize]) -> Shape {
    Shape::from_dims(dims)
}

// FLAME provides built-in gradient tracking, no need for separate grad store
