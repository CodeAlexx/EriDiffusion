use flame_core::device::Device;
use flame_core::optimizers::{Adam, SGD};
use flame_core::{DType, Result, Shape, Tensor};

// Hybrid tensor operations - NO LONGER NEEDED
// This module is deprecated as we've fully migrated to FLAME
// All tensors are now FLAME tensors

/// Deprecated - all tensors are now FLAME tensors
pub type HybridTensor = Tensor;

/// Compatibility stub for gradual matmul
pub fn gradual_matmul(a: &Tensor, b: &Tensor) -> flame_core::Result<Tensor> {
    Ok(a.matmul(b)?)
}

/// Compatibility stub for gradual add
pub fn gradual_add(a: &Tensor, b: &Tensor) -> flame_core::Result<Tensor> {
    Ok(a.add(b)?)
}

/// Compatibility stub for gradual mul
pub fn gradual_mul(a: &Tensor, b: &Tensor) -> flame_core::Result<Tensor> {
    Ok(a.mul(b)?)
}
