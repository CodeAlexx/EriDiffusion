use flame_core::{DType, Result, Shape, Tensor};
use flame_core::device::Device;
use flame_core::optimizers::{Adam, SGD};

/// Tensor device fix utilities
//
// Utilities to ensure tensors are on the correct device by reconstructing them

// FLAME uses flame_core::device::Device instead of Device

/// Force a tensor to be on the cached device by reconstructing it

pub fn force_to_cached_device(tensor: &Tensor) -> flame_core::Result<Tensor> {
let device = crate::trainers::cached_device::get_single_device()?;

// If already on the correct device, still reconstruct to ensure clean context
if tensor.device().same_device(&device) {
// Get the data and reconstruct
let shape = tensor.shape().dims();
let dtype = tensor.dtype();

// Move to CPU first to break any device context
let cpu_tensor = tensor.unwrap())?;

// Then move to our cached device
cpu_tensor
} else {
// Different device, just move it
tensor
}

/// Ensure all tensors in a vec are on the cached device
pub fn ensure_all_on_cached_device(tensors: Vec<Tensor>) -> flame_core::Result<Tensor> {
tensors.into_iter()
.map(|t| force_to_cached_device(&t))
.collect()
}
}
