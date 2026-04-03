use flame_core::device::Device;
use flame_core::optimizers::{Adam, SGD};
use flame_core::{DType, Result, Shape, Tensor};
use std::sync::Arc;

type FlameShape = flame_core::Shape;
type FlameDevice = flame_core::device::Device;

// Tensor conversion utilities - NO LONGER NEEDED
// This module is deprecated as we've fully migrated to FLAME
// Keeping stub functions to avoid breaking existing code

// Re-export FLAME types with old names for compatibility

/// No-op conversion - tensor is already FLAME
pub fn convert_to_flame_tensor(
    tensor: &Tensor,
    _device: Arc<FlameDevice>,
) -> flame_core::Result<Tensor> {
    Ok(tensor.clone())
}

/// No-op device conversion - device is already FLAME
pub fn convert_device(device: Arc<FlameDevice>) -> Arc<FlameDevice> {
    device
}

/// No-op device conversion - device is already FLAME
pub fn flame_device_to_flame(device: Arc<FlameDevice>) -> Arc<FlameDevice> {
    device
}
