use flame_core::device::Device;
use flame_core::optimizers::{Adam, SGD};
use flame_core::{DType, Result, Shape, Tensor};
use half::{bf16, f16};
use std::{collections::HashMap, sync::Mutex};

// Device fix module to ensure consistent device usage
//
// This module provides utilities to work around multi-GPU device issues
// when only one GPU is available but FLAME creates tensors on wrong devices

// FLAME uses flame_core::device::Device instead of Device

// Global device to ensure consistency
static GLOBAL_DEVICE: Mutex<Option<Device>> = Mutex::new(None);

/// Initialize the global device
pub trait WithDType {
    fn dtype() -> DType;
}

// bf16 and f16 are already imported from half crate above

impl WithDType for f32 {
    fn dtype() -> DType {
        DType::F32
    }
}
impl WithDType for f16 {
    fn dtype() -> DType {
        DType::F16
    }
}
impl WithDType for bf16 {
    fn dtype() -> DType {
        DType::BF16
    }
}

pub fn init_global_device() -> flame_core::Result<()> {
    let mut global = GLOBAL_DEVICE.lock().unwrap();
    if global.is_none() {
        // Force device 0
        let device = Device::cuda(0)?;
        *global = Some(device);
    }
    Ok(())
}

/// Get the global device (always returns cuda:0)
pub fn get_device() -> flame_core::Result<Device> {
    let global = GLOBAL_DEVICE.lock().unwrap();
    if let Some(ref device) = *global {
        Ok(device.clone())
    } else {
        // Initialize if not done
        drop(global);
        init_global_device()?;
        let global = GLOBAL_DEVICE.lock().unwrap();
        Ok(global.as_ref().unwrap().clone())
    }
}

/// Create a tensor with the global device
pub fn create_tensor_f32(data: &[f32], shape: &[usize]) -> flame_core::Result<Tensor> {
    let device = get_device()?;
    Ok(Tensor::from_slice(data, Shape::from_dims(shape), device.cuda_device().clone())?)
}

/// Ensure a tensor is on the correct device
pub fn ensure_device(tensor: &Tensor) -> flame_core::Result<Tensor> {
    let device = get_device()?;
    // Compare device ordinals to check if they're the same
    if tensor.device().ordinal() != device.ordinal() {
        // Different device, need to transfer by copying data
        let data = tensor.to_vec()?;
        Tensor::from_vec(data, tensor.shape().clone(), device.cuda_device().clone())
    } else {
        // Same device, just clone
        Ok(tensor.clone())
    }
}

/// Load safetensors with forced device
pub fn load_safetensors_forced<P: AsRef<std::path::Path>>(
    path: P,
) -> flame_core::Result<HashMap<String, Tensor>> {
    let device = get_device()?;

    // Load to the forced device
    let loader = crate::loaders::WeightLoader::from_safetensors(path, device.clone())?;

    // Double-check all tensors are on the right device
    let mut fixed_tensors = std::collections::HashMap::new();
    for (name, tensor) in loader.weights {
        let fixed = ensure_device(&tensor)?;
        fixed_tensors.insert(name, fixed);
    }

    Ok(fixed_tensors)
}
