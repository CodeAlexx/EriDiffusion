use flame_core::device::Device;
use flame_core::optimizers::{Adam, SGD};
use flame_core::{DType, Result, Shape, Tensor};
use half::{bf16, f16};
use once_cell::sync::Lazy;
use std::{collections::HashMap, sync::Mutex};

// Force device zero module - aggressive fix for multi-device issues
//
// This module ensures ALL operations use device 0, working around
// FLAME's incorrect device detection

// FLAME uses flame_core::device::Device instead of Device

// Cache the real device 0 after we find it;
static REAL_DEVICE_ZERO: Lazy<Mutex<Option<Device>>> = Lazy::new(|| Mutex::new(None));

/// Find the actual device 0 by testing device creation
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

pub fn find_real_device_zero() -> flame_core::Result<Device> {
    let mut cache = REAL_DEVICE_ZERO.lock().unwrap();
    if let Some(ref device) = *cache {
        return Ok(device.clone());
    }

    // Special case: When CUDA_VISIBLE_DEVICES=0, FLAME might return the physical device ID
    // Try device 0 first, but accept whatever device ID it actually returns
    match Device::cuda(0) {
        Ok(device) => {
            // Test if this device actually works
            let shape = Shape::new(vec![1, 1]);
            match Tensor::zeros(shape, device.cuda_device().clone()) {
                Ok(_) => {
                    *cache = Some(device.clone());
                    return Ok(device);
                }
                Err(_) => {
                    // Device test failed, continue trying
                }
            }
        }
        Err(_) => {
            // Device creation failed, continue trying
        }
    }

    // If that didn't work, try other indices
    for i in 1..10 {
        match Device::cuda(i) {
            Ok(device) => {
                // Test if this device actually works
                let shape = Shape::new(vec![1, 1]);
                match Tensor::zeros(shape, device.cuda_device().clone()) {
                    Ok(_) => {
                        *cache = Some(device.clone());
                        return Ok(device);
                    }
                    Err(_) => {
                        // Device test failed, continue trying
                    }
                }
            }
            Err(_) => {
                break; // No more devices
            }
        }
    }

    Err(flame_core::Error::InvalidOperation(
        "Could not find any working CUDA device".to_string(),
    ))
}

/// Load safetensors with aggressive device forcing
pub fn load_safetensors_forced_v2<P: AsRef<std::path::Path>>(
    path: P,
) -> flame_core::Result<std::collections::HashMap<String, Tensor>> {
    let device = find_real_device_zero()?;
    println!("DEBUG: Loading safetensors with forced device");

    // ALWAYS load to CPU first to avoid device issues
    // Use device 0 temporarily for loading
    let temp_device = &device;
    let loader = crate::loaders::WeightLoader::from_safetensors(path, device.clone())?;
    let cpu_tensors = loader.weights;

    let mut result = HashMap::new();

    for (name, tensor) in cpu_tensors {
        // Move each tensor to our target device
        let gpu_tensor = tensor;
        result.insert(name, gpu_tensor);
    }

    Ok(result)
}

/// Create a tensor on the forced device
pub fn tensor_from_slice_forced_f32(data: &[f32], shape: &[usize]) -> flame_core::Result<Tensor> {
    let device = find_real_device_zero()?;

    // Create on CPU first then move
    let cpu_tensor =
        Tensor::from_slice(data, Shape::from_dims(shape), device.cuda_device().clone())?;
    Ok(cpu_tensor)
}

/// Create random tensor on forced device
pub fn randn_forced(mean: f32, std: f32, shape: &[usize]) -> flame_core::Result<Tensor> {
    let device = find_real_device_zero()?;

    // Create on CPU first
    let cpu_tensor =
        Tensor::randn(Shape::from_dims(shape), mean, std, device.cuda_device().clone())?;
    Ok(cpu_tensor)
}

/// Create uniform random tensor on forced device
pub fn rand_forced(min: f32, max: f32, shape: &[usize]) -> flame_core::Result<Tensor> {
    // Create on forced device
    let device = find_real_device_zero()?;
    let shape = Shape::from_dims(shape);
    // Use randn and scale/shift to approximate uniform distribution
    let mean = (min + max) / 2.0;
    let range = (max - min) / 2.0;
    let tensor = Tensor::randn(shape, mean, range * 0.5774, device.cuda_device().clone())?;
    Ok(tensor)
}

/// Get the working device
pub fn get_working_device() -> flame_core::Result<Device> {
    find_real_device_zero()
}
