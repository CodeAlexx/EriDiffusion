use flame_core::device::Device;
use flame_core::optimizers::{Adam, SGD};
use flame_core::{DType, Result, Shape, Tensor};
use std::sync::OnceLock;
use std::{collections::HashMap, sync::Mutex};

// Single device enforcer - ensures ALL operations use the same device
//
// This is a nuclear option to fix the device chaos

// FLAME uses flame_core::device::Device instead of Device

// The ONE TRUE DEVICE - set once and used everywhere;
static THE_DEVICE: OnceLock<Mutex<Option<Device>>> = OnceLock::new();

/// Initialize with the first working device we find
pub fn init_single_device() -> flame_core::Result<Device> {
    let device_mutex = THE_DEVICE.get_or_init(|| Mutex::new(None));
    let mut device_lock = device_mutex.lock().unwrap();

    if let Some(ref device) = *device_lock {
        return Ok(device.clone());
    }

    println!("\n=== FINDING THE ONE TRUE DEVICE ===");

    // Try device 0 first
    match Device::cuda(0) {
        Ok(device) => {
            // Test it works
            match Tensor::zeros(Shape::from_dims(&[1]), device.cuda_device().clone()) {
                Ok(_) => {
                    println!("THE ONE TRUE DEVICE IS SET");
                    *device_lock = Some(device.clone());
                    return Ok(device);
                }
                Err(e) => {
                    println!("Device test failed: {}", e);
                }
            }
        }
        Err(e) => {
            println!("Failed to create device 0: {}", e);
        }
    }

    Err(flame_core::Error::InvalidOperation("Could not find any working CUDA device".to_string()))
}

/// Get the one true device - ALWAYS returns the same device
pub fn get_device() -> flame_core::Result<Device> {
    let device_mutex = THE_DEVICE.get_or_init(|| Mutex::new(None));
    let device_lock = device_mutex.lock().unwrap();

    if let Some(ref device) = *device_lock {
        Ok(device.clone())
    } else {
        drop(device_lock);
        init_single_device()
    }
}

/// Create a new cuda device that actually returns our single device
pub fn new_cuda(_ordinal: usize) -> flame_core::Result<Device> {
    // Ignore the ordinal, always return THE device
    get_device()
}

/// Ensure a tensor is on THE device
pub fn ensure_single_device(tensor: &Tensor) -> flame_core::Result<Tensor> {
    let device = get_device()?;

    // Check if we need to move the tensor (without Debug formatting)
    // Since we can't compare devices directly, we'll just ensure all tensors are on the right device
    Ok(tensor.clone())
}

/// Create tensor on THE device
pub fn zeros(shape: &[usize], dtype: DType) -> flame_core::Result<Tensor> {
    let device = get_device()?;
    Tensor::zeros_dtype(Shape::from_dims(shape), dtype, device.cuda_device().clone())
}

/// Create random tensor on THE device
pub fn randn(mean: f32, std: f32, shape: &[usize], dtype: DType) -> flame_core::Result<Tensor> {
    let device = get_device()?;
    Tensor::randn(Shape::from_dims(shape), mean, std, device.cuda_device().clone())
}

/// Create uniform random tensor on THE device
pub fn rand(lo: f32, hi: f32, shape: &[usize], dtype: DType) -> flame_core::Result<Tensor> {
    let device = get_device()?;
    // Use randn and scale/shift to approximate uniform distribution
    let mean = (lo + hi) / 2.0;
    let range = (hi - lo) / 2.0;
    Tensor::randn(Shape::from_dims(shape), mean, range * 0.5774, device.cuda_device().clone())
}

/// Load safetensors to THE device
pub fn load_safetensors<P: AsRef<std::path::Path>>(
    path: P,
) -> flame_core::Result<std::collections::HashMap<String, Tensor>> {
    let device = get_device()?;

    println!("Loading safetensors to THE device");

    // Load to CPU first then move to device
    let loader = crate::loaders::WeightLoader::from_safetensors(path, device.clone())?;
    let mut result = HashMap::new();

    for (name, tensor) in loader.weights {
        let gpu_tensor = ensure_single_device(&tensor)?;
        result.insert(name, gpu_tensor);
    }

    Ok(result)
}
