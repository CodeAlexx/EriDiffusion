//! Single device enforcer - ensures ALL operations use the same device
//! 
//! This is a nuclear option to fix the device chaos

use anyhow::Result;
use candle_core::{Device, Tensor, DType};
use std::sync::Mutex;
use once_cell::sync::Lazy;

// The ONE TRUE DEVICE - set once and used everywhere
static THE_DEVICE: Lazy<Mutex<Option<Device>>> = Lazy::new(|| Mutex::new(None));

/// Initialize with the first working device we find
pub fn init_single_device() -> Result<Device> {
    let mut device_lock = THE_DEVICE.lock().unwrap();
    
    if let Some(ref device) = *device_lock {
        return Ok(device.clone());
    }
    
    println!("\n=== FINDING THE ONE TRUE DEVICE ===");
    
    // Try device 0 first
    match Device::new_cuda(0) {
        Ok(device) => {
            // Test it works
            match Tensor::zeros(&[1], DType::F32, &device) {
                Ok(_) => {
                    println!("THE ONE TRUE DEVICE IS: {:?}", device);
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
    
    Err(anyhow::anyhow!("Could not find any working CUDA device"))
}

/// Get the one true device - ALWAYS returns the same device
pub fn get_device() -> Result<Device> {
    let device_lock = THE_DEVICE.lock().unwrap();
    
    if let Some(ref device) = *device_lock {
        Ok(device.clone())
    } else {
        drop(device_lock);
        init_single_device()
    }
}

/// Create a new cuda device that actually returns our single device
pub fn new_cuda(_ordinal: usize) -> Result<Device> {
    // Ignore the ordinal, always return THE device
    get_device()
}

/// Ensure a tensor is on THE device
pub fn ensure_single_device(tensor: &Tensor) -> Result<Tensor> {
    let device = get_device()?;
    
    if format!("{:?}", tensor.device()) != format!("{:?}", device) {
        println!("WARNING: Moving tensor from {:?} to {:?}", tensor.device(), device);
        Ok(tensor.to_device(&device)?)
    } else {
        Ok(tensor.clone())
    }
}

/// Create tensor on THE device
pub fn zeros(shape: &[usize], dtype: DType) -> Result<Tensor> {
    let device = get_device()?;
    Ok(Tensor::zeros(shape, dtype, &device)?)
}

/// Create random tensor on THE device
pub fn randn(mean: f32, std: f32, shape: &[usize]) -> Result<Tensor> {
    let device = get_device()?;
    
    // Create on CPU first to avoid device issues
    let cpu_tensor = Tensor::randn(mean, std, shape, &Device::Cpu)?;
    Ok(cpu_tensor.to_device(&device)?)
}

/// Create uniform random tensor on THE device
pub fn rand(min: f32, max: f32, shape: &[usize]) -> Result<Tensor> {
    let device = get_device()?;
    
    // Create on CPU first to avoid device issues
    let cpu_tensor = Tensor::rand(min, max, shape, &Device::Cpu)?;
    Ok(cpu_tensor.to_device(&device)?)
}

/// Load safetensors to THE device
pub fn load_safetensors<P: AsRef<std::path::Path>>(
    path: P,
) -> Result<std::collections::HashMap<String, Tensor>> {
    let device = get_device()?;
    
    println!("Loading safetensors to THE device: {:?}", device);
    
    // Always load to CPU first
    let cpu_tensors = candle_core::safetensors::load(&path, &Device::Cpu)?;
    
    let mut result = std::collections::HashMap::new();
    for (name, tensor) in cpu_tensors {
        let gpu_tensor = tensor.to_device(&device)?;
        result.insert(name, gpu_tensor);
    }
    
    Ok(result)
}