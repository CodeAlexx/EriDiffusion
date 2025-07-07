//! Force device zero module - aggressive fix for multi-device issues
//! 
//! This module ensures ALL operations use device 0, working around
//! Candle's incorrect device detection

use anyhow::Result;
use candle_core::{Device, Tensor, DType};
use std::sync::Mutex;
use std::collections::HashMap;
use once_cell::sync::Lazy;

// Cache the real device 0 after we find it
static REAL_DEVICE_ZERO: Lazy<Mutex<Option<Device>>> = Lazy::new(|| Mutex::new(None));

/// Find the actual device 0 by testing device creation
pub fn find_real_device_zero() -> Result<Device> {
    let mut cache = REAL_DEVICE_ZERO.lock().unwrap();
    if let Some(ref device) = *cache {
        return Ok(device.clone());
    }
    
    println!("DEBUG: Finding real device 0...");
    
    // Special case: When CUDA_VISIBLE_DEVICES=0, Candle might return the physical device ID
    // Try device 0 first, but accept whatever device ID it actually returns
    match Device::new_cuda(0) {
        Ok(device) => {
            println!("DEBUG: Device::new_cuda(0) returned {:?}", device);
            
            // Test if this device actually works
            match Tensor::zeros(&[1], DType::F32, &device) {
                Ok(_) => {
                    println!("DEBUG: Device works! Using {:?} as our device", device);
                    *cache = Some(device.clone());
                    return Ok(device);
                }
                Err(e) => {
                    println!("DEBUG: Device failed test: {}", e);
                }
            }
        }
        Err(e) => {
            println!("DEBUG: Device::new_cuda(0) failed: {}", e);
        }
    }
    
    // If that didn't work, try other indices
    for i in 1..10 {
        match Device::new_cuda(i) {
            Ok(device) => {
                println!("DEBUG: Device::new_cuda({}) returned {:?}", i, device);
                
                // Test if this device actually works
                match Tensor::zeros(&[1], DType::F32, &device) {
                    Ok(_) => {
                        println!("DEBUG: Device {} works! Using as device 0", i);
                        *cache = Some(device.clone());
                        return Ok(device);
                    }
                    Err(e) => {
                        println!("DEBUG: Device {} failed test: {}", i, e);
                    }
                }
            }
            Err(e) => {
                println!("DEBUG: Device::new_cuda({}) failed: {}", i, e);
                break; // No more devices
            }
        }
    }
    
    Err(anyhow::anyhow!("Could not find any working CUDA device"))
}

/// Load safetensors with aggressive device forcing
pub fn load_safetensors_forced_v2<P: AsRef<std::path::Path>>(
    path: P,
) -> Result<HashMap<String, Tensor>> {
    let device = find_real_device_zero()?;
    println!("DEBUG: Loading safetensors with forced device: {:?}", device);
    
    // ALWAYS load to CPU first to avoid device issues
    println!("DEBUG: Loading to CPU first to ensure correct device placement");
    let cpu_tensors = candle_core::safetensors::load(&path, &Device::Cpu)?;
    println!("DEBUG: Loaded {} tensors to CPU", cpu_tensors.len());
    
    let mut result = HashMap::new();
    
    for (name, tensor) in cpu_tensors {
        // Move each tensor to our target device
        let gpu_tensor = tensor.to_device(&device)?;
        println!("DEBUG: Moved {} to {:?}", name, gpu_tensor.device());
        result.insert(name, gpu_tensor);
    }
    
    println!("DEBUG: All tensors moved to target device");
    Ok(result)
}

/// Create a tensor on the forced device
pub fn create_tensor_forced<T: candle_core::WithDType>(
    data: &[T],
    shape: &[usize],
) -> Result<Tensor> {
    let device = find_real_device_zero()?;
    
    // Create on CPU first then move
    let cpu_tensor = Tensor::from_slice(data, shape, &Device::Cpu)?;
    Ok(cpu_tensor.to_device(&device)?)
}

/// Create random tensor on forced device
pub fn randn_forced(
    mean: f32,
    std: f32,
    shape: &[usize],
) -> Result<Tensor> {
    let device = find_real_device_zero()?;
    
    // Create on CPU first
    let cpu_tensor = Tensor::randn(mean, std, shape, &Device::Cpu)?;
    Ok(cpu_tensor.to_device(&device)?)
}

/// Create uniform random tensor on forced device
pub fn rand_forced(
    min: f32,
    max: f32,
    shape: &[usize],
) -> Result<Tensor> {
    let device = find_real_device_zero()?;
    
    // Create on CPU first
    let cpu_tensor = Tensor::rand(min, max, shape, &Device::Cpu)?;
    Ok(cpu_tensor.to_device(&device)?)
}

/// Get the working device
pub fn get_working_device() -> Result<Device> {
    find_real_device_zero()
}