//! Device fix module to ensure consistent device usage
//! 
//! This module provides utilities to work around multi-GPU device issues
//! when only one GPU is available but candle creates tensors on wrong devices

use anyhow::Result;
use candle_core::{Device as CandleDevice, Tensor};
use std::sync::Mutex;

// Global device to ensure consistency
static GLOBAL_DEVICE: Mutex<Option<CandleDevice>> = Mutex::new(None);

/// Initialize the global device
pub fn init_global_device() -> Result<()> {
    let mut global = GLOBAL_DEVICE.lock().unwrap();
    if global.is_none() {
        // Force device 0
        let device = CandleDevice::new_cuda(0)?;
        println!("DEBUG: Initialized global device to {:?}", device);
        *global = Some(device);
    }
    Ok(())
}

/// Get the global device (always returns cuda:0)
pub fn get_device() -> Result<CandleDevice> {
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
pub fn create_tensor_on_device<T: candle_core::WithDType>(
    data: &[T],
    shape: &[usize],
) -> Result<Tensor> {
    let device = get_device()?;
    Ok(Tensor::from_slice(data, shape, &device)?)
}

/// Ensure a tensor is on the correct device
pub fn ensure_device(tensor: &Tensor) -> Result<Tensor> {
    let device = get_device()?;
    if tensor.device().location() != device.location() {
        println!("DEBUG: Moving tensor from {:?} to {:?}", tensor.device(), device);
        Ok(tensor.to_device(&device)?)
    } else {
        Ok(tensor.clone())
    }
}

/// Load safetensors with forced device
pub fn load_safetensors_forced<P: AsRef<std::path::Path>>(
    path: P,
) -> Result<std::collections::HashMap<String, Tensor>> {
    let device = get_device()?;
    println!("DEBUG: Loading safetensors with forced device: {:?}", device);
    
    // Load to the forced device
    let tensors = candle_core::safetensors::load(path, &device)?;
    
    // Double-check all tensors are on the right device
    let mut fixed_tensors = std::collections::HashMap::new();
    for (name, tensor) in tensors {
        let fixed = ensure_device(&tensor)?;
        fixed_tensors.insert(name, fixed);
    }
    
    Ok(fixed_tensors)
}