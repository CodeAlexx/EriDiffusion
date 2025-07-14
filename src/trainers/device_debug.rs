//! Device debugging utilities to track down the CUDA mismatch issue

use candle_core::{Device, Tensor};
use std::collections::HashMap;
use std::sync::Mutex;

lazy_static::lazy_static! {
    static ref TENSOR_REGISTRY: Mutex<HashMap<String, DeviceInfo>> = Mutex::new(HashMap::new());
}

#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub device_debug: String,
    pub created_at: String,
    pub tensor_shape: Vec<usize>,
}

/// Register a tensor with debugging info
pub fn register_tensor(name: &str, tensor: &Tensor, location: &str) {
    let device_info = DeviceInfo {
        device_debug: format!("{:?}", tensor.device()),
        created_at: location.to_string(),
        tensor_shape: tensor.dims().to_vec(),
    };
    
    let mut registry = TENSOR_REGISTRY.lock().unwrap();
    registry.insert(name.to_string(), device_info);
}

/// Print all registered tensors and their devices
pub fn print_tensor_registry() {
    println!("\n=== TENSOR DEVICE REGISTRY ===");
    let registry = TENSOR_REGISTRY.lock().unwrap();
    
    // Group by device
    let mut by_device: HashMap<String, Vec<String>> = HashMap::new();
    
    for (name, info) in registry.iter() {
        by_device.entry(info.device_debug.clone())
            .or_insert_with(Vec::new)
            .push(format!("{} (created at: {})", name, info.created_at));
    }
    
    for (device, tensors) in by_device {
        println!("\nDevice: {}", device);
        for tensor in tensors {
            println!("  - {}", tensor);
        }
    }
    println!("==============================\n");
}

/// Check if all tensors have the same device
pub fn check_device_consistency() -> bool {
    let registry = TENSOR_REGISTRY.lock().unwrap();
    if registry.is_empty() {
        return true;
    }
    
    let devices: Vec<_> = registry.values()
        .map(|info| &info.device_debug)
        .collect();
    
    let first_device = &devices[0];
    let all_same = devices.iter().all(|d| d == first_device);
    
    if !all_same {
        println!("⚠️  DEVICE MISMATCH DETECTED!");
        print_tensor_registry();
    }
    
    all_same
}

/// Get device ID from debug string
pub fn extract_device_id(device: &Device) -> Option<usize> {
    let debug_str = format!("{:?}", device);
    // Parse "Cuda(CudaDevice(DeviceId(N)))" to extract N
    if let Some(start) = debug_str.find("DeviceId(") {
        let start = start + 9;
        if let Some(end) = debug_str[start..].find(')') {
            return debug_str[start..start + end].parse().ok();
        }
    }
    None
}

/// Compare two devices thoroughly
pub fn compare_devices(d1: &Device, d2: &Device, name1: &str, name2: &str) {
    println!("\n=== DEVICE COMPARISON ===");
    println!("{}: {:?}", name1, d1);
    println!("{}: {:?}", name2, d2);
    
    let id1 = extract_device_id(d1);
    let id2 = extract_device_id(d2);
    
    println!("Device IDs: {:?} vs {:?}", id1, id2);
    println!("Format match: {}", format!("{:?}", d1) == format!("{:?}", d2));
    
    // Try pointer comparison if both are CUDA
    match (d1, d2) {
        (Device::Cuda(_), Device::Cuda(_)) => {
            println!("Both are CUDA devices");
            // The issue is likely that Candle creates new CudaDevice structs
            // even for the same physical device
        }
        _ => println!("Not both CUDA devices"),
    }
    println!("========================\n");
}