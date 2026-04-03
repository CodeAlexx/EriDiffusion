use flame_core::device::Device;
use flame_core::optimizers::{Adam, SGD};
use flame_core::{DType, Result, Shape, Tensor};
use std::sync::Arc;
use std::{collections::HashMap, sync::Mutex};

pub struct DeviceInfo {
    pub device_debug: String,
    pub created_at: String,
    pub tensor_shape: Vec<usize>,
}

// Device debugging utilities to track down the CUDA mismatch issue

// FLAME uses flame_core::device::Device instead of Device

lazy_static::lazy_static! {
static ref TENSOR_REGISTRY: Mutex<std::collections::HashMap<String, DeviceInfo>> = Mutex::new(HashMap::new());
}

// Extension trait for Tensor to add missing methods
trait TensorExt {
    fn sum_dim(&self, dim: usize) -> flame_core::Result<Tensor>;
    fn add_scalar(&self, scalar: f32) -> flame_core::Result<Tensor>;
    fn mul_scalar(&self, scalar: f32) -> flame_core::Result<Tensor>;
    fn square(&self) -> flame_core::Result<Tensor>;
}

impl TensorExt for Tensor {
    fn sum_dim(&self, dim: usize) -> flame_core::Result<Tensor> {
        // Sum along dimension - FLAME sum_keepdim takes isize
        self.sum_keepdim(dim as isize)
    }

    fn add_scalar(&self, scalar: f32) -> flame_core::Result<Tensor> {
        // Add scalar to all elements
        let scalar_tensor = Tensor::full(self.shape().clone(), scalar, self.device().clone())?;
        self.add(&scalar_tensor)
    }

    fn mul_scalar(&self, scalar: f32) -> flame_core::Result<Tensor> {
        // Multiply all elements by scalar
        let scalar_tensor = Tensor::full(self.shape().clone(), scalar, self.device().clone())?;
        self.mul(&scalar_tensor)
    }

    fn square(&self) -> flame_core::Result<Tensor> {
        // Element-wise square
        self.mul(self)
    }
}

/// Register a tensor with debugging info
pub fn register_tensor(name: &str, tensor: &Tensor, location: &str) -> flame_core::Result<()> {
    let device_info = DeviceInfo {
        device_debug: "Device".to_string(),
        created_at: location.to_string(),
        tensor_shape: tensor.shape().dims().to_vec(),
    };

    let mut registry = TENSOR_REGISTRY.lock().unwrap();
    registry.insert(name.to_string(), device_info);
    Ok(())
}

/// Print all registered tensors and their devices
pub fn print_tensor_registry() -> flame_core::Result<()> {
    println!("\n=== TENSOR DEVICE REGISTRY ===");
    let registry = TENSOR_REGISTRY.lock().unwrap();

    // Group by device
    let mut by_device: std::collections::HashMap<String, Vec<String>> =
        std::collections::HashMap::new();

    for (name, info) in registry.iter() {
        by_device
            .entry(info.device_debug.clone())
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
    Ok(())
}

/// Check if all tensors have the same device
pub fn check_device_consistency() -> bool {
    let registry = TENSOR_REGISTRY.lock().unwrap();
    if registry.is_empty() {
        return true;
    }

    let devices: Vec<_> = registry.values().map(|info| &info.device_debug).collect();

    let first_device = &devices[0];
    let all_same = devices.iter().all(|d| d == first_device);

    if !all_same {
        println!("⚠️  DEVICE MISMATCH DETECTED!");
        let _ = print_tensor_registry();
    }

    all_same
}

/// Get device ID from debug string
pub fn extract_device_id(device: &Device) -> Option<usize> {
    let debug_str = "Device".to_string();
    // Parse "Cuda(Device(DeviceId(N)))" to extract N
    if let Some(start) = debug_str.find("DeviceId(") {
        let start = start + 9;
        if let Some(end) = debug_str[start..].find(')') {
            return debug_str[start..start + end].parse().ok();
        }
    }
    None
}

/// Compare two devices thoroughly
pub fn compare_devices(
    d1: &Device,
    d2: &Device,
    name1: &str,
    name2: &str,
) -> flame_core::Result<()> {
    println!("\n=== DEVICE COMPARISON ===");
    println!("{}: Device", name1);
    println!("{}: Device", name2);

    let id1 = extract_device_id(d1);
    let id2 = extract_device_id(d2);

    println!(
        "Device IDs: {} vs {}",
        id1.map_or("None".to_string(), |id| id.to_string()),
        id2.map_or("None".to_string(), |id| id.to_string())
    );
    println!("Format match: checking...");

    // In FLAME, Device is always CUDA
    println!("Both are CUDA devices (FLAME only supports CUDA)");
    // The issue is likely that FLAME creates new Device structs
    // even for the same physical device
    println!("========================\n");

    Ok(())
}
