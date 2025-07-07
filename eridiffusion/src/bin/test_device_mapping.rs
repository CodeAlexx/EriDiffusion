//! Test device mapping to understand Candle's behavior

use anyhow::Result;
use candle_core::{Device, Tensor, DType};

fn main() -> Result<()> {
    println!("=== DEVICE MAPPING TEST ===\n");
    
    // Check environment
    if let Ok(visible) = std::env::var("CUDA_VISIBLE_DEVICES") {
        println!("CUDA_VISIBLE_DEVICES: {}", visible);
    } else {
        println!("CUDA_VISIBLE_DEVICES: not set");
    }
    
    // Test device creation
    println!("\nTesting device creation:");
    for i in 0..5 {
        match Device::new_cuda(i) {
            Ok(device) => {
                println!("Device::new_cuda({}) => {:?}", i, device);
                
                // Try to create a tensor on this device
                match Tensor::zeros(&[1], DType::F32, &device) {
                    Ok(_) => println!("  ✓ Can create tensors on this device"),
                    Err(e) => println!("  ✗ Cannot create tensors: {}", e),
                }
            }
            Err(e) => {
                println!("Device::new_cuda({}) => Error: {}", i, e);
                break;
            }
        }
    }
    
    // Test safetensors loading
    println!("\nTesting tensor creation on each device:");
    
    // Get the working device
    let device = Device::new_cuda(0)?;
    println!("Using device: {:?}", device);
    
    // Create a test tensor
    let tensor = Tensor::randn(0.0, 1.0, &[2, 2], &device)?;
    println!("Created tensor on: {:?}", tensor.device());
    
    // Try moving between devices
    let cpu_tensor = tensor.to_device(&Device::Cpu)?;
    println!("Moved to CPU: {:?}", cpu_tensor.device());
    
    let gpu_tensor = cpu_tensor.to_device(&device)?;
    println!("Moved back to GPU: {:?}", gpu_tensor.device());
    
    println!("\n=== END TEST ===");
    Ok(())
}