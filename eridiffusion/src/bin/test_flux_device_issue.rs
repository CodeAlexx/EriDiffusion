//! Focused test to reproduce the Flux device issue

use anyhow::Result;
use candle_core::{Device, Tensor, DType, Module};
use candle_nn::{VarBuilder, VarMap, linear};

fn main() -> Result<()> {
    println!("=== Testing Flux Device Issue ===\n");
    
    // Test 1: Create model parameters with empty VarMap
    println!("Test 1: Empty VarMap model creation");
    let device = Device::new_cuda(0)?;
    println!("Created device: {:?}", device);
    
    let var_map = VarMap::new();
    let vb = VarBuilder::from_varmap(&var_map, DType::F32, &device);
    println!("Created VarBuilder with device: {:?}", device);
    
    // Create a linear layer
    let linear_layer = linear(10, 5, vb.pp("test_linear"))?;
    println!("Created linear layer");
    
    // Create input tensor
    let input = Tensor::randn(0.0f32, 1.0, &[2, 10], &device)?;
    println!("Input device: {:?}", input.device());
    
    // Try forward pass
    match linear_layer.forward(&input) {
        Ok(output) => println!("✅ Forward pass works: {:?}", output.shape()),
        Err(e) => println!("❌ Forward pass failed: {}", e),
    }
    
    // Test 2: Mixed tensor operations
    println!("\nTest 2: Cached tensor interaction");
    
    // Simulate cached tensor (created earlier)
    let cached_device = Device::new_cuda(0)?;
    let cached_tensor = Tensor::randn(0.0f32, 1.0, &[2, 10], &cached_device)?;
    println!("Cached tensor device: {:?}", cached_tensor.device());
    
    // Try to use with model
    match linear_layer.forward(&cached_tensor) {
        Ok(output) => println!("✅ Cached tensor forward works: {:?}", output.shape()),
        Err(e) => println!("❌ Cached tensor forward failed: {}", e),
    }
    
    // Test 3: VarMap parameters
    println!("\nTest 3: Accessing VarMap parameters");
    let vars = var_map.all_vars();
    println!("Number of variables in VarMap: {}", vars.len());
    
    for var in vars.iter().take(2) {
        println!("  Variable device: {:?}", var.device());
        
        // Try to use the variable's tensor
        let var_tensor = var.as_tensor();
        match var_tensor.matmul(&Tensor::ones(&[5, 3], DType::F32, &device)?) {
            Ok(_) => println!("    ✅ Can use variable tensor"),
            Err(e) => println!("    ❌ Cannot use variable tensor: {}", e),
        }
    }
    
    // Test 4: Device comparison at lower level
    println!("\nTest 4: Device internals");
    println!("Device format: {:?}", device);
    println!("Cached device format: {:?}", cached_device);
    
    // Test 5: Create tensors with get_or_init pattern
    println!("\nTest 5: OnceLock device pattern");
    use std::sync::OnceLock;
    static DEVICE: OnceLock<Device> = OnceLock::new();
    
    let d1 = DEVICE.get_or_init(|| Device::new_cuda(0).unwrap()).clone();
    let d2 = DEVICE.get_or_init(|| Device::new_cuda(0).unwrap()).clone();
    
    let t1 = Tensor::randn(0.0f32, 1.0, &[2, 2], &d1)?;
    let t2 = Tensor::randn(0.0f32, 1.0, &[2, 2], &d2)?;
    
    match t1.matmul(&t2) {
        Ok(_) => println!("✅ OnceLock pattern works"),
        Err(e) => println!("❌ OnceLock pattern failed: {}", e),
    }
    
    println!("\n=== Test Complete ===");
    Ok(())
}