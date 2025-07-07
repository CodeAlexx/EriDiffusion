//! Test program to isolate the Candle device mismatch issue

use anyhow::Result;
use candle_core::{Device, Tensor, DType, Module};
use candle_nn::{VarBuilder, VarMap, linear, Linear};

fn main() -> Result<()> {
    println!("=== Testing Candle Device Issue ===\n");
    
    // Test 1: Basic tensor operations with cached device
    println!("Test 1: Basic tensor operations");
    let device = Device::new_cuda(0)?;
    let t1 = Tensor::randn(0.0f32, 1.0, &[2, 3], &device)?;
    let t2 = Tensor::randn(0.0f32, 1.0, &[3, 4], &device)?;
    let result = t1.matmul(&t2)?;
    println!("✅ Basic matmul works: shape {:?}", result.shape());
    
    // Test 2: Multiple device instances
    println!("\nTest 2: Multiple device instances");
    let device1 = Device::new_cuda(0)?;
    let device2 = Device::new_cuda(0)?;
    println!("Device 1: {:?}", device1);
    println!("Device 2: {:?}", device2);
    println!("Same device? {}", format!("{:?}", device1) == format!("{:?}", device2));
    
    let t3 = Tensor::randn(0.0f32, 1.0, &[2, 3], &device1)?;
    let t4 = Tensor::randn(0.0f32, 1.0, &[3, 4], &device2)?;
    match t3.matmul(&t4) {
        Ok(r) => println!("✅ Cross-device matmul works: shape {:?}", r.shape()),
        Err(e) => println!("❌ Cross-device matmul failed: {}", e),
    }
    
    // Test 3: VarMap and linear layers
    println!("\nTest 3: VarMap and linear layers");
    let var_map = VarMap::new();
    let vb = VarBuilder::from_varmap(&var_map, DType::F32, &device);
    let linear_layer = linear(10, 5, vb.pp("linear"))?;
    let input = Tensor::randn(0.0f32, 1.0, &[2, 10], &device)?;
    let output = linear_layer.forward(&input)?;
    println!("✅ Linear layer works: shape {:?}", output.shape());
    
    // Test 4: Mixed tensor creation
    println!("\nTest 4: Mixed tensor creation");
    let device3 = Device::new_cuda(0)?;
    let t5 = Tensor::zeros(&[2, 2], DType::F32, &device)?;
    let t6 = Tensor::ones(&[2, 2], DType::F32, &device3)?;
    match t5.add(&t6) {
        Ok(r) => println!("✅ Mixed device add works: shape {:?}", r.shape()),
        Err(e) => println!("❌ Mixed device add failed: {}", e),
    }
    
    // Test 5: Clone and move between devices
    println!("\nTest 5: Clone and device movement");
    let t7 = Tensor::randn(0.0f32, 1.0, &[2, 2], &device)?;
    let t7_clone = t7.clone();
    let t8 = Tensor::randn(0.0f32, 1.0, &[2, 2], &device)?;
    match t7_clone.matmul(&t8) {
        Ok(r) => println!("✅ Clone matmul works: shape {:?}", r.shape()),
        Err(e) => println!("❌ Clone matmul failed: {}", e),
    }
    
    // Test 6: Cached device pattern
    println!("\nTest 6: Cached device pattern");
    use std::sync::OnceLock;
    static CACHED_DEVICE: OnceLock<Device> = OnceLock::new();
    
    fn get_cached_device() -> Result<Device> {
        Ok(CACHED_DEVICE.get_or_init(|| {
            Device::new_cuda(0).unwrap_or(Device::Cpu)
        }).clone())
    }
    
    let cached1 = get_cached_device()?;
    let cached2 = get_cached_device()?;
    println!("Cached device 1: {:?}", cached1);
    println!("Cached device 2: {:?}", cached2);
    
    let t9 = Tensor::randn(0.0f32, 1.0, &[2, 3], &cached1)?;
    let t10 = Tensor::randn(0.0f32, 1.0, &[3, 4], &cached2)?;
    match t9.matmul(&t10) {
        Ok(r) => println!("✅ Cached device matmul works: shape {:?}", r.shape()),
        Err(e) => println!("❌ Cached device matmul failed: {}", e),
    }
    
    println!("\n=== Test Complete ===");
    Ok(())
}