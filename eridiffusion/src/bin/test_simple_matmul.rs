//! Test simple 2D matmul

use anyhow::Result;
use candle_core::{Device, Tensor, DType};

fn main() -> Result<()> {
    println!("=== Testing Simple MatMul ===\n");
    
    // Create device
    let device = Device::new_cuda(0)?;
    println!("Device: {:?}", device);
    
    // Test 1: Simple 2D matmul
    println!("\nTest 1: 2D matmul");
    let a = Tensor::randn(0.0f32, 1.0, &[10, 20], &device)?;
    let b = Tensor::randn(0.0f32, 1.0, &[20, 5], &device)?;
    
    println!("a shape: {:?}", a.shape());
    println!("b shape: {:?}", b.shape());
    
    match a.matmul(&b) {
        Ok(c) => println!("✅ 2D matmul works! Result shape: {:?}", c.shape()),
        Err(e) => println!("❌ 2D matmul failed: {}", e),
    }
    
    // Test 2: 3D matmul (batch matmul)
    println!("\nTest 2: 3D matmul (batch)");
    let a3d = Tensor::randn(0.0f32, 1.0, &[2, 10, 20], &device)?;
    let b2d = Tensor::randn(0.0f32, 1.0, &[20, 5], &device)?;
    
    println!("a3d shape: {:?}", a3d.shape());
    println!("b2d shape: {:?}", b2d.shape());
    
    match a3d.matmul(&b2d) {
        Ok(c) => println!("✅ 3D @ 2D matmul works! Result shape: {:?}", c.shape()),
        Err(e) => println!("❌ 3D @ 2D matmul failed: {}", e),
    }
    
    // Test 3: The exact failing case but smaller
    println!("\nTest 3: Smaller version of failing case");
    let x = Tensor::randn(0.0f32, 1.0, &[1, 10, 20], &device)?;
    let w = Tensor::randn(0.0f32, 1.0, &[20, 5], &device)?;
    
    println!("x shape: {:?}", x.shape());
    println!("w shape: {:?}", w.shape());
    
    match x.matmul(&w) {
        Ok(y) => println!("✅ Small 3D @ 2D works! Result shape: {:?}", y.shape()),
        Err(e) => println!("❌ Small 3D @ 2D failed: {}", e),
    }
    
    // Test 4: Try flattening first dimension
    println!("\nTest 4: Flatten and matmul");
    let x_flat = x.reshape((10, 20))?;
    println!("x_flat shape: {:?}", x_flat.shape());
    
    match x_flat.matmul(&w) {
        Ok(y) => {
            println!("✅ Flattened matmul works! Result shape: {:?}", y.shape());
            // Reshape back
            let y_3d = y.reshape((1, 10, 5))?;
            println!("Reshaped back to: {:?}", y_3d.shape());
        }
        Err(e) => println!("❌ Flattened matmul failed: {}", e),
    }
    
    println!("\n=== Test Complete ===");
    Ok(())
}