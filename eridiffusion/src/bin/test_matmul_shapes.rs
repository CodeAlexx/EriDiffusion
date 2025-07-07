//! Test matmul with the exact shapes that are failing

use anyhow::Result;
use candle_core::{Device, Tensor, DType};

fn main() -> Result<()> {
    println!("=== Testing MatMul with Exact Shapes ===\n");
    
    // Create device
    let device = Device::new_cuda(0)?;
    println!("Device: {:?}", device);
    
    // Create tensors with exact shapes from error
    let lhs = Tensor::randn(0.0f32, 1.0, &[1, 4096, 3072], &device)?;
    let rhs = Tensor::randn(0.0f32, 1.0, &[3072, 16], &device)?;
    
    println!("LHS shape: {:?}", lhs.shape());
    println!("RHS shape: {:?}", rhs.shape());
    
    // Try matmul
    println!("\nTrying matmul...");
    match lhs.matmul(&rhs) {
        Ok(result) => {
            println!("✅ MatMul successful!");
            println!("Result shape: {:?}", result.shape());
        }
        Err(e) => {
            println!("❌ MatMul failed: {}", e);
            
            // Try with contiguous tensors
            println!("\nTrying with contiguous tensors...");
            let lhs_c = lhs.contiguous()?;
            let rhs_c = rhs.contiguous()?;
            
            match lhs_c.matmul(&rhs_c) {
                Ok(result) => {
                    println!("✅ MatMul with contiguous successful!");
                    println!("Result shape: {:?}", result.shape());
                }
                Err(e2) => {
                    println!("❌ MatMul with contiguous also failed: {}", e2);
                }
            }
        }
    }
    
    // Also test the transpose case
    println!("\n--- Testing transpose case ---");
    let a = Tensor::randn(0.0f32, 1.0, &[16, 3072], &device)?;
    let a_t = a.t()?;
    
    println!("a shape: {:?}", a.shape());
    println!("a_t shape: {:?}", a_t.shape());
    println!("a_t contiguous: {}", a_t.is_contiguous());
    
    // Make it contiguous
    let a_t_c = a_t.contiguous()?;
    println!("a_t_c contiguous: {}", a_t_c.is_contiguous());
    
    match lhs.matmul(&a_t_c) {
        Ok(result) => {
            println!("✅ MatMul with transposed tensor successful!");
            println!("Result shape: {:?}", result.shape());
        }
        Err(e) => {
            println!("❌ MatMul with transposed tensor failed: {}", e);
        }
    }
    
    println!("\n=== Test Complete ===");
    Ok(())
}