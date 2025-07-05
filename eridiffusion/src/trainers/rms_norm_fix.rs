use anyhow::Result;
use candle_core::{DType, Device, Tensor, D};
use candle_nn::{Module, VarBuilder};

/// This module provides a fix for the RMS norm CUDA dispatch issue in candle.
/// The CUDA kernel exists but isn't being called due to a dispatch problem.
/// This implementation ensures CUDA kernels are used when available.

/// Check if candle was compiled with CUDA support
pub fn has_cuda_support() -> bool {
    cfg!(feature = "cuda")
}

/// Test if RMS norm works on CUDA
pub fn test_rms_norm_cuda() -> Result<()> {
    println!("\n=== Testing RMS Norm CUDA Support ===");
    
    // First check if CUDA is available
    match Device::cuda_if_available(0) {
        Ok(device) => {
            println!("✓ CUDA device found");
            
            // Create test tensors
            let x = Tensor::randn(0f32, 1.0, &[2, 8, 512], &device)?;
            let weight = Tensor::ones(&[512], DType::F32, &device)?;
            
            // Try the standard candle_nn rms_norm
            println!("Testing candle_nn::ops::rms_norm...");
            match candle_nn::ops::rms_norm(&x, &weight, 1e-6) {
                Ok(_) => {
                    println!("✓ Standard RMS norm works on CUDA!");
                    Ok(())
                }
                Err(e) => {
                    if e.to_string().contains("no cuda implementation") {
                        println!("✗ RMS norm CUDA dispatch failed: {}", e);
                        println!("  This confirms the bug - CUDA kernel exists but isn't being called");
                        
                        // Try our workaround
                        println!("\nTesting workaround implementation...");
                        let output = rms_norm_workaround(&x, &weight, 1e-6)?;
                        println!("✓ Workaround successful! Output shape: {:?}", output.shape());
                    }
                    Err(e.into())
                }
            }
        }
        Err(_) => {
            println!("No CUDA device available, skipping test");
            Ok(())
        }
    }
}

/// Workaround RMS norm implementation that uses GPU-friendly operations
pub fn rms_norm_workaround(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    // This implementation uses only operations that have CUDA support
    let x_dtype = x.dtype();
    
    // Use F32 for internal calculations for stability
    let x_f32 = match x_dtype {
        DType::F16 | DType::BF16 => x.to_dtype(DType::F32)?,
        _ => x.clone(),
    };
    
    let hidden_size = x_f32.dim(D::Minus1)? as f64;
    
    // All these operations have CUDA implementations
    let x_squared = x_f32.sqr()?;
    let mean = x_squared.sum_keepdim(D::Minus1)?.affine(1.0 / hidden_size, 0.0)?;
    let rsqrt = (mean.affine(1.0, eps as f64))?.sqrt()?.recip()?;
    let normalized = x_f32.broadcast_mul(&rsqrt)?;
    
    // Convert back to original dtype and apply weight
    let normalized = match x_dtype {
        DType::F16 => normalized.to_dtype(DType::F16)?,
        DType::BF16 => normalized.to_dtype(DType::BF16)?,
        _ => normalized,
    };
    
    normalized.broadcast_mul(weight).map_err(|e| e.into())
}

/// RMSNorm module that uses the workaround
#[derive(Clone)]
pub struct RMSNorm {
    weight: Tensor,
    eps: f64,
}

impl RMSNorm {
    pub fn new(weight: Tensor, eps: f64) -> Self {
        Self { weight, eps }
    }
    
    pub fn from_vars(dim: usize, eps: f64, vb: VarBuilder) -> candle_core::Result<Self> {
        let weight = vb.get(dim, "weight")?;
        Ok(Self::new(weight, eps))
    }
}

impl Module for RMSNorm {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        rms_norm_workaround(xs, &self.weight, self.eps as f32)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))
    }
}

/// Initialize the RMS norm fix
pub fn init_rms_norm_fix() -> Result<()> {
    println!("\n=== Initializing RMS Norm CUDA Fix ===");
    
    // Check CUDA compilation support
    if has_cuda_support() {
        println!("✓ Candle compiled with CUDA support");
    } else {
        println!("⚠ Candle NOT compiled with CUDA support!");
        println!("  To enable CUDA, rebuild with: cargo build --features cuda");
    }
    
    // Test the fix
    test_rms_norm_cuda()?;
    
    println!("\n=== RMS Norm Fix Ready ===");
    Ok(())
}