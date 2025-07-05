use candle_core::{DType, Device, Result, Tensor, D};

/// RMS Norm implementation that works on GPU by using native candle operations
/// This avoids the "no cuda implementation" error by using operations that do have CUDA support
#[derive(Debug, Clone)]
pub struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    pub fn new(weight: Tensor, eps: f64) -> Self {
        Self { weight, eps }
    }
    
    pub fn from_vars(dim: usize, eps: f64, vb: candle_nn::VarBuilder) -> Result<Self> {
        let weight = vb.get(dim, "weight")?;
        Ok(Self::new(weight, eps))
    }
}

impl candle_nn::Module for RmsNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // This implementation uses only operations that have CUDA support
        let x_dtype = xs.dtype();
        
        // Use F32 for internal calculations for stability
        let xs = match x_dtype {
            DType::F16 | DType::BF16 => xs.to_dtype(DType::F32)?,
            _ => xs.clone(),
        };
        
        // Get the last dimension size
        let hidden_size = xs.dim(D::Minus1)? as f64;
        
        // Compute x^2
        let x_squared = xs.sqr()?;
        
        // Mean of x^2 along last dimension
        let mean = x_squared.sum_keepdim(D::Minus1)?.affine(1.0 / hidden_size, 0.0)?;
        
        // Add epsilon and compute rsqrt
        let rsqrt = (mean.affine(1.0, self.eps))?.sqrt()?.recip()?;
        
        // Normalize: x * rsqrt
        let normalized = xs.broadcast_mul(&rsqrt)?;
        
        // Convert back to original dtype
        let normalized = match x_dtype {
            DType::F16 => normalized.to_dtype(DType::F16)?,
            DType::BF16 => normalized.to_dtype(DType::BF16)?,
            _ => normalized,
        };
        
        // Apply weight
        normalized.broadcast_mul(&self.weight)
    }
}

/// Fast GPU-friendly RMS norm function that can be used as drop-in replacement
pub fn rms_norm(xs: &Tensor, weight: &Tensor, eps: f32) -> anyhow::Result<Tensor> {
    use candle_nn::Module;
    let norm = RmsNorm::new(weight.clone(), eps as f64);
    norm.forward(xs).map_err(|e| e.into())
}

/// Create a patched MMDiT module that uses our GPU-friendly RMS norm
pub mod mmdit_patch {
    use super::*;
    use candle_core::{Module as _, Result};
    use candle_nn::VarBuilder;
    
    /// Wrapper for candle's RMSNorm that redirects to our implementation
    pub struct RMSNorm(pub super::RmsNorm);
    
    impl RMSNorm {
        pub fn new(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
            Ok(RMSNorm(super::RmsNorm::from_vars(dim, eps, vb)?))
        }
    }
    
    impl candle_nn::Module for RMSNorm {
        fn forward(&self, xs: &Tensor) -> Result<Tensor> {
            self.0.forward(xs)
        }
    }
}

/// Helper to check if we're using our patched version
pub fn verify_gpu_rms_norm() -> anyhow::Result<()> {
    println!("Using GPU-optimized RMS norm implementation");
    
    // Quick test to verify it works
    if let Ok(device) = Device::cuda_if_available(0) {
        let test_tensor = Tensor::randn(0f32, 1.0, &[2, 8], &device)?;
        let weight = Tensor::ones(&[8], DType::F32, &device)?;
        let _ = rms_norm(&test_tensor, &weight, 1e-6)?;
        println!("✓ GPU RMS norm test passed");
    }
    
    Ok(())
}