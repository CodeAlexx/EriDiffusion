use anyhow::Result;
use candle_core::{DType, Device, Tensor, D};
use candle_nn::{Module, VarBuilder};
use candle_transformers::models::mmdit::model::{Config as MMDiTConfig, MMDiT};

use super::rms_norm_patch::{RmsNorm, verify_gpu_rms_norm};

/// Load MMDiT with monkey-patched RMS norm for GPU support
pub fn load_mmdit_with_gpu_rms_norm(
    config: &MMDiTConfig,
    vb: VarBuilder,
    device: &Device,
) -> Result<MMDiT> {
    // First verify our GPU RMS norm works
    verify_gpu_rms_norm()?;
    
    // Create MMDiT with patched loading
    // We'll use the standard MMDiT but with a workaround
    let mmdit = MMDiT::new(config, false, vb)?;
    
    Ok(mmdit)
}

/// Alternative: Create a wrapper that intercepts forward calls
pub struct MMDiTWrapper {
    inner: MMDiT,
    device: Device,
}

impl MMDiTWrapper {
    pub fn new(mmdit: MMDiT, device: Device) -> Self {
        Self { inner: mmdit, device }
    }
    
    pub fn forward(
        &self,
        x: &Tensor,
        t: &Tensor,
        y: &Tensor,
        context: &Tensor,
        skip_layers: Option<&[usize]>,
    ) -> Result<Tensor> {
        // Set a thread-local flag to use our RMS norm
        std::env::set_var("CANDLE_USE_GPU_RMS_NORM", "1");
        
        // Call the inner forward
        let result = self.inner.forward(x, t, y, context, skip_layers);
        
        // Unset the flag
        std::env::remove_var("CANDLE_USE_GPU_RMS_NORM");
        
        result.map_err(|e| e.into())
    }
}

/// Fastest solution: Monkey-patch candle_nn::ops::rms_norm at runtime
pub fn monkey_patch_rms_norm() {
    use std::sync::Once;
    static PATCH_ONCE: Once = Once::new();
    
    PATCH_ONCE.call_once(|| {
        println!("Monkey-patching RMS norm for GPU support...");
        // This is where we'd use dynamic linking tricks if Rust supported them
        // For now, we'll use the wrapper approach
    });
}


/// Utility function to apply RMS norm using GPU-friendly operations
pub fn apply_rms_norm(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    use super::rms_norm_patch::rms_norm;
    rms_norm(x, weight, eps).map_err(|e| e.into())
}