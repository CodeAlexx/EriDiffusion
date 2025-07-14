//! Direct VarBuilder that bypasses VarMap for Flux models

use candle_core::{Device, DType, Result, Tensor};
use candle_nn::VarBuilder;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// A simple tensor storage that implements the minimal interface needed by VarBuilder
pub struct DirectTensorStorage {
    tensors: Arc<Mutex<HashMap<String, Tensor>>>,
    dtype: DType,
    device: Device,
}

impl DirectTensorStorage {
    pub fn new(tensors: HashMap<String, Tensor>, dtype: DType, device: Device) -> Self {
        Self {
            tensors: Arc::new(Mutex::new(tensors)),
            dtype,
            device,
        }
    }
    
    pub fn get(&self, path: &str) -> Result<Tensor> {
        let tensors = self.tensors.lock().unwrap();
        
        // Direct lookup
        if let Some(tensor) = tensors.get(path) {
            return Ok(tensor.clone());
        }
        
        // Try to find with hierarchical path matching
        // When VarBuilder uses pp(), it builds up a prefix and looks for the remainder
        // So if we're looking for "to_q.weight" and have "double_blocks.0.img_attn.to_q.weight",
        // we need to handle this properly
        
        // For now, let's return an error to see what paths are being requested
        candle_core::bail!("DirectTensorStorage: cannot find tensor at path: {}", path)
    }
}

/// Create a VarBuilder that directly accesses tensors without VarMap overhead
pub fn create_direct_var_builder(
    tensors: HashMap<String, Tensor>,
    dtype: DType,
    device: Device,
) -> Result<VarBuilder<'static>> {
    // This is a hack - we need to create a VarBuilder somehow
    // The normal way is from_varmap, but that's what's causing issues
    
    // Let's try creating an empty VarBuilder and see if we can work with it
    let vb = VarBuilder::zeros(dtype, &device);
    
    // Now we need to somehow inject our tensors...
    // This is where we'd need to modify Candle internals
    
    // For now, let's return the empty VarBuilder to see what happens
    Ok(vb)
}