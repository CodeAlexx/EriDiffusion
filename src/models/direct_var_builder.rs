//! Direct VarBuilder implementation that bypasses VarMap's Variable requirement
//! This allows us to load tensors without duplicating memory

use candle_core::{Device, DType, Result, Tensor, Error, Shape};
use candle_nn::{VarBuilder, Init};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// A custom backend for VarBuilder that stores tensors directly
pub struct DirectTensorBackend {
    tensors: Arc<Mutex<HashMap<String, Tensor>>>,
    dtype: DType,
    device: Device,
    prefix_stack: Arc<Mutex<Vec<String>>>,
}

impl DirectTensorBackend {
    pub fn new(tensors: HashMap<String, Tensor>, dtype: DType, device: Device) -> Self {
        Self {
            tensors: Arc::new(Mutex::new(tensors)),
            dtype,
            device,
            prefix_stack: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

/// Implement the SimpleBackend trait for our custom backend
impl candle_nn::var_builder::SimpleBackend for DirectTensorBackend {
    fn get(
        &self,
        s: Shape,
        name: &str,
        _h: Init,
        dtype: DType,
        dev: &Device,
    ) -> Result<Tensor> {
        let tensors = self.tensors.lock().unwrap();
        
        // Debug output for missing tensors
        if !tensors.contains_key(name) {
            println!("DirectTensorBackend: Looking for '{}' but not found", name);
            // Show similar keys
            let prefix = name.split('.').take(3).collect::<Vec<_>>().join(".");
            let similar: Vec<_> = tensors.keys()
                .filter(|k| k.starts_with(&prefix))
                .take(5)
                .collect();
            if !similar.is_empty() {
                println!("  Similar keys: {:?}", similar);
            }
        }
        
        // Direct lookup
        if let Some(tensor) = tensors.get(name) {
            // Verify shape matches
            if tensor.shape() != &s {
                return Err(Error::Msg(format!(
                    "shape mismatch for {}: expected {:?}, got {:?}",
                    name, s, tensor.shape()
                )));
            }
            
            // Return tensor on the requested device/dtype
            return tensor.to_device(dev)?.to_dtype(dtype);
        }
        
        // If not found, return error
        Err(Error::Msg(format!("cannot find tensor at path: {}", name)))
    }

    fn contains_tensor(&self, name: &str) -> bool {
        let tensors = self.tensors.lock().unwrap();
        tensors.contains_key(name)
    }
}

/// Create a VarBuilder with our custom backend
pub fn create_direct_var_builder(
    tensors: HashMap<String, Tensor>,
    dtype: DType,
    device: Device,
) -> VarBuilder<'static> {
    let backend = DirectTensorBackend::new(tensors, dtype, device.clone());
    VarBuilder::from_backend(Box::new(backend), dtype, device)
}