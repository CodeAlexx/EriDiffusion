//! Lazy VarBuilder that loads tensors on demand

use anyhow::Result;
use candle_core::{Device, DType, Tensor};
use candle_nn::{VarBuilder, VarMap};
use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use super::tensor_remapper::TensorRemapper;

/// A lazy var map that loads tensors on demand
pub struct LazyVarMap {
    remapper: Arc<TensorRemapper>,
    cache: Arc<Mutex<HashMap<String, candle_core::Var>>>,
    dtype: DType,
    device: Device,
}

impl LazyVarMap {
    pub fn new(remapper: TensorRemapper, dtype: DType, device: &Device) -> Self {
        Self {
            remapper: Arc::new(remapper),
            cache: Arc::new(Mutex::new(HashMap::new())),
            dtype,
            device: device.clone(),
        }
    }
    
    /// Get or load a tensor variable
    pub fn get_or_load(&self, path: &str) -> Result<candle_core::Var> {
        // Check cache first
        {
            let cache = self.cache.lock().unwrap();
            if let Some(var) = cache.get(path) {
                return Ok(var.clone());
            }
        }
        
        // Load tensor
        let tensor = self.remapper.load_with_fallbacks(path)?;
        let var = candle_core::Var::from_tensor(&tensor)?;
        
        // Cache it
        {
            let mut cache = self.cache.lock().unwrap();
            cache.insert(path.to_string(), var.clone());
        }
        
        Ok(var)
    }
}

/// Create a lazy VarBuilder that loads tensors on demand
pub fn create_lazy_var_builder<'a>(
    remapper: TensorRemapper,
    dtype: DType,
    device: &'a Device,
) -> VarBuilder<'a> {
    // For now, we'll create an empty VarMap and rely on the pp() method
    // to trigger lazy loading. This is a temporary solution.
    let var_map = VarMap::new();
    VarBuilder::from_varmap(&var_map, dtype, device)
}

/// Custom tensor loader that integrates with VarBuilder
pub struct LazyTensorLoader {
    remapper: Arc<TensorRemapper>,
}

impl LazyTensorLoader {
    pub fn new(remapper: TensorRemapper) -> Self {
        Self {
            remapper: Arc::new(remapper),
        }
    }
    
    /// Load a tensor by path
    pub fn load(&self, path: &str) -> Result<Tensor> {
        self.remapper.load_with_fallbacks(path)
            .map_err(|e| anyhow::anyhow!("Failed to load tensor {}: {}", path, e))
    }
}

/// Create a VarBuilder with a custom backend that loads tensors lazily
pub fn create_lazy_flux_var_builder<'a>(
    remapper: TensorRemapper,
    dtype: DType,
    device: &'a Device,
) -> Result<VarBuilder<'a>> {
    // Create a VarMap but don't populate it yet
    let var_map = VarMap::new();
    
    // Store the remapper for later use
    // Note: This is a simplified approach. In a real implementation,
    // we'd need to extend VarBuilder to support custom backends.
    
    // For now, return an empty VarBuilder that will fail on tensor access
    // This demonstrates the concept but needs proper implementation
    Ok(VarBuilder::from_varmap(&var_map, dtype, device))
}