//! Patched VarBuilder that works with Flux tensor name mapping
//! 
//! This is a modified version of candle-nn's VarBuilder that can handle
//! multiple possible paths for tensor lookup.

use candle_core::{DType, Device, Result, Shape, Tensor, Var};
use candle_nn::{Init, VarMap};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// A patched VarMap that tries multiple possible tensor paths
pub struct FluxVarMap {
    data: Arc<Mutex<HashMap<String, Var>>>,
    // Maps from requested paths to actual paths in the data
    path_mappings: Arc<Mutex<HashMap<String, String>>>,
}

impl FluxVarMap {
    pub fn new() -> Self {
        Self {
            data: Arc::new(Mutex::new(HashMap::new())),
            path_mappings: Arc::new(Mutex::new(HashMap::new())),
        }
    }
    
    /// Set a tensor with its original name
    pub fn set_one<S: AsRef<str>>(&self, name: S, tensor: Tensor) -> Result<()> {
        let name = name.as_ref();
        let var = Var::from_tensor(&tensor)?;
        self.data.lock().unwrap().insert(name.to_string(), var);
        Ok(())
    }
    
    /// Set a tensor with multiple possible paths
    pub fn set_with_paths<S: AsRef<str>>(&self, paths: Vec<S>, tensor: Tensor) -> Result<()> {
        if paths.is_empty() {
            return Err(candle_core::Error::Msg("No paths provided".to_string()));
        }
        
        // Store the tensor with the first path
        let primary_path = paths[0].as_ref().to_string();
        let var = Var::from_tensor(&tensor)?;
        self.data.lock().unwrap().insert(primary_path.clone(), var);
        
        // Map all other paths to the primary path
        let mut mappings = self.path_mappings.lock().unwrap();
        for path in paths.iter().skip(1) {
            mappings.insert(path.as_ref().to_string(), primary_path.clone());
        }
        
        Ok(())
    }
    
    /// Get a tensor, trying multiple possible paths
    pub fn get<S: Into<Shape>>(
        &self,
        shape: S,
        path: &str,
        init: Init,
        dtype: DType,
        device: &Device,
    ) -> Result<Tensor> {
        let shape = shape.into();
        let mut tensor_data = self.data.lock().unwrap();
        
        // First try direct lookup
        if let Some(tensor) = tensor_data.get(path) {
            let tensor_shape = tensor.shape();
            if &shape != tensor_shape {
                return Err(candle_core::Error::Msg(format!(
                    "shape mismatch on {}: {:?} <> {:?}", path, shape, tensor_shape
                )));
            }
            return Ok(tensor.as_tensor().clone());
        }
        
        // Try mapped paths
        let mappings = self.path_mappings.lock().unwrap();
        if let Some(actual_path) = mappings.get(path) {
            if let Some(tensor) = tensor_data.get(actual_path) {
                let tensor_shape = tensor.shape();
                if &shape != tensor_shape {
                    return Err(candle_core::Error::Msg(format!(
                        "shape mismatch on {} (mapped to {}): {:?} <> {:?}", 
                        path, actual_path, shape, tensor_shape
                    )));
                }
                return Ok(tensor.as_tensor().clone());
            }
        }
        
        // Try to find similar paths for debugging
        let similar: Vec<String> = tensor_data.keys()
            .filter(|k| k.contains(&path[path.len().saturating_sub(20)..]))
            .take(5)
            .cloned()
            .collect();
            
        if !similar.is_empty() {
            return Err(candle_core::Error::Msg(format!(
                "cannot find {} in VarMap. Similar keys: {:?}", path, similar
            )));
        }
        
        // Not found - create new variable
        Err(candle_core::Error::Msg(format!(
            "cannot find {} in VarMap", path
        )))
    }
    
    pub fn contains_key(&self, name: &str) -> bool {
        let data = self.data.lock().unwrap();
        if data.contains_key(name) {
            return true;
        }
        
        let mappings = self.path_mappings.lock().unwrap();
        if let Some(actual_path) = mappings.get(name) {
            return data.contains_key(actual_path);
        }
        
        false
    }
    
    pub fn all_vars(&self) -> HashMap<String, Var> {
        self.data.lock().unwrap().clone()
    }
}

/// Create a VarBuilder from a FluxVarMap
pub fn var_builder_from_flux_varmap(
    varmap: &FluxVarMap,
    dtype: DType,
    device: &Device,
) -> candle_nn::VarBuilder {
    // We need to convert our FluxVarMap to a regular VarMap
    // This is a workaround since we can't modify candle-nn directly
    let regular_varmap = VarMap::new();
    
    // Copy all tensors
    let data = varmap.data.lock().unwrap();
    for (name, var) in data.iter() {
        regular_varmap.data().lock().unwrap().insert(name.clone(), var.clone());
    }
    
    candle_nn::VarBuilder::from_varmap(&regular_varmap, dtype, device)
}