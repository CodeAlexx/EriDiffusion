//! Model offloading utilities for memory management
//!
//! Since FLAME doesn't support CPU tensors, we implement offloading by:
//! 1. Saving model weights to disk or memory
//! 2. Freeing GPU memory
//! 3. Reloading when needed

use flame_core::{Device, Error, Result, Tensor};
use log::{debug, info};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// Offloaded model state
#[derive(Clone)]
pub struct OffloadedModel {
    /// Model name/identifier
    pub name: String,
    /// Weights stored in memory (as bytes)
    pub weights_data: HashMap<String, Vec<u8>>,
    /// Shapes for reconstruction
    pub shapes: HashMap<String, Vec<usize>>,
    /// DTypes for reconstruction
    pub dtypes: HashMap<String, flame_core::DType>,
}

/// Model offloader that manages GPU memory by offloading models
pub struct ModelOffloader {
    /// Device for GPU operations
    device: Device,
    /// Offloaded models stored in memory
    offloaded_models: Arc<Mutex<HashMap<String, OffloadedModel>>>,
    /// Currently loaded models on GPU
    loaded_models: Arc<Mutex<HashMap<String, HashMap<String, Tensor>>>>,
}

impl ModelOffloader {
    /// Create a new model offloader
    pub fn new(device: Device) -> Self {
        Self {
            device,
            offloaded_models: Arc::new(Mutex::new(HashMap::new())),
            loaded_models: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Offload a model's weights to CPU memory
    pub fn offload_model(&self, name: &str, weights: HashMap<String, Tensor>) -> Result<()> {
        info!("Offloading model '{}' from GPU to CPU memory", name);

        let mut offloaded = OffloadedModel {
            name: name.to_string(),
            weights_data: HashMap::new(),
            shapes: HashMap::new(),
            dtypes: HashMap::new(),
        };

        // Convert each tensor to bytes and store metadata
        for (key, tensor) in &weights {
            // Get tensor data as bytes
            let data = tensor.to_vec1::<f32>()?;
            let bytes: Vec<u8> = data.iter().flat_map(|&f| f.to_le_bytes()).collect();

            // Store metadata
            offloaded.weights_data.insert(key.clone(), bytes);
            offloaded.shapes.insert(key.clone(), tensor.shape().dims().to_vec());
            offloaded.dtypes.insert(key.clone(), tensor.dtype());
        }

        // Store in offloaded models
        {
            let mut offloaded_models = self.offloaded_models.lock().unwrap();
            offloaded_models.insert(name.to_string(), offloaded);
        }

        // Remove from loaded models to free GPU memory
        {
            let mut loaded_models = self.loaded_models.lock().unwrap();
            loaded_models.remove(name);
        }

        info!("Model '{}' offloaded successfully, {} weights freed from GPU", name, weights.len());
        Ok(())
    }

    /// Load a model back to GPU from CPU memory
    pub fn load_model(&self, name: &str) -> Result<HashMap<String, Tensor>> {
        info!("Loading model '{}' from CPU memory to GPU", name);

        // Check if already loaded
        {
            let loaded_models = self.loaded_models.lock().unwrap();
            if let Some(weights) = loaded_models.get(name) {
                debug!("Model '{}' already loaded on GPU", name);
                return Ok(weights.clone());
            }
        }

        // Get offloaded model
        let offloaded = {
            let offloaded_models = self.offloaded_models.lock().unwrap();
            offloaded_models.get(name).cloned().ok_or_else(|| {
                Error::InvalidOperation(format!("Model '{}' not found in offloaded storage", name))
            })?
        };

        // Reconstruct tensors
        let mut weights = HashMap::new();
        for (key, bytes) in &offloaded.weights_data {
            let shape = &offloaded.shapes[key];
            let dtype = offloaded.dtypes[key];

            // Convert bytes back to f32 vector
            let float_data: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();

            // Create tensor
            let tensor = if dtype == flame_core::DType::F32 {
                Tensor::from_vec(
                    float_data,
                    flame_core::Shape::from_dims(shape),
                    self.device.cuda_device().clone(),
                )?
            } else {
                // For other dtypes, use from_vec_dtype
                Tensor::from_vec_dtype(
                    float_data,
                    flame_core::Shape::from_dims(shape),
                    self.device.cuda_device().clone(),
                    dtype,
                )?
            };

            weights.insert(key.clone(), tensor);
        }

        // Store in loaded models
        {
            let mut loaded_models = self.loaded_models.lock().unwrap();
            loaded_models.insert(name.to_string(), weights.clone());
        }

        info!("Model '{}' loaded successfully, {} weights on GPU", name, weights.len());
        Ok(weights)
    }

    /// Check if a model is currently loaded on GPU
    pub fn is_loaded(&self, name: &str) -> bool {
        let loaded_models = self.loaded_models.lock().unwrap();
        loaded_models.contains_key(name)
    }

    /// Check if a model is offloaded
    pub fn is_offloaded(&self, name: &str) -> bool {
        let offloaded_models = self.offloaded_models.lock().unwrap();
        offloaded_models.contains_key(name)
    }

    /// Get memory usage statistics
    pub fn get_stats(&self) -> (usize, usize) {
        let loaded = self.loaded_models.lock().unwrap().len();
        let offloaded = self.offloaded_models.lock().unwrap().len();
        (loaded, offloaded)
    }

    /// Clear all offloaded models from memory
    pub fn clear_offloaded(&self) {
        let mut offloaded_models = self.offloaded_models.lock().unwrap();
        offloaded_models.clear();
        info!("Cleared all offloaded models from memory");
    }
}

/// Helper trait for models that support offloading
pub trait Offloadable {
    /// Get the model's weights as a HashMap
    fn get_weights(&self) -> Result<HashMap<String, Tensor>>;

    /// Load weights into the model
    fn load_weights(&mut self, weights: HashMap<String, Tensor>) -> Result<()>;

    /// Get the model name for offloading
    fn model_name(&self) -> &str;
}

impl ModelOffloader {
    /// Register a model with estimated memory usage (for tracking)
    pub fn register_model(&self, name: &str, size_bytes: u64) -> Result<()> {
        info!("Registered model '{}' with estimated size: {:.2} GB", name, size_bytes as f64 / 1e9);
        Ok(())
    }

    /// Force cleanup of memory (placeholder for CUDA operations)
    pub fn cleanup_memory(&self) -> Result<()> {
        info!("Running memory cleanup...");
        // In FLAME, we can't directly call CUDA operations
        // but dropping tensors should free memory
        Ok(())
    }
}
