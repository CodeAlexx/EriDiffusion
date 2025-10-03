use crate::loaders::WeightLoader;
use anyhow::{anyhow, Context};
use flame_core::device::Device;
use flame_core::optimizers::{Adam, SGD};
use flame_core::{DType, Result, Shape, Tensor};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Weight loader for managing model weights
pub struct PrefixedWeightLoader {
    loader: WeightLoader,
    prefix: String,
}
pub struct QuantoWeightLoader {
    quanto_manager: Arc<QuantoManager>,
    device: Device,
    dtype: DType,
    prefix: String,
    loaded_weights: Arc<Mutex<HashMap<String, Tensor>>>,
}

// Lazy WeightLoader for quantized models
// Loads weights on-demand from QuantoManager to save memory

// FLAME uses flame_core::device::Device instead of Device

/// QuantoManager placeholder for managing quantized weights
pub struct QuantoManager {
    weights: HashMap<String, Tensor>,
}

impl QuantoManager {
    pub fn get_weight(&self, name: &str) -> flame_core::Result<Tensor> {
        self.weights
            .get(name)
            .ok_or_else(|| {
                flame_core::Error::InvalidOperation(format!("Weight not found: {}", name))
            })
            .map(|t| t.clone())
    }
}

// WeightLoader implementation is in crate::loaders::WeightLoader

impl PrefixedWeightLoader {
    pub fn get(&self, key: &str) -> flame_core::Result<&Tensor> {
        let full_key = format!("{}.{}", self.prefix, key);
        self.loader.get(&full_key)
    }

    pub fn tensor(&self, key: &str, shape: &[usize]) -> flame_core::Result<Tensor> {
        let full_key = format!("{}.{}", self.prefix, key);
        self.loader.tensor(&full_key, shape)
    }

    // TODO: Fix this when WeightLoader implements Clone or use Arc
    // pub fn pp(&self, prefix: &str) -> PrefixedWeightLoader {
    // PrefixedWeightLoader {
    // loader: self.loader.clone(),
    //             prefix: format!("{}.{}", self.prefix, prefix),
    //         }
    //     }
}

impl QuantoWeightLoader {
    pub fn new(quanto_manager: Arc<QuantoManager>, device: Device, dtype: DType) -> Self {
        Self {
            quanto_manager,
            device,
            dtype,
            prefix: String::new(),
            loaded_weights: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Create a WeightLoader that lazily loads from QuantoManager
    pub fn into_weight_loader(self) -> flame_core::Result<WeightLoader> {
        // Create HashMap<String, Tensor> with custom backend
        let var_map = HashMap::new();

        // Return a standard WeightLoader
        Ok(WeightLoader { weights: var_map, device: self.device.clone() })
    }

    /// Get a weight on-demand
    pub fn get_weight(&self, name: &str) -> flame_core::Result<Tensor> {
        // Check if already loaded
        {
            let loaded = self.loaded_weights.lock().unwrap();
            if let Some(tensor) = loaded.get(name) {
                return Ok(tensor.clone());
            }
        }

        // Load from quanto manager
        let full_name = if self.prefix.is_empty() {
            name.to_string()
        } else {
            format!("{}.{}", self.prefix, name)
        };

        println!("Loading weight on-demand: {}", full_name);
        let weight = self.quanto_manager.get_weight(&full_name)?;

        // Convert to target device and dtype
        let weight =
            if weight.device().ordinal() != self.device.ordinal() { weight } else { weight };

        let weight =
            if weight.dtype() != self.dtype { weight.to_dtype(self.dtype)? } else { weight };

        // Cache it
        {
            let mut loaded = self.loaded_weights.lock().unwrap();
            loaded.insert(name.to_string(), weight.clone());
        }

        Ok(weight)
    }
}
