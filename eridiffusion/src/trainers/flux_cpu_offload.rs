//! CPU offloading for Flux model weights
//! 
//! This allows training large models on limited VRAM by keeping
//! most weights on CPU and only loading them to GPU when needed.

use anyhow::{Result, Context};
use candle_core::{Tensor, Device, DType};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Manages weight offloading between CPU and GPU
#[derive(Clone)]
pub struct WeightOffloadManager {
    /// Weights stored on CPU
    cpu_weights: Arc<Mutex<HashMap<String, Tensor>>>,
    /// Active weights on GPU (limited cache)
    gpu_cache: Arc<Mutex<HashMap<String, Tensor>>>,
    /// Maximum number of weights to keep on GPU
    max_gpu_weights: usize,
    /// Access order for LRU eviction
    access_order: Arc<Mutex<Vec<String>>>,
    /// GPU device
    gpu_device: Device,
}

impl WeightOffloadManager {
    pub fn new(gpu_device: Device, max_gpu_weights: usize) -> Self {
        Self {
            cpu_weights: Arc::new(Mutex::new(HashMap::new())),
            gpu_cache: Arc::new(Mutex::new(HashMap::new())),
            max_gpu_weights,
            access_order: Arc::new(Mutex::new(Vec::new())),
            gpu_device,
        }
    }
    
    /// Store a weight on CPU
    pub fn store_weight(&self, name: String, weight: Tensor) -> Result<()> {
        // Move to CPU if not already there
        let cpu_weight = if weight.device().is_cpu() {
            weight
        } else {
            weight.to_device(&Device::Cpu)?
        };
        
        let mut cpu_weights = self.cpu_weights.lock().unwrap();
        cpu_weights.insert(name, cpu_weight);
        Ok(())
    }
    
    /// Get a weight, loading it to GPU if necessary
    pub fn get_weight(&self, name: &str) -> Result<Tensor> {
        // Check GPU cache first
        {
            let mut gpu_cache = self.gpu_cache.lock().unwrap();
            if let Some(weight) = gpu_cache.get(name) {
                // Update access order
                self.update_access_order(name);
                return Ok(weight.clone());
            }
        }
        
        // Not in GPU cache, load from CPU
        let cpu_weight = {
            let cpu_weights = self.cpu_weights.lock().unwrap();
            cpu_weights.get(name)
                .ok_or_else(|| anyhow::anyhow!("Weight '{}' not found", name))?
                .clone()
        };
        
        // Move to GPU
        let gpu_weight = cpu_weight.to_device(&self.gpu_device)?;
        
        // Add to GPU cache with eviction if needed
        self.add_to_gpu_cache(name.to_string(), gpu_weight.clone())?;
        
        Ok(gpu_weight)
    }
    
    /// Add weight to GPU cache with LRU eviction
    fn add_to_gpu_cache(&self, name: String, weight: Tensor) -> Result<()> {
        let mut gpu_cache = self.gpu_cache.lock().unwrap();
        let mut access_order = self.access_order.lock().unwrap();
        
        // Evict if cache is full
        while gpu_cache.len() >= self.max_gpu_weights && !gpu_cache.is_empty() {
            // Find least recently used
            if let Some(lru_name) = access_order.first() {
                let lru_name = lru_name.clone();
                gpu_cache.remove(&lru_name);
                access_order.retain(|n| n != &lru_name);
                println!("Evicted '{}' from GPU cache", lru_name);
            }
        }
        
        // Add new weight
        gpu_cache.insert(name.clone(), weight);
        access_order.push(name);
        
        Ok(())
    }
    
    /// Update access order for LRU
    fn update_access_order(&self, name: &str) {
        let mut access_order = self.access_order.lock().unwrap();
        access_order.retain(|n| n != name);
        access_order.push(name.to_string());
    }
    
    /// Clear GPU cache to free memory
    pub fn clear_gpu_cache(&self) {
        let mut gpu_cache = self.gpu_cache.lock().unwrap();
        let mut access_order = self.access_order.lock().unwrap();
        gpu_cache.clear();
        access_order.clear();
        println!("Cleared GPU cache");
    }
    
    /// Get memory usage statistics
    pub fn get_stats(&self) -> (usize, usize) {
        let cpu_weights = self.cpu_weights.lock().unwrap();
        let gpu_cache = self.gpu_cache.lock().unwrap();
        (cpu_weights.len(), gpu_cache.len())
    }
}