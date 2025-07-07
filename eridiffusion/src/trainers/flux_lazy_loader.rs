//! Lazy loading for Flux model weights
//! 
//! This module provides a lazy loading mechanism that keeps weights on CPU
//! and only loads them to GPU when needed, avoiding OOM errors.

use anyhow::{Result, Context};
use candle_core::{Tensor, Device, DType};
use candle_nn::{VarBuilder, VarMap};
use std::collections::HashMap;
use std::path::Path;
use safetensors::SafeTensors;
use memmap2::Mmap;
use std::fs::File;
use std::sync::{Arc, Mutex};

use crate::models::flux_custom::{FluxConfig, FluxModelWithLoRA};
use crate::models::flux_custom::lora::LoRAConfig;

/// Lazy weight storage that keeps weights on CPU until needed
pub struct LazyWeightStorage {
    /// Memory-mapped file
    mmap: Arc<Mmap>,
    /// SafeTensors view
    tensors: SafeTensors<'static>,
    /// Tensor metadata
    metadata: HashMap<String, TensorInfo>,
    /// Cache of loaded tensors (on GPU)
    cache: Arc<Mutex<HashMap<String, Tensor>>>,
    /// Target device
    device: Device,
    /// Target dtype
    dtype: DType,
}

#[derive(Clone)]
struct TensorInfo {
    shape: Vec<usize>,
    dtype: DType,
    offset: usize,
    size: usize,
}

unsafe impl Send for LazyWeightStorage {}
unsafe impl Sync for LazyWeightStorage {}

impl LazyWeightStorage {
    pub fn new(model_path: &Path, device: Device, dtype: DType) -> Result<Self> {
        println!("Creating lazy weight storage for: {}", model_path.display());
        
        // Memory-map the file
        let file = File::open(model_path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let mmap = Arc::new(mmap);
        
        // Create SafeTensors view - need to leak to get 'static lifetime
        let tensors = unsafe {
            let mmap_ptr = mmap.as_ptr();
            let mmap_len = mmap.len();
            let mmap_slice = std::slice::from_raw_parts(mmap_ptr, mmap_len);
            SafeTensors::deserialize(mmap_slice)?
        };
        
        // Build metadata
        let mut metadata = HashMap::new();
        for name in tensors.names() {
            let tensor_view = tensors.tensor(name)?;
            let info = TensorInfo {
                shape: tensor_view.shape().to_vec(),
                dtype: convert_dtype(tensor_view.dtype())?,
                offset: 0, // SafeTensors handles offsets internally
                size: tensor_view.shape().iter().product::<usize>() * dtype_size(tensor_view.dtype()),
            };
            metadata.insert(name.to_string(), info);
        }
        
        println!("Indexed {} tensors for lazy loading", metadata.len());
        
        Ok(Self {
            mmap,
            tensors,
            metadata,
            cache: Arc::new(Mutex::new(HashMap::new())),
            device,
            dtype,
        })
    }
    
    /// Get a tensor by name, loading it if necessary
    pub fn get(&self, name: &str) -> Result<Tensor> {
        // Check cache first
        {
            let cache = self.cache.lock().unwrap();
            if let Some(tensor) = cache.get(name) {
                return Ok(tensor.clone());
            }
        }
        
        // Load from file
        let tensor = self.load_tensor(name)?;
        
        // Cache it
        {
            let mut cache = self.cache.lock().unwrap();
            cache.insert(name.to_string(), tensor.clone());
        }
        
        Ok(tensor)
    }
    
    /// Load a specific tensor from the file
    fn load_tensor(&self, name: &str) -> Result<Tensor> {
        let tensor_view = self.tensors.tensor(name)
            .with_context(|| format!("Tensor '{}' not found", name))?;
        
        // Load to CPU first
        let cpu_tensor = Tensor::from_raw_buffer(
            tensor_view.data(),
            convert_dtype(tensor_view.dtype())?,
            tensor_view.shape(),
            &Device::Cpu,
        )?;
        
        // Convert dtype if needed
        let cpu_tensor = if cpu_tensor.dtype() != self.dtype {
            cpu_tensor.to_dtype(self.dtype)?
        } else {
            cpu_tensor
        };
        
        // Move to target device
        Ok(cpu_tensor.to_device(&self.device)?)
    }
    
    /// Clear the cache to free GPU memory
    pub fn clear_cache(&self) {
        let mut cache = self.cache.lock().unwrap();
        cache.clear();
    }
    
    /// Get the size of a tensor without loading it
    pub fn tensor_size(&self, name: &str) -> Option<usize> {
        self.metadata.get(name).map(|info| info.size)
    }
    
    /// Get all tensor names
    pub fn tensor_names(&self) -> Vec<String> {
        self.metadata.keys().cloned().collect()
    }
}

/// Convert safetensors dtype to candle dtype
fn convert_dtype(dtype: safetensors::Dtype) -> Result<DType> {
    match dtype {
        safetensors::Dtype::F32 => Ok(DType::F32),
        safetensors::Dtype::F16 => Ok(DType::F16),
        safetensors::Dtype::BF16 => Ok(DType::BF16),
        safetensors::Dtype::U8 => Ok(DType::U8),
        _ => anyhow::bail!("Unsupported dtype: {:?}", dtype),
    }
}

/// Get the size in bytes of a dtype
fn dtype_size(dtype: safetensors::Dtype) -> usize {
    match dtype {
        safetensors::Dtype::F32 => 4,
        safetensors::Dtype::F16 => 2,
        safetensors::Dtype::BF16 => 2,
        safetensors::Dtype::U8 => 1,
        _ => 4, // Default to F32 size
    }
}

/// Lazy VarBuilder that loads weights on demand
pub struct LazyVarBuilder {
    storage: Arc<LazyWeightStorage>,
    prefix: String,
}

impl LazyVarBuilder {
    pub fn new(storage: Arc<LazyWeightStorage>) -> Self {
        Self {
            storage,
            prefix: String::new(),
        }
    }
    
    pub fn pp<S: Into<String>>(&self, prefix: S) -> Self {
        let new_prefix = if self.prefix.is_empty() {
            prefix.into()
        } else {
            format!("{}.{}", self.prefix, prefix.into())
        };
        
        Self {
            storage: self.storage.clone(),
            prefix: new_prefix,
        }
    }
    
    pub fn get<S: Into<String>>(&self, name: S, shape: &[usize]) -> Result<Tensor> {
        let name = name.into();
        let full_name = if self.prefix.is_empty() {
            name
        } else {
            format!("{}.{}", self.prefix, name)
        };
        
        let tensor = self.storage.get(&full_name)?;
        
        // Verify shape matches
        if tensor.dims() != shape {
            anyhow::bail!(
                "Shape mismatch for {}: expected {:?}, got {:?}",
                full_name, shape, tensor.dims()
            );
        }
        
        Ok(tensor)
    }
    
    pub fn get_with_hints<S: Into<String>>(
        &self,
        name: S,
        shape: &[usize],
        _dtype: DType,
    ) -> Result<Tensor> {
        self.get(name, shape)
    }
    
    pub fn device(&self) -> &Device {
        &self.storage.device
    }
    
    pub fn dtype(&self) -> DType {
        self.storage.dtype
    }
}

/// Weight name mapping for Flux
fn get_flux_weight_mapping() -> HashMap<String, String> {
    let mut mapping = HashMap::new();
    
    // Direct mappings
    mapping.insert("img_in.weight".to_string(), "img_in.weight".to_string());
    mapping.insert("img_in.bias".to_string(), "img_in.bias".to_string());
    mapping.insert("txt_in.weight".to_string(), "txt_in.weight".to_string());
    mapping.insert("txt_in.bias".to_string(), "txt_in.bias".to_string());
    
    // Time and vector embeddings
    mapping.insert("time_in.in_layer.weight".to_string(), "time_in.in_layer.weight".to_string());
    mapping.insert("time_in.in_layer.bias".to_string(), "time_in.in_layer.bias".to_string());
    mapping.insert("time_in.out_layer.weight".to_string(), "time_in.out_layer.weight".to_string());
    mapping.insert("time_in.out_layer.bias".to_string(), "time_in.out_layer.bias".to_string());
    
    mapping.insert("vector_in.in_layer.weight".to_string(), "vector_in.in_layer.weight".to_string());
    mapping.insert("vector_in.in_layer.bias".to_string(), "vector_in.in_layer.bias".to_string());
    mapping.insert("vector_in.out_layer.weight".to_string(), "vector_in.out_layer.weight".to_string());
    mapping.insert("vector_in.out_layer.bias".to_string(), "vector_in.out_layer.bias".to_string());
    
    // Final layer
    mapping.insert("final_layer.linear.weight".to_string(), "final_layer.linear.weight".to_string());
    mapping.insert("final_layer.linear.bias".to_string(), "final_layer.linear.bias".to_string());
    
    // Guidance embeddings if present
    mapping.insert("guidance_in.in_layer.weight".to_string(), "guidance_in.in_layer.weight".to_string());
    mapping.insert("guidance_in.in_layer.bias".to_string(), "guidance_in.in_layer.bias".to_string());
    mapping.insert("guidance_in.out_layer.weight".to_string(), "guidance_in.out_layer.weight".to_string());
    mapping.insert("guidance_in.out_layer.bias".to_string(), "guidance_in.out_layer.bias".to_string());
    
    // Double blocks (0-18)
    for i in 0..19 {
        let prefix = format!("double_blocks.{}", i);
        // Add all the layer mappings for double blocks
        mapping.insert(format!("{}.img_norm1.linear.weight", prefix), format!("{}.img_norm1.linear.weight", prefix));
        mapping.insert(format!("{}.img_norm1.linear.bias", prefix), format!("{}.img_norm1.linear.bias", prefix));
        mapping.insert(format!("{}.txt_norm1.linear.weight", prefix), format!("{}.txt_norm1.linear.weight", prefix));
        mapping.insert(format!("{}.txt_norm1.linear.bias", prefix), format!("{}.txt_norm1.linear.bias", prefix));
        // ... add more mappings as needed
    }
    
    // Single blocks (0-37)
    for i in 0..38 {
        let prefix = format!("single_blocks.{}", i);
        // Add single block mappings
        mapping.insert(format!("{}.norm.linear.weight", prefix), format!("{}.norm.linear.weight", prefix));
        mapping.insert(format!("{}.norm.linear.bias", prefix), format!("{}.norm.linear.bias", prefix));
        // ... add more mappings as needed
    }
    
    mapping
}

/// Lazy weight storage with name remapping
pub struct LazyWeightStorageWithRemap {
    base_storage: LazyWeightStorage,
    name_mapping: HashMap<String, String>,
}

impl LazyWeightStorageWithRemap {
    pub fn new(base_storage: LazyWeightStorage) -> Self {
        Self {
            base_storage,
            name_mapping: get_flux_weight_mapping(),
        }
    }
    
    pub fn get(&self, name: &str) -> Result<Tensor> {
        // Try direct name first
        if let Ok(tensor) = self.base_storage.get(name) {
            return Ok(tensor);
        }
        
        // Try mapped name
        if let Some(mapped_name) = self.name_mapping.get(name) {
            return self.base_storage.get(mapped_name);
        }
        
        // If neither works, return error
        anyhow::bail!("Tensor '{}' not found (tried direct and mapped names)", name)
    }
}

/// Load Flux model with lazy loading  
pub fn load_flux_lazy(
    model_path: &Path,
    flux_config: &FluxConfig,
    lora_config: &LoRAConfig,
    device: Device,
    dtype: DType,
) -> Result<(FluxModelWithLoRA, Arc<LazyWeightStorage>)> {
    println!("=== Lazy Loading Flux Model ===");
    
    // For now, return error - we need to implement proper lazy loading in FluxModelWithLoRA first
    anyhow::bail!("Lazy loading not yet fully implemented - need to update FluxModelWithLoRA");
}