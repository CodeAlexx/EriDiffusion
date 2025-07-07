//! Lazy loading for large safetensors files
//!
//! This module provides memory-efficient loading of large model files
//! by using memory mapping and loading tensors on demand.

use anyhow::{Result, Context};
use candle_core::{Device, DType, Tensor};
use memmap2::Mmap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

/// Tensor metadata from safetensors header
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorInfo {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offsets: [usize; 2],
}

/// Lazy safetensors loader using memory mapping
pub struct LazySafetensorsLoader {
    mmap: Mmap,
    tensors: HashMap<String, TensorInfo>,
    header_size: usize,
    device: Device,
    dtype: DType,
}

impl LazySafetensorsLoader {
    /// Create a new lazy loader for a safetensors file
    pub fn new(path: &Path, device: Device, dtype: DType) -> Result<Self> {
        println!("Opening safetensors file for lazy loading: {:?}", path);
        
        let file = File::open(path)
            .with_context(|| format!("Failed to open file: {:?}", path))?;
        
        // Memory map the file
        let mmap = unsafe { Mmap::map(&file)? };
        
        // Read header size (first 8 bytes)
        let header_size = u64::from_le_bytes(
            mmap[0..8].try_into()
                .context("Failed to read header size")?
        ) as usize;
        
        // Parse header JSON
        let header_json = &mmap[8..8 + header_size];
        let header_str = std::str::from_utf8(header_json)
            .context("Invalid UTF-8 in header")?;
        
        // First parse as generic JSON to handle metadata
        let parsed: serde_json::Value = serde_json::from_str(header_str)
            .context("Failed to parse header as JSON")?;
        
        let mut header_data = HashMap::new();
        
        // Extract tensor entries, skipping metadata
        if let Some(obj) = parsed.as_object() {
            for (key, value) in obj {
                if key == "__metadata__" {
                    continue;
                }
                
                // Try to parse as TensorInfo
                match serde_json::from_value::<TensorInfo>(value.clone()) {
                    Ok(info) => {
                        header_data.insert(key.clone(), info);
                    }
                    Err(e) => {
                        println!("Warning: Failed to parse tensor info for '{}': {}", key, e);
                    }
                }
            }
        }
        
        println!("Lazy loader initialized with {} tensors", header_data.len());
        
        Ok(Self {
            mmap,
            tensors: header_data,
            header_size: 8 + header_size,
            device,
            dtype,
        })
    }
    
    /// Check if a tensor exists
    pub fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }
    
    /// Get list of all tensor names
    pub fn tensor_names(&self) -> Vec<String> {
        self.tensors.keys().cloned().collect()
    }
    
    /// Load a specific tensor by name
    pub fn load_tensor(&self, name: &str) -> Result<Tensor> {
        let info = self.tensors.get(name)
            .with_context(|| format!("Tensor not found: {}", name))?;
        
        // Calculate byte offsets
        let start_offset = self.header_size + info.data_offsets[0];
        let end_offset = self.header_size + info.data_offsets[1];
        let tensor_bytes = &self.mmap[start_offset..end_offset];
        
        // Parse dtype
        let tensor_dtype = match info.dtype.as_str() {
            "F32" => DType::F32,
            "F16" => DType::F16,
            "BF16" => DType::BF16,
            "I64" => DType::I64,
            "U8" => DType::U8,
            "U32" => DType::U32,
            _ => return Err(anyhow::anyhow!("Unsupported dtype: {}", info.dtype)),
        };
        
        // Create tensor from bytes
        let tensor = match tensor_dtype {
            DType::F32 => {
                let data: Vec<f32> = tensor_bytes.chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                    .collect();
                Tensor::from_vec(data, info.shape.as_slice(), &Device::Cpu)?
            }
            DType::F16 => {
                let data: Vec<half::f16> = tensor_bytes.chunks_exact(2)
                    .map(|chunk| half::f16::from_le_bytes(chunk.try_into().unwrap()))
                    .collect();
                Tensor::from_vec(data, info.shape.as_slice(), &Device::Cpu)?
            }
            DType::BF16 => {
                let data: Vec<half::bf16> = tensor_bytes.chunks_exact(2)
                    .map(|chunk| half::bf16::from_le_bytes(chunk.try_into().unwrap()))
                    .collect();
                Tensor::from_vec(data, info.shape.as_slice(), &Device::Cpu)?
            }
            _ => return Err(anyhow::anyhow!("Unsupported dtype for loading: {:?}", tensor_dtype)),
        };
        
        // Move to target device and dtype if needed
        let tensor = if tensor.device().location() != self.device.location() {
            tensor.to_device(&self.device)?
        } else {
            tensor
        };
        
        let tensor = if tensor.dtype() != self.dtype {
            tensor.to_dtype(self.dtype)?
        } else {
            tensor
        };
        
        Ok(tensor)
    }
    
    /// Load multiple tensors by prefix
    pub fn load_tensors_with_prefix(&self, prefix: &str) -> Result<HashMap<String, Tensor>> {
        let mut result = HashMap::new();
        
        for (name, _) in &self.tensors {
            if name.starts_with(prefix) {
                let tensor = self.load_tensor(name)?;
                result.insert(name.clone(), tensor);
            }
        }
        
        Ok(result)
    }
}

/// Create a lazy tensor provider function for VarBuilder
pub fn create_lazy_tensor_provider(
    path: &Path,
    device: Device,
    dtype: DType,
) -> Result<impl Fn(&str) -> Result<Tensor>> {
    let loader = LazySafetensorsLoader::new(path, device, dtype)?;
    let loader = std::sync::Arc::new(loader);
    
    Ok(move |tensor_name: &str| {
        loader.load_tensor(tensor_name)
            .with_context(|| format!("Failed to load tensor: {}", tensor_name))
    })
}