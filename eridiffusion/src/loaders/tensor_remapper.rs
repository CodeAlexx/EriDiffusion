//! Tensor remapper for handling checkpoint format mismatches
//! 
//! This implements Option 3 from ops.txt - a flexible tensor remapper
//! that can handle missing tensors, name mismatches, and tensor synthesis.

use anyhow::Result;
use candle_core::{Device, DType, Tensor};
use std::collections::HashMap;
use std::path::Path;
use super::lazy_safetensors::LazySafetensorsLoader;
use std::sync::Arc;

/// Tensor remapper that handles various checkpoint format issues
pub struct TensorRemapper {
    checkpoint: HashMap<String, Tensor>,
    lazy_loader: Option<Arc<LazySafetensorsLoader>>,
    mappings: HashMap<String, String>,
    device: Device,
    dtype: DType,
}

impl TensorRemapper {
    /// Create a new tensor remapper from a checkpoint file
    pub fn from_checkpoint(path: &Path, device: Device, dtype: DType) -> Result<Self> {
        println!("Loading checkpoint with TensorRemapper from: {:?}", path);
        
        // Check if file exists
        if !path.exists() {
            return Err(anyhow::anyhow!("Checkpoint file not found: {:?}", path));
        }
        
        // Check file size to decide on loading strategy
        let metadata = std::fs::metadata(path)?;
        let size_gb = metadata.len() as f64 / (1024.0 * 1024.0 * 1024.0);
        
        if size_gb > 10.0 {
            // Use lazy loading for large files
            println!("Large file detected ({:.2} GB), using lazy loading...", size_gb);
            let lazy_loader = Arc::new(LazySafetensorsLoader::new(path, device.clone(), dtype)?);
            
            Ok(Self {
                checkpoint: HashMap::new(),
                lazy_loader: Some(lazy_loader),
                mappings: Self::create_default_mappings(),
                device,
                dtype,
            })
        } else {
            // Use regular loading for smaller files
            println!("File size {:.2} GB, using regular loading...", size_gb);
            
            use memmap2::Mmap;
            use std::fs::File;
            
            let file = File::open(path)?;
            let mmap = unsafe { Mmap::map(&file)? };
            
            let checkpoint = candle_core::safetensors::load_buffer(&mmap[..], &Device::Cpu)?;
            println!("Loaded {} tensors via mmap", checkpoint.len());
            
            Ok(Self {
                checkpoint,
                lazy_loader: None,
                mappings: Self::create_default_mappings(),
                device,
                dtype,
            })
        }
    }
    
    /// Create default mappings for common name mismatches
    fn create_default_mappings() -> HashMap<String, String> {
        let mut mappings = HashMap::new();
        
        // Flux-specific mappings
        mappings.insert("time_in.mlp.0.fc1.weight".to_string(), "time_in.in_layer.weight".to_string());
        mappings.insert("time_in.mlp.0.fc1.bias".to_string(), "time_in.in_layer.bias".to_string());
        mappings.insert("time_in.mlp.0.fc2.weight".to_string(), "time_in.out_layer.weight".to_string());
        mappings.insert("time_in.mlp.0.fc2.bias".to_string(), "time_in.out_layer.bias".to_string());
        
        mappings.insert("vector_in.mlp.0.fc1.weight".to_string(), "vector_in.in_layer.weight".to_string());
        mappings.insert("vector_in.mlp.0.fc1.bias".to_string(), "vector_in.in_layer.bias".to_string());
        mappings.insert("vector_in.mlp.0.fc2.weight".to_string(), "vector_in.out_layer.weight".to_string());
        mappings.insert("vector_in.mlp.0.fc2.bias".to_string(), "vector_in.out_layer.bias".to_string());
        
        mappings.insert("final_layer.weight".to_string(), "final_layer.linear.weight".to_string());
        mappings.insert("final_layer.bias".to_string(), "final_layer.linear.bias".to_string());
        
        mappings
    }
    
    /// Load a tensor with fallbacks and synthesis
    pub fn load_with_fallbacks(&self, model_path: &str) -> Result<Tensor> {
        // If using lazy loader, try to load from it first
        if let Some(ref lazy_loader) = self.lazy_loader {
            // Try original path
            if lazy_loader.contains(model_path) {
                return lazy_loader.load_tensor(model_path);
            }
            
            // Try mapped path
            if let Some(mapped_path) = self.mappings.get(model_path) {
                if lazy_loader.contains(mapped_path) {
                    return lazy_loader.load_tensor(mapped_path);
                }
            }
            
            // Try to synthesize
            if let Some(tensor) = self.synthesize_tensor_lazy(model_path, lazy_loader)? {
                return Ok(tensor);
            }
        } else {
            // Use regular checkpoint
            // Try original path
            if let Some(tensor) = self.checkpoint.get(model_path) {
                return self.prepare_tensor(tensor);
            }
            
            // Try mapped path
            if let Some(mapped_path) = self.mappings.get(model_path) {
                if let Some(tensor) = self.checkpoint.get(mapped_path) {
                    return self.prepare_tensor(tensor);
                }
            }
            
            // Try to synthesize
            if let Some(tensor) = self.synthesize_tensor(model_path)? {
                return Ok(tensor);
            }
        }
        
        Err(anyhow::anyhow!("No tensor found for: {}", model_path))
    }
    
    /// Prepare tensor by moving to target device and dtype
    fn prepare_tensor(&self, tensor: &Tensor) -> Result<Tensor> {
        let tensor = if tensor.device().location() != self.device.location() {
            tensor.to_device(&self.device)?
        } else {
            tensor.clone()
        };
        
        let tensor = if tensor.dtype() != self.dtype {
            tensor.to_dtype(self.dtype)?
        } else {
            tensor
        };
        
        Ok(tensor)
    }
    
    /// Synthesize missing tensors
    fn synthesize_tensor(&self, model_path: &str) -> Result<Option<Tensor>> {
        // Handle QKV splits
        if model_path.contains(".to_q.") || model_path.contains(".to_k.") || model_path.contains(".to_v.") {
            return self.synthesize_qkv_tensor(model_path);
        }
        
        // Handle layer norms that might be missing
        if model_path.contains("norm") && (model_path.ends_with(".weight") || model_path.ends_with(".bias")) {
            return self.synthesize_layer_norm(model_path);
        }
        
        // Handle projection renames
        if model_path.contains(".to_out.0.") {
            let proj_path = model_path.replace(".to_out.0.", ".proj.");
            if let Some(tensor) = self.checkpoint.get(&proj_path) {
                return Ok(Some(self.prepare_tensor(tensor)?));
            }
        }
        
        Ok(None)
    }
    
    /// Synthesize QKV tensors from combined tensor
    fn synthesize_qkv_tensor(&self, model_path: &str) -> Result<Option<Tensor>> {
        // Extract block prefix (e.g., "double_blocks.0.img_attn")
        let parts: Vec<&str> = model_path.split('.').collect();
        if parts.len() < 4 {
            return Ok(None);
        }
        
        let prefix = parts[..parts.len()-2].join(".");
        let qkv_path = format!("{}.qkv.{}", prefix, parts.last().unwrap());
        
        if let Some(qkv_tensor) = self.checkpoint.get(&qkv_path) {
            let is_weight = model_path.ends_with(".weight");
            
            if is_weight {
                let (total_dim, in_dim) = qkv_tensor.dims2()?;
                let head_dim = total_dim / 3;
                
                let tensor = if model_path.contains(".to_q.") {
                    qkv_tensor.narrow(0, 0, head_dim)?
                } else if model_path.contains(".to_k.") {
                    qkv_tensor.narrow(0, head_dim, head_dim)?
                } else {
                    qkv_tensor.narrow(0, head_dim * 2, head_dim)?
                };
                
                return Ok(Some(self.prepare_tensor(&tensor)?));
            } else {
                // Handle bias
                let total_dim = qkv_tensor.dims1()?;
                let head_dim = total_dim / 3;
                
                let tensor = if model_path.contains(".to_q.") {
                    qkv_tensor.narrow(0, 0, head_dim)?
                } else if model_path.contains(".to_k.") {
                    qkv_tensor.narrow(0, head_dim, head_dim)?
                } else {
                    qkv_tensor.narrow(0, head_dim * 2, head_dim)?
                };
                
                return Ok(Some(self.prepare_tensor(&tensor)?));
            }
        }
        
        Ok(None)
    }
    
    /// Synthesize layer norm tensors (identity transform)
    fn synthesize_layer_norm(&self, model_path: &str) -> Result<Option<Tensor>> {
        // Determine hidden size from context
        let hidden_size = if model_path.contains("double_blocks") || model_path.contains("single_blocks") {
            3072 // Flux hidden size
        } else {
            return Ok(None);
        };
        
        let tensor = if model_path.ends_with(".weight") {
            Tensor::ones(hidden_size, self.dtype, &Device::Cpu)?
        } else {
            Tensor::zeros(hidden_size, self.dtype, &Device::Cpu)?
        };
        
        Ok(Some(self.prepare_tensor(&tensor)?))
    }
    
    /// Synthesize missing tensors with lazy loading
    fn synthesize_tensor_lazy(&self, model_path: &str, lazy_loader: &Arc<LazySafetensorsLoader>) -> Result<Option<Tensor>> {
        // Handle QKV splits
        if model_path.contains(".to_q.") || model_path.contains(".to_k.") || model_path.contains(".to_v.") {
            return self.synthesize_qkv_tensor_lazy(model_path, lazy_loader);
        }
        
        // Handle layer norms that might be missing
        if model_path.contains("norm") && (model_path.ends_with(".weight") || model_path.ends_with(".bias")) {
            return self.synthesize_layer_norm(model_path);
        }
        
        // Handle projection renames
        if model_path.contains(".to_out.0.") {
            let proj_path = model_path.replace(".to_out.0.", ".proj.");
            if lazy_loader.contains(&proj_path) {
                return Ok(Some(lazy_loader.load_tensor(&proj_path)?));
            }
        }
        
        Ok(None)
    }
    
    /// Synthesize QKV tensors from combined tensor with lazy loading
    fn synthesize_qkv_tensor_lazy(&self, model_path: &str, lazy_loader: &Arc<LazySafetensorsLoader>) -> Result<Option<Tensor>> {
        // Extract block prefix (e.g., "double_blocks.0.img_attn")
        let parts: Vec<&str> = model_path.split('.').collect();
        if parts.len() < 4 {
            return Ok(None);
        }
        
        let prefix = parts[..parts.len()-2].join(".");
        let qkv_path = format!("{}.qkv.{}", prefix, parts.last().unwrap());
        
        if lazy_loader.contains(&qkv_path) {
            let qkv_tensor = lazy_loader.load_tensor(&qkv_path)?;
            let is_weight = model_path.ends_with(".weight");
            
            if is_weight {
                let (total_dim, _in_dim) = qkv_tensor.dims2()?;
                let head_dim = total_dim / 3;
                
                let tensor = if model_path.contains(".to_q.") {
                    qkv_tensor.narrow(0, 0, head_dim)?
                } else if model_path.contains(".to_k.") {
                    qkv_tensor.narrow(0, head_dim, head_dim)?
                } else {
                    qkv_tensor.narrow(0, head_dim * 2, head_dim)?
                };
                
                return Ok(Some(tensor));
            } else {
                // Handle bias
                let total_dim = qkv_tensor.dims1()?;
                let head_dim = total_dim / 3;
                
                let tensor = if model_path.contains(".to_q.") {
                    qkv_tensor.narrow(0, 0, head_dim)?
                } else if model_path.contains(".to_k.") {
                    qkv_tensor.narrow(0, head_dim, head_dim)?
                } else {
                    qkv_tensor.narrow(0, head_dim * 2, head_dim)?
                };
                
                return Ok(Some(tensor));
            }
        }
        
        Ok(None)
    }
    
    /// Get all available tensors
    pub fn tensors(&self) -> &HashMap<String, Tensor> {
        &self.checkpoint
    }
    
    /// Add custom mapping
    pub fn add_mapping(&mut self, from: String, to: String) {
        self.mappings.insert(from, to);
    }
    
    /// Create a tensor provider function for VarBuilder
    pub fn tensor_provider(&self) -> impl Fn(&str) -> Result<Tensor> + '_ {
        move |path: &str| self.load_with_fallbacks(path)
    }
}

/// Helper to create a Flux-compatible tensor remapper
pub fn create_flux_remapper(
    checkpoint_path: &Path,
    device: Device,
    dtype: DType,
) -> Result<TensorRemapper> {
    let mut remapper = TensorRemapper::from_checkpoint(checkpoint_path, device, dtype)?;
    
    // Add Flux-specific mappings
    // Handle MLP naming differences
    for i in 0..19 {
        for block_type in ["img_mlp", "txt_mlp"] {
            remapper.add_mapping(
                format!("double_blocks.{}.{}.linear1.weight", i, block_type),
                format!("double_blocks.{}.{}.0.weight", i, block_type),
            );
            remapper.add_mapping(
                format!("double_blocks.{}.{}.linear1.bias", i, block_type),
                format!("double_blocks.{}.{}.0.bias", i, block_type),
            );
            remapper.add_mapping(
                format!("double_blocks.{}.{}.linear2.weight", i, block_type),
                format!("double_blocks.{}.{}.2.weight", i, block_type),
            );
            remapper.add_mapping(
                format!("double_blocks.{}.{}.linear2.bias", i, block_type),
                format!("double_blocks.{}.{}.2.bias", i, block_type),
            );
        }
    }
    
    Ok(remapper)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_mappings() {
        let mappings = TensorRemapper::create_default_mappings();
        assert_eq!(
            mappings.get("time_in.mlp.0.fc1.weight"),
            Some(&"time_in.in_layer.weight".to_string())
        );
    }
}