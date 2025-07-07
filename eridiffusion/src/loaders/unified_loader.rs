//! Unified weight loader with automatic adaptation
//! 
//! This implements the loader component from the ML framework design,
//! focused on solving the tensor name mapping issues for Flux models.

use anyhow::{Result, Context};
use candle_core::{Device, DType, Tensor};
use candle_nn::{VarBuilder, VarMap};
use std::collections::HashMap;
use std::path::Path;

/// Weight adapter trait for converting between different naming conventions
pub trait WeightAdapter: Send + Sync {
    /// Check if this adapter can handle the conversion
    fn can_adapt(&self, from_arch: &str, to_arch: &str) -> bool;
    
    /// Adapt a single weight name
    fn adapt_name(&self, name: &str) -> String;
    
    /// Adapt tensor shapes if needed (e.g., splitting QKV)
    fn adapt_tensor(&self, name: &str, tensor: Tensor) -> Result<Vec<(String, Tensor)>>;
}

/// Flux weight adapter for handling checkpoint format differences
pub struct FluxAdapter {
    hidden_size: usize,
}

impl FluxAdapter {
    pub fn new(hidden_size: usize) -> Self {
        Self { hidden_size }
    }
}

impl WeightAdapter for FluxAdapter {
    fn can_adapt(&self, from_arch: &str, to_arch: &str) -> bool {
        (from_arch == "flux" || from_arch == "fluxdev" || from_arch == "fluxschnell") 
            && (to_arch == "flux" || to_arch == "flux_custom")
    }
    
    fn adapt_name(&self, name: &str) -> String {
        // Handle time embedding naming differences
        name.replace("time_in.in_layer", "time_in.mlp.0.fc1")
            .replace("time_in.out_layer", "time_in.mlp.0.fc2")
            .replace("vector_in.in_layer", "vector_in.mlp.0.fc1")
            .replace("vector_in.out_layer", "vector_in.mlp.0.fc2")
            // Handle final layer
            .replace("final_layer.linear", "final_layer")
    }
    
    fn adapt_tensor(&self, name: &str, tensor: Tensor) -> Result<Vec<(String, Tensor)>> {
        let mut results = Vec::new();
        
        // Split QKV tensors
        if name.contains(".qkv.weight") {
            let prefix = name.replace(".qkv.weight", "");
            let (total_dim, in_dim) = tensor.dims2()?;
            let head_dim = total_dim / 3;
            
            results.push((
                format!("{}.to_q.weight", prefix),
                tensor.narrow(0, 0, head_dim)?
            ));
            results.push((
                format!("{}.to_k.weight", prefix),
                tensor.narrow(0, head_dim, head_dim)?
            ));
            results.push((
                format!("{}.to_v.weight", prefix),
                tensor.narrow(0, head_dim * 2, head_dim)?
            ));
        } else if name.contains(".qkv.bias") {
            let prefix = name.replace(".qkv.bias", "");
            let total_dim = tensor.dims1()?;
            let head_dim = total_dim / 3;
            
            results.push((
                format!("{}.to_q.bias", prefix),
                tensor.narrow(0, 0, head_dim)?
            ));
            results.push((
                format!("{}.to_k.bias", prefix),
                tensor.narrow(0, head_dim, head_dim)?
            ));
            results.push((
                format!("{}.to_v.bias", prefix),
                tensor.narrow(0, head_dim * 2, head_dim)?
            ));
        } else if name.contains(".proj.") {
            // Rename projection layers
            let new_name = name.replace(".proj.", ".to_out.0.");
            results.push((new_name, tensor));
        } else {
            // Default: just adapt the name
            results.push((self.adapt_name(name), tensor));
        }
        
        Ok(results)
    }
}

/// Architecture detection from weight dictionary
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Architecture {
    Flux,
    FluxDev,
    FluxSchnell,
    SD35,
    SDXL,
    Unknown,
}

impl Architecture {
    /// Detect architecture from tensor names
    pub fn detect(tensors: &HashMap<String, Tensor>) -> Self {
        // Check for Flux-specific tensors
        if tensors.contains_key("double_blocks.0.img_attn.qkv.weight") {
            // Check for dev vs schnell - guidance_in exists in Dev model
            if tensors.contains_key("guidance_in.in_layer.weight") || tensors.contains_key("guidance_in.weight") {
                Architecture::FluxDev
            } else {
                Architecture::FluxSchnell
            }
        } else if tensors.contains_key("time_embed.timestep_embedder.mlp.0.weight") {
            Architecture::SD35
        } else if tensors.contains_key("down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight") {
            Architecture::SDXL
        } else {
            Architecture::Unknown
        }
    }
}

/// Unified loader that can handle multiple formats
pub struct UnifiedLoader {
    adapters: Vec<Box<dyn WeightAdapter>>,
    device: Device,
    dtype: DType,
}

impl UnifiedLoader {
    pub fn new(device: Device, dtype: DType) -> Self {
        Self {
            adapters: vec![
                Box::new(FluxAdapter::new(3072)), // Default hidden size
            ],
            device,
            dtype,
        }
    }
    
    /// Add a custom adapter
    pub fn with_adapter(mut self, adapter: Box<dyn WeightAdapter>) -> Self {
        self.adapters.push(adapter);
        self
    }
    
    /// Load weights from a file
    pub fn load(&self, path: &Path) -> Result<HashMap<String, Tensor>> {
        println!("Loading weights from: {:?}", path);
        
        // ALWAYS load to CPU first to avoid OOM
        let tensors = candle_core::safetensors::load(path, &Device::Cpu)
            .context("Failed to load safetensors file")?;
        
        println!("Loaded {} tensors to CPU", tensors.len());
        Ok(tensors)
    }
    
    /// Create a VarBuilder with adapted weights
    pub fn create_var_builder(
        &self,
        weights: HashMap<String, Tensor>,
        target_arch: &str,
    ) -> Result<VarBuilder<'static>> {
        let source_arch = Architecture::detect(&weights);
        println!("Detected architecture: {:?}", source_arch);
        
        // Find suitable adapter
        let adapter = self.adapters.iter()
            .find(|a| a.can_adapt(&format!("{:?}", source_arch).to_lowercase(), target_arch))
            .context("No suitable adapter found")?;
        
        // Adapt all tensors
        let mut adapted = HashMap::new();
        for (name, tensor) in weights {
            let adapted_tensors = adapter.adapt_tensor(&name, tensor)?;
            for (new_name, new_tensor) in adapted_tensors {
                // Move to target device and dtype
                let tensor = if new_tensor.device().location() != self.device.location() {
                    new_tensor.to_device(&self.device)?
                } else {
                    new_tensor
                };
                
                let tensor = if tensor.dtype() != self.dtype {
                    tensor.to_dtype(self.dtype)?
                } else {
                    tensor
                };
                
                adapted.insert(new_name, tensor);
            }
        }
        
        println!("Adapted {} tensors", adapted.len());
        
        // Create VarMap
        let var_map = VarMap::new();
        {
            let mut data = var_map.data().lock().unwrap();
            for (name, tensor) in adapted {
                let var = candle_core::Var::from_tensor(&tensor)?;
                data.insert(name, var);
            }
        }
        
        Ok(VarBuilder::from_varmap(&var_map, self.dtype, &self.device))
    }
    
    /// Load and create VarBuilder in one step
    pub fn load_into_var_builder(
        &self,
        path: &Path,
        target_arch: &str,
    ) -> Result<VarBuilder<'static>> {
        let weights = self.load(path)?;
        self.create_var_builder(weights, target_arch)
    }
}

/// Helper function for common Flux loading scenario
pub fn load_flux_weights(
    checkpoint_path: &Path,
    device: Device,
    dtype: DType,
    hidden_size: usize,
) -> Result<VarBuilder<'static>> {
    let loader = UnifiedLoader::new(device, dtype)
        .with_adapter(Box::new(FluxAdapter::new(hidden_size)));
    
    loader.load_into_var_builder(checkpoint_path, "flux")
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_flux_adapter() {
        let adapter = FluxAdapter::new(3072);
        
        // Test name adaptation
        assert_eq!(
            adapter.adapt_name("time_in.in_layer.weight"),
            "time_in.mlp.0.fc1.weight"
        );
        
        assert_eq!(
            adapter.adapt_name("final_layer.linear.weight"),
            "final_layer.weight"
        );
    }
    
    #[test]
    fn test_architecture_detection() {
        let mut tensors = HashMap::new();
        
        // Flux architecture
        tensors.insert(
            "double_blocks.0.img_attn.qkv.weight".to_string(),
            Tensor::zeros(&[1], DType::F32, &Device::Cpu).unwrap()
        );
        assert_eq!(Architecture::detect(&tensors), Architecture::FluxSchnell);
        
        // Flux Dev
        tensors.insert(
            "guidance_in.weight".to_string(),
            Tensor::zeros(&[1], DType::F32, &Device::Cpu).unwrap()
        );
        assert_eq!(Architecture::detect(&tensors), Architecture::FluxDev);
    }
}