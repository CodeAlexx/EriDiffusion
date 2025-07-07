// flux_tensor_mapping.rs - Proper tensor name mapping for Flux model

use candle_core::{Device, DType, Result, Tensor};
use candle_nn::{VarBuilder, VarMap};
use std::collections::HashMap;
use safetensors::SafeTensors;

/// Maps custom model structure names to actual Flux checkpoint names
pub struct FluxTensorMapper {
    name_map: HashMap<String, String>,
}

impl FluxTensorMapper {
    pub fn new() -> Self {
        let mut name_map = HashMap::new();
        
        // Time embedding mappings
        name_map.insert("time_in.mlp.0".to_string(), "time_in.in_layer.weight".to_string());
        name_map.insert("time_in.mlp.0.bias".to_string(), "time_in.in_layer.bias".to_string());
        name_map.insert("time_in.mlp.2".to_string(), "time_in.out_layer.weight".to_string());
        name_map.insert("time_in.mlp.2.bias".to_string(), "time_in.out_layer.bias".to_string());
        
        // Vector embedding mappings
        name_map.insert("vector_in.mlp.0".to_string(), "vector_in.in_layer.weight".to_string());
        name_map.insert("vector_in.mlp.0.bias".to_string(), "vector_in.in_layer.bias".to_string());
        name_map.insert("vector_in.mlp.2".to_string(), "vector_in.out_layer.weight".to_string());
        name_map.insert("vector_in.mlp.2.bias".to_string(), "vector_in.out_layer.bias".to_string());
        
        // Guidance embedding mappings
        name_map.insert("guidance_in.mlp.0".to_string(), "guidance_in.in_layer.weight".to_string());
        name_map.insert("guidance_in.mlp.0.bias".to_string(), "guidance_in.in_layer.bias".to_string());
        name_map.insert("guidance_in.mlp.2".to_string(), "guidance_in.out_layer.weight".to_string());
        name_map.insert("guidance_in.mlp.2.bias".to_string(), "guidance_in.out_layer.bias".to_string());
        
        // Input projections
        name_map.insert("img_in".to_string(), "img_in.weight".to_string());
        name_map.insert("img_in.bias".to_string(), "img_in.bias".to_string());
        name_map.insert("txt_in".to_string(), "txt_in.weight".to_string());
        name_map.insert("txt_in.bias".to_string(), "txt_in.bias".to_string());
        
        // Final layer
        name_map.insert("final_layer".to_string(), "final_layer.linear.weight".to_string());
        name_map.insert("final_layer.bias".to_string(), "final_layer.linear.bias".to_string());
        
        Self { name_map }
    }
    
    /// Add double block mappings
    pub fn add_double_block_mappings(&mut self, num_blocks: usize) {
        for i in 0..num_blocks {
            let prefix = format!("double_blocks.{}", i);
            
            // Image attention mappings
            self.add_attention_mappings(&prefix, "img_attn");
            
            // Text attention mappings
            self.add_attention_mappings(&prefix, "txt_attn");
            
            // Image MLP mappings
            self.add_mlp_mappings(&prefix, "img_mlp");
            
            // Text MLP mappings
            self.add_mlp_mappings(&prefix, "txt_mlp");
            
            // Layer norm mappings
            self.name_map.insert(
                format!("{}.img_norm1.weight", prefix),
                format!("{}.img_norm1.scale", prefix),
            );
            self.name_map.insert(
                format!("{}.img_norm2.weight", prefix),
                format!("{}.img_norm2.scale", prefix),
            );
            self.name_map.insert(
                format!("{}.txt_norm1.weight", prefix),
                format!("{}.txt_norm1.scale", prefix),
            );
            self.name_map.insert(
                format!("{}.txt_norm2.weight", prefix),
                format!("{}.txt_norm2.scale", prefix),
            );
            
            // Modulation mappings
            self.add_modulation_mappings(&prefix, "img_mod");
            self.add_modulation_mappings(&prefix, "txt_mod");
        }
    }
    
    /// Add single block mappings
    pub fn add_single_block_mappings(&mut self, num_blocks: usize) {
        for i in 0..num_blocks {
            let prefix = format!("single_blocks.{}", i);
            
            // Attention mappings
            self.add_attention_mappings(&prefix, "attn");
            
            // MLP mappings
            self.add_mlp_mappings(&prefix, "mlp");
            
            // Layer norm mappings
            self.name_map.insert(
                format!("{}.norm1.weight", prefix),
                format!("{}.norm.scale", prefix),
            );
            
            // Modulation mappings
            self.add_modulation_mappings(&prefix, "modulation");
        }
    }
    
    fn add_attention_mappings(&mut self, block_prefix: &str, attn_name: &str) {
        // QKV projections
        self.name_map.insert(
            format!("{}.{}.to_q", block_prefix, attn_name),
            format!("{}.{}.qkv.weight", block_prefix, attn_name),
        );
        self.name_map.insert(
            format!("{}.{}.to_k", block_prefix, attn_name),
            format!("{}.{}.qkv.weight", block_prefix, attn_name),
        );
        self.name_map.insert(
            format!("{}.{}.to_v", block_prefix, attn_name),
            format!("{}.{}.qkv.weight", block_prefix, attn_name),
        );
        
        // Output projection
        self.name_map.insert(
            format!("{}.{}.to_out.0", block_prefix, attn_name),
            format!("{}.{}.proj.weight", block_prefix, attn_name),
        );
        self.name_map.insert(
            format!("{}.{}.to_out.0.bias", block_prefix, attn_name),
            format!("{}.{}.proj.bias", block_prefix, attn_name),
        );
        
        // Normalization for QK
        self.name_map.insert(
            format!("{}.{}.norm_q.weight", block_prefix, attn_name),
            format!("{}.{}.norm.query_norm.scale", block_prefix, attn_name),
        );
        self.name_map.insert(
            format!("{}.{}.norm_k.weight", block_prefix, attn_name),
            format!("{}.{}.norm.key_norm.scale", block_prefix, attn_name),
        );
    }
    
    fn add_mlp_mappings(&mut self, block_prefix: &str, mlp_name: &str) {
        // For double blocks, MLPs use direct numbering (0, 2)
        if block_prefix.contains("double_blocks") {
            // First linear layer (0)
            self.name_map.insert(
                format!("{}.{}.0", block_prefix, mlp_name),
                format!("{}.{}.0.weight", block_prefix, mlp_name),
            );
            self.name_map.insert(
                format!("{}.{}.0.bias", block_prefix, mlp_name),
                format!("{}.{}.0.bias", block_prefix, mlp_name),
            );
            
            // Second linear layer (2)
            self.name_map.insert(
                format!("{}.{}.2", block_prefix, mlp_name),
                format!("{}.{}.2.weight", block_prefix, mlp_name),
            );
            self.name_map.insert(
                format!("{}.{}.2.bias", block_prefix, mlp_name),
                format!("{}.{}.2.bias", block_prefix, mlp_name),
            );
        } else {
            // For single blocks, use linear1/linear2
            self.name_map.insert(
                format!("{}.{}.linear1", block_prefix, mlp_name),
                format!("{}.{}.linear1.weight", block_prefix, mlp_name),
            );
            self.name_map.insert(
                format!("{}.{}.linear1.bias", block_prefix, mlp_name),
                format!("{}.{}.linear1.bias", block_prefix, mlp_name),
            );
            
            self.name_map.insert(
                format!("{}.{}.linear2", block_prefix, mlp_name),
                format!("{}.{}.linear2.weight", block_prefix, mlp_name),
            );
            self.name_map.insert(
                format!("{}.{}.linear2.bias", block_prefix, mlp_name),
                format!("{}.{}.linear2.bias", block_prefix, mlp_name),
            );
        }
    }
    
    fn add_modulation_mappings(&mut self, block_prefix: &str, mod_name: &str) {
        self.name_map.insert(
            format!("{}.{}.lin.weight", block_prefix, mod_name),
            format!("{}.{}.lin.weight", block_prefix, mod_name),
        );
        self.name_map.insert(
            format!("{}.{}.lin.bias", block_prefix, mod_name),
            format!("{}.{}.lin.bias", block_prefix, mod_name),
        );
    }
    
    /// Get the actual tensor name from checkpoint
    pub fn get_checkpoint_name(&self, model_name: &str) -> Option<&String> {
        self.name_map.get(model_name)
    }
    
    /// Create a mapped VarBuilder
    pub fn create_mapped_var_builder(
        &self,
        tensors: HashMap<String, Tensor>,
        dtype: DType,
        device: &Device,
    ) -> Result<VarBuilder> {
        let mut var_map = VarMap::new();
        
        // Map tensors to expected names
        for (model_name, checkpoint_name) in &self.name_map {
            if let Some(tensor) = tensors.get(checkpoint_name) {
                var_map.set_one(model_name, tensor.to_device(device)?.to_dtype(dtype)?)?;
            }
        }
        
        Ok(VarBuilder::from_varmap(&var_map, dtype, device))
    }
}

/// Load Flux checkpoint with proper name mapping
pub fn load_flux_checkpoint(
    checkpoint_path: &Path,
    config: &ModelConfig,
    device: &Device,
    dtype: DType,
) -> Result<FluxModel> {
    println!("Loading Flux checkpoint from {:?}", checkpoint_path);
    
    // Load tensors from checkpoint
    let tensors = if checkpoint_path.extension() == Some(std::ffi::OsStr::new("safetensors")) {
        safetensors::load(checkpoint_path, device)?
    } else {
        // Handle other formats if needed
        return Err(candle_core::Error::Msg(
            "Only safetensors format is currently supported".to_string()
        ));
    };
    
    // Create mapper
    let mut mapper = FluxTensorMapper::new();
    mapper.add_double_block_mappings(config.num_double_blocks);
    mapper.add_single_block_mappings(config.num_single_blocks);
    
    // Create mapped VarBuilder
    let vb = mapper.create_mapped_var_builder(tensors, dtype, device)?;
    
    // Create model with mapped weights
    let model = FluxModel::from_vb(config.clone(), vb, device.clone(), dtype)?;
    
    println!("Successfully loaded Flux model");
    Ok(model)
}

/// Handle combined QKV weights that need to be split
pub fn split_qkv_weights(
    combined_qkv: &Tensor,
    hidden_size: usize,
) -> Result<(Tensor, Tensor, Tensor)> {
    let (total_size, in_features) = combined_qkv.dims2()?;
    assert_eq!(total_size, hidden_size * 3, "QKV weight size mismatch");
    
    let chunk_size = hidden_size;
    let q = combined_qkv.narrow(0, 0, chunk_size)?;
    let k = combined_qkv.narrow(0, chunk_size, chunk_size)?;
    let v = combined_qkv.narrow(0, chunk_size * 2, chunk_size)?;
    
    Ok((q, k, v))
}

/// Custom VarBuilder that handles Flux-specific weight loading
pub struct FluxVarBuilder {
    inner: VarBuilder,
    tensors: HashMap<String, Tensor>,
    mapper: FluxTensorMapper,
}

impl FluxVarBuilder {
    pub fn from_checkpoint(
        checkpoint_path: &Path,
        config: &ModelConfig,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let tensors = safetensors::load(checkpoint_path, device)?;
        
        let mut mapper = FluxTensorMapper::new();
        mapper.add_double_block_mappings(config.num_double_blocks);
        mapper.add_single_block_mappings(config.num_single_blocks);
        
        let inner = VarBuilder::zeros(dtype, device);
        
        Ok(Self {
            inner,
            tensors,
            mapper,
        })
    }
    
    pub fn get(&self, name: &str) -> Result<Tensor> {
        // First check if we have a mapping
        if let Some(checkpoint_name) = self.mapper.get_checkpoint_name(name) {
            if let Some(tensor) = self.tensors.get(checkpoint_name) {
                return Ok(tensor.clone());
            }
        }
        
        // Check for direct match
        if let Some(tensor) = self.tensors.get(name) {
            return Ok(tensor.clone());
        }
        
        // Handle special cases like combined QKV
        if name.ends_with(".to_q") || name.ends_with(".to_k") || name.ends_with(".to_v") {
            let base = name.trim_end_matches(".to_q")
                .trim_end_matches(".to_k")
                .trim_end_matches(".to_v");
            let qkv_name = format!("{}.qkv.weight", base);
            
            if let Some(qkv_tensor) = self.tensors.get(&qkv_name) {
                let hidden_size = qkv_tensor.dim(0)? / 3;
                let (q, k, v) = split_qkv_weights(qkv_tensor, hidden_size)?;
                
                return Ok(match name {
                    n if n.ends_with(".to_q") => q,
                    n if n.ends_with(".to_k") => k,
                    n if n.ends_with(".to_v") => v,
                    _ => unreachable!(),
                });
            }
        }
        
        // Fall back to zeros
        Err(candle_core::Error::Msg(format!("Tensor {} not found in checkpoint", name)))
    }
}

/// Example usage
pub fn load_flux_model_properly(
    checkpoint_path: &Path,
    device: &Device,
) -> Result<FluxModel> {
    // Load config
    let config_path = checkpoint_path.parent().unwrap().join("config.json");
    let config: ModelConfig = if config_path.exists() {
        let config_str = std::fs::read_to_string(config_path)?;
        serde_json::from_str(&config_str)?
    } else {
        ModelConfig::default()
    };
    
    // Use the proper loader
    load_flux_checkpoint(checkpoint_path, &config, device, DType::BF16)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_tensor_mapping() {
        let mapper = FluxTensorMapper::new();
        
        // Test time embedding mapping
        assert_eq!(
            mapper.get_checkpoint_name("time_in.mlp.0"),
            Some(&"time_in.in_layer.weight".to_string())
        );
        
        // Test that we handle missing mappings
        assert_eq!(
            mapper.get_checkpoint_name("nonexistent.weight"),
            None
        );
    }
}
