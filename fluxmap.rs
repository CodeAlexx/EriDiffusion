// Proper tensor mapping from model structure to checkpoint names

use candle_core::{Device, DType, Result, Tensor};
use candle_nn::{VarBuilder, VarMap};
use std::collections::HashMap;

/// Maps from our model's expected tensor names to Flux checkpoint names
pub struct FluxTensorMapper {
    /// Maps from model name -> checkpoint name
    model_to_checkpoint: HashMap<String, String>,
}

impl FluxTensorMapper {
    pub fn new() -> Self {
        let mut model_to_checkpoint = HashMap::new();
        
        // Time embedding: our model expects mlp.0/mlp.2, checkpoint has in_layer/out_layer
        model_to_checkpoint.insert(
            "time_in.mlp.0.weight".to_string(),
            "time_in.in_layer.weight".to_string(),
        );
        model_to_checkpoint.insert(
            "time_in.mlp.0.bias".to_string(),
            "time_in.in_layer.bias".to_string(),
        );
        model_to_checkpoint.insert(
            "time_in.mlp.2.weight".to_string(),
            "time_in.out_layer.weight".to_string(),
        );
        model_to_checkpoint.insert(
            "time_in.mlp.2.bias".to_string(),
            "time_in.out_layer.bias".to_string(),
        );
        
        // Vector embedding: same pattern
        model_to_checkpoint.insert(
            "vector_in.mlp.0.weight".to_string(),
            "vector_in.in_layer.weight".to_string(),
        );
        model_to_checkpoint.insert(
            "vector_in.mlp.0.bias".to_string(),
            "vector_in.in_layer.bias".to_string(),
        );
        model_to_checkpoint.insert(
            "vector_in.mlp.2.weight".to_string(),
            "vector_in.out_layer.weight".to_string(),
        );
        model_to_checkpoint.insert(
            "vector_in.mlp.2.bias".to_string(),
            "vector_in.out_layer.bias".to_string(),
        );
        
        // Final layer: our model expects final_layer.weight, checkpoint might have final_layer.linear.weight
        model_to_checkpoint.insert(
            "final_layer.weight".to_string(),
            "final_layer.linear.weight".to_string(),
        );
        model_to_checkpoint.insert(
            "final_layer.bias".to_string(),
            "final_layer.linear.bias".to_string(),
        );
        
        Self { model_to_checkpoint }
    }
    
    /// Add double block mappings
    pub fn add_double_block_mappings(&mut self, num_blocks: usize) {
        for i in 0..num_blocks {
            let prefix = format!("double_blocks.{}", i);
            
            // Our model expects separated Q/K/V, but checkpoint has combined QKV
            // Image attention
            self.model_to_checkpoint.insert(
                format!("{}.img_attn.to_q.weight", prefix),
                format!("{}.img_attn.qkv.weight", prefix),
            );
            self.model_to_checkpoint.insert(
                format!("{}.img_attn.to_k.weight", prefix),
                format!("{}.img_attn.qkv.weight", prefix),
            );
            self.model_to_checkpoint.insert(
                format!("{}.img_attn.to_v.weight", prefix),
                format!("{}.img_attn.qkv.weight", prefix),
            );
            
            // Output projection
            self.model_to_checkpoint.insert(
                format!("{}.img_attn.to_out.0.weight", prefix),
                format!("{}.img_attn.proj.weight", prefix),
            );
            self.model_to_checkpoint.insert(
                format!("{}.img_attn.to_out.0.bias", prefix),
                format!("{}.img_attn.proj.bias", prefix),
            );
            
            // Text attention - same pattern
            self.model_to_checkpoint.insert(
                format!("{}.txt_attn.to_q.weight", prefix),
                format!("{}.txt_attn.qkv.weight", prefix),
            );
            self.model_to_checkpoint.insert(
                format!("{}.txt_attn.to_k.weight", prefix),
                format!("{}.txt_attn.qkv.weight", prefix),
            );
            self.model_to_checkpoint.insert(
                format!("{}.txt_attn.to_v.weight", prefix),
                format!("{}.txt_attn.qkv.weight", prefix),
            );
            self.model_to_checkpoint.insert(
                format!("{}.txt_attn.to_out.0.weight", prefix),
                format!("{}.txt_attn.proj.weight", prefix),
            );
            self.model_to_checkpoint.insert(
                format!("{}.txt_attn.to_out.0.bias", prefix),
                format!("{}.txt_attn.proj.bias", prefix),
            );
            
            // MLPs - our model expects fc1/fc2, checkpoint has 0/2
            self.model_to_checkpoint.insert(
                format!("{}.img_mlp.fc1.weight", prefix),
                format!("{}.img_mlp.0.weight", prefix),
            );
            self.model_to_checkpoint.insert(
                format!("{}.img_mlp.fc1.bias", prefix),
                format!("{}.img_mlp.0.bias", prefix),
            );
            self.model_to_checkpoint.insert(
                format!("{}.img_mlp.fc2.weight", prefix),
                format!("{}.img_mlp.2.weight", prefix),
            );
            self.model_to_checkpoint.insert(
                format!("{}.img_mlp.fc2.bias", prefix),
                format!("{}.img_mlp.2.bias", prefix),
            );
            
            // Text MLP
            self.model_to_checkpoint.insert(
                format!("{}.txt_mlp.fc1.weight", prefix),
                format!("{}.txt_mlp.0.weight", prefix),
            );
            self.model_to_checkpoint.insert(
                format!("{}.txt_mlp.fc1.bias", prefix),
                format!("{}.txt_mlp.0.bias", prefix),
            );
            self.model_to_checkpoint.insert(
                format!("{}.txt_mlp.fc2.weight", prefix),
                format!("{}.txt_mlp.2.weight", prefix),
            );
            self.model_to_checkpoint.insert(
                format!("{}.txt_mlp.fc2.bias", prefix),
                format!("{}.txt_mlp.2.bias", prefix),
            );
            
            // Modulation layers (if your model uses them)
            self.model_to_checkpoint.insert(
                format!("{}.img_mod.lin.weight", prefix),
                format!("{}.img_mod.lin.weight", prefix),
            );
            self.model_to_checkpoint.insert(
                format!("{}.img_mod.lin.bias", prefix),
                format!("{}.img_mod.lin.bias", prefix),
            );
            self.model_to_checkpoint.insert(
                format!("{}.txt_mod.lin.weight", prefix),
                format!("{}.txt_mod.lin.weight", prefix),
            );
            self.model_to_checkpoint.insert(
                format!("{}.txt_mod.lin.bias", prefix),
                format!("{}.txt_mod.lin.bias", prefix),
            );
        }
    }
    
    /// Add single block mappings
    pub fn add_single_block_mappings(&mut self, num_blocks: usize) {
        for i in 0..num_blocks {
            let prefix = format!("single_blocks.{}", i);
            
            // Attention mappings
            self.model_to_checkpoint.insert(
                format!("{}.attn.to_q.weight", prefix),
                format!("{}.attn.qkv.weight", prefix),
            );
            self.model_to_checkpoint.insert(
                format!("{}.attn.to_k.weight", prefix),
                format!("{}.attn.qkv.weight", prefix),
            );
            self.model_to_checkpoint.insert(
                format!("{}.attn.to_v.weight", prefix),
                format!("{}.attn.qkv.weight", prefix),
            );
            self.model_to_checkpoint.insert(
                format!("{}.attn.to_out.0.weight", prefix),
                format!("{}.attn.proj.weight", prefix),
            );
            self.model_to_checkpoint.insert(
                format!("{}.attn.to_out.0.bias", prefix),
                format!("{}.attn.proj.bias", prefix),
            );
            
            // MLP - our model expects fc1/fc2, checkpoint has linear1/linear2
            self.model_to_checkpoint.insert(
                format!("{}.mlp.fc1.weight", prefix),
                format!("{}.linear1.weight", prefix),
            );
            self.model_to_checkpoint.insert(
                format!("{}.mlp.fc1.bias", prefix),
                format!("{}.linear1.bias", prefix),
            );
            self.model_to_checkpoint.insert(
                format!("{}.mlp.fc2.weight", prefix),
                format!("{}.linear2.weight", prefix),
            );
            self.model_to_checkpoint.insert(
                format!("{}.mlp.fc2.bias", prefix),
                format!("{}.linear2.bias", prefix),
            );
            
            // Modulation
            self.model_to_checkpoint.insert(
                format!("{}.modulation.lin.weight", prefix),
                format!("{}.modulation.lin.weight", prefix),
            );
            self.model_to_checkpoint.insert(
                format!("{}.modulation.lin.bias", prefix),
                format!("{}.modulation.lin.bias", prefix),
            );
        }
    }
    
    /// Get checkpoint name for a model tensor name
    pub fn get_checkpoint_name(&self, model_name: &str) -> Option<&String> {
        self.model_to_checkpoint.get(model_name)
    }
    
    /// Check if this is a QKV weight that needs splitting
    pub fn is_qkv_weight(&self, model_name: &str) -> bool {
        model_name.ends_with(".to_q.weight") || 
        model_name.ends_with(".to_k.weight") || 
        model_name.ends_with(".to_v.weight")
    }
    
    /// Get QKV index for splitting (0 for Q, 1 for K, 2 for V)
    pub fn get_qkv_index(&self, model_name: &str) -> Option<usize> {
        if model_name.ends_with(".to_q.weight") {
            Some(0)
        } else if model_name.ends_with(".to_k.weight") {
            Some(1)
        } else if model_name.ends_with(".to_v.weight") {
            Some(2)
        } else {
            None
        }
    }
}

/// Create a VarBuilder that properly maps tensors
pub fn create_mapped_var_builder(
    checkpoint_tensors: &HashMap<String, Tensor>,
    mapper: &FluxTensorMapper,
    dtype: DType,
    device: &Device,
) -> Result<VarBuilder> {
    let mut var_map = VarMap::new();
    let mut mapped_count = 0;
    let mut qkv_cache: HashMap<String, (Tensor, Tensor, Tensor)> = HashMap::new();
    
    // Process all expected model tensor names
    for (model_name, checkpoint_name) in &mapper.model_to_checkpoint {
        if mapper.is_qkv_weight(model_name) {
            // Handle QKV splitting
            let base_name = model_name
                .trim_end_matches(".to_q.weight")
                .trim_end_matches(".to_k.weight")
                .trim_end_matches(".to_v.weight");
            
            let cache_key = format!("{}.qkv", base_name);
            
            // Split QKV if not already cached
            if !qkv_cache.contains_key(&cache_key) {
                if let Some(qkv_tensor) = checkpoint_tensors.get(checkpoint_name) {
                    let (out_features, in_features) = qkv_tensor.dims2()?;
                    let head_dim = out_features / 3;
                    
                    let q = qkv_tensor.narrow(0, 0, head_dim)?;
                    let k = qkv_tensor.narrow(0, head_dim, head_dim)?;
                    let v = qkv_tensor.narrow(0, head_dim * 2, head_dim)?;
                    
                    qkv_cache.insert(cache_key.clone(), (q, k, v));
                }
            }
            
            // Get the appropriate part
            if let Some((q, k, v)) = qkv_cache.get(&cache_key) {
                let tensor = match mapper.get_qkv_index(model_name) {
                    Some(0) => q,
                    Some(1) => k,
                    Some(2) => v,
                    _ => continue,
                };
                
                var_map.set_one(model_name, tensor.to_device(device)?.to_dtype(dtype)?)?;
                mapped_count += 1;
            }
        } else {
            // Regular tensor mapping
            if let Some(tensor) = checkpoint_tensors.get(checkpoint_name) {
                var_map.set_one(model_name, tensor.to_device(device)?.to_dtype(dtype)?)?;
                mapped_count += 1;
            }
        }
    }
    
    // Also include any tensors that don't need mapping (direct matches)
    for (name, tensor) in checkpoint_tensors {
        if !mapper.model_to_checkpoint.contains_key(name) {
            // This tensor doesn't need mapping, add it directly
            var_map.set_one(name, tensor.to_device(device)?.to_dtype(dtype)?)?;
            mapped_count += 1;
        }
    }
    
    println!("Mapped {} tensors from checkpoint", mapped_count);
    
    Ok(VarBuilder::from_varmap(&var_map, dtype, device))
}

/// Debug function to show what mappings would be applied
pub fn debug_tensor_mapping(
    checkpoint_tensors: &HashMap<String, Tensor>,
    mapper: &FluxTensorMapper,
) {
    println!("\n=== Tensor Mapping Debug ===");
    
    // Show some checkpoint tensor names
    println!("\nSample checkpoint tensors:");
    for (i, name) in checkpoint_tensors.keys().enumerate() {
        if i < 20 || name.contains("mlp") || name.contains("time_in") {
            println!("  {}", name);
        }
    }
    
    // Show some mappings
    println!("\nSample mappings (model -> checkpoint):");
    for (model_name, checkpoint_name) in &mapper.model_to_checkpoint {
        if model_name.contains("time_in") || model_name.contains("blocks.0") {
            println!("  {} -> {}", model_name, checkpoint_name);
            if checkpoint_tensors.contains_key(checkpoint_name) {
                println!("    ✓ Found in checkpoint");
            } else {
                println!("    ✗ NOT found in checkpoint");
            }
        }
    }
}

/// Usage example
pub fn load_flux_with_proper_mapping(
    checkpoint_path: &Path,
    config: &ModelConfig,
    device: &Device,
    dtype: DType,
) -> Result<FluxModel> {
    // Load checkpoint tensors
    let checkpoint_tensors = safetensors::load(checkpoint_path, device)?;
    
    // Create mapper
    let mut mapper = FluxTensorMapper::new();
    mapper.add_double_block_mappings(config.num_double_blocks);
    mapper.add_single_block_mappings(config.num_single_blocks);
    
    // Debug the mapping
    debug_tensor_mapping(&checkpoint_tensors, &mapper);
    
    // Create mapped VarBuilder
    let vb = create_mapped_var_builder(&checkpoint_tensors, &mapper, dtype, device)?;
    
    // Create model with mapped weights
    let model = FluxModel::from_vb(config, vb, device.clone(), dtype)?;
    
    Ok(model)
}
