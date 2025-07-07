//! Save Flux LoRA weights in AI-Toolkit format

use anyhow::Result;
use candle_core::{Tensor, DType, Device};
use safetensors::{serialize_to_file, View, Dtype as SafeDtype};
use std::collections::HashMap;
use std::path::Path;
use std::borrow::Cow;
use crate::networks::lora::LoRAConfig;

/// Wrapper to make Candle tensors work with safetensors
#[derive(Clone, Copy)]
pub struct TensorView<'a> {
    tensor: &'a Tensor,
}

impl<'a> View for TensorView<'a> {
    fn dtype(&self) -> SafeDtype {
        match self.tensor.dtype() {
            DType::F16 => SafeDtype::F16,
            DType::F32 => SafeDtype::F32,
            DType::BF16 => SafeDtype::BF16,
            _ => panic!("Unsupported dtype for safetensors"),
        }
    }

    fn shape(&self) -> &[usize] {
        self.tensor.dims()
    }

    fn data(&self) -> Cow<[u8]> {
        // Flatten and convert to bytes
        let data = match self.tensor.dtype() {
            DType::F32 => {
                let vec = self.tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                bytemuck::cast_slice(&vec).to_vec()
            },
            DType::F16 => {
                let vec = self.tensor.flatten_all().unwrap().to_vec1::<half::f16>().unwrap();
                bytemuck::cast_slice(&vec).to_vec()
            },
            DType::BF16 => {
                let vec = self.tensor.flatten_all().unwrap().to_vec1::<half::bf16>().unwrap();
                bytemuck::cast_slice(&vec).to_vec()
            },
            _ => panic!("Unsupported dtype"),
        };
        Cow::Owned(data)
    }

    fn data_len(&self) -> usize {
        self.tensor.elem_count() * self.tensor.dtype().size_in_bytes()
    }
}


/// Save Flux LoRA weights in AI-Toolkit format
pub fn save_flux_lora(
    lora_weights: HashMap<String, (Tensor, Tensor)>, // (lora_A, lora_B) pairs
    output_path: &Path,
    config: &LoRAConfig,
) -> Result<()> {
    // First pass: collect all tensors including slices
    let mut all_tensors: Vec<(String, Tensor)> = Vec::new();
    
    // Convert our internal naming to AI-Toolkit naming
    for (key, (lora_a, lora_b)) in lora_weights.iter() {
        // Check if this is a QKV layer that needs splitting
        if key.contains(".qkv") {
            // Split QKV into separate Q, K, V for AI-Toolkit compatibility
            let base_key = key.replace(".qkv", "");
            let ai_toolkit_base = convert_to_ai_toolkit_naming(&base_key);
            
            // Get dimensions
            let rank = lora_a.dims()[1]; // lora_a: [in_features, rank]
            let out_features = lora_b.dims()[1]; // lora_b: [rank, out_features]
            let split_size = out_features / 3; // Assume Q, K, V are equal size
            
            // Split lora_b into Q, K, V (lora_a is shared)
            for (idx, target) in ["to_q", "to_k", "to_v"].iter().enumerate() {
                let start = idx * split_size;
                let end = (idx + 1) * split_size;
                
                // Extract the slice for this target
                let lora_b_slice = lora_b.narrow(1, start, split_size)?;
                
                // Save lora_A (same for all three)
                let key_a = format!("{}.{}.lora_A", ai_toolkit_base, target);
                all_tensors.push((key_a, lora_a.clone()));
                
                // Save lora_B (sliced)
                let key_b = format!("{}.{}.lora_B", ai_toolkit_base, target);
                all_tensors.push((key_b, lora_b_slice));
            }
        } else {
            // Direct mapping for non-QKV layers
            let ai_toolkit_key = convert_to_ai_toolkit_naming(key);
            
            // Save lora_A
            let key_a = format!("{}.lora_A", ai_toolkit_key);
            all_tensors.push((key_a, lora_a.clone()));
            
            // Save lora_B
            let key_b = format!("{}.lora_B", ai_toolkit_key);
            all_tensors.push((key_b, lora_b.clone()));
        }
    }
    
    // Prepare metadata
    let mut metadata = HashMap::new();
    metadata.insert("format".to_string(), "pt".to_string());
    metadata.insert("lora_type".to_string(), "flux".to_string());
    metadata.insert("base_model".to_string(), "flux.1-dev".to_string());
    metadata.insert("rank".to_string(), config.rank.to_string());
    metadata.insert("alpha".to_string(), config.alpha.to_string());
    metadata.insert("target_modules".to_string(), config.target_modules.join(","));
    
    // Convert to format expected by serialize_to_file
    let tensor_views: Vec<(String, TensorView)> = all_tensors.iter()
        .map(|(key, tensor)| (key.clone(), TensorView { tensor }))
        .collect();
    
    // Sort by key for consistent output
    let mut sorted_views = tensor_views;
    sorted_views.sort_by(|(a, _), (b, _)| a.cmp(b));
    
    // Convert to &str references
    let tensor_vec: Vec<(&str, TensorView)> = sorted_views.iter()
        .map(|(key, view)| (key.as_str(), *view))
        .collect();
    
    // Save file
    serialize_to_file(tensor_vec, &Some(metadata), output_path)?;
    
    println!("Saved {} LoRA tensors to {:?}", sorted_views.len(), output_path);
    
    Ok(())
}

/// Convert our internal naming to AI-Toolkit format
fn convert_to_ai_toolkit_naming(internal_key: &str) -> String {
    // Examples of conversion:
    // "double_blocks.0.img_attn" -> "transformer.double_blocks.0.img_attn"
    // "double_blocks.0.img_mlp.0" -> "transformer.double_blocks.0.img_mlp.0"
    // "single_blocks.0.attn" -> "transformer.single_transformer_blocks.0.attn"
    
    let key = if internal_key.starts_with("double_blocks") {
        internal_key.replace("double_blocks", "transformer.double_blocks")
    } else if internal_key.starts_with("single_blocks") {
        internal_key.replace("single_blocks", "transformer.single_transformer_blocks")
    } else {
        format!("transformer.{}", internal_key)
    };
    
    // QKV splitting is now handled in save_flux_lora function
    key
}

/// Create example LoRA weights for testing
pub fn create_example_lora_weights(
    device: &Device,
    dtype: DType,
    config: &LoRAConfig,
) -> Result<HashMap<String, (Tensor, Tensor)>> {
    let mut weights = HashMap::new();
    let hidden_size = 3072;
    
    // Create LoRA for first double block as example
    for target in ["to_q", "to_k", "to_v"] {
        let key = format!("double_blocks.0.img_attn.{}", target);
        let lora_a = Tensor::randn(0.0, 0.02, (hidden_size, config.rank), device)?;
        let lora_b = Tensor::zeros((config.rank, hidden_size), dtype, device)?;
        weights.insert(key, (lora_a, lora_b));
    }
    
    // MLP
    let key = "double_blocks.0.img_mlp.0".to_string();
    let lora_a = Tensor::randn(0.0, 0.02, (hidden_size, config.rank), device)?;
    let lora_b = Tensor::zeros((config.rank, hidden_size * 4), dtype, device)?;
    weights.insert(key, (lora_a, lora_b));
    
    Ok(weights)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_naming_conversion() {
        assert_eq!(
            convert_to_ai_toolkit_naming("double_blocks.0.img_attn.qkv"),
            "transformer.double_blocks.0.img_attn.qkv"
        );
        
        assert_eq!(
            convert_to_ai_toolkit_naming("single_blocks.0.attn.qkv"),
            "transformer.single_transformer_blocks.0.attn.qkv"
        );
    }
}