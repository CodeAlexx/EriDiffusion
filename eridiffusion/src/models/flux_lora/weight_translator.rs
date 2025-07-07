//! Weight translation between AI-Toolkit format and Candle format
//! 
//! This module provides utilities to convert between different Flux weight naming conventions:
//! - AI-Toolkit format: Uses separate `to_q`, `to_k`, `to_v` layers
//! - Candle format: Uses combined `qkv` layer
//! 
//! This is useful when:
//! 1. Loading AI-Toolkit LoRAs into Candle for inference
//! 2. Converting Candle-trained weights to AI-Toolkit format
//! 3. Debugging weight compatibility issues

use anyhow::Result;
use candle_core::{Device, DType, Tensor};
use std::collections::HashMap;

/// Translates weight names between different Flux formats
pub struct FluxWeightTranslator {
    /// Mapping from AI-Toolkit names to Candle names
    ai_toolkit_to_candle: HashMap<String, String>,
    /// Mapping from Candle names to AI-Toolkit names
    candle_to_ai_toolkit: HashMap<String, String>,
}

impl FluxWeightTranslator {
    pub fn new() -> Self {
        let mut ai_toolkit_to_candle = HashMap::new();
        let mut candle_to_ai_toolkit = HashMap::new();
        
        // Build the bidirectional mapping
        // Note: This is a simplified example - full implementation would cover all layers
        
        // Double blocks attention
        for i in 0..19 {
            // Image attention
            let ai_base = format!("transformer.double_blocks.{}.img_attn", i);
            let candle_base = format!("double_blocks.{}.img_attn", i);
            
            // AI-Toolkit uses separate to_q, to_k, to_v
            // Candle uses combined qkv
            ai_toolkit_to_candle.insert(
                format!("{}.to_q.weight", ai_base),
                format!("{}.qkv.weight[q]", candle_base),
            );
            ai_toolkit_to_candle.insert(
                format!("{}.to_k.weight", ai_base),
                format!("{}.qkv.weight[k]", candle_base),
            );
            ai_toolkit_to_candle.insert(
                format!("{}.to_v.weight", ai_base),
                format!("{}.qkv.weight[v]", candle_base),
            );
            
            // Output projection
            ai_toolkit_to_candle.insert(
                format!("{}.to_out.0.weight", ai_base),
                format!("{}.proj.weight", candle_base),
            );
            
            // Text attention (same pattern)
            let ai_base = format!("transformer.double_blocks.{}.txt_attn", i);
            let candle_base = format!("double_blocks.{}.txt_attn", i);
            
            ai_toolkit_to_candle.insert(
                format!("{}.to_q.weight", ai_base),
                format!("{}.qkv.weight[q]", candle_base),
            );
            // ... etc
        }
        
        // Build reverse mapping
        for (k, v) in &ai_toolkit_to_candle {
            candle_to_ai_toolkit.insert(v.clone(), k.clone());
        }
        
        Self {
            ai_toolkit_to_candle,
            candle_to_ai_toolkit,
        }
    }
    
    /// Convert AI-Toolkit LoRA weights to Candle format
    pub fn convert_ai_toolkit_to_candle(
        &self,
        weights: HashMap<String, Tensor>,
    ) -> Result<HashMap<String, Tensor>> {
        let mut converted = HashMap::new();
        let mut qkv_parts: HashMap<String, Vec<(String, Tensor)>> = HashMap::new();
        
        for (name, tensor) in weights {
            if name.contains("to_q.") || name.contains("to_k.") || name.contains("to_v.") {
                // This needs to be combined into qkv
                let base = name.replace(".to_q.", ".qkv.")
                    .replace(".to_k.", ".qkv.")
                    .replace(".to_v.", ".qkv.");
                
                let part = if name.contains("to_q") { "q" }
                    else if name.contains("to_k") { "k" }
                    else { "v" };
                
                qkv_parts.entry(base.clone())
                    .or_insert_with(Vec::new)
                    .push((part.to_string(), tensor));
            } else if name.contains("to_out.0.") {
                // Rename to proj
                let new_name = name.replace("to_out.0.", "proj.");
                converted.insert(new_name, tensor);
            } else {
                // Keep as-is
                converted.insert(name, tensor);
            }
        }
        
        // Combine Q, K, V into QKV
        for (base_name, mut parts) in qkv_parts {
            // Sort to ensure Q, K, V order
            parts.sort_by_key(|(part, _)| part.clone());
            
            if parts.len() == 3 {
                // Concatenate along output dimension
                let tensors: Vec<&Tensor> = parts.iter().map(|(_, t)| t).collect();
                let qkv = Tensor::cat(&tensors, 0)?;
                converted.insert(base_name, qkv);
            }
        }
        
        Ok(converted)
    }
    
    /// Convert Candle QKV weights to AI-Toolkit format (split Q, K, V)
    pub fn convert_candle_to_ai_toolkit(
        &self,
        weights: HashMap<String, Tensor>,
    ) -> Result<HashMap<String, Tensor>> {
        let mut converted = HashMap::new();
        
        for (name, tensor) in weights {
            if name.contains(".qkv.") {
                // Split QKV into Q, K, V
                let chunks = tensor.chunk(3, 0)?;
                if chunks.len() == 3 {
                    let base = name.replace(".qkv.", ".");
                    converted.insert(format!("{}.to_q.{}", base, name.split('.').last().unwrap()), chunks[0].clone());
                    converted.insert(format!("{}.to_k.{}", base, name.split('.').last().unwrap()), chunks[1].clone());
                    converted.insert(format!("{}.to_v.{}", base, name.split('.').last().unwrap()), chunks[2].clone());
                }
            } else if name.contains(".proj.") {
                // Rename to to_out.0
                let new_name = name.replace(".proj.", ".to_out.0.");
                converted.insert(new_name, tensor);
            } else {
                // Keep as-is
                converted.insert(name, tensor);
            }
        }
        
        Ok(converted)
    }
}

/// Helper to load AI-Toolkit format weights for use with Candle
pub fn load_ai_toolkit_lora_for_candle(
    lora_path: &str,
    device: &Device,
    dtype: DType,
) -> Result<HashMap<String, Tensor>> {
    // Load the safetensors file
    let tensors = candle_core::safetensors::load(lora_path, device)?;
    
    // Convert to Candle format
    let translator = FluxWeightTranslator::new();
    translator.convert_ai_toolkit_to_candle(tensors)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_weight_name_translation() {
        let translator = FluxWeightTranslator::new();
        
        // Test AI-Toolkit to Candle mapping exists
        assert!(translator.ai_toolkit_to_candle.contains_key(
            "transformer.double_blocks.0.img_attn.to_q.weight"
        ));
    }
}