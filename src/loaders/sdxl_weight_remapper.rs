use std::collections::HashMap;
use candle_core::Tensor;

/// Check if weights need prefix stripping
pub fn check_and_strip_prefix(weights: &HashMap<String, Tensor>) -> HashMap<String, Tensor> {
    // Check if weights have common prefixes
    let has_model_prefix = weights.keys().any(|k| k.starts_with("model.diffusion_model."));
    let has_diffusion_prefix = weights.keys().any(|k| k.starts_with("diffusion_model."));
    
    if has_model_prefix {
        println!("Stripping 'model.diffusion_model.' prefix from weights");
        let mut stripped = HashMap::new();
        for (key, tensor) in weights {
            let new_key = key.strip_prefix("model.diffusion_model.")
                .unwrap_or(key)
                .to_string();
            stripped.insert(new_key, tensor.clone());
        }
        stripped
    } else if has_diffusion_prefix {
        println!("Stripping 'diffusion_model.' prefix from weights");
        let mut stripped = HashMap::new();
        for (key, tensor) in weights {
            let new_key = key.strip_prefix("diffusion_model.")
                .unwrap_or(key)
                .to_string();
            stripped.insert(new_key, tensor.clone());
        }
        stripped
    } else {
        println!("No prefix stripping needed");
        weights.clone()
    }
}

/// Remap SDXL weights from SD checkpoint format to Diffusers format
pub fn remap_sdxl_weights(weights: &HashMap<String, Tensor>) -> HashMap<String, Tensor> {
    // First strip any common prefixes
    let weights = check_and_strip_prefix(weights);
    
    // Check what format we have
    let has_input_blocks = weights.keys().any(|k| k.starts_with("input_blocks."));
    let has_conv_in = weights.keys().any(|k| k.starts_with("conv_in."));
    let has_first_stage = weights.keys().any(|k| k.starts_with("first_stage_model."));
    
    if has_conv_in && !has_first_stage {
        println!("Weights already in Diffusers format, no remapping needed");
        return weights;
    }
    
    if !has_input_blocks {
        println!("Warning: Weights don't match expected SD checkpoint format");
        println!("First few keys: {:?}", weights.keys().take(5).collect::<Vec<_>>());
        return weights;
    }
    
    println!("Remapping SD checkpoint format to Diffusers format...");
    let mut remapped = HashMap::new();
    
    for (key, tensor) in &weights {
        let new_key = match key.as_str() {
            // Initial conv - SD uses input_blocks.0.0
            "input_blocks.0.0.weight" => "conv_in.weight",
            "input_blocks.0.0.bias" => "conv_in.bias",
            
            // Time embeddings
            "time_embed.0.weight" => "time_embedding.linear_1.weight",
            "time_embed.0.bias" => "time_embedding.linear_1.bias",
            "time_embed.2.weight" => "time_embedding.linear_2.weight",
            "time_embed.2.bias" => "time_embedding.linear_2.bias",
            
            // Label embeddings (for SDXL conditioning)
            "label_emb.0.0.weight" => "add_embedding.linear_1.weight",
            "label_emb.0.0.bias" => "add_embedding.linear_1.bias",
            "label_emb.0.2.weight" => "add_embedding.linear_2.weight",
            "label_emb.0.2.bias" => "add_embedding.linear_2.bias",
            
            // Down blocks - map input_blocks to down_blocks
            // SD: input_blocks.1.0 -> down_blocks.0.resnets.0
            // SD: input_blocks.1.1 -> down_blocks.0.attentions.0
            // SD: input_blocks.2.0 -> down_blocks.0.resnets.1
            // etc.
            key if key.starts_with("input_blocks.") => {
                // This is complex mapping, keep original for now
                // TODO: Implement full SD->Diffusers block mapping
                key
            }
            
            // Middle blocks
            key if key.starts_with("middle_block.") => {
                // Map middle_block to mid_block
                &key.replace("middle_block.", "mid_block.")
            }
            
            // Output blocks -> up_blocks
            key if key.starts_with("output_blocks.") => {
                // TODO: Implement full mapping
                key
            }
            
            // Final layers
            "out.0.weight" => "conv_norm_out.weight",
            "out.0.bias" => "conv_norm_out.bias",
            "out.2.weight" => "conv_out.weight",
            "out.2.bias" => "conv_out.bias",
            
            // VAE encoder mappings - SD format uses first_stage_model prefix
            key if key.starts_with("first_stage_model.encoder.") => {
                // Strip the first_stage_model prefix for encoder weights
                &key["first_stage_model.".len()..]
            }
            key if key.starts_with("first_stage_model.decoder.") => {
                // Strip the first_stage_model prefix for decoder weights
                &key["first_stage_model.".len()..]
            }
            key if key.starts_with("first_stage_model.quant_conv.") => {
                // Map quantization conv
                &key["first_stage_model.".len()..]
            }
            key if key.starts_with("first_stage_model.post_quant_conv.") => {
                // Map post quantization conv
                &key["first_stage_model.".len()..]
            }
            
            // Keep other weights as-is
            _ => key,
        };
        
        remapped.insert(new_key.to_string(), tensor.clone());
    }
    
    println!("Remapped {} weights", remapped.len());
    println!("Sample remapped keys: {:?}", remapped.keys().take(10).collect::<Vec<_>>());
    
    // Check if we have essential keys
    if remapped.contains_key("conv_in.weight") {
        println!("✓ Found conv_in.weight after remapping");
    } else {
        println!("✗ Missing conv_in.weight after remapping");
        println!("Looking for alternative keys...");
        if weights.contains_key("input_blocks.0.0.weight") {
            println!("  Found input_blocks.0.0.weight in original weights");
        }
        // Show keys that might be the conv_in
        for (key, _) in weights {
            if key.contains("conv") && key.contains("in") || key == "input_blocks.0.0.weight" {
                println!("  Potential conv_in key: {}", key);
            }
        }
    }
    
    remapped
}