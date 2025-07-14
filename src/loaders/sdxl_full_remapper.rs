//! Complete SDXL weight remapping from SD checkpoint format to Diffusers format
//! This properly maps input_blocks -> down_blocks, output_blocks -> up_blocks, etc.

use std::collections::HashMap;
use candle_core::Tensor;

/// Map SD 1.x checkpoint format to SDXL Diffusers format
pub fn remap_sdxl_unet_weights(weights: &HashMap<String, Tensor>) -> HashMap<String, Tensor> {
    let mut remapped = HashMap::new();
    
    // Track which block indices map to which down/up blocks
    // SDXL structure:
    // - input_blocks: 0=conv_in, 1-2=down.0, 3-5=down.1, 6-8=down.2
    // - output_blocks: 0-2=up.3, 3-5=up.2, 6-8=up.1, 9-11=up.0
    
    for (key, tensor) in weights {
        let new_key = if key.starts_with("time_embed.") {
            // Time embedding: time_embed -> time_embedding
            key.replace("time_embed.", "time_embedding.linear_")
                .replace(".0.", ".1.")
                .replace(".2.", ".2.")
        } else if key.starts_with("label_emb.") {
            // Label embedding for SDXL conditioning
            key.replace("label_emb.0.0.", "add_embedding.linear_1.")
                .replace("label_emb.0.2.", "add_embedding.linear_2.")
        } else if key == "input_blocks.0.0.weight" || key == "input_blocks.0.0.bias" {
            // Initial convolution
            key.replace("input_blocks.0.0.", "conv_in.")
        } else if key.starts_with("input_blocks.") {
            // Map input_blocks to down_blocks
            map_input_block_to_down(key)
        } else if key.starts_with("middle_block.") {
            // Middle block: middle_block -> mid_block
            key.replace("middle_block.", "mid_block.")
        } else if key.starts_with("output_blocks.") {
            // Map output_blocks to up_blocks
            map_output_block_to_up(key)
        } else if key == "out.0.weight" || key == "out.0.bias" {
            // Final norm
            key.replace("out.0.", "conv_norm_out.")
        } else if key == "out.2.weight" || key == "out.2.bias" {
            // Final conv
            key.replace("out.2.", "conv_out.")
        } else {
            // Keep as-is
            key.clone()
        };
        
        remapped.insert(new_key, tensor.clone());
    }
    
    remapped
}

/// Map input_blocks to down_blocks
fn map_input_block_to_down(key: &str) -> String {
    // Parse block index
    let parts: Vec<&str> = key.split('.').collect();
    if parts.len() < 3 {
        return key.to_string();
    }
    
    let block_idx: usize = parts[1].parse().unwrap_or(0);
    
    // SDXL mapping:
    // input_blocks.0 = conv_in (handled separately)
    // input_blocks.1-2 = down_blocks.0.resnets.0-1
    // input_blocks.3 = down_blocks.0.downsamplers.0
    // input_blocks.4-5 = down_blocks.1.resnets.0-1
    // input_blocks.6 = down_blocks.1.downsamplers.0
    // input_blocks.7-9 = down_blocks.2.resnets.0-2
    // input_blocks.10 = down_blocks.2.downsamplers.0
    // input_blocks.11-12 = down_blocks.3.resnets.0-1
    
    let (down_idx, res_idx, is_downsample) = match block_idx {
        1..=2 => (0, block_idx - 1, false),
        3 => (0, 0, true),
        4..=5 => (1, block_idx - 4, false),
        6 => (1, 0, true),
        7..=9 => (2, block_idx - 7, false),
        10 => (2, 0, true),
        11..=12 => (3, block_idx - 11, false),
        _ => return key.to_string(),
    };
    
    let layer_type = parts[2];
    let remaining = parts[3..].join(".");
    
    match layer_type {
        "0" => {
            // ResNet block
            if is_downsample {
                format!("down_blocks.{}.downsamplers.0.{}", down_idx, 
                    remaining.replace("op.", "conv."))
            } else {
                // Map SD resnet naming to diffusers
                let new_remaining = remaining
                    .replace("in_layers.0.", "norm1.")
                    .replace("in_layers.2.", "conv1.")
                    .replace("emb_layers.1.", "time_emb_proj.")
                    .replace("out_layers.0.", "norm2.")
                    .replace("out_layers.3.", "conv2.")
                    .replace("skip_connection.", "conv_shortcut.");
                format!("down_blocks.{}.resnets.{}.{}", down_idx, res_idx, new_remaining)
            }
        }
        "1" => {
            // Attention block
            let new_remaining = remaining
                .replace("norm.", "norm.")
                .replace("q.", "to_q.")
                .replace("k.", "to_k.")
                .replace("v.", "to_v.")
                .replace("proj_out.", "to_out.0.");
            format!("down_blocks.{}.attentions.{}.{}", down_idx, res_idx, new_remaining)
        }
        _ => key.to_string(),
    }
}

/// Map output_blocks to up_blocks
fn map_output_block_to_up(key: &str) -> String {
    // Parse block index
    let parts: Vec<&str> = key.split('.').collect();
    if parts.len() < 3 {
        return key.to_string();
    }
    
    let block_idx: usize = parts[1].parse().unwrap_or(0);
    
    // SDXL output blocks are in reverse order
    // output_blocks.0-2 = up_blocks.3.resnets.0-2
    // output_blocks.3-5 = up_blocks.2.resnets.0-2
    // output_blocks.6-8 = up_blocks.1.resnets.0-2
    // output_blocks.9-11 = up_blocks.0.resnets.0-2
    
    let (up_idx, res_idx) = match block_idx {
        0..=2 => (3, block_idx),
        3..=5 => (2, block_idx - 3),
        6..=8 => (1, block_idx - 6),
        9..=11 => (0, block_idx - 9),
        _ => return key.to_string(),
    };
    
    let layer_type = parts[2];
    let remaining = parts[3..].join(".");
    
    match layer_type {
        "0" => {
            // ResNet block (same mapping as input blocks)
            let new_remaining = remaining
                .replace("in_layers.0.", "norm1.")
                .replace("in_layers.2.", "conv1.")
                .replace("emb_layers.1.", "time_emb_proj.")
                .replace("out_layers.0.", "norm2.")
                .replace("out_layers.3.", "conv2.")
                .replace("skip_connection.", "conv_shortcut.");
            format!("up_blocks.{}.resnets.{}.{}", up_idx, res_idx, new_remaining)
        }
        "1" => {
            // Attention block
            let new_remaining = remaining
                .replace("norm.", "norm.")
                .replace("q.", "to_q.")
                .replace("k.", "to_k.")
                .replace("v.", "to_v.")
                .replace("proj_out.", "to_out.0.");
            format!("up_blocks.{}.attentions.{}.{}", up_idx, res_idx, new_remaining)
        }
        "2" => {
            // Upsample conv
            format!("up_blocks.{}.upsamplers.0.conv.{}", up_idx, 
                remaining.replace("conv.", ""))
        }
        _ => key.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_remap_conv_in() {
        let mut weights = HashMap::new();
        weights.insert("input_blocks.0.0.weight".to_string(), Tensor::zeros(&[320, 4, 3, 3], candle_core::DType::F32, &candle_core::Device::Cpu).unwrap());
        
        let remapped = remap_sdxl_unet_weights(&weights);
        assert!(remapped.contains_key("conv_in.weight"));
    }
    
    #[test]
    fn test_remap_down_blocks() {
        let mut weights = HashMap::new();
        weights.insert("input_blocks.1.0.in_layers.0.weight".to_string(), Tensor::zeros(&[320], candle_core::DType::F32, &candle_core::Device::Cpu).unwrap());
        
        let remapped = remap_sdxl_unet_weights(&weights);
        assert!(remapped.contains_key("down_blocks.0.resnets.0.norm1.weight"));
    }
}