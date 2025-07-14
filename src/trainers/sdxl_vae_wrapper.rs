//! SDXL VAE wrapper using Candle's built-in AutoEncoderKL with direct tensor loading
//! This is for inference only during training, so we can load from tensors directly

use log::{info, debug, warn, error};
use anyhow::{Result, Context};
use candle_core::{Device, DType, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::stable_diffusion;
use std::collections::HashMap;

/// SDXL VAE wrapper that uses Candle's pure Rust VAE
pub struct SDXLVAEWrapper {
    vae: stable_diffusion::vae::AutoEncoderKL,
    device: Device,
    dtype: DType,
}

impl SDXLVAEWrapper {
    /// Create new VAE from weights
    pub fn new(weights: HashMap<String, Tensor>, device: Device, dtype: DType) -> Result<Self> {
        info!("Creating SDXL VAE with Candle's pure Rust implementation...");
        
        // Debug: Print first 20 VAE weight keys before remapping
        info!("\n=== VAE Weight Keys BEFORE Remapping (first 20) ===");
        let mut sorted_keys: Vec<_> = weights.keys().cloned().collect();
        sorted_keys.sort();
        for (i, key) in sorted_keys.iter().take(20).enumerate() {
            info!("{:2}. {}", i + 1, key);
        }
        info!("Total VAE weight keys: {}\n", weights.len());
        
        // Debug: Check for any keys that might be downsamplers before remapping
        info!("\n=== Checking for potential downsampler keys BEFORE remapping ===");
        let potential_downsampler_keys: Vec<_> = sorted_keys.iter()
            .filter(|k| k.contains("downsample") || k.contains("down_sample") || 
                        (k.contains("down") && k.contains("conv") && !k.contains("resnets")))
            .cloned()
            .collect();
        info!("Found {} potential downsampler keys:", potential_downsampler_keys.len());
        for key in &potential_downsampler_keys {
            info!("  - {}", key);
        }
        
        // Debug: Show all encoder.down keys that contain conv
        info!("\n=== All encoder.down keys with 'conv' BEFORE remapping ===");
        let encoder_down_conv_keys: Vec<_> = sorted_keys.iter()
            .filter(|k| k.contains("encoder.down") && k.contains("conv"))
            .cloned()
            .collect();
        info!("Found {} encoder.down conv keys:", encoder_down_conv_keys.len());
        for key in &encoder_down_conv_keys {
            info!("  - {}", key);
        }
        
        // Remap weights first (outside unsafe block)
        let mut remapped_weights = HashMap::new();
        
        // Check if we need to remap weight keys
        let has_encoder_prefix = weights.keys().any(|k| k.starts_with("encoder."));
        
        // Always apply remapping to handle various formats
        info!("Remapping VAE weights to Candle format...");
        let mut remap_count = 0;
        let mut downsample_remap_count = 0;
        for (key, tensor) in &weights {
            if let Some(new_key) = remap_vae_key(key) {
                if key != &new_key {
                    remap_count += 1;
                    if new_key.contains("downsample") {
                        downsample_remap_count += 1;
                        info!("  Downsampler remap: {} -> {}", key, new_key);
                    }
                }
                remapped_weights.insert(new_key, tensor.clone());
            }
        }
        info!("Remapped {} keys total, {} were downsampler keys", remap_count, downsample_remap_count);
        
        // Create VarBuilder directly from tensors (for inference only)
        // This uses unsafe but is fine for inference-only models
        let vs = unsafe {
            
            // Debug: Print first 20 VAE weight keys after remapping
            info!("\n=== VAE Weight Keys AFTER Remapping (first 20) ===");
            let mut sorted_remapped_keys: Vec<_> = remapped_weights.keys().cloned().collect();
            sorted_remapped_keys.sort();
            for (i, key) in sorted_remapped_keys.iter().take(20).enumerate() {
                info!("{:2}. {}", i + 1, key);
            }
            info!("Total remapped VAE weight keys: {}\n", remapped_weights.len());
            
            // Debug: Print ALL encoder keys to diagnose missing downsamplers
            info!("\n=== ALL Encoder Keys After Remapping ===");
            let encoder_keys: Vec<_> = sorted_remapped_keys.iter()
                .filter(|k| k.starts_with("encoder."))
                .cloned()
                .collect();
            info!("Total encoder keys: {}", encoder_keys.len());
            for (i, key) in encoder_keys.iter().enumerate() {
                info!("{:3}. {}", i + 1, key);
            }
            
            // Specifically check for downsampler keys
            info!("\n=== Checking for Downsampler Keys ===");
            let downsampler_keys: Vec<_> = encoder_keys.iter()
                .filter(|k| k.contains("downsample"))
                .cloned()
                .collect();
            if downsampler_keys.is_empty() {
                info!("WARNING: No downsampler keys found!");
            } else {
                info!("Found {} downsampler keys:", downsampler_keys.len());
                for key in &downsampler_keys {
                    info!("  - {}", key);
                }
            }
            
            // Check what down_blocks keys exist
            info!("\n=== Down Block Structure ===");
            for i in 0..4 {  // SDXL has 4 down blocks
                let block_prefix = format!("encoder.down_blocks.{}", i);
                let block_keys: Vec<_> = encoder_keys.iter()
                    .filter(|k| k.starts_with(&block_prefix))
                    .cloned()
                    .collect();
                info!("Block {}: {} keys", i, block_keys.len());
                
                // Show unique patterns in this block
                let mut patterns = std::collections::HashSet::new();
                for key in &block_keys {
                    let parts: Vec<_> = key.split('.').collect();
                    if parts.len() > 3 {
                        patterns.insert(format!("{}.{}.{}", parts[0], parts[1], parts[2]));
                    }
                }
                for pattern in patterns {
                    info!("  Pattern: {}", pattern);
                }
            }
            
            // Save weights to temporary file for VarBuilder
            let temp_file = std::env::temp_dir().join("vae_weights_temp.safetensors");
            info!("Saving VAE weights to temporary file: {:?}", temp_file);
            
            // Convert to safetensors format
            use safetensors::{serialize, tensor::TensorView, Dtype as SafeDtype};
            // First collect all tensor data
            let mut tensor_data: HashMap<String, (SafeDtype, Vec<usize>, Vec<u8>)> = HashMap::new();
            
            for (name, tensor) in &remapped_weights {
                let shape = tensor.dims().to_vec();
                
                // Convert tensor data to bytes based on dtype
                let (dtype, data) = match tensor.dtype() {
                    DType::F32 => {
                        let vec_data = tensor.flatten_all()?.to_vec1::<f32>()?;
                        let bytes: Vec<u8> = vec_data.iter()
                            .flat_map(|f| f.to_le_bytes())
                            .collect();
                        (SafeDtype::F32, bytes)
                    },
                    DType::F16 => {
                        let vec_data = tensor.flatten_all()?.to_vec1::<half::f16>()?;
                        let bytes: Vec<u8> = vec_data.iter()
                            .flat_map(|f| f.to_le_bytes())
                            .collect();
                        (SafeDtype::F16, bytes)
                    },
                    DType::BF16 => {
                        let vec_data = tensor.flatten_all()?.to_vec1::<half::bf16>()?;
                        let bytes: Vec<u8> = vec_data.iter()
                            .flat_map(|f| f.to_le_bytes())
                            .collect();
                        (SafeDtype::BF16, bytes)
                    },
                    _ => {
                        // Default to F32
                        let vec_data = tensor.flatten_all()?.to_vec1::<f32>()?;
                        let bytes: Vec<u8> = vec_data.iter()
                            .flat_map(|f| f.to_le_bytes())
                            .collect();
                        (SafeDtype::F32, bytes)
                    }
                };
                
                // Store the data
                tensor_data.insert(name.clone(), (dtype, shape, data));
            }
            
            // Then create TensorViews
            let mut tensors = HashMap::new();
            for (name, (dtype, shape, data)) in &tensor_data {
                tensors.insert(
                    name.clone(),
                    TensorView::new(*dtype, shape.clone(), data)?
                );
            }
            
            let serialized = serialize(&tensors, &None)?;
            std::fs::write(&temp_file, serialized)?;
            
            // Create VarBuilder from the temporary file
            let vb = VarBuilder::from_mmaped_safetensors(&[&temp_file], dtype, &device)?;
            
            // Clean up temp file
            let _ = std::fs::remove_file(&temp_file);
            
            vb
        };
        
        // SDXL VAE configuration
        // Note: For SDXL, the VAE uses specific channel configurations
        // that match the expected architecture
        let config = stable_diffusion::vae::AutoEncoderKLConfig {
            block_out_channels: vec![128, 256, 512, 512],
            layers_per_block: 2,
            latent_channels: 4,
            norm_num_groups: 32,
            use_quant_conv: true,
            use_post_quant_conv: true,
        };
        
        // Create VAE with VarBuilder
        info!("Creating AutoEncoderKL with config: block_out_channels={:?}", vec![128, 256, 512, 512]);
        let vae = match stable_diffusion::vae::AutoEncoderKL::new(vs, 3, 3, config) {
            Ok(vae) => vae,
            Err(e) => {
                info!("Error creating AutoEncoderKL: {}", e);
                
                // Print detailed shape information for debugging
                info!("\n=== VAE Weight Shape Debugging ===");
                let problematic_keys = vec![
                    "decoder.up_blocks.0.resnets.0.norm1.weight",
                    "decoder.up_blocks.0.resnets.0.norm1.bias",
                    "decoder.up_blocks.0.resnets.0.conv1.weight",
                    "decoder.up_blocks.0.resnets.0.conv1.bias",
                    "decoder.up_blocks.1.resnets.0.norm1.weight",
                    "decoder.up_blocks.1.resnets.0.norm1.bias",
                    "decoder.up_blocks.2.resnets.0.norm1.weight",
                    "decoder.up_blocks.2.resnets.0.norm1.bias",
                    "decoder.up_blocks.3.resnets.0.norm1.weight",
                    "decoder.up_blocks.3.resnets.0.norm1.bias",
                ];
                
                for key in &problematic_keys {
                    if let Some(tensor) = remapped_weights.get(*key) {
                        debug!("{}: shape = {:?}"));
                    } else {
                        info!("{}: NOT FOUND", key);
                    }
                }
                
                // Also check encoder down blocks for comparison
                info!("\n=== Encoder Down Blocks (for comparison) ===");
                let encoder_keys = vec![
                    "encoder.down_blocks.0.resnets.0.norm1.weight",
                    "encoder.down_blocks.1.resnets.0.norm1.weight",
                    "encoder.down_blocks.2.resnets.0.norm1.weight",
                    "encoder.down_blocks.3.resnets.0.norm1.weight",
                ];
                
                for key in &encoder_keys {
                    if let Some(tensor) = remapped_weights.get(*key) {
                        debug!("{}: shape = {:?}"));
                    } else {
                        info!("{}: NOT FOUND", key);
                    }
                }
                
                return Err(anyhow::anyhow!("Failed to create AutoEncoderKL: {}", e));
            }
        };
        
        Ok(Self { vae, device, dtype })
    }
    
    /// Encode image to latent space
    pub fn encode(&self, image: &Tensor) -> Result<Tensor> {
        // SDXL VAE expects input normalized to [-1, 1]
        // Input should already be in [0, 1] range
        let x = ((image * 2.0)? - 1.0)?;
        
        // Ensure correct dtype
        let x = if x.dtype() != self.dtype {
            x.to_dtype(self.dtype)?
        } else {
            x
        };
        
        // Encode to latent distribution
        let dist = self.vae.encode(&x)?;
        
        // Sample from distribution and scale
        let latent = dist.sample()?;
        Ok((latent * 0.18215)?)
    }
    
    /// Decode latent to image
    pub fn decode(&self, latent: &Tensor) -> Result<Tensor> {
        // Unscale latent
        let z = (latent / 0.18215)?;
        
        // Ensure correct dtype
        let z = if z.dtype() != self.dtype {
            z.to_dtype(self.dtype)?
        } else {
            z
        };
        
        // Decode
        let decoded = self.vae.decode(&z)?;
        
        // Convert from [-1, 1] to [0, 1]
        Ok(((decoded + 1.0)? / 2.0)?)
    }
}

/// Remap VAE weight keys from various formats to Candle's expected format
fn remap_vae_key(key: &str) -> Option<String> {
    // Remove first_stage_model prefix if present
    let key = key.strip_prefix("first_stage_model.").unwrap_or(key);
    
    // Remap the key based on patterns
    let new_key = if key.starts_with("encoder.down.") {
        // encoder.down.0.block.0.norm1.weight -> encoder.down_blocks.0.resnets.0.norm1.weight
        // encoder.down.0.downsample.conv.weight -> encoder.down_blocks.0.downsamplers.0.conv.weight
        // encoder.down.0.2.0.weight -> encoder.down_blocks.0.downsamplers.0.conv.weight (SD checkpoint format)
        let parts: Vec<&str> = key.split('.').collect();
        
        if key.contains(".downsample.") {
            // Handle explicit downsampler keys
            if parts.len() >= 4 {
                let block_idx = parts[2]; // The block index
                let rest = parts[4..].join(".");
                format!("encoder.down_blocks.{}.downsamplers.0.{}", block_idx, rest)
            } else {
                key.replace("encoder.down.", "encoder.down_blocks.")
            }
        } else if parts.len() >= 5 && parts[3] == "2" && parts[4] == "0" {
            // Handle SD checkpoint format: encoder.down.X.2.0.weight/bias -> downsamplers
            let block_idx = parts[2];
            let param_type = if parts.len() > 5 { parts[5] } else { "weight" };
            format!("encoder.down_blocks.{}.downsamplers.0.conv.{}", block_idx, param_type)
        } else {
            key.replace("encoder.down.", "encoder.down_blocks.")
               .replace(".block.", ".resnets.")
        }
    } else if key.starts_with("decoder.up.") {
        // decoder.up.0.block.0.norm1.weight -> decoder.up_blocks.0.resnets.0.norm1.weight
        // decoder.up.0.upsample.conv.weight -> decoder.up_blocks.0.upsamplers.0.conv.weight
        // decoder.up.0.2.0.weight -> decoder.up_blocks.0.upsamplers.0.conv.weight (SD checkpoint format)
        let parts: Vec<&str> = key.split('.').collect();
        
        if key.contains(".upsample.") {
            // Handle explicit upsampler keys
            if parts.len() >= 4 {
                let block_idx = parts[2]; // The block index
                let rest = parts[4..].join(".");
                format!("decoder.up_blocks.{}.upsamplers.0.{}", block_idx, rest)
            } else {
                key.replace("decoder.up.", "decoder.up_blocks.")
            }
        } else if parts.len() >= 5 && parts[3] == "2" && parts[4] == "0" {
            // Handle SD checkpoint format: decoder.up.X.2.0.weight/bias -> upsamplers
            let block_idx = parts[2];
            let param_type = if parts.len() > 5 { parts[5] } else { "weight" };
            format!("decoder.up_blocks.{}.upsamplers.0.conv.{}", block_idx, param_type)
        } else {
            key.replace("decoder.up.", "decoder.up_blocks.")
               .replace(".block.", ".resnets.")
        }
    } else if key.starts_with("encoder.mid.") {
        // encoder.mid.block_1.norm1.weight -> encoder.mid_block.resnets.0.norm1.weight
        // encoder.mid.attn_1.k.weight -> encoder.mid_block.attentions.0.to_k.weight
        if key.contains("block_") {
            let block_num = key.chars()
                .find(|c| c.is_numeric())
                .and_then(|c| c.to_digit(10))
                .unwrap_or(1);
            key.replace("encoder.mid.", "encoder.mid_block.")
               .replace(&format!("block_{}", block_num), &format!("resnets.{}", block_num - 1))
        } else if key.contains("attn_") {
            key.replace("encoder.mid.", "encoder.mid_block.")
               .replace("attn_1.", "attentions.0.")
               .replace(".k.", ".to_k.")
               .replace(".q.", ".to_q.")
               .replace(".v.", ".to_v.")
               .replace(".proj_out.", ".to_out.0.")
               .replace(".norm.", ".group_norm.")
        } else {
            key.replace("encoder.mid.", "encoder.mid_block.")
        }
    } else if key.starts_with("decoder.mid.") {
        // Similar to encoder.mid
        if key.contains("block_") {
            let block_num = key.chars()
                .find(|c| c.is_numeric())
                .and_then(|c| c.to_digit(10))
                .unwrap_or(1);
            key.replace("decoder.mid.", "decoder.mid_block.")
               .replace(&format!("block_{}", block_num), &format!("resnets.{}", block_num - 1))
        } else if key.contains("attn_") {
            key.replace("decoder.mid.", "decoder.mid_block.")
               .replace("attn_1.", "attentions.0.")
               .replace(".k.", ".to_k.")
               .replace(".q.", ".to_q.")
               .replace(".v.", ".to_v.")
               .replace(".proj_out.", ".to_out.0.")
               .replace(".norm.", ".group_norm.")
        } else {
            key.replace("decoder.mid.", "decoder.mid_block.")
        }
    } else if key == "encoder.norm_out.weight" {
        // encoder.norm_out.weight -> encoder.conv_norm_out.weight
        "encoder.conv_norm_out.weight".to_string()
    } else if key == "encoder.norm_out.bias" {
        // encoder.norm_out.bias -> encoder.conv_norm_out.bias
        "encoder.conv_norm_out.bias".to_string()
    } else if key == "decoder.norm_out.weight" {
        // decoder.norm_out.weight -> decoder.conv_norm_out.weight
        "decoder.conv_norm_out.weight".to_string()
    } else if key == "decoder.norm_out.bias" {
        // decoder.norm_out.bias -> decoder.conv_norm_out.bias
        "decoder.conv_norm_out.bias".to_string()
    } else if key == "quant_conv.weight" || key == "quant_conv.bias" {
        // These are correct as-is
        key.to_string()
    } else if key == "post_quant_conv.weight" || key == "post_quant_conv.bias" {
        // These are correct as-is
        key.to_string()
    } else {
        // Keep other keys as-is
        key.to_string()
    };
    
    // Apply nin_shortcut -> conv_shortcut remapping on the result
    let new_key = new_key.replace(".nin_shortcut.", ".conv_shortcut.");
    
    Some(new_key)
}


#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_vae_key_remapping() {
        assert_eq!(
            remap_vae_key("first_stage_model.encoder.conv_in.weight"),
            Some("encoder.conv_in.weight".to_string())
        );
        
        assert_eq!(
            remap_vae_key("encoder.conv_in.weight"),
            Some("encoder.conv_in.weight".to_string())
        );
        
        // Test downsampler remapping
        assert_eq!(
            remap_vae_key("encoder.down.0.downsample.conv.weight"),
            Some("encoder.down_blocks.0.downsamplers.0.conv.weight".to_string())
        );
        
        // Test SD checkpoint format downsampler
        assert_eq!(
            remap_vae_key("encoder.down.0.2.0.weight"),
            Some("encoder.down_blocks.0.downsamplers.0.conv.weight".to_string())
        );
        
        assert_eq!(
            remap_vae_key("encoder.down.1.2.0.bias"),
            Some("encoder.down_blocks.1.downsamplers.0.conv.bias".to_string())
        );
        
        // Test upsampler remapping
        assert_eq!(
            remap_vae_key("decoder.up.0.upsample.conv.weight"),
            Some("decoder.up_blocks.0.upsamplers.0.conv.weight".to_string())
        );
        
        // Test SD checkpoint format upsampler
        assert_eq!(
            remap_vae_key("decoder.up.0.2.0.weight"),
            Some("decoder.up_blocks.0.upsamplers.0.conv.weight".to_string())
        );
        
        // Test nin_shortcut remapping
        assert_eq!(
            remap_vae_key("encoder.down_blocks.0.resnets.0.nin_shortcut.weight"),
            Some("encoder.down_blocks.0.resnets.0.conv_shortcut.weight".to_string())
        );
        
        assert_eq!(
            remap_vae_key("decoder.up_blocks.2.resnets.1.nin_shortcut.bias"),
            Some("decoder.up_blocks.2.resnets.1.conv_shortcut.bias".to_string())
        );
        
        // Test normalization layer remapping
        assert_eq!(
            remap_vae_key("encoder.norm_out.weight"),
            Some("encoder.conv_norm_out.weight".to_string())
        );
        
        assert_eq!(
            remap_vae_key("encoder.norm_out.bias"),
            Some("encoder.conv_norm_out.bias".to_string())
        );
        
        assert_eq!(
            remap_vae_key("decoder.norm_out.weight"),
            Some("decoder.conv_norm_out.weight".to_string())
        );
        
        assert_eq!(
            remap_vae_key("decoder.norm_out.bias"),
            Some("decoder.conv_norm_out.bias".to_string())
        );
    }
}