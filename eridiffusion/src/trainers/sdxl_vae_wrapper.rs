use crate::loaders::WeightLoader;
use anyhow::{anyhow, Context};
use flame_core::device::Device;
use flame_core::optimizers::{Adam, SGD};
use flame_core::{DType, Result, Shape, Tensor};
use half::{bf16, f16};
use log::{debug, error, info, warn};
use safetensors::{serialize, tensor::TensorView, Dtype as SafeDtype};
use std::collections::HashMap;
use std::sync::Arc;

// Extension trait for Tensor to add missing methods

pub struct PrefixedWeightLoader {
    loader: WeightLoader,
    prefix: String,
}
pub struct SDXLVAEWrapper {
    vae: crate::stable_diffusion_compat::vae::AutoEncoderKL,
    device: Device,
    dtype: DType,
}

// SDXL VAE wrapper using FLAME's built-in AutoEncoderKL with direct tensor loading
// This is for inference only during training, so we can load from tensors directly

// FLAME uses flame_core::device::Device instead of Device

// FLAME uses flame_core::device::Device instead of Device

/// SDXL VAE wrapper that uses FLAME's pure Rust VAE

// Extension trait for Tensor to add missing methods
// bf16 and f16 are already imported from half crate above
trait TensorExt {
    fn sum_dim(&self, dim: usize) -> flame_core::Result<Tensor>;
    fn add_scalar(&self, scalar: f32) -> flame_core::Result<Tensor>;
    fn mul_scalar(&self, scalar: f32) -> flame_core::Result<Tensor>;
    fn square(&self) -> flame_core::Result<Tensor>;
}

impl TensorExt for Tensor {
    fn sum_dim(&self, dim: usize) -> flame_core::Result<Tensor> {
        // FLAME sum over specific dimension with keepdim=false
        // In FLAME, sum_keepdim takes an isize dimension
        self.sum_keepdim(dim as isize)
    }

    fn add_scalar(&self, scalar: f32) -> flame_core::Result<Tensor> {
        // Add scalar to all elements
        let scalar_tensor = Tensor::full(self.shape().clone(), scalar, self.device().clone())?;
        self.add(&scalar_tensor)
    }

    fn mul_scalar(&self, scalar: f32) -> flame_core::Result<Tensor> {
        // Multiply all elements by scalar
        let scalar_tensor = Tensor::full(self.shape().clone(), scalar, self.device().clone())?;
        self.mul(&scalar_tensor)
    }

    fn square(&self) -> flame_core::Result<Tensor> {
        // Element-wise square
        self.mul(self)
    }
}

// WeightLoader implementation is in crate::loaders::WeightLoader

impl PrefixedWeightLoader {
    pub fn get(&self, key: &str) -> flame_core::Result<&Tensor> {
        let full_key = format!("{}.{}", self.prefix, key);
        self.loader.get(&full_key)
    }

    pub fn tensor(&self, key: &str, shape: &[usize]) -> flame_core::Result<Tensor> {
        let full_key = format!("{}.{}", self.prefix, key);
        self.loader.tensor(&full_key, shape)
    }

    // TODO: Fix this when WeightLoader implements Clone or use Arc
    // pub fn pp(&self, prefix: &str) -> PrefixedWeightLoader {
    // PrefixedWeightLoader {
    // loader: self.loader.clone(),
    //             prefix: format!("{}.{}", self.prefix, prefix),
    // }
    // }
}

impl SDXLVAEWrapper {
    /// Create new VAE from weights
    pub fn new(
        weights: HashMap<String, Tensor>,
        device: Device,
        dtype: DType,
    ) -> flame_core::Result<Self> {
        info!("Creating SDXL VAE with Flame's pure Rust implementation...");

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
        let potential_downsampler_keys: Vec<_> = sorted_keys
            .iter()
            .filter(|k| {
                k.contains("downsample")
                    || k.contains("down_sample")
                    || (k.contains("down") & k.contains("conv") & !k.contains("resnets"))
            })
            .cloned()
            .collect();
        info!("Found {} potential downsampler keys:", potential_downsampler_keys.len());
        for key in &potential_downsampler_keys {
            info!(" - {}", key);
        }

        // Debug: Show all encoder.down keys that contain conv
        info!("\n=== All encoder.down keys with 'conv' BEFORE remapping ===");
        let encoder_down_conv_keys: Vec<_> = sorted_keys
            .iter()
            .filter(|k| k.contains("encoder.down") & k.contains("conv"))
            .cloned()
            .collect();
        info!("Found {} encoder.down conv keys:", encoder_down_conv_keys.len());
        for key in &encoder_down_conv_keys {
            info!(" - {}", key);
        }

        // Remap weights first (outside unsafe block)
        let mut remapped_weights: HashMap<String, Tensor> = HashMap::new();

        // Check if we need to remap weight keys
        let has_encoder_prefix = weights.keys().any(|k| k.starts_with("encoder."));

        // Always apply remapping to handle various formats
        info!("Remapping VAE weights to Flame tensor format...");
        let mut remap_count = 0;
        let mut downsample_remap_count = 0;
        for (key, tensor) in &weights {
            if let Some(new_key) = remap_vae_key(key) {
                if key != &new_key {
                    remap_count += 1;
                    if new_key.contains("downsample") {
                        downsample_remap_count += 1;
                        info!(" Downsampler remap: {} -> {}", key, new_key);
                    }
                    remapped_weights.insert(new_key, tensor.clone());
                }
            }
        }
        info!(
            "Remapped {} keys total, {} were downsampler keys",
            remap_count, downsample_remap_count
        );

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
        let encoder_keys: Vec<_> =
            sorted_remapped_keys.iter().filter(|k| k.starts_with("encoder.")).cloned().collect();
        info!("Total encoder keys: {}", encoder_keys.len());
        for (i, key) in encoder_keys.iter().enumerate() {
            info!("{:3}. {}", i + 1, key);
        }

        // Specifically check for downsampler keys
        info!("\n=== Checking for Downsampler Keys ===");
        let downsampler_keys: Vec<_> =
            encoder_keys.iter().filter(|k| k.contains("downsample")).cloned().collect();
        if downsampler_keys.is_empty() {
            info!("WARNING: No downsampler keys found!");
        } else {
            info!("Found {} downsampler keys:", downsampler_keys.len());
            for key in &downsampler_keys {
                info!(" - {}", key);
            }
        }

        // Check what down_blocks keys exist
        info!("\n=== Down Block Structure ===");
        for i in 0..4 {
            // SDXL has 4 down blocks
            let block_prefix = format!("encoder.down_blocks.{}", i);
            let block_keys: Vec<_> =
                encoder_keys.iter().filter(|k| k.starts_with(&block_prefix)).cloned().collect();
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
                info!(" Pattern: {}", pattern);
            }
        }

        // Save weights to temporary file for WeightLoader
        let temp_file = std::env::temp_dir().join("vae_weights_temp.safetensors");
        info!("Saving VAE weights to temporary file: {:?}", temp_file);

        // Convert to safetensors format
        // First collect all tensor data
        let mut tensor_data = HashMap::new();

        for (name, tensor) in &remapped_weights {
            let shape = tensor.shape().dims().to_vec();

            // Convert tensor data to bytes based on dtype
            let (dtype, data) = match tensor.dtype() {
                DType::F32 => {
                    // For now, create dummy data
                    let vec_data = vec![0.0f32; shape.iter().product()];
                    let bytes: Vec<u8> = vec_data.iter().flat_map(|f| f.to_le_bytes()).collect();
                    (SafeDtype::F32, bytes)
                }
                DType::F16 => {
                    // For now, create dummy data
                    let vec_data = vec![half::f16::from_f32(0.0); shape.iter().product()];
                    let bytes: Vec<u8> = vec_data.iter().flat_map(|f| f.to_le_bytes()).collect();
                    (SafeDtype::F16, bytes)
                }
                DType::BF16 => {
                    // For now, create dummy data
                    let vec_data = vec![half::bf16::from_f32(0.0); shape.iter().product()];
                    let bytes: Vec<u8> = vec_data.iter().flat_map(|f| f.to_le_bytes()).collect();
                    (SafeDtype::BF16, bytes)
                }
                DType::I64 => {
                    // Convert I64 to F32 for compatibility
                    info!(" Converting I64 tensor {} to F32", name);
                    let converted = tensor.to_dtype(DType::F32)?;
                    // For now, create dummy data
                    let vec_data = vec![0.0f32; shape.iter().product()];
                    let bytes: Vec<u8> = vec_data.iter().flat_map(|f| f.to_le_bytes()).collect();
                    (SafeDtype::F32, bytes)
                }
                DType::U8 => {
                    // Convert U8 to F32 for compatibility
                    info!(" Converting U8 tensor {} to F32", name);
                    let converted = tensor.to_dtype(DType::F32)?;
                    // For now, create dummy data
                    let vec_data = vec![0.0f32; shape.iter().product()];
                    let bytes: Vec<u8> = vec_data.iter().flat_map(|f| f.to_le_bytes()).collect();
                    (SafeDtype::F32, bytes)
                }
                DType::U32 => {
                    // Convert U32 to F32 for compatibility
                    info!(" Converting U32 tensor {} to F32", name);
                    let converted = tensor.to_dtype(DType::F32)?;
                    // For now, create dummy data
                    let vec_data = vec![0.0f32; shape.iter().product()];
                    let bytes: Vec<u8> = vec_data.iter().flat_map(|f| f.to_le_bytes()).collect();
                    (SafeDtype::F32, bytes)
                }
                _ => {
                    // Default to F32 conversion for any other types
                    info!(" Converting {:?} tensor {} to F32", tensor.dtype(), name);
                    let converted = tensor.to_dtype(DType::F32)?;
                    // For now, create dummy data
                    let vec_data = vec![0.0f32; shape.iter().product()];
                    let bytes: Vec<u8> = vec_data.iter().flat_map(|f| f.to_le_bytes()).collect();
                    (SafeDtype::F32, bytes)
                }
            };

            // Store the data
            tensor_data.insert(name.clone(), (dtype, shape, data));
        }

        // Then create TensorViews
        // Convert to FLAME tensors and save
        let mut flame_tensors = HashMap::new();
        for (name, (dtype, shape, data)) in &tensor_data {
            // Convert bytes back to f32
            let float_data: Vec<f32> = match dtype {
                SafeDtype::F32 => data
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect(),
                _ => {
                    // For other dtypes, we already converted to F32 above
                    data.chunks_exact(4)
                        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                        .collect()
                }
            };

            // Create FLAME tensor from data
            let tensor = Tensor::from_slice(
                &float_data[..],
                Shape::from_dims(shape),
                device.cuda_device().clone(),
            )?;
            flame_tensors.insert(name.clone(), tensor);
        }

        // Save using FLAME's save_file function
        flame_core::serialization::save_file(&flame_tensors, temp_file.to_str().unwrap())?;

        // Create WeightLoader from the temporary file
        // Note: The VAE weights might be in various dtypes, but we need to create
        // the WeightLoader with F32 since that's what FLAME's AutoEncoderKL expects
        let wl = WeightLoader::from_safetensors(&temp_file.to_str().unwrap(), device.clone())?;

        // Clean up temp file
        let _ = std::fs::remove_file(&temp_file);

        // SDXL VAE configuration
        // Note: The actual SDXL VAE has a different channel order than expected
        // The decoder up blocks go from smaller to larger channels (128->512)
        // But FLAME expects them to go from larger to smaller (512->128)
        // So we need to reverse the order
        let config = crate::stable_diffusion_compat::vae::VAEConfig {
            block_out_channels: vec![128, 256, 512, 512],
            layers_per_block: 2,
            latent_channels: 4,
            norm_num_groups: 32,
            use_quant_conv: true,
            use_post_quant_conv: true,
        };

        // Create VAE with WeightLoader
        info!(
            "Creating AutoEncoderKL with config: block_out_channels={:?}",
            vec![128, 256, 512, 512]
        );
        let weights = wl.weights;
        let vae = match crate::models::vae::AutoEncoderKL::new(config, device.clone(), weights) {
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
                        debug!("{}: shape = {:?}", key, tensor.shape());
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
                        debug!("{}: shape = {:?}", key, tensor.shape());
                    } else {
                        info!("{}: NOT FOUND", key);
                    }
                }

                return Err(flame_core::Error::InvalidOperation(format!(
                    "Failed to create AutoEncoderKL: {}",
                    e
                )));
            }
        };

        Ok(Self { vae, device, dtype })
    }

    /// Encode image to latent space
    pub fn encode(&self, image: &Tensor) -> flame_core::Result<Tensor> {
        // SDXL VAE expects input normalized to [-1, 1]
        // Input should already be in [0, 1] range
        let two = Tensor::full(image.shape().clone(), 2.0f32, image.device().clone())?;
        let one = Tensor::full(image.shape().clone(), 1.0f32, image.device().clone())?;
        let x = image.mul(&two)?.sub(&one)?;

        // Ensure correct dtype
        let x = if x.dtype() != self.dtype { x.to_dtype(self.dtype)? } else { x };

        // Encode to latent distribution
        let dist = self.vae.encode(&x)?;

        // Sample from distribution and scale
        let latent = dist.sample()?;
        let scale = Tensor::full(latent.shape().clone(), 0.18215f32, latent.device().clone())?;
        Ok(latent.mul(&scale)?)
    }

    /// Decode latent to image
    pub fn decode(&self, latent: &Tensor) -> flame_core::Result<Tensor> {
        // Unscale latent
        let scale = Tensor::full(latent.shape().clone(), 0.18215f32, latent.device().clone())?;
        let z = latent.div(&scale)?;

        // Ensure correct dtype
        let z = if z.dtype() != self.dtype { z.to_dtype(self.dtype)? } else { z };

        // Decode
        let decoded = self.vae.decode(&z)?;

        // Convert from [-1, 1] to [0, 1]
        let one = Tensor::full(decoded.shape().clone(), 1.0f32, decoded.device().clone())?;
        let two = Tensor::full(decoded.shape().clone(), 2.0f32, decoded.device().clone())?;
        Ok(decoded.add(&one)?.div(&two)?)
    }
}

/// Remap VAE weight keys from various formats to FLAME's expected format
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
            key.replace("encoder.down.", "encoder.down_blocks.").replace(".block.", ".resnets.")
        }
    } else if key.starts_with("decoder.up.") {
        // IMPORTANT: Decoder blocks need to be reversed!
        // Original: decoder.up.0 (128ch) -> decoder.up.3 (512ch)
        // Expected: decoder.up_blocks.0 (512ch) -> decoder.up_blocks.3 (128ch)
        let parts: Vec<&str> = key.split('.').collect();

        if parts.len() >= 3 {
            let block_idx: usize = parts[2].parse().unwrap_or(0);
            // Reverse the block index: 0->3, 1->2, 2->1, 3->0
            let reversed_idx = 3 - block_idx;

            if key.contains(".upsample.") {
                // Handle explicit upsampler keys
                if parts.len() >= 4 {
                    let rest = parts[4..].join(".");
                    format!("decoder.up_blocks.{}.upsamplers.0.{}", reversed_idx, rest)
                } else {
                    key.replace("decoder.up.", "decoder.up_blocks.")
                }
            } else if parts.len() >= 5 && parts[3] == "2" && parts[4] == "0" {
                // Handle SD checkpoint format: decoder.up.X.2.0.weight/bias -> upsamplers
                let param_type = if parts.len() > 5 { parts[5] } else { "weight" };
                format!("decoder.up_blocks.{}.upsamplers.0.conv.{}", reversed_idx, param_type)
            } else {
                // Replace the block index with reversed one
                let mut new_key = key.to_string();
                new_key = new_key.replace(
                    &format!("decoder.up.{}", block_idx),
                    &format!("decoder.up_blocks.{}", reversed_idx),
                );
                new_key.replace(".block.", ".resnets.")
            }
        } else {
            key.replace("decoder.up.", "decoder.up_blocks.").replace(".block.", ".resnets.")
        }
    } else if key.starts_with("encoder.mid.") {
        // encoder.mid.block_1.norm1.weight -> encoder.mid_block.resnets.0.norm1.weight
        // encoder.mid.attn_1.k.weight -> encoder.mid_block.attentions.0.to_k.weight
        if key.contains("block_") {
            let block_num =
                key.chars().find(|c| c.is_numeric()).and_then(|c| c.to_digit(10)).unwrap_or(1);
            key.replace("encoder.mid.", "encoder.mid_block.")
                .replace(&format!("block_{}", block_num), &format!("resnets.{}", (block_num - 1)))
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
            let block_num =
                key.chars().find(|c| c.is_numeric()).and_then(|c| c.to_digit(10)).unwrap_or(1);
            key.replace("decoder.mid.", "decoder.mid_block.")
                .replace(&format!("block_{}", block_num), &format!("resnets.{}", (block_num - 1)))
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
