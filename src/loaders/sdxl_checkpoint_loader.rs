//! SDXL checkpoint loader that handles different model formats

use anyhow::{Result, Context};
use candle_core::{Device, DType, Tensor};
use candle_nn::VarBuilder;
use std::collections::HashMap;
use std::path::Path;

use crate::models::sdxl_unet::SDXLUNet2DConditionModel;
use crate::loaders::sdxl_weight_remapper::remap_sdxl_weights;

/// Load SDXL UNet from a checkpoint that might be in SD format
pub fn load_sdxl_checkpoint(
    model_path: &Path,
    device: &Device,
    dtype: DType,
) -> Result<SDXLUNet2DConditionModel> {
    println!("Loading SDXL checkpoint with format detection...");
    
    // Load all tensors
    let tensors = unsafe {
        candle_core::safetensors::load(model_path.to_str().unwrap(), device)?
    };
    
    // Check format
    let has_diffusers_format = tensors.contains_key("conv_in.weight");
    let has_model_prefix = tensors.contains_key("model.diffusion_model.conv_in.weight");
    
    if has_diffusers_format {
        println!("Detected Diffusers format - loading directly");
        let vb = VarBuilder::from_tensors(tensors, dtype, device);
        return SDXLUNet2DConditionModel::new(vb, Default::default());
    }
    
    if has_model_prefix {
        println!("Detected model.diffusion_model prefix - using prefix loader");
        let vb = VarBuilder::from_tensors(tensors, dtype, device);
        let vb_prefixed = vb.pp("model.diffusion_model");
        return SDXLUNet2DConditionModel::new(vb_prefixed, Default::default());
    }
    
    // If neither format matches, try to remap from SD checkpoint format
    println!("Attempting to remap from SD checkpoint format...");
    
    // Create a mapping of expected names to possible source names
    let remapped_tensors = remap_sd_to_diffusers(tensors)?;
    
    let vb = VarBuilder::from_tensors(remapped_tensors, dtype, device);
    SDXLUNet2DConditionModel::new(vb, Default::default())
}

/// Remap SD checkpoint format to Diffusers format
fn remap_sd_to_diffusers(tensors: HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>> {
    // Use the comprehensive remapper
    let remapped = remap_sdxl_weights(&tensors);
    
    // Check if we have essential layers after remapping
    let has_conv_in = remapped.contains_key("conv_in.weight");
    let has_time_embed = remapped.contains_key("time_embedding.linear_1.weight") || 
                         remapped.contains_key("time_embed.0.weight");
    
    if !has_conv_in {
        println!("Warning: conv_in.weight not found after remapping");
        println!("Available keys sample: {:?}", remapped.keys().take(10).collect::<Vec<_>>());
    }
    
    if !has_time_embed {
        println!("Warning: time embedding weights not found after remapping");
    }
    
    Ok(remapped)
}

/// Load text encoders for SDXL
pub fn load_text_encoders_sdxl(
    model_path: &Path,
    device: &Device,
    dtype: DType,
) -> Result<(
    candle_transformers::models::stable_diffusion::clip::ClipTextTransformer,
    candle_transformers::models::stable_diffusion::clip::ClipTextTransformer,
)> {
    use candle_transformers::models::stable_diffusion::clip::{self, ClipTextTransformer};
    use candle_nn::VarBuilder;
    
    // For SDXL, use separate CLIP model files
    let clip_dir = Path::new("/home/alex/SwarmUI/Models/clip");
    let clip_l_path = clip_dir.join("clip_l.safetensors");
    let clip_g_path = clip_dir.join("clip_g.safetensors");
    
    println!("Loading CLIP models from separate files:");
    println!("  CLIP-L: {:?}", clip_l_path);
    println!("  CLIP-G: {:?}", clip_g_path);
    
    // Load CLIP-L
    let text_encoder = if clip_l_path.exists() {
        println!("Loading CLIP-L from {:?}", clip_l_path);
        let tensors = candle_core::safetensors::load(&clip_l_path, device)?;
        let vb = VarBuilder::from_tensors(tensors, dtype, device);
        let config = clip::Config::v1_5();
        ClipTextTransformer::new(vb, &config)?
    } else {
        anyhow::bail!("CLIP-L model not found at {:?}", clip_l_path);
    };
    
    // Load CLIP-G
    let text_encoder_2 = if clip_g_path.exists() {
        println!("Loading CLIP-G from {:?}", clip_g_path);
        let tensors = candle_core::safetensors::load(&clip_g_path, device)?;
        let vb = VarBuilder::from_tensors(tensors, dtype, device);
        let config = clip::Config::sdxl2();
        ClipTextTransformer::new(vb, &config)?
    } else {
        anyhow::bail!("CLIP-G model not found at {:?}", clip_g_path);
    };
    
    Ok((text_encoder, text_encoder_2))
}