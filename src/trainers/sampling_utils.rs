//! Common utilities for sampling/inference across all models
//! Provides image saving, tensor conversion, and directory management

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use std::path::{Path, PathBuf};

/// Save a tensor as an image file (JPG or PNG)
/// 
/// # Arguments
/// * `tensor` - Image tensor in shape [C, H, W] with values in [-1, 1]
/// * `path` - Output path for the image
/// * `format` - Image format ("jpg" or "png")
pub fn save_tensor_as_image(
    tensor: &Tensor,
    path: &Path,
    format: &str,
) -> Result<()> {
    // Convert from [-1, 1] to [0, 255]
    let tensor = ((tensor.clamp(-1.0, 1.0)? + 1.0)? * 127.5)?;
    let tensor = tensor.to_dtype(DType::U8)?;
    
    // Get dimensions and convert CHW to HWC
    let (channels, height, width) = tensor.dims3()
        .context("Expected tensor with 3 dimensions [C, H, W]")?;
    
    if channels != 3 {
        anyhow::bail!("Expected 3 channels (RGB), got {}", channels);
    }
    
    // Permute from CHW to HWC and flatten
    let tensor = tensor.permute((1, 2, 0))?; // CHW -> HWC
    let pixels = tensor.flatten_all()?.to_vec1::<u8>()?;
    
    // Create image buffer
    let img = image::ImageBuffer::<image::Rgb<u8>, Vec<u8>>::from_raw(
        width as u32,
        height as u32,
        pixels,
    ).context("Failed to create image buffer")?;
    
    // Save based on format
    match format.to_lowercase().as_str() {
        "jpg" | "jpeg" => {
            img.save_with_format(path, image::ImageFormat::Jpeg)
                .context("Failed to save as JPEG")?;
        }
        "png" => {
            img.save_with_format(path, image::ImageFormat::Png)
                .context("Failed to save as PNG")?;
        }
        _ => anyhow::bail!("Unsupported format: {}. Use 'jpg' or 'png'", format),
    }
    
    Ok(())
}

/// Create the standard output directory structure
/// Returns the samples directory path
pub fn create_output_directory(lora_name: &str) -> Result<PathBuf> {
    let output_dir = PathBuf::from("/outputs").join(lora_name).join("samples");
    std::fs::create_dir_all(&output_dir)
        .with_context(|| format!("Failed to create output directory: {:?}", output_dir))?;
    Ok(output_dir)
}

/// Generate a sample filename with step number and index
pub fn generate_sample_filename(
    step: usize,
    index: usize,
    format: &str,
) -> String {
    format!("sample_step{:06}_idx{:02}.{}", step, index, format)
}

/// Save generation metadata alongside the image
pub fn save_metadata(
    path: &Path,
    prompt: &str,
    negative_prompt: Option<&str>,
    step: usize,
    cfg_scale: f32,
    seed: u64,
) -> Result<()> {
    let metadata_path = path.with_extension("txt");
    let mut content = format!(
        "Prompt: {}\n\
         Step: {}\n\
         CFG Scale: {}\n\
         Seed: {}\n",
        prompt, step, cfg_scale, seed
    );
    
    if let Some(neg) = negative_prompt {
        content.push_str(&format!("Negative Prompt: {}\n", neg));
    }
    
    std::fs::write(&metadata_path, content)
        .with_context(|| format!("Failed to write metadata to {:?}", metadata_path))?;
    
    Ok(())
}

/// Decode VAE latents to image tensor
/// Handles different VAE scaling factors for different models
pub fn decode_latents(
    vae: &impl VaeDecoder,
    latents: &Tensor,
    vae_scale: f32,
) -> Result<Tensor> {
    // Scale latents
    let scaled_latents = (latents / vae_scale)?;
    
    // Decode through VAE
    let images = vae.decode(&scaled_latents)?;
    
    // Ensure we have the right shape
    match images.dims() {
        Ok(dims) if dims.len() == 4 => Ok(images),
        Ok(dims) => anyhow::bail!("Expected 4D tensor from VAE, got {:?}", dims),
        Err(e) => Err(e.into()),
    }
}

/// Trait for VAE decoders across different models
pub trait VaeDecoder {
    fn decode(&self, latents: &Tensor) -> Result<Tensor>;
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_output_directory() {
        let dir = create_output_directory("test_lora").unwrap();
        assert_eq!(dir, PathBuf::from("/outputs/test_lora/samples"));
    }
    
    #[test]
    fn test_filename_generation() {
        let filename = generate_sample_filename(1000, 0, "jpg");
        assert_eq!(filename, "sample_step001000_idx00.jpg");
    }
}