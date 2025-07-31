//! Image utilities following exact candle-examples patterns
//! Provides proper image saving for sampling during training

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor, D};
use std::path::Path;

/// Save a tensor as an image following candle-examples pattern
/// This matches the implementation in candle-examples/src/lib.rs
pub fn save_image<P: AsRef<Path>>(tensor: &Tensor, path: P) -> Result<()> {
    // Convert from [-1, 1] to [0, 255] using exact candle pattern
    let tensor = ((tensor.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?;
    let tensor = tensor.to_dtype(DType::U8)?;
    
    // Get dimensions - expecting CHW format
    let (channel, height, width) = tensor.dims3()
        .context("Expected 3D tensor [C, H, W]")?;
    
    if channel != 3 {
        anyhow::bail!("Expected 3 channels (RGB), got {}", channel);
    }
    
    // Permute from CHW to HWC for image crate
    let tensor = tensor.permute((1, 2, 0))?;
    let data = tensor.flatten_all()?.to_vec1::<u8>()?;
    
    // Create and save image
    let img = image::ImageBuffer::<image::Rgb<u8>, Vec<u8>>::from_raw(
        width as u32,
        height as u32,
        data,
    ).context("Failed to create image buffer")?;
    
    // Determine format from path extension
    let path = path.as_ref();
    match path.extension().and_then(|s| s.to_str()) {
        Some("jpg") | Some("jpeg") => {
            img.save_with_format(path, image::ImageFormat::Jpeg)?;
        }
        Some("png") => {
            img.save_with_format(path, image::ImageFormat::Png)?;
        }
        _ => {
            // Default to PNG
            img.save_with_format(path, image::ImageFormat::Png)?;
        }
    }
    
    Ok(())
}

/// Decode VAE latents following candle patterns
/// Handles model-specific scaling factors
pub fn decode_vae_latents(
    latents: &Tensor,
    vae_scale: f32,
    model_type: ModelType,
) -> Result<Tensor> {
    // Apply model-specific pre-scaling
    let scaled_latents = match model_type {
        ModelType::SD35 => {
            // SD3.5 uses TAESD3 scaling
            let x = (latents / 1.5305)?;
            (x + 0.0609)?
        }
        _ => {
            // Standard scaling for Flux and SDXL
            (latents / vae_scale)?
        }
    };
    
    Ok(scaled_latents)
}

/// Convert decoded VAE output to image range
/// Following exact candle pattern for image conversion
pub fn vae_output_to_image(decoded: &Tensor) -> Result<Tensor> {
    // Standard conversion from [-1, 1] to [0, 1]
    let images = ((decoded / 2.0)? + 0.5)?;
    let images = images.clamp(0.0, 1.0)?;
    
    // Scale to [0, 255]
    let images = (images * 255.0)?;
    
    Ok(images)
}

/// Model types for VAE scaling
#[derive(Debug, Clone, Copy)]
pub enum ModelType {
    SDXL,
    SD35,
    Flux,
}

impl ModelType {
    /// Get the VAE scaling factor for each model
    pub fn vae_scale(&self) -> f32 {
        match self {
            ModelType::SDXL => 0.08,
            ModelType::SD35 => 0.18215,
            ModelType::Flux => 0.13025,
        }
    }
}

/// Create output directory following user requirements
pub fn create_sample_directory(lora_name: &str) -> Result<std::path::PathBuf> {
    let output_dir = std::path::PathBuf::from("/outputs")
        .join(lora_name)
        .join("samples");
    
    std::fs::create_dir_all(&output_dir)
        .with_context(|| format!("Failed to create directory: {:?}", output_dir))?;
    
    Ok(output_dir)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::Device;
    
    #[test]
    fn test_model_vae_scales() {
        assert_eq!(ModelType::SDXL.vae_scale(), 0.08);
        assert_eq!(ModelType::SD35.vae_scale(), 0.18215);
        assert_eq!(ModelType::Flux.vae_scale(), 0.13025);
    }
    
    #[test]
    fn test_tensor_conversion() -> Result<()> {
        let device = Device::Cpu;
        
        // Create test tensor in [-1, 1] range
        let tensor = Tensor::randn(0.0, 0.5, &[3, 64, 64], &device)?;
        let tensor = tensor.clamp(-1.0, 1.0)?;
        
        // Test conversion
        let converted = vae_output_to_image(&tensor)?;
        
        // Check range is [0, 255]
        let min = converted.min(D::Flat)?.to_scalar::<f32>()?;
        let max = converted.max(D::Flat)?.to_scalar::<f32>()?;
        
        assert!(min >= 0.0);
        assert!(max <= 255.0);
        
        Ok(())
    }
}