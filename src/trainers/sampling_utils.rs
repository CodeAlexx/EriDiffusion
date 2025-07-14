//! Common sampling utilities for all diffusion models
//! Provides a simple interface for generating samples during training

use anyhow::Result;
use candle_core::{Device, DType, Tensor};
use std::path::PathBuf;
use std::fs;
use image::{ImageBuffer, Rgb};

/// Save a tensor as an image
pub fn save_tensor_as_image(tensor: &Tensor, path: &PathBuf) -> Result<()> {
    // Ensure directory exists
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    
    // Convert tensor to image format
    // Assume tensor is in shape [C, H, W] with values in [-1, 1]
    let (c, h, w) = tensor.dims3()?;
    
    // Move to CPU if needed
    let tensor = if tensor.device().is_cuda() {
        tensor.to_device(&Device::Cpu)?
    } else {
        tensor.clone()
    };
    
    // Denormalize from [-1, 1] to [0, 255]
    let tensor = ((tensor + 1.0)? * 127.5)?;
    let tensor = tensor.clamp(0.0, 255.0)?;
    
    // Convert to u8
    let data: Vec<f32> = tensor.flatten_all()?.to_vec1()?;
    let data: Vec<u8> = data.iter().map(|&x| x as u8).collect();
    
    // Create image buffer
    if c == 3 {
        // RGB image
        let img = ImageBuffer::<Rgb<u8>, _>::from_raw(w as u32, h as u32, data)
            .ok_or_else(|| anyhow::anyhow!("Failed to create image buffer"))?;
        img.save(path)?;
    } else if c == 1 {
        // Grayscale - convert to RGB
        let mut rgb_data = Vec::with_capacity((w * h * 3) as usize);
        for i in 0..((w * h) as usize) {
            rgb_data.push(data[i]);
            rgb_data.push(data[i]);
            rgb_data.push(data[i]);
        }
        let img = ImageBuffer::<Rgb<u8>, _>::from_raw(w as u32, h as u32, rgb_data)
            .ok_or_else(|| anyhow::anyhow!("Failed to create image buffer"))?;
        img.save(path)?;
    } else {
        return Err(anyhow::anyhow!("Unsupported number of channels: {}", c));
    }
    
    Ok(())
}

/// Create a sample directory structure
pub fn create_sample_directory(base_dir: &PathBuf, step: usize) -> Result<PathBuf> {
    let sample_dir = base_dir.join("samples").join(format!("step_{:06}", step));
    fs::create_dir_all(&sample_dir)?;
    Ok(sample_dir)
}

/// Log sampling information
pub fn log_sampling_start(model_type: &str, step: usize, num_samples: usize) {
    println!("\n=== {} Sampling at Step {} ===", model_type, step);
    println!("Generating {} samples...", num_samples);
}

/// Log sampling completion
pub fn log_sampling_complete(model_type: &str, step: usize, sample_dir: &PathBuf) {
    println!("✓ {} sampling complete at step {}", model_type, step);
    println!("  Samples saved to: {}", sample_dir.display());
}

/// Simple sampling message for models without full implementation
pub fn log_sampling_placeholder(model_type: &str, reason: &str) {
    println!("\n⚠️  {} sampling not available: {}", model_type, reason);
    println!("   Training continues without sample generation");
}