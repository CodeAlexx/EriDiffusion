use anyhow::Context;
use flame_core::device::Device;
use flame_core::{DType, Error, Result, Tensor};
use image::{ImageBuffer, RgbImage};
use log::{debug, info};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::sampler::{SDXLSampler, SamplerConfig};
use crate::config::trainer_config::SampleConfig;
use crate::trainers::sdxl_vae_native::SDXLVAENative;
use crate::trainers::text_encoders::TextEncoders;

/// Generate samples during training for monitoring progress
pub fn generate_samples(
    unet_weights: &HashMap<String, Tensor>,
    vae: &SDXLVAENative,
    text_encoders: &TextEncoders,
    lora_collection: &crate::trainers::lora::LoRACollection,
    config: &SampleConfig,
    device: &Device,
    step: usize,
    output_dir: &Path,
) -> flame_core::Result<Vec<PathBuf>> {
    info!("Generating {} samples at step {}", config.prompts.len(), step);

    let sampler_config = SamplerConfig {
        num_inference_steps: config.sample_steps,
        guidance_scale: config.guidance_scale,
        eta: 0.0, // Default eta value
        scheduler_type: super::SchedulerType::DDPM,
    };

    // Create a new HashMap by cloning individual tensors
    let mut unet_weights_owned = HashMap::new();
    for (k, v) in unet_weights {
        unet_weights_owned.insert(k.clone(), v.clone());
    }

    let mut sampler = SDXLSampler::new(
        unet_weights_owned,
        Some(lora_collection.clone()),
        sampler_config,
        device.clone(),
        DType::F16,
    )?;

    let mut generated_paths = Vec::new();

    // Generate samples for each prompt
    for (i, prompt) in config.prompts.iter().enumerate() {
        debug!("Generating sample {} with prompt: {}", i, prompt);

        // Encode prompt
        let (text_embeds, pooled_embeds) = text_encoders.encode_sdxl(prompt, 77)?;

        // Sample
        let latents = sampler.sample(
            &text_embeds,
            &pooled_embeds,
            config.height,
            config.width,
            Some(config.seed + i as u64),
        )?;

        // Decode latents
        let images = vae.decode(&latents).map_err(|e| {
            flame_core::Error::InvalidOperation(format!("VAE decode failed: {}", e))
        })?;

        // Convert to RGB and save
        let image = tensor_to_image(&images)?;
        let filename = format!("step_{:06}_sample_{:02}.png", step, i);
        let path = output_dir.join("samples").join(filename);

        // Ensure directory exists
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| flame_core::Error::InvalidOperation(e.to_string()))?;
        }

        // Save image
        image.save(&path).map_err(|e| {
            flame_core::Error::InvalidOperation(format!("Failed to save image: {}", e))
        })?;
        generated_paths.push(path);
    }

    info!("Generated {} samples successfully", generated_paths.len());
    Ok(generated_paths)
}

/// Save samples in various formats
pub fn save_samples(
    images: &Tensor,
    output_dir: &Path,
    prefix: &str,
    format: &str,
) -> flame_core::Result<Vec<PathBuf>> {
    let batch_size = images.shape().dims()[0];
    let mut saved_paths = Vec::new();

    for i in 0..batch_size {
        // Extract single image
        let image = images.narrow(0, i, 1)?.squeeze(Some(0))?;

        // Convert to RGB image
        let rgb_image = tensor_to_image(&image)?;

        // Generate filename
        let filename = format!("{}_{:03}.{}", prefix, i, format);
        let path = output_dir.join(filename);

        // Save based on format
        match format {
            "png" => rgb_image.save(&path).map_err(|e| {
                flame_core::Error::InvalidOperation(format!("Failed to save PNG: {}", e))
            })?,
            "jpg" | "jpeg" => {
                let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(
                    std::fs::File::create(&path).map_err(|e| {
                        flame_core::Error::InvalidOperation(format!(
                            "Failed to create file: {}",
                            e
                        ))
                    })?,
                    95,
                );
                encoder.encode_image(&rgb_image).map_err(|e| {
                    flame_core::Error::InvalidOperation(format!(
                        "Failed to encode JPEG: {}",
                        e
                    ))
                })?;
            }
            _ => {
                return Err(flame_core::Error::InvalidOperation(format!(
                    "Unsupported image format: {}",
                    format
                ))
                .into())
            }
        }

        saved_paths.push(path);
    }

    Ok(saved_paths)
}

/// Convert tensor to RGB image
fn tensor_to_image(tensor: &Tensor) -> flame_core::Result<RgbImage> {
    // Assume tensor is [C, H, W] with values in [-1, 1]
    let shape = tensor.shape();
    let channels = shape.dims()[0];
    let height = shape.dims()[1];
    let width = shape.dims()[2];

    if channels != 3 {
        return Err(flame_core::Error::InvalidOperation(format!(
            "Expected 3 channels, got {}",
            channels
        )));
    }

    // Convert to f32 and get data
    let tensor_f32 =
        if tensor.dtype() != DType::F32 { tensor.to_dtype(DType::F32)? } else { tensor.clone() };
    let data = tensor_f32.to_vec1::<f32>()?;

    // Create image buffer
    let mut img = ImageBuffer::new(width as u32, height as u32);

    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let r = ((data[idx] + 1.0) * 127.5).clamp(0.0, 255.0) as u8;
            let g = ((data[idx + height * width] + 1.0) * 127.5).clamp(0.0, 255.0) as u8;
            let b = ((data[idx + 2 * height * width] + 1.0) * 127.5).clamp(0.0, 255.0) as u8;

            img.put_pixel(x as u32, y as u32, image::Rgb([r, g, b]));
        }
    }

    Ok(img)
}

/// Batch generate samples with different settings
pub fn batch_generate_samples(
    unet_weights: &HashMap<String, Tensor>,
    vae: &SDXLVAENative,
    text_encoders: &TextEncoders,
    lora_collection: Option<&crate::trainers::lora::LoRACollection>,
    prompts: &[String],
    settings: Vec<SamplerConfig>,
    device: &Device,
    output_dir: &Path,
) -> flame_core::Result<()> {
    for (i, config) in settings.iter().enumerate() {
        // Create a new HashMap by cloning individual tensors
        let mut unet_weights_owned = HashMap::new();
        for (k, v) in unet_weights {
            unet_weights_owned.insert(k.clone(), v.clone());
        }

        let mut sampler = SDXLSampler::new(
            unet_weights_owned,
            lora_collection.cloned(),
            config.clone(),
            device.clone(),
            DType::F16,
        )?;

        for (j, prompt) in prompts.iter().enumerate() {
            let (text_embeds, pooled_embeds) = text_encoders.encode_sdxl(prompt, 77)?;

            let latents =
                sampler.sample(&text_embeds, &pooled_embeds, 1024, 1024, Some(42 + j as u64))?;

            let images = vae.decode(&latents).map_err(|e| {
                flame_core::Error::InvalidOperation(format!("VAE decode failed: {}", e))
            })?;
            let paths =
                save_samples(&images, output_dir, &format!("batch_{}_prompt_{}", i, j), "png")?;

            debug!("Saved batch {} prompt {} to {:?}", i, j, paths);
        }
    }

    Ok(())
}
