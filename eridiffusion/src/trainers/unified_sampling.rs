use anyhow::Context;
use flame_core::device::Device;
use flame_core::optimizers::{Adam, SGD};
use flame_core::{DType, Result, Shape, Tensor};
use half::{bf16, f16};
use image::{ImageBuffer, Rgb};
use rand::{Rng, SeedableRng};
use std::collections::HashMap;
use std::{
    io::Write,
    path::{Path, PathBuf},
};

// Unified sampling/inference for SDXL, SD 3.5, and Flux
// Provides sampling capabilities integrated with training pipelines

/// Configuration for sampling
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    pub num_inference_steps: usize,
    pub guidance_scale: f64,
    pub height: usize,
    pub width: usize,
    pub seed: Option<u64>,
    pub output_dir: PathBuf,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            num_inference_steps: 30,
            guidance_scale: 7.5,
            height: 1024,
            width: 1024,
            seed: None,
            output_dir: PathBuf::from("outputs"),
        }
    }
}

/// Save tensor as image
pub fn save_image(
    image_tensor: &Tensor,
    output_dir: &Path,
    model_name: &str,
    step: usize,
    idx: usize,
    prompt: &str,
) -> flame_core::Result<PathBuf> {
    // Ensure output directory exists
    std::fs::create_dir_all(output_dir).map_err(|e| {
        flame_core::Error::Io(format!("Failed to create output directory: {}", e))
    })?;

    // Convert to RGB image
    let image = image_tensor
        .clamp(-1.0, 1.0)?
        .add_scalar(1.0)?
        .mul_scalar(127.5)?
        .to_dtype(DType::U8)?
        .get(0)?;

    // Create filename
    let safe_prompt = prompt
        .chars()
        .map(|c| if c.is_alphanumeric() || c == ' ' { c } else { '_' })
        .collect::<String>()
        .trim()
        .chars()
        .take(30)
        .collect::<String>()
        .replace(' ', "_");

    let filename = format!("{}_step_{:06}_sample_{:02}_{}.png", model_name, step, idx, safe_prompt);
    let path = output_dir.join(filename);

    // Save image using image crate
    save_tensor_as_image(&image, &path)?;

    println!("Saved {} sample to: {}", model_name, path.display());
    Ok(path)
}

/// Helper function to save a tensor as an image file
pub fn save_tensor_as_image(tensor: &Tensor, path: &PathBuf) -> flame_core::Result<()> {
    let shape = tensor.shape();
    if shape.rank() != 3 {
        return Err(flame_core::Error::InvalidOperation(format!(
            "Expected 3D tensor (C, H, W), got shape: {:?}",
            shape
        )));
    }

    let channels = shape.dims()[0];
    let height = shape.dims()[0];
    let width = shape.dims()[0];
    // First get as f32, then convert to u8
    let data_f32 = tensor.flatten_all()?.to_vec1::<f32>()?;
    let data: Vec<u8> = data_f32.iter().map(|&x| x as u8).collect();

    if channels == 3 {
        // RGB image
        let mut img = ImageBuffer::<Rgb<u8>, Vec<u8>>::new(width as u32, height as u32);

        for y in 0..height {
            for x in 0..width {
                let r = data[y * width + x];
                let g = data[height * width + y * width + x];
                let b = data[2 * height * width + y * width + x];
                img.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
            }
        }

        img.save(path)
            .map_err(|e| flame_core::Error::Io(format!("Failed to save image: {}", e)))?;
    } else if channels == 1 {
        // Grayscale image
        let img = image::ImageBuffer::<image::Luma<u8>, Vec<u8>>::from_raw(
            width as u32,
            height as u32,
            data,
        )
        .ok_or_else(|| {
            flame_core::Error::InvalidOperation("Failed to create image buffer".into())
        })?;

        img.save(path)
            .map_err(|e| flame_core::Error::Io(format!("Failed to save image: {}", e)))?;
    } else {
        return Err(flame_core::Error::InvalidOperation(format!(
            "Unsupported number of channels: {}",
            channels
        )));
    }

    Ok(())
}

/// Generate validation samples during training
pub fn generate_validation_samples(
    model_type: &str,
    vae: &Tensor, // VAE weights
    text_embeddings: &Tensor,
    pooled_embeddings: Option<&Tensor>,
    prompts: &[String],
    config: &SamplingConfig,
    step: usize,
    device: &Device,
) -> flame_core::Result<Vec<PathBuf>> {
    match model_type {
        "sdxl" => {
            println!("Generating SDXL validation samples...");
            generate_sdxl_samples(
                vae,
                text_embeddings,
                pooled_embeddings,
                prompts,
                config,
                step,
                device,
            )
        }
        "sd35" => {
            println!("Generating SD 3.5 validation samples...");
            generate_sd35_samples(
                vae,
                text_embeddings,
                pooled_embeddings,
                prompts,
                config,
                step,
                device,
            )
        }
        "flux" => {
            println!("Generating Flux validation samples...");
            generate_flux_samples(
                vae,
                text_embeddings,
                pooled_embeddings,
                prompts,
                config,
                step,
                device,
            )
        }
        _ => Err(flame_core::Error::InvalidOperation(format!(
            "Unknown model type: {}",
            model_type
        ))),
    }
}

/// Generate SDXL samples
fn generate_sdxl_samples(
    vae: &Tensor,
    text_embeddings: &Tensor,
    pooled_embeddings: Option<&Tensor>,
    prompts: &[String],
    config: &SamplingConfig,
    step: usize,
    device: &Device,
) -> flame_core::Result<Vec<PathBuf>> {
    let mut saved_paths = Vec::new();

    // TODO: Implement actual SDXL sampling
    println!("SDXL sampling not yet implemented in unified sampler");

    Ok(saved_paths)
}

/// Generate SD 3.5 samples
fn generate_sd35_samples(
    vae: &Tensor,
    text_embeddings: &Tensor,
    pooled_embeddings: Option<&Tensor>,
    prompts: &[String],
    config: &SamplingConfig,
    step: usize,
    device: &Device,
) -> flame_core::Result<Vec<PathBuf>> {
    let mut saved_paths = Vec::new();

    // TODO: Implement actual SD 3.5 sampling
    println!("SD 3.5 sampling not yet implemented in unified sampler");

    Ok(saved_paths)
}

/// Generate Flux samples
fn generate_flux_samples(
    vae: &Tensor,
    text_embeddings: &Tensor,
    pooled_embeddings: Option<&Tensor>,
    prompts: &[String],
    config: &SamplingConfig,
    step: usize,
    device: &Device,
) -> flame_core::Result<Vec<PathBuf>> {
    let mut saved_paths = Vec::new();

    // TODO: Implement actual Flux sampling
    println!("Flux sampling not yet implemented in unified sampler");

    Ok(saved_paths)
}

/// Initialize noise for sampling
pub fn get_initial_noise(
    batch_size: usize,
    num_channels: usize,
    height: usize,
    width: usize,
    seed: Option<u64>,
    device: &Device,
) -> flame_core::Result<Tensor> {
    if let Some(seed) = seed {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let data: Vec<f32> = (0..batch_size * num_channels * height * width)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();
        Tensor::from_vec(
            data,
            Shape::from_dims(&[batch_size, num_channels, height, width]),
            device.cuda_device().clone(),
        )
    } else {
        Tensor::randn(
            Shape::from_dims(&[batch_size, num_channels, height, width]),
            0.0,
            1.0,
            device.cuda_device().clone(),
        )
    }
}

/// Apply classifier-free guidance
pub fn apply_cfg(
    noise_pred_cond: &Tensor,
    noise_pred_uncond: &Tensor,
    guidance_scale: f64,
) -> flame_core::Result<Tensor> {
    let diff = noise_pred_cond.sub(noise_pred_uncond)?;
    let scaled = diff.mul_scalar(guidance_scale as f32)?;
    noise_pred_uncond.add(&scaled)
}

/// Rescale noise for scheduling
pub fn rescale_noise_cfg(
    noise_cfg: &Tensor,
    noise_pred_cond: &Tensor,
    guidance_rescale: f64,
) -> flame_core::Result<Tensor> {
    if guidance_rescale == 0.0 {
        return Ok(noise_cfg.clone());
    }

    let std_cond =
        noise_pred_cond.std(Some(&[3]), false)?.std(Some(&[2]), false)?.std(Some(&[1]), false)?;
    let std_cfg =
        noise_cfg.std(Some(&[3]), false)?.std(Some(&[2]), false)?.std(Some(&[1]), false)?;

    let factor = std_cond.div(&std_cfg.add_scalar(1e-10)?)?;
    let factor =
        factor.mul_scalar(guidance_rescale as f32)?.add_scalar((1.0 - guidance_rescale) as f32)?;
    let factor = factor.unsqueeze(1)?.unsqueeze(2)?.unsqueeze(3)?;

    Ok(noise_cfg.mul(&factor)?)
}
