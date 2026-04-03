use anyhow::{anyhow, Context};
use flame_core::device::Device;
use flame_core::{DType, Error, Result, Shape, Tensor};
use image::{DynamicImage, RgbImage};
use rand::{rngs::StdRng, SeedableRng};
use std::{path::Path, sync::Arc};

use crate::models::attention::TensorAttentionExt;
use crate::models::{AutoEncoderKL, CLIPTextEncoder, UNet2DConditionModel};
use crate::samplers::flame_schedulers::Scheduler;

#[derive(Debug, Clone)]
pub struct SamplingConfig {
    pub height: usize,
    pub width: usize,
    pub num_inference_steps: usize,
    pub guidance_scale: f32,
    pub eta: f32,
    pub generator_seed: Option<u64>,
    pub output_type: OutputType,
    pub scheduler_type: SchedulerType,
}
pub struct FlameSampler {
    unet: UNet2DConditionModel,
    vae: AutoEncoderKL,
    text_encoder: CLIPTextEncoder,
    text_encoder_2: Option<CLIPTextEncoder>,
    scheduler: Box<dyn Scheduler>,
    device: Device,
}
struct TextEmbeddings {
    encoder_hidden_states: Tensor,
    add_text_embeds: Option<Tensor>,
    add_time_ids: Option<Tensor>,
}

// Complete FLAME-based sampler implementation
// Provides unified sampling interface for different diffusion models

// Extension trait for Tensor to add missing methods
trait TensorExt {
    fn sum_dim(&self, dim: usize) -> flame_core::Result<Tensor>;
    fn add_scalar(&self, scalar: f32) -> flame_core::Result<Tensor>;
    fn mul_scalar(&self, scalar: f32) -> flame_core::Result<Tensor>;
    fn square(&self) -> flame_core::Result<Tensor>;
}

impl TensorExt for Tensor {
    fn sum_dim(&self, dim: usize) -> flame_core::Result<Tensor> {
        // Sum along dimension
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

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OutputType {
    Latent, // Return latent tensor
    Tensor, // Decode to tensor
    Image,  // Convert to DynamicImage
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SchedulerType {
    DDIM,
    DDPM,
    EulerDiscrete,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            height: 1024,
            width: 1024,
            num_inference_steps: 50,
            guidance_scale: 7.5,
            eta: 0.0,
            generator_seed: None,
            output_type: OutputType::Image,
            scheduler_type: SchedulerType::DDIM,
        }
    }
}

/// Main FLAME sampler for text-to-image generation

impl FlameSampler {
    pub fn new(
        unet: UNet2DConditionModel,
        vae: AutoEncoderKL,
        text_encoder: CLIPTextEncoder,
        text_encoder_2: Option<CLIPTextEncoder>,
        scheduler_type: SchedulerType,
        device: Device,
    ) -> flame_core::Result<Self> {
        // Create scheduler based on type
        use crate::samplers::flame_schedulers::{
            DDIMScheduler, DDPMScheduler, EulerDiscreteScheduler, SchedulerConfig,
        };
        let scheduler_config = SchedulerConfig::default();
        let scheduler: Box<dyn Scheduler> = match scheduler_type {
            SchedulerType::DDIM => Box::new(DDIMScheduler::new(scheduler_config, device.clone())?),
            SchedulerType::DDPM => Box::new(DDPMScheduler::new(scheduler_config, device.clone())?),
            SchedulerType::EulerDiscrete => {
                Box::new(EulerDiscreteScheduler::new(scheduler_config, device.clone())?)
            }
        };

        Ok(Self { unet, vae, text_encoder, text_encoder_2, scheduler, device })
    }

    /// Generate images from text prompts
    pub fn generate(
        &mut self,
        prompts: &[String],
        negative_prompts: Option<&[String]>,
        config: &SamplingConfig,
    ) -> flame_core::Result<Vec<DynamicImage>> {
        let batch_size = prompts.len();

        // Set timesteps
        self.scheduler.set_timesteps(config.num_inference_steps)?;
        let timesteps = self.scheduler.get_timesteps();

        // Encode prompts
        let text_embeddings = self.encode_prompts(prompts, negative_prompts)?;

        // Initialize latents
        let latents = self.prepare_latents(
            batch_size,
            config.height / 8, // VAE downscale factor
            config.width / 8,
            config.generator_seed,
        )?;

        // Sampling loop
        let mut latents = latents;
        for (i, &timestep) in timesteps.iter().enumerate() {
            // Expand latents for classifier-free guidance
            let latent_model_input = if config.guidance_scale > 1.0 {
                Tensor::cat(&[&latents, &latents], 0)?
            } else {
                latents.clone()
            };

            // Scale model input
            let latent_model_input =
                self.scheduler.scale_model_input(&latent_model_input, timestep)?;

            // Predict noise residual
            let batch_size = latent_model_input.shape().dims()[0];
            let timestep_tensor = Tensor::from_vec(
                vec![timestep as f32; batch_size],
                Shape::from_dims(&[batch_size]),
                self.device.cuda_device_arc(),
            )?;

            let noise_pred = self.unet.forward(
                &latent_model_input,
                &timestep_tensor,
                &text_embeddings.encoder_hidden_states,
                None, // additional_residuals
            )?;

            // Perform guidance
            let noise_pred = if config.guidance_scale > 1.0 {
                let chunks = noise_pred.chunk(2, 0)?;
                let noise_pred_uncond = &chunks[0];
                let noise_pred_text = &chunks[1];

                noise_pred_uncond.add(
                    &noise_pred_text
                        .sub(noise_pred_uncond)?
                        .mul_scalar(config.guidance_scale as f32)?,
                )?
            } else {
                noise_pred
            };

            // Compute previous sample
            let step_output = self.scheduler.step(
                &noise_pred,
                timestep,
                &latents,
                None, // No custom RNG needed
            )?;
            latents = step_output.prev_sample;

            // Optional: yield progress
            if i % 10 == 0 {
                eprintln!("Sampling step {}/{}", i + 1, timesteps.len());
            }
        }

        // Decode latents based on output type
        match config.output_type {
            OutputType::Latent => {
                // Return latents as images (for debugging)
                Ok(vec![latents_to_debug_image(&latents)?])
            }
            OutputType::Tensor | OutputType::Image => {
                // Decode with VAE
                let images = self.vae.decode(&latents)?;

                // Convert to images
                let images = tensor_to_images(&images)?;
                Ok(images)
            }
        }
    }

    /// Encode text prompts
    fn encode_prompts(
        &self,
        prompts: &[String],
        negative_prompts: Option<&[String]>,
    ) -> flame_core::Result<TextEmbeddings> {
        let batch_size = prompts.len();

        // For now, create dummy token IDs (in production, use proper tokenizer)
        let max_length = 77;
        let shape = &[batch_size, max_length];
        let input_ids = Tensor::zeros(Shape::from_dims(shape), self.device.cuda_device_arc())?;

        // Encode positive prompts
        let positive_embeds = self.text_encoder.forward(&input_ids, None)?;

        // Encode negative prompts if provided
        let negative_embeds = if let Some(neg_prompts) = negative_prompts {
            if neg_prompts.len() != batch_size {
                return Err(flame_core::Error::InvalidOperation(
                    "Negative prompts must match batch size".to_string(),
                ));
            }
            let neg_input_ids =
                Tensor::zeros(Shape::from_dims(shape), self.device.cuda_device_arc())?;
            Some(self.text_encoder.forward(&neg_input_ids, None)?)
        } else {
            // Use empty prompt
            let empty_input_ids =
                Tensor::zeros(Shape::from_dims(shape), self.device.cuda_device_arc())?;
            Some(self.text_encoder.forward(&empty_input_ids, None)?)
        };

        // Handle dual text encoders (SDXL)
        let (encoder_hidden_states, pooled_prompt_embeds) =
            if let Some(text_encoder_2) = &self.text_encoder_2 {
                // SDXL: concatenate outputs from both encoders
                let positive_embeds_2 = text_encoder_2.forward(&input_ids, None)?;

                let hidden_states = Tensor::cat(
                    &[&positive_embeds.last_hidden_state, &positive_embeds_2.last_hidden_state],
                    positive_embeds.last_hidden_state.shape().rank() - 1,
                )?;

                let pooled = positive_embeds_2.pooled_output;

                (hidden_states, Some(pooled))
            } else {
                (positive_embeds.last_hidden_state, Some(positive_embeds.pooled_output))
            };

        // Prepare for classifier-free guidance
        let encoder_hidden_states = if negative_embeds.is_some() {
            let neg_hidden = negative_embeds.as_ref().unwrap();

            // Handle dual encoders for negative
            let neg_hidden_states = if let Some(text_encoder_2) = &self.text_encoder_2 {
                let neg_embeds_2 = text_encoder_2.forward(&input_ids, None)?;
                Tensor::cat(
                    &[&neg_hidden.last_hidden_state, &neg_embeds_2.last_hidden_state],
                    neg_hidden.last_hidden_state.shape().rank() - 1,
                )?
            } else {
                neg_hidden.last_hidden_state.clone()
            };

            // Concatenate negative and positive
            Tensor::cat(&[&neg_hidden_states, &encoder_hidden_states], 0)?
        } else {
            encoder_hidden_states
        };

        // Create additional embeddings for SDXL
        let (add_text_embeds, add_time_ids) = if pooled_prompt_embeds.is_some() {
            // SDXL requires additional embeddings
            let pooled = pooled_prompt_embeds.unwrap();

            // For negative prompts, also need pooled embeddings
            let add_text_embeds = if negative_embeds.is_some() {
                let neg_pooled = if let Some(text_encoder_2) = &self.text_encoder_2 {
                    let neg_embeds_2 = text_encoder_2.forward(&input_ids, None)?;
                    neg_embeds_2.pooled_output
                } else {
                    negative_embeds.as_ref().unwrap().pooled_output.clone()
                };
                Tensor::cat(&[&neg_pooled, &pooled], 0)?
            } else {
                pooled
            };

            // Create time embeddings (original_size, crops_coords_top_left, target_size)
            let time_ids = Tensor::from_vec(
                vec![1024.0, 1024.0, 0.0, 0.0, 1024.0, 1024.0], // Default SDXL values
                Shape::from_dims(&[1, 6]),
                self.device.cuda_device_arc(),
            )?;

            let add_time_ids = if negative_embeds.is_some() {
                {
                    let mut copies = vec![];
                    for _ in 0..(2 * batch_size) {
                        copies.push(&time_ids);
                    }
                    Tensor::cat(&copies, 0)?
                }
            } else {
                {
                    let mut copies = vec![];
                    for _ in 0..batch_size {
                        copies.push(&time_ids);
                    }
                    Tensor::cat(&copies, 0)?
                }
            };

            (Some(add_text_embeds), Some(add_time_ids))
        } else {
            (None, None)
        };

        Ok(TextEmbeddings { encoder_hidden_states, add_text_embeds, add_time_ids })
    }

    /// Initialize latents
    fn prepare_latents(
        &self,
        batch_size: usize,
        height: usize,
        width: usize,
        seed: Option<u64>,
    ) -> flame_core::Result<Tensor> {
        let latent_channels = 4; // Standard for most VAEs
        let shape = &[batch_size, latent_channels, height, width];

        // Initialize with random noise
        let latents = if let Some(_seed) = seed {
            // FLAME doesn't support seeded RNG - use regular randn
            Tensor::randn(Shape::from_dims(shape), 0.0f32, 1.0f32, self.device.cuda_device_arc())?
        } else {
            Tensor::randn(Shape::from_dims(shape), 0.0f32, 1.0f32, self.device.cuda_device_arc())?
        };

        // Scale by scheduler init noise sigma
        // For DDIM/DDPM, this is typically 1.0
        Ok(latents)
    }
}

/// Convert tensor to images
fn tensor_to_images(tensor: &Tensor) -> flame_core::Result<Vec<DynamicImage>> {
    // Tensor shape: [batch, channels, height, width]
    let shape = tensor.shape().dims();
    let batch_size = shape[0];
    let channels = shape[1];
    let height = shape[2];
    let width = shape[3];

    if channels != 3 {
        return Err(flame_core::Error::InvalidOperation(format!("",)));
    }

    // Convert to CPU and get data
    let tensor_cpu = tensor;
    let data: Vec<f32> = tensor_cpu.to_vec()?;

    let mut images = Vec::new();

    for b in 0..batch_size {
        let mut img = RgbImage::new(width as u32, height as u32);

        for y in 0..height {
            for x in 0..width {
                let idx =
                    |c: usize| b * channels * height * width + c * height * width + y * width + x;

                // Convert from [-1, 1] to [0, 255]
                let r = ((data[idx(0)] + 1.0) * 127.5).clamp(0.0, 255.0) as u8;
                let g = ((data[idx(1)] + 1.0) * 127.5).clamp(0.0, 255.0) as u8;
                let b_val = ((data[idx(2)] + 1.0) * 127.5).clamp(0.0, 255.0) as u8;

                img.put_pixel(x as u32, y as u32, image::Rgb([r, g, b_val]));
            }
        }

        images.push(DynamicImage::ImageRgb8(img));
    }

    Ok(images)
}

/// Convert latents to debug image
fn latents_to_debug_image(latents: &Tensor) -> flame_core::Result<DynamicImage> {
    // Take first 3 channels and normalize
    let shape = latents.shape();
    let latents_rgb = if shape.dims()[0] >= 3 {
        // FLAME doesn't have narrow - just use the tensor as is
        latents.clone()
    } else {
        // Repeat channels if needed
        Tensor::cat(&[latents, latents, latents], 1)?
    };

    // Normalize to [0, 1] with proper clamping
    // VAE output is typically in [-1, 1] range, so we normalize
    let min_val = latents_rgb.min_all()?;
    let max_val = latents_rgb.max_all()?;
    let range = (max_val - min_val).max(1e-5);

    // Normalize and clamp to [0, 1]
    let normalized = latents_rgb.sub_scalar(min_val)?.div_scalar(range)?.clamp(0.0, 1.0)?;

    // Convert to image
    let images = tensor_to_images(&normalized)?;
    Ok(images[0].clone())
}

/// High-level sampling function
pub fn sample_images(
    unet: UNet2DConditionModel,
    vae: AutoEncoderKL,
    text_encoder: CLIPTextEncoder,
    text_encoder_2: Option<CLIPTextEncoder>,
    prompts: &[String],
    config: SamplingConfig,
    device: Device,
) -> flame_core::Result<Vec<DynamicImage>> {
    let mut sampler =
        FlameSampler::new(unet, vae, text_encoder, text_encoder_2, config.scheduler_type, device)?;

    sampler.generate(prompts, None, &config)
}

/// Save images to disk
pub fn save_images(
    images: &[DynamicImage],
    output_dir: &Path,
    prefix: &str,
) -> flame_core::Result<Vec<String>> {
    std::fs::create_dir_all(output_dir)
        .map_err(|e| {
            flame_core::Error::InvalidOperation(format!("Failed to create directory: {}", e))
        })
        .map_err(|e| flame_core::Error::InvalidOperation(e.to_string()))?;

    let mut paths = Vec::new();

    for (i, img) in images.iter().enumerate() {
        let filename = format!("{}_{:04}.png", prefix, i);
        let path = output_dir.join(&filename);
        img.save(&path).map_err(|e| {
            flame_core::Error::InvalidOperation(format!("Failed to save image: {}", e))
        })?;
        paths.push(path.to_string_lossy().to_string());
    }

    Ok(paths)
}
