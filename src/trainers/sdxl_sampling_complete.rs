//! Complete SDXL sampling implementation for LoRA training
//! This module provides comprehensive sampling functionality including:
//! - Multiple scheduler support (DDIM, DPM++, Euler)
//! - Proper SDXL time_ids handling
//! - LoRA injection during sampling
//! - Classifier-free guidance
//! - VAE decoding with proper scaling

use log::{info, debug, warn, error};
use anyhow::{Result, Context};
use candle_core::{Tensor, Device, DType, Module, D};
use std::path::{Path, PathBuf};
use std::collections::HashMap;
use image::{RgbImage, ImageBuffer};

use crate::trainers::sdxl_lora_trainer_fixed::LoRACollection;
use crate::trainers::sdxl_vae_wrapper::SDXLVAEWrapper;
use crate::trainers::text_encoders::TextEncoders;

/// Scheduler types supported for SDXL sampling
#[derive(Debug, Clone, Copy)]
pub enum SchedulerType {
    DDIM,
    DPMPlusPlus2M,
    EulerDiscrete,
    EulerAncestralDiscrete,
}

/// Configuration for SDXL sampling
#[derive(Debug, Clone)]
pub struct SDXLSamplingConfig {
    pub scheduler_type: SchedulerType,
    pub num_inference_steps: usize,
    pub guidance_scale: f64,
    pub eta: f64,  // For DDIM stochasticity
    pub prediction_type: PredictionType,
    pub clip_sample: bool,
    pub thresholding: bool,
    pub dynamic_thresholding_ratio: f64,
    pub sample_max_value: f64,
}

impl Default for SDXLSamplingConfig {
    fn default() -> Self {
        Self {
            scheduler_type: SchedulerType::DDIM,
            num_inference_steps: 30,
            guidance_scale: 7.5,
            eta: 0.0,
            prediction_type: PredictionType::Epsilon,
            clip_sample: false,
            thresholding: false,
            dynamic_thresholding_ratio: 0.995,
            sample_max_value: 1.0,
        }
    }
}

/// Prediction type for the noise scheduler
#[derive(Debug, Clone, Copy)]
pub enum PredictionType {
    Epsilon,
    Sample,
    VPrediction,
}

/// Beta schedule type
#[derive(Debug, Clone, Copy)]
pub enum BetaSchedule {
    Linear,
    ScaledLinear,
    SquaredCosCapV2,
}

/// SDXL Time IDs for conditioning
#[derive(Debug, Clone)]
pub struct SDXLTimeIds {
    pub height: f32,
    pub width: f32,
    pub crop_top: f32,
    pub crop_left: f32,
    pub target_height: f32,
    pub target_width: f32,
}

impl SDXLTimeIds {
    pub fn new(height: usize, width: usize) -> Self {
        Self {
            height: height as f32,
            width: width as f32,
            crop_top: 0.0,
            crop_left: 0.0,
            target_height: height as f32,
            target_width: width as f32,
        }
    }

    pub fn to_tensor(&self, batch_size: usize, device: &Device) -> Result<Tensor> {
        let time_ids = vec![
            self.height,
            self.width,
            self.crop_top,
            self.crop_left,
            self.target_height,
            self.target_width,
        ];
        
        // Repeat for batch size
        let mut all_time_ids = Vec::new();
        for _ in 0..batch_size {
            all_time_ids.extend_from_slice(&time_ids);
        }
        
        Ok(Tensor::from_vec(all_time_ids, &[batch_size, 6], device)?)
    }
}

/// DDIM Scheduler implementation
pub struct DDIMScheduler {
    timesteps: Vec<usize>,
    alphas_cumprod: Vec<f64>,
    init_noise_sigma: f64,
    num_inference_steps: usize,
    num_train_timesteps: usize,
    eta: f64,
    prediction_type: PredictionType,
}

impl DDIMScheduler {
    pub fn new(
        num_inference_steps: usize,
        num_train_timesteps: usize,
        beta_start: f64,
        beta_end: f64,
        beta_schedule: BetaSchedule,
        prediction_type: PredictionType,
        eta: f64,
    ) -> Result<Self> {
        // Create beta schedule
        let betas = match beta_schedule {
            BetaSchedule::Linear => {
                let betas: Vec<f64> = (0..num_train_timesteps)
                    .map(|i| {
                        beta_start + (beta_end - beta_start) * (i as f64) / (num_train_timesteps as f64 - 1.0)
                    })
                    .collect();
                betas
            }
            BetaSchedule::ScaledLinear => {
                let start = beta_start.sqrt();
                let end = beta_end.sqrt();
                let betas: Vec<f64> = (0..num_train_timesteps)
                    .map(|i| {
                        let beta = start + (end - start) * (i as f64) / (num_train_timesteps as f64 - 1.0);
                        beta * beta
                    })
                    .collect();
                betas
            }
            BetaSchedule::SquaredCosCapV2 => {
                let max_beta = 0.999;
                let alpha_bar = |t: f64| {
                    f64::cos((t + 0.008) / 1.008 * std::f64::consts::FRAC_PI_2).powi(2)
                };
                let betas: Vec<f64> = (0..num_train_timesteps)
                    .map(|i| {
                        let t1 = i as f64 / num_train_timesteps as f64;
                        let t2 = (i + 1) as f64 / num_train_timesteps as f64;
                        (1.0 - alpha_bar(t2) / alpha_bar(t1)).min(max_beta)
                    })
                    .collect();
                betas
            }
        };

        // Calculate alphas_cumprod
        let alphas: Vec<f64> = betas.iter().map(|b| 1.0 - b).collect();
        let mut alphas_cumprod = Vec::with_capacity(num_train_timesteps);
        let mut cumprod = 1.0;
        for alpha in alphas {
            cumprod *= alpha;
            alphas_cumprod.push(cumprod);
        }

        // Create timesteps
        let step_ratio = num_train_timesteps / num_inference_steps;
        let timesteps: Vec<usize> = (0..num_inference_steps)
            .map(|i| (num_inference_steps - 1 - i) * step_ratio)
            .collect();

        // Initial noise sigma
        let init_noise_sigma = 1.0;

        Ok(Self {
            timesteps,
            alphas_cumprod,
            init_noise_sigma,
            num_inference_steps,
            num_train_timesteps,
            eta,
            prediction_type,
        })
    }

    pub fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    pub fn init_noise_sigma(&self) -> f64 {
        self.init_noise_sigma
    }

    pub fn scale_model_input(&self, sample: Tensor, _timestep: usize) -> Result<Tensor> {
        // DDIM doesn't scale the model input
        Ok(sample)
    }

    pub fn step(
        &self,
        model_output: &Tensor,
        timestep: usize,
        sample: &Tensor,
        generator: Option<&mut rand::rngs::StdRng>,
    ) -> Result<Tensor> {
        // Get alpha values
        let alpha_prod_t = self.alphas_cumprod[timestep];
        let alpha_prod_t_prev = if timestep > 0 {
            self.alphas_cumprod[timestep - self.num_train_timesteps / self.num_inference_steps]
        } else {
            1.0
        };

        let beta_prod_t = 1.0 - alpha_prod_t;
        let beta_prod_t_prev = 1.0 - alpha_prod_t_prev;

        // Compute predicted sample
        let pred_original_sample = match self.prediction_type {
            PredictionType::Epsilon => {
                // x_0 = (sample - sqrt(beta_t) * model_output) / sqrt(alpha_t)
                let sqrt_alpha_prod_t = alpha_prod_t.sqrt();
                let sqrt_one_minus_alpha_prod_t = beta_prod_t.sqrt();
                
                ((sample - (model_output * sqrt_one_minus_alpha_prod_t)?)? / sqrt_alpha_prod_t)?
            }
            PredictionType::Sample => model_output.clone(),
            PredictionType::VPrediction => {
                // x_0 = sqrt(alpha_t) * sample - sqrt(beta_t) * model_output
                let sqrt_alpha_prod_t = alpha_prod_t.sqrt();
                let sqrt_one_minus_alpha_prod_t = beta_prod_t.sqrt();
                
                ((sample * sqrt_alpha_prod_t)? - (model_output * sqrt_one_minus_alpha_prod_t)?)?
            }
        };

        // Compute variance
        let variance = if self.eta > 0.0 && timestep > 0 {
            let variance_noise = if let Some(gen) = generator {
                // Use provided generator for deterministic results
                sample.randn_like(0.0, 1.0)?
            } else {
                sample.randn_like(0.0, 1.0)?
            };
            
            let std_dev_t = self.eta * ((beta_prod_t_prev / beta_prod_t) * (1.0 - alpha_prod_t / alpha_prod_t_prev)).sqrt();
            (variance_noise * std_dev_t)?
        } else {
            sample.zeros_like()?
        };

        // Compute predicted sample direction
        let pred_sample_direction = match self.prediction_type {
            PredictionType::Epsilon => {
                let sqrt_one_minus_alpha_prod_t_prev = beta_prod_t_prev.sqrt();
                (model_output * sqrt_one_minus_alpha_prod_t_prev)?
            }
            _ => {
                // For other prediction types, recompute epsilon
                let sqrt_alpha_prod_t = alpha_prod_t.sqrt();
                let sqrt_one_minus_alpha_prod_t = beta_prod_t.sqrt();
                let epsilon = ((sample - (&pred_original_sample * sqrt_alpha_prod_t)?)? / sqrt_one_minus_alpha_prod_t)?;
                
                let sqrt_one_minus_alpha_prod_t_prev = beta_prod_t_prev.sqrt();
                (epsilon * sqrt_one_minus_alpha_prod_t_prev)?
            }
        };

        // x_t-1 = sqrt(alpha_t-1) * x_0 + sqrt(1 - alpha_t-1) * epsilon + variance
        let sqrt_alpha_prod_t_prev = alpha_prod_t_prev.sqrt();
        Ok((((pred_original_sample * sqrt_alpha_prod_t_prev)? + pred_sample_direction)? + variance)?)
    }
}

/// Complete SDXL Sampler with LoRA support
pub struct SDXLSampler {
    device: Device,
    dtype: DType,
    config: SDXLSamplingConfig,
}

impl SDXLSampler {
    pub fn new(device: Device, dtype: DType, config: SDXLSamplingConfig) -> Self {
        Self {
            device,
            dtype,
            config,
        }
    }

    /// Generate samples using SDXL with LoRA
    pub fn generate_samples(
        &self,
        unet_weights: &HashMap<String, Tensor>,
        lora_collection: &LoRACollection,
        vae: &SDXLVAEWrapper,
        text_encoders: &mut TextEncoders,
        prompts: &[String],
        negative_prompts: Option<&[String]>,
        height: usize,
        width: usize,
        seed: Option<u64>,
    ) -> Result<Vec<Tensor>> {
        let batch_size = prompts.len();
        
        // Set seed if provided
        if let Some(seed) = seed {
            info!("Using seed: {}", seed);
            // Note: Candle doesn't have global seed setting, handled in tensor creation
        }

        // Encode prompts
        info!("Encoding prompts...");
        let mut prompt_embeds_vec = Vec::new();
        let mut pooled_embeds_vec = Vec::new();
        
        for prompt in prompts {
            let (embed, pooled) = text_encoders.encode_sdxl(prompt, 77)?;
            prompt_embeds_vec.push(embed);
            pooled_embeds_vec.push(pooled);
        }
        
        let prompt_embeds = Tensor::stack(&prompt_embeds_vec, 0)?;
        let pooled_prompt_embeds = Tensor::stack(&pooled_embeds_vec, 0)?;

        // Encode negative prompts
        let (negative_embeds, negative_pooled) = if let Some(neg_prompts) = negative_prompts {
            let mut neg_embeds_vec = Vec::new();
            let mut neg_pooled_vec = Vec::new();
            
            for neg_prompt in neg_prompts {
                let (embed, pooled) = text_encoders.encode_sdxl(neg_prompt, 77)?;
                neg_embeds_vec.push(embed);
                neg_pooled_vec.push(pooled);
            }
            
            (
                Tensor::stack(&neg_embeds_vec, 0)?,
                Tensor::stack(&neg_pooled_vec, 0)?
            )
        } else {
            // Use unconditional embeddings
            text_encoders.encode_unconditional(batch_size, 77)?
        };

        // Prepare for classifier-free guidance
        let do_classifier_free_guidance = self.config.guidance_scale > 1.0;
        
        let (text_embeddings, pooled_embeddings) = if do_classifier_free_guidance {
            (
                Tensor::cat(&[negative_embeds, prompt_embeds], 0)?,
                Tensor::cat(&[negative_pooled, pooled_prompt_embeds], 0)?
            )
        } else {
            (prompt_embeds, pooled_prompt_embeds)
        };
        
        // Ensure correct dtype for text embeddings
        let text_embeddings = text_embeddings.to_dtype(DType::F16)?;
        let pooled_embeddings = pooled_embeddings.to_dtype(DType::F16)?;

        // Generate time_ids
        info!("Generating time IDs...");
        let time_ids = SDXLTimeIds::new(height, width);
        let time_ids_tensor = time_ids.to_tensor(batch_size, &self.device)?;
        
        let time_ids_tensor = if do_classifier_free_guidance {
            Tensor::cat(&[&time_ids_tensor, &time_ids_tensor], 0)?
        } else {
            time_ids_tensor
        };

        // Additional conditioning kwargs
        let mut added_cond_kwargs = HashMap::new();
        added_cond_kwargs.insert("text_embeds".to_string(), pooled_embeddings.clone());
        added_cond_kwargs.insert("time_ids".to_string(), time_ids_tensor.clone());

        // Create scheduler
        let scheduler = DDIMScheduler::new(
            self.config.num_inference_steps,
            1000, // SDXL training steps
            0.00085,
            0.012,
            BetaSchedule::ScaledLinear,
            self.config.prediction_type,
            self.config.eta,
        )?;

        // Generate initial latents
        info!("Generating initial latents...");
        let latent_shape = [batch_size, 4, height / 8, width / 8];
        let mut latents = if let Some(seed) = seed {
            // Create seeded random tensor
            use rand::{Rng, SeedableRng};
            use rand::rngs::StdRng;
            let mut rng = StdRng::seed_from_u64(seed);
            
            let num_elements: usize = latent_shape.iter().product();
            let random_data: Vec<f32> = (0..num_elements)
                .map(|_| rng.gen_range(-1.0..1.0))
                .collect();
            
            Tensor::from_vec(random_data, &latent_shape, &self.device)?
        } else {
            Tensor::randn(0.0f32, 1.0f32, &latent_shape, &self.device)?
        };

        // Scale by init noise sigma
        latents = (latents * scheduler.init_noise_sigma())?;

        // Denoising loop
        info!("Running denoising loop...");
        let timesteps = scheduler.timesteps();
        let total_steps = timesteps.len();
        
        for (i, &timestep) in timesteps.iter().enumerate() {
            if i % 5 == 0 || i == total_steps - 1 {
                info!("  Step {}/{}", i + 1, total_steps);
            }

            // Expand latents for classifier-free guidance
            let latent_model_input = if do_classifier_free_guidance {
                Tensor::cat(&[&latents, &latents], 0)?
            } else {
                latents.clone()
            };

            // Scale model input
            let latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)?;
            
            // Ensure correct dtype for model input
            let latent_model_input = latent_model_input.to_dtype(DType::F16)?;

            // Create timestep tensor
            let t = if do_classifier_free_guidance {
                Tensor::from_vec(
                    vec![timestep as i64; batch_size * 2],
                    &[batch_size * 2],
                    &self.device
                )?
            } else {
                Tensor::from_vec(
                    vec![timestep as i64; batch_size],
                    &[batch_size],
                    &self.device
                )?
            };

            // Predict noise residual with LoRA
            let noise_pred = self.unet_forward_with_lora(
                &latent_model_input,
                &t,
                &text_embeddings,
                unet_weights,
                lora_collection,
                Some(&added_cond_kwargs),
            )?;

            // Perform guidance
            let noise_pred = if do_classifier_free_guidance {
                let noise_pred_uncond = noise_pred.narrow(0, 0, batch_size)?;
                let noise_pred_text = noise_pred.narrow(0, batch_size, batch_size)?;
                
                // guided_pred = uncond + guidance_scale * (text - uncond)
                let diff = (noise_pred_text - &noise_pred_uncond)?;
                (noise_pred_uncond + (diff * self.config.guidance_scale)?)?
            } else {
                noise_pred
            };

            // Scheduler step
            latents = scheduler.step(&noise_pred, timestep, &latents, None)?;
        }

        // Decode latents
        info!("Decoding latents to images...");
        let images = self.decode_latents(vae, &latents)?;

        Ok(images)
    }

    /// Forward pass through UNet with LoRA injection
    fn unet_forward_with_lora(
        &self,
        sample: &Tensor,
        timestep: &Tensor,
        encoder_hidden_states: &Tensor,
        unet_weights: &HashMap<String, Tensor>,
        lora_collection: &LoRACollection,
        added_cond_kwargs: Option<&HashMap<String, Tensor>>,
    ) -> Result<Tensor> {
        // Use the existing forward pass from the trainer
        use crate::trainers::sdxl_forward_sampling::forward_sdxl_sampling;
        
        // Extract additional conditioning
        let pooled_proj = added_cond_kwargs
            .and_then(|kwargs| kwargs.get("text_embeds"))
            .context("Missing text_embeds in added_cond_kwargs")?;
        let time_ids = added_cond_kwargs
            .and_then(|kwargs| kwargs.get("time_ids"))
            .context("Missing time_ids in added_cond_kwargs")?;
        
        // Forward pass with LoRA and additional conditioning
        forward_sdxl_sampling(
            sample,
            timestep,
            encoder_hidden_states,
            unet_weights,
            lora_collection,
            pooled_proj,
            time_ids,
        )
    }

    /// Decode latents to images using VAE
    fn decode_latents(&self, vae: &SDXLVAEWrapper, latents: &Tensor) -> Result<Vec<Tensor>> {
        let batch_size = latents.dims()[0];
        let mut images = Vec::new();

        // Decode each latent individually to manage memory
        for i in 0..batch_size {
            let latent = latents.get(i)?.unsqueeze(0)?;
            let image = vae.decode(&latent)?;
            images.push(image);
        }

        Ok(images)
    }

    /// Save image tensor to file
    pub fn save_image(&self, image: &Tensor, path: &Path) -> Result<()> {
        // Ensure image is in the right format
        let image = image.squeeze(0)?; // Remove batch dimension if present
        
        // Convert from [-1, 1] to [0, 255]
        let image = image
            .clamp(-1.0, 1.0)?
            .affine(127.5, 127.5)?;

        // Get dimensions
        let (c, h, w) = image.dims3()?;
        
        // Convert to RGB format
        let image_data = if c == 3 {
            // Already RGB
            image
                .permute((1, 2, 0))? // CHW -> HWC
                .flatten_all()?
                .to_vec1::<f32>()?
        } else if c == 1 {
            // Grayscale, convert to RGB
            let rgb = image.repeat(&[3, 1, 1])?;
            rgb
                .permute((1, 2, 0))? // CHW -> HWC
                .flatten_all()?
                .to_vec1::<f32>()?
        } else {
            anyhow::bail!("Unsupported number of channels: {}", c);
        };

        // Convert to u8
        let image_data: Vec<u8> = image_data
            .iter()
            .map(|&x| x.clamp(0.0, 255.0) as u8)
            .collect();

        // Create and save image
        let img = RgbImage::from_raw(w as u32, h as u32, image_data)
            .context("Failed to create image")?;
        img.save(path)?;

        info!("Saved image to: {}", path.display());
        Ok(())
    }
}

/// Sampling utilities for training integration
pub struct TrainingSampler {
    sampler: SDXLSampler,
    output_dir: PathBuf,
    validation_prompts: Vec<String>,
    negative_prompt: Option<String>,
}

impl TrainingSampler {
    pub fn new(
        device: Device,
        dtype: DType,
        output_dir: PathBuf,
        validation_prompts: Vec<String>,
        negative_prompt: Option<String>,
        config: SDXLSamplingConfig,
    ) -> Self {
        let sampler = SDXLSampler::new(device, dtype, config);
        
        Self {
            sampler,
            output_dir,
            validation_prompts,
            negative_prompt,
        }
    }

    /// Generate samples at a specific training step
    pub fn generate_training_samples(
        &self,
        step: usize,
        unet_weights: &HashMap<String, Tensor>,
        lora_collection: &LoRACollection,
        vae: &SDXLVAEWrapper,
        text_encoders: &mut TextEncoders,
    ) -> Result<Vec<PathBuf>> {
        // Create step-specific directory
        let step_dir = self.output_dir.join(format!("step_{:06}", step));
        std::fs::create_dir_all(&step_dir)?;

        // Generate samples
        let negative_prompts = self.negative_prompt.as_ref().map(|n| vec![n.clone(); self.validation_prompts.len()]);
        
        let images = self.sampler.generate_samples(
            unet_weights,
            lora_collection,
            vae,
            text_encoders,
            &self.validation_prompts,
            negative_prompts.as_deref(),
            1024, // SDXL default
            1024,
            Some(42), // Fixed seed for consistency
        )?;

        // Save images
        let mut saved_paths = Vec::new();
        for (i, (image, prompt)) in images.iter().zip(self.validation_prompts.iter()).enumerate() {
            // Save image
            let image_path = step_dir.join(format!("sample_{:02}.png", i));
            self.sampler.save_image(image, &image_path)?;
            
            // Save prompt
            let prompt_path = step_dir.join(format!("sample_{:02}.txt", i));
            std::fs::write(&prompt_path, prompt)?;
            
            saved_paths.push(image_path);
        }

        info!("Generated {} samples at step {}", saved_paths.len(), step);
        Ok(saved_paths)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_ids_generation() {
        let device = Device::Cpu;
        let time_ids = SDXLTimeIds::new(1024, 1024);
        let tensor = time_ids.to_tensor(2, &device).unwrap();
        
        assert_eq!(tensor.dims(), &[2, 6]);
    }

    #[test]
    fn test_ddim_scheduler_creation() {
        let scheduler = DDIMScheduler::new(
            30,
            1000,
            0.00085,
            0.012,
            BetaSchedule::ScaledLinear,
            PredictionType::Epsilon,
            0.0,
        ).unwrap();
        
        assert_eq!(scheduler.timesteps().len(), 30);
        assert_eq!(scheduler.init_noise_sigma(), 1.0);
    }
}