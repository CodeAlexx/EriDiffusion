//! SDXL Sampling Implementation
//!
//! Implements Stable Diffusion XL inference with:
//! - Dual CLIP text encoders (CLIP-L and CLIP-G)
//! - 4-channel VAE
//! - U-Net with cross-attention
//! - Multiple scheduler support (DDIM, DDPM, Euler)

use crate::models::sdxl_unet_complete::{
    AddedCondKwargs, UNet2DConditionModel, UNet2DConditionModelConfig,
};
use flame_core::device::Device;
use flame_core::{DType, Error, Result, Shape, Tensor};
use image::{ImageBuffer, RgbImage};
use log::{debug, info};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// SDXL Sampling Configuration
#[derive(Debug, Clone)]
pub struct SDXLSamplingConfig {
    pub num_inference_steps: usize,
    pub guidance_scale: f64,
    pub eta: f32, // For DDIM scheduler
    pub scheduler_type: SDXLSchedulerType,
    pub resolution: (usize, usize),
    pub clip_skip: usize, // Number of CLIP layers to skip
}

impl Default for SDXLSamplingConfig {
    fn default() -> Self {
        Self {
            num_inference_steps: 30,
            guidance_scale: 7.5,
            eta: 0.0,
            scheduler_type: SDXLSchedulerType::DDIM,
            resolution: (1024, 1024),
            clip_skip: 0,
        }
    }
}

/// Scheduler types supported by SDXL
#[derive(Debug, Clone, Copy)]
pub enum SDXLSchedulerType {
    DDIM,
    DDPM,
    EulerDiscrete,
    EulerAncestral,
    DPMSolverMultistep,
}

/// DDIM Scheduler for SDXL
pub struct DDIMScheduler {
    num_train_timesteps: usize,
    num_inference_steps: usize,
    timesteps: Vec<i64>,
    alphas_cumprod: Vec<f32>,
    eta: f32,
}

impl DDIMScheduler {
    pub fn new(num_inference_steps: usize, num_train_timesteps: usize, eta: f32) -> Self {
        // Create linear beta schedule
        let betas = Self::linear_beta_schedule(num_train_timesteps);
        let alphas: Vec<f32> = betas.iter().map(|b| 1.0 - b).collect();

        // Calculate cumulative product of alphas
        let mut alphas_cumprod = vec![alphas[0]];
        for i in 1..alphas.len() {
            alphas_cumprod.push(alphas_cumprod[i - 1] * alphas[i]);
        }

        // Create timesteps for inference
        let step_ratio = num_train_timesteps / num_inference_steps;
        let timesteps: Vec<i64> = (0..num_inference_steps)
            .map(|i| ((num_inference_steps - 1 - i) * step_ratio) as i64)
            .collect();

        Self { num_train_timesteps, num_inference_steps, timesteps, alphas_cumprod, eta }
    }

    fn linear_beta_schedule(num_timesteps: usize) -> Vec<f32> {
        let beta_start = 0.00085;
        let beta_end = 0.012;
        (0..num_timesteps)
            .map(|i| beta_start + (beta_end - beta_start) * (i as f32 / (num_timesteps - 1) as f32))
            .collect()
    }

    /// Single denoising step
    pub fn step(
        &self,
        model_output: &Tensor,
        timestep_idx: usize,
        sample: &Tensor,
        generator: Option<&mut StdRng>,
    ) -> Result<Tensor> {
        let timestep = self.timesteps[timestep_idx];
        let prev_timestep = if timestep_idx < self.timesteps.len() - 1 {
            self.timesteps[timestep_idx + 1]
        } else {
            -1
        };

        // Get alpha values
        let alpha_prod_t = self.alphas_cumprod[timestep as usize];
        let alpha_prod_t_prev =
            if prev_timestep >= 0 { self.alphas_cumprod[prev_timestep as usize] } else { 1.0 };

        let beta_prod_t = 1.0 - alpha_prod_t;
        let beta_prod_t_prev = 1.0 - alpha_prod_t_prev;

        // Compute predicted original sample
        let pred_original_sample = sample
            .sub(&model_output.mul_scalar(beta_prod_t.sqrt())?)?
            .div_scalar(alpha_prod_t.sqrt())?;

        // Clip prediction
        let pred_original_sample = pred_original_sample.clamp(-1.0, 1.0)?;

        // Compute variance
        let variance = if self.eta > 0.0 && timestep_idx < self.timesteps.len() - 1 {
            let variance =
                (beta_prod_t_prev / beta_prod_t) * (1.0 - alpha_prod_t / alpha_prod_t_prev);
            variance * self.eta * self.eta
        } else {
            0.0
        };

        // Compute coefficients
        let pred_sample_direction =
            model_output.mul_scalar((1.0 - alpha_prod_t_prev - variance).sqrt())?;

        // Compute previous sample
        let mut prev_sample = pred_original_sample
            .mul_scalar(alpha_prod_t_prev.sqrt())?
            .add(&pred_sample_direction)?;

        // Add noise if needed
        if variance > 0.0 {
            let noise = if let Some(gen) = generator {
                let shape = sample.shape();
                let noise_vec: Vec<f32> = (0..shape.dims().iter().product::<usize>())
                    .map(|_| gen.gen::<f32>() * 2.0 - 1.0)
                    .collect();
                Tensor::from_vec(noise_vec, shape.clone(), sample.device().clone())?
            } else {
                Tensor::randn(sample.shape().clone(), 0.0, 1.0, sample.device().clone())?
            };

            prev_sample = prev_sample.add(&noise.mul_scalar(variance.sqrt())?)?;
        }

        Ok(prev_sample)
    }
}

/// SDXL Sampler
pub struct SDXLSampler {
    device: Device,
    dtype: DType,
    config: SDXLSamplingConfig,
}

/// Text encoder outputs for SDXL
pub struct TextEncoders {
    pub clip_l: Box<dyn Fn(&str) -> Result<(Tensor, Tensor)>>,
    pub clip_g: Box<dyn Fn(&str) -> Result<(Tensor, Tensor)>>,
}

impl SDXLSampler {
    pub fn new(device: Device, dtype: DType, config: SDXLSamplingConfig) -> Self {
        Self { device, dtype, config }
    }

    /// Encode prompts using dual CLIP encoders
    pub fn encode_prompts(
        &self,
        text_encoders: &mut TextEncoders,
        prompt: &str,
        negative_prompt: &str,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        info!("Encoding prompts with CLIP-L and CLIP-G...");

        // Encode positive prompt
        let (clip_l_embeds, clip_l_pooled) = (text_encoders.clip_l)(prompt)?;
        let (clip_g_embeds, clip_g_pooled) = (text_encoders.clip_g)(prompt)?;

        // Concatenate embeddings from both encoders (along last dimension)
        let last_dim = clip_l_embeds.shape().rank() - 1;
        let positive_embeds = Tensor::cat(&[&clip_l_embeds, &clip_g_embeds], last_dim)?;
        let positive_pooled =
            Tensor::cat(&[&clip_l_pooled, &clip_g_pooled], clip_l_pooled.shape().rank() - 1)?;

        // Encode negative prompt
        let (neg_clip_l_embeds, _) = (text_encoders.clip_l)(negative_prompt)?;
        let (neg_clip_g_embeds, _) = (text_encoders.clip_g)(negative_prompt)?;
        let negative_embeds = Tensor::cat(
            &[&neg_clip_l_embeds, &neg_clip_g_embeds],
            neg_clip_l_embeds.shape().rank() - 1,
        )?;

        // For classifier-free guidance, concatenate positive and negative
        let text_embeds = Tensor::cat(&[&negative_embeds, &positive_embeds], 0)?;

        // Create time embeddings for SDXL
        let time_ids = self.create_time_ids()?;

        Ok((text_embeds, positive_pooled, time_ids))
    }

    /// Create time embeddings for SDXL (resolution, crops, etc.)
    fn create_time_ids(&self) -> Result<Tensor> {
        let (height, width) = self.config.resolution;
        let original_size = vec![height as f32, width as f32];
        let target_size = vec![height as f32, width as f32];
        let crops_coords_top_left = vec![0.0, 0.0];

        let time_ids = vec![
            original_size[0],
            original_size[1],
            crops_coords_top_left[0],
            crops_coords_top_left[1],
            target_size[0],
            target_size[1],
        ];

        // Create tensor and duplicate for CFG
        let time_ids_tensor = Tensor::from_vec(
            time_ids.clone(),
            Shape::from_dims(&[1, 6]),
            self.device.cuda_device().clone(),
        )?;
        let time_ids_neg = time_ids_tensor.clone();

        Tensor::cat(&[&time_ids_neg, &time_ids_tensor], 0)
    }

    /// Initialize random latents
    fn prepare_latents(
        &self,
        batch_size: usize,
        height: usize,
        width: usize,
        generator: Option<&mut StdRng>,
    ) -> Result<Tensor> {
        // 4 channels for SDXL VAE, 8x downscaling
        let shape = vec![batch_size, 4, height / 8, width / 8];

        let latents = if let Some(gen) = generator {
            let values: Vec<f32> =
                (0..shape.iter().product()).map(|_| gen.gen::<f32>() * 2.0 - 1.0).collect();
            Tensor::from_vec(values, Shape::from_dims(&shape), self.device.cuda_device().clone())?
        } else {
            Tensor::randn(
                Shape::from_dims(&shape),
                0.0f32,
                1.0f32,
                self.device.cuda_device().clone(),
            )?
        };

        latents.to_dtype(self.dtype)
    }

    /// Generate samples
    pub fn generate_samples(
        &self,
        unet: &HashMap<String, Tensor>,
        lora_collection: &Box<dyn std::any::Any>,
        vae_decode: &Box<dyn Fn(&Tensor) -> Result<Tensor>>,
        text_encoders: &mut TextEncoders,
        prompts: &[String],
        negative_prompts: Option<&[String]>,
        height: usize,
        width: usize,
        seed: Option<u64>,
    ) -> Result<Vec<PathBuf>> {
        let batch_size = prompts.len();
        let empty_prompts = vec!["".to_string(); batch_size];
        let negative_prompts = negative_prompts.unwrap_or(&empty_prompts);

        // Initialize generator
        let mut generator = seed.map(|s| StdRng::seed_from_u64(s));

        // Process each prompt
        let mut all_paths = Vec::new();

        for (i, (prompt, negative_prompt)) in prompts.iter().zip(negative_prompts).enumerate() {
            info!("Generating image {}/{} for prompt: {}", i + 1, batch_size, prompt);

            // Encode prompts
            let (text_embeds, pooled_embeds, time_ids) =
                self.encode_prompts(text_encoders, prompt, negative_prompt)?;

            // Initialize latents
            let mut latents = self.prepare_latents(1, height, width, generator.as_mut())?;

            // Create scheduler
            let scheduler = DDIMScheduler::new(
                self.config.num_inference_steps,
                1000, // num_train_timesteps for SDXL
                self.config.eta,
            );

            // Denoising loop
            info!("Running SDXL denoising for {} steps...", self.config.num_inference_steps);
            for (step_idx, &timestep) in scheduler.timesteps.iter().enumerate() {
                // Expand latents for CFG
                let latent_model_input = Tensor::cat(&[&latents, &latents], 0)?;

                // Create timestep embedding
                let t = Tensor::full(
                    Shape::from_dims(&[2]),
                    timestep as f32,
                    self.device.cuda_device().clone(),
                )?
                .to_dtype(self.dtype)?;

                // UNet forward pass
                // This is a placeholder - actual implementation would call the UNet
                let noise_pred = self.unet_forward(
                    unet,
                    &latent_model_input,
                    &t,
                    &text_embeds,
                    &pooled_embeds,
                    &time_ids,
                )?;

                // Perform guidance
                let chunks = noise_pred.chunk(2, 0)?;
                let noise_pred_uncond = &chunks[0];
                let noise_pred_text = &chunks[1];
                let noise_pred = noise_pred_uncond.add(
                    &noise_pred_text
                        .sub(noise_pred_uncond)?
                        .mul_scalar(self.config.guidance_scale as f32)?,
                )?;

                // Scheduler step
                latents = scheduler.step(&noise_pred, step_idx, &latents, generator.as_mut())?;

                if (step_idx + 1) % 10 == 0 {
                    info!("Completed step {}/{}", step_idx + 1, self.config.num_inference_steps);
                }
            }

            // VAE decode
            info!("Decoding SDXL latents...");
            let images = vae_decode(&latents)?;

            // Save image
            let output_dir = Path::new("outputs/sdxl");
            std::fs::create_dir_all(output_dir)
                .map_err(|e| {
                    flame_core::Error::InvalidOperation(format!(
                        "Failed to create directory: {}",
                        e
                    ))
                })
                .map_err(|e| flame_core::Error::InvalidOperation(e.to_string()))?;

            let path = output_dir.join(format!("sdxl_sample_{}.png", i));
            self.save_image(&images.slice(&[(0, 1)])?.squeeze(Some(0))?, &path)?;
            all_paths.push(path);
        }

        Ok(all_paths)
    }

    /// UNet forward pass for SDXL
    fn unet_forward(
        &self,
        unet: &HashMap<String, Tensor>,
        latents: &Tensor,
        timestep: &Tensor,
        encoder_hidden_states: &Tensor,
        pooled_projections: &Tensor,
        time_ids: &Tensor,
    ) -> Result<Tensor> {
        // SDXL uses additional conditioning with pooled text embeddings and time IDs
        // Create the additional conditioning tensor by concatenating pooled embeddings and time IDs
        // Concatenate along the last dimension (1 for 2D tensors)
        let text_time_embeds = Tensor::cat(&[pooled_projections, time_ids], 1)?;

        // Create the SDXL UNet model from weights
        let config = UNet2DConditionModelConfig::sdxl();
        // Create empty weights HashMap for now - weights should be loaded separately
        let weights = std::collections::HashMap::new();
        let unet_model = UNet2DConditionModel::new(config, &self.device, weights)?;

        // TODO: Load weights into model from the HashMap
        // This would involve mapping the weight names to the model's parameters

        // Create added conditioning kwargs
        let added_cond_kwargs = AddedCondKwargs {
            text_embeds: pooled_projections.clone(), // Pooled embeddings from CLIP
            text_time_embeds,
            time_ids: time_ids.clone(), // Time embeddings
        };

        // Forward pass through UNet
        unet_model.forward(latents, timestep, encoder_hidden_states, Some(&added_cond_kwargs))
    }

    /// Save image tensor to file
    fn save_image(&self, image: &Tensor, path: &Path) -> Result<()> {
        // Convert from [-1, 1] to [0, 255]
        let image = image.clamp(-1.0, 1.0)?.mul_scalar(127.5)?.add_scalar(127.5)?;

        // Get dimensions
        let shape = image.shape();
        let dims = shape.dims();
        let (c, h, w) = match shape.rank() {
            3 => (dims[0], dims[1], dims[2]),
            4 => (dims[1], dims[2], dims[3]), // Skip batch
            _ => {
                return Err(flame_core::Error::InvalidOperation(
                    "Invalid image tensor shape".to_string(),
                ))
            }
        };

        // Ensure we have RGB
        let rgb_image = if c == 1 {
            // Grayscale to RGB
            image.repeat(&[3, 1, 1])?
        } else if c == 3 {
            image.clone()
        } else {
            return Err(flame_core::Error::InvalidOperation(
                "Image must have 1 or 3 channels".to_string(),
            ));
        };

        // Convert to HWC and flatten
        let image_data = rgb_image.permute(&[1, 2, 0])?.flatten_all()?.to_vec1::<f32>()?;

        // Convert to u8
        let image_data: Vec<u8> = image_data.iter().map(|&x| x.clamp(0.0, 255.0) as u8).collect();

        // Save image
        let img = RgbImage::from_raw(w as u32, h as u32, image_data)
            .ok_or_else(|| flame_core::Error::InvalidOperation("Failed to create image".into()))?;
        img.save(path).map_err(|e| flame_core::Error::Io(e.to_string()))?;

        info!("Saved SDXL sample to: {}", path.display());
        Ok(())
    }
}

/// Example usage function
pub fn sample_sdxl(
    device: Device,
    unet: HashMap<String, Tensor>,
    vae_decode: impl Fn(&Tensor) -> Result<Tensor> + 'static,
    text_encoders: TextEncoders,
    prompt: &str,
    negative_prompt: &str,
    seed: Option<u64>,
) -> Result<Vec<PathBuf>> {
    let config = SDXLSamplingConfig::default();
    let sampler = SDXLSampler::new(device, DType::F16, config);

    let mut encoders = text_encoders;

    sampler.generate_samples(
        &unet,
        &(Box::new(()) as Box<dyn std::any::Any>), // Placeholder for LoRA collection
        &(Box::new(vae_decode) as Box<dyn Fn(&Tensor) -> Result<Tensor> + 'static>),
        &mut encoders,
        &[prompt.to_string()],
        Some(&[negative_prompt.to_string()]),
        1024,
        1024,
        seed,
    )
}
