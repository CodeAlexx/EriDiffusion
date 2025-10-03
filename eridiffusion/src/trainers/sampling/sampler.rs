use anyhow::Context;
use flame_core::device::Device;
use flame_core::Result;
use flame_core::{DType, Shape, Tensor};
use log::{debug, info};
use rand::{Rng, SeedableRng};
use std::collections::HashMap;

use super::scheduler::{DDIMScheduler, DDPMScheduler, Scheduler, SchedulerType, SchedulerWrapper};
use crate::trainers::lora::LoRACollection;

#[derive(Clone)]
pub struct SamplerConfig {
    pub num_inference_steps: usize,
    pub guidance_scale: f32,
    pub eta: f32,
    pub scheduler_type: SchedulerType,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            num_inference_steps: 25,
            guidance_scale: 7.5,
            eta: 0.0,
            scheduler_type: SchedulerType::DDPM,
        }
    }
}

pub struct SDXLSampler {
    unet_weights: HashMap<String, Tensor>,
    lora_collection: Option<LoRACollection>,
    config: SamplerConfig,
    device: Device,
    dtype: DType,
}

impl SDXLSampler {
    pub fn new(
        unet_weights: HashMap<String, Tensor>,
        lora_collection: Option<LoRACollection>,
        config: SamplerConfig,
        device: Device,
        dtype: DType,
    ) -> flame_core::Result<Self> {
        Ok(Self { unet_weights, lora_collection, config, device, dtype })
    }

    /// Sample from the model
    pub fn sample(
        &mut self,
        text_embeds: &Tensor,
        pooled_embeds: &Tensor,
        height: usize,
        width: usize,
        seed: Option<u64>,
    ) -> flame_core::Result<Tensor> {
        let batch_size = text_embeds.shape().dims()[0];
        let latent_height = height / 8;
        let latent_width = width / 8;

        // Initialize random latents
        let mut rng = if let Some(s) = seed {
            rand::rngs::StdRng::seed_from_u64(s)
        } else {
            rand::rngs::StdRng::from_entropy()
        };

        let latents = Tensor::randn(
            Shape::from_dims(&[batch_size, 4, latent_height, latent_width]),
            0.0,
            1.0,
            self.device.cuda_device().clone(),
        )?;

        // Create scheduler
        let mut scheduler = match self.config.scheduler_type {
            SchedulerType::DDPM => {
                SchedulerWrapper::DDPM(DDPMScheduler::new(self.config.num_inference_steps))
            }
            SchedulerType::DDIM => SchedulerWrapper::DDIM(DDIMScheduler::new(
                self.config.num_inference_steps,
                self.config.eta,
            )),
            _ => {
                return Err(flame_core::Error::InvalidOperation(
                    "Unsupported scheduler type".to_string(),
                ))
            }
        };

        // Set initial noise
        let mut latents = scheduler.add_noise(&latents, &latents, 0)?;

        // Create time embeddings for SDXL
        let time_ids = self.create_time_ids(height, width)?;

        // Denoising loop
        for (i, &t) in scheduler.timesteps().iter().enumerate() {
            debug!("Denoising step {}/{}", i + 1, self.config.num_inference_steps);

            // Expand latents for classifier-free guidance
            let latent_model_input = if self.config.guidance_scale > 1.0 {
                Tensor::cat(&[&latents, &latents], 0)?
            } else {
                latents.clone()
            };

            // Predict noise
            let noise_pred =
                self.unet_forward(&latent_model_input, t, text_embeds, pooled_embeds, &time_ids)?;

            // Perform guidance
            let noise_pred = if self.config.guidance_scale > 1.0 {
                // Split tensor into unconditional and conditional predictions
                let shape = noise_pred.shape().dims();
                let uncond_shape = vec![batch_size, shape[1], shape[2], shape[3]];
                let cond_shape = uncond_shape.clone();

                // Use narrow to extract the slices
                let noise_pred_uncond = noise_pred.narrow(0, 0, batch_size)?;
                let noise_pred_cond = noise_pred.narrow(0, batch_size, batch_size)?;

                noise_pred_uncond.add(
                    &noise_pred_cond
                        .sub(&noise_pred_uncond)?
                        .mul_scalar(self.config.guidance_scale as f64 as f32)?,
                )?
            } else {
                noise_pred
            };

            // Compute previous sample
            latents = scheduler.step(&noise_pred, t, &latents)?; // TODO: Use gradient_map instead of individual tensor
        }

        Ok(latents)
    }

    /// Forward pass through UNet with LoRA
    fn unet_forward(
        &self,
        latents: &Tensor,
        timestep: usize,
        text_embeds: &Tensor,
        pooled_embeds: &Tensor,
        time_ids: &Tensor,
    ) -> flame_core::Result<Tensor> {
        // This is a simplified forward pass
        // In reality, this would call the full SDXL UNet forward
        use crate::trainers::sdxl_forward_sampling::forward_sdxl_sampling;

        // Convert timestep to tensor
        let device = latents.device();
        let timestep_tensor =
            Tensor::from_slice(&[timestep as f32], Shape::from_dims(&[1]), device.clone())?;

        // Handle optional LoRA collection
        if let Some(lora) = &self.lora_collection {
            forward_sdxl_sampling(
                latents,
                &timestep_tensor,
                text_embeds,
                &self.unet_weights,
                lora,
                Some(pooled_embeds),
                Some(time_ids),
            )
        } else {
            // Create a temporary empty LoRA collection for sampling without LoRA
            let empty_config = crate::trainers::lora::LoRAConfig {
                rank: 0,
                alpha: 0.0,
                dtype: "f32".to_string(),
                target_modules: vec![],
                dropout: Some(0.0),
            };
            // Convert Arc<CudaDevice> to &Device
            let device_wrapper = Device::from(latents.device().clone());
            let empty_lora = LoRACollection::new(empty_config, &device_wrapper)?;
            forward_sdxl_sampling(
                latents,
                &timestep_tensor,
                text_embeds,
                &self.unet_weights,
                &empty_lora,
                Some(pooled_embeds),
                Some(time_ids),
            )
        }
    }

    /// Create time embeddings for SDXL
    fn create_time_ids(&self, height: usize, width: usize) -> flame_core::Result<Tensor> {
        // SDXL time ids: [original_height, original_width, crop_top, crop_left, target_height, target_width]
        let time_ids = vec![
            height as f32,
            width as f32,
            0.0, // crop_top
            0.0, // crop_left
            height as f32,
            width as f32,
        ];

        Tensor::from_vec(time_ids, Shape::from_dims(&[1, 6]), self.device.cuda_device().clone())
    }

    /// Sample with custom conditioning
    pub fn sample_with_conditioning<F>(
        &mut self,
        conditioning_fn: F,
        height: usize,
        width: usize,
        seed: Option<u64>,
    ) -> flame_core::Result<Tensor>
    where
        F: Fn(usize) -> flame_core::Result<(Tensor, Tensor)>, // Returns (text_embeds, pooled_embeds)
    {
        let batch_size = 1; // For simplicity
        let latent_height = height / 8;
        let latent_width = width / 8;

        // Initialize latents
        let mut rng = if let Some(s) = seed {
            rand::rngs::StdRng::seed_from_u64(s)
        } else {
            rand::rngs::StdRng::from_entropy()
        };

        let mut latents = Tensor::randn(
            Shape::from_dims(&[batch_size, 4, latent_height, latent_width]),
            0.0,
            1.0,
            self.device.cuda_device().clone(),
        )?;

        // Create scheduler
        let mut scheduler =
            SchedulerWrapper::DDPM(DDPMScheduler::new(self.config.num_inference_steps));
        latents = scheduler.add_noise(&latents, &latents, 0)?;

        let time_ids = self.create_time_ids(height, width)?;

        // Denoising loop with dynamic conditioning
        for (i, &t) in scheduler.timesteps().iter().enumerate() {
            // Get conditioning for this timestep
            let (text_embeds, pooled_embeds) = conditioning_fn(t)?;

            // Predict and denoise
            let noise_pred =
                self.unet_forward(&latents, t, &text_embeds, &pooled_embeds, &time_ids)?;

            latents = scheduler.step(&noise_pred, t, &latents)?; // TODO: Use gradient_map instead of individual tensor
        }

        Ok(latents)
    }
}
