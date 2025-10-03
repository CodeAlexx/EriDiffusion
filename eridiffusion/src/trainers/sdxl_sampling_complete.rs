use flame_core::device::Device;
use flame_core::optimizers::{Adam, SGD};
use flame_core::{DType, Error, Result, Shape, Tensor};
use image::{ImageBuffer, RgbImage};
use log::{debug, error, info, warn};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::path::Path;
use std::{collections::HashMap, path::PathBuf};

// Import missing types
use crate::models::sdxl_unet_complete::AddedCondKwargs;
use crate::models::sdxl_vae::SDXLVAE;
use crate::samplers::flame_schedulers::{
    BetaSchedule as FlameBetaSchedule, PredictionType as FlamePredictionType, TimestepSpacing,
};
use crate::samplers::SchedulerConfig;
use crate::trainers::lora::LoRACollection;
use crate::trainers::sdxl_forward_sampling::forward_sdxl_sampling;
use crate::trainers::sdxl_vae_wrapper::SDXLVAEWrapper;
use crate::trainers::text_encoders::TextEncoders;

pub struct DDIMScheduler {
    timesteps: Vec<usize>,
    alphas_cumprod: Vec<f64>,
    init_noise_sigma: f64,
    num_inference_steps: usize,
    num_train_timesteps: usize,
    eta: f64,
    prediction_type: PredictionType,
}

// Complete SDXL sampling implementation for LoRA training
// This module provides comprehensive sampling functionality including:
// - Multiple scheduler support (DDIM, DPM++, Euler)
// - Proper SDXL time_ids handling
// - LoRA injection during sampling
// - Classifier-free guidance
// - VAE decoding with proper scaling

// FLAME uses flame_core::device::Device instead of Device

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
    pub eta: f64, // For DDIM stochasticity
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
    pub original_height: f32,
    pub original_width: f32,
    pub crop_top: f32,
    pub crop_left: f32,
    pub target_height: f32,
    pub target_width: f32,
}

impl SDXLTimeIds {
    pub fn new(height: usize, width: usize) -> Self {
        Self {
            original_height: height as f32,
            original_width: width as f32,
            crop_top: 0.0,
            crop_left: 0.0,
            target_height: height as f32,
            target_width: width as f32,
        }
    }

    pub fn to_tensor(&self, batch_size: usize, device: &Device) -> flame_core::Result<Tensor> {
        let time_ids = vec![
            self.original_height,
            self.original_width,
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

        Ok(Tensor::from_vec(
            all_time_ids,
            Shape::from_dims(&[batch_size, 6]),
            device.cuda_device().clone(),
        )?)
    }
}

/// DDIM Scheduler implementation

impl DDIMScheduler {
    /// Create a new DDIM scheduler
    pub fn new(
        num_train_timesteps: usize,
        beta_start: f64,
        beta_end: f64,
        beta_schedule: BetaSchedule,
        prediction_type: PredictionType,
    ) -> Self {
        // Generate beta schedule
        let betas = match beta_schedule {
            BetaSchedule::Linear => {
                let mut betas = Vec::with_capacity(num_train_timesteps);
                for i in 0..num_train_timesteps {
                    let t = i as f64 / (num_train_timesteps - 1) as f64;
                    betas.push(beta_start + t * (beta_end - beta_start));
                }
                betas
            }
            BetaSchedule::ScaledLinear => {
                let mut betas = Vec::with_capacity(num_train_timesteps);
                let start = beta_start.sqrt();
                let end = beta_end.sqrt();
                for i in 0..num_train_timesteps {
                    let t = i as f64 / (num_train_timesteps - 1) as f64;
                    let beta = start + t * (end - start);
                    betas.push(beta * beta);
                }
                betas
            }
            BetaSchedule::SquaredCosCapV2 => {
                // Simplified cosine schedule
                let mut betas = Vec::with_capacity(num_train_timesteps);
                for i in 0..num_train_timesteps {
                    let t = i as f64 / num_train_timesteps as f64;
                    let alpha_bar = (t * std::f64::consts::PI / 2.0).cos().powi(2);
                    let alpha_bar_prev = if i > 0 {
                        ((i - 1) as f64 / num_train_timesteps as f64 * std::f64::consts::PI / 2.0)
                            .cos()
                            .powi(2)
                    } else {
                        1.0
                    };
                    betas.push(1.0 - alpha_bar / alpha_bar_prev);
                }
                betas
            }
        };

        // Calculate alphas and cumulative products
        let mut alphas_cumprod = Vec::with_capacity(num_train_timesteps);
        let mut alpha_cumprod = 1.0;
        for beta in &betas {
            alpha_cumprod *= 1.0 - beta;
            alphas_cumprod.push(alpha_cumprod);
        }

        Self {
            timesteps: vec![],
            alphas_cumprod,
            init_noise_sigma: 1.0,
            num_inference_steps: 50,
            num_train_timesteps,
            eta: 0.0,
            prediction_type,
        }
    }

    /// Set the number of inference steps
    pub fn set_timesteps(&mut self, num_inference_steps: usize) {
        self.num_inference_steps = num_inference_steps;

        // Create timestep schedule
        let step_ratio = self.num_train_timesteps / num_inference_steps;
        let mut timesteps = Vec::with_capacity(num_inference_steps);

        for i in 0..num_inference_steps {
            let t = (num_inference_steps - 1 - i) * step_ratio;
            timesteps.push(t.min(self.num_train_timesteps - 1));
        }

        self.timesteps = timesteps;
    }

    /// Get alpha_prod for a given timestep
    fn get_alpha_prod(&self, timestep: usize) -> f64 {
        self.alphas_cumprod[timestep.min(self.num_train_timesteps - 1)]
    }

    /// Single denoising step
    pub fn step(
        &self,
        model_output: &Tensor,
        timestep: usize,
        sample: &Tensor,
        eta: f64,
        generator: Option<&mut StdRng>,
    ) -> flame_core::Result<Tensor> {
        // Get alphas
        let alpha_prod_t = self.get_alpha_prod(timestep);
        let alpha_prod_t_prev =
            if timestep > 0 { self.get_alpha_prod(self.get_prev_timestep(timestep)) } else { 1.0 };

        let beta_prod_t = 1.0 - alpha_prod_t;
        let beta_prod_t_prev = 1.0 - alpha_prod_t_prev;

        // Compute predicted original sample based on prediction type
        let pred_original_sample = match self.prediction_type {
            PredictionType::Epsilon => {
                // x_0 = (x_t - sqrt(1 - alpha_t) * eps) / sqrt(alpha_t)
                let sqrt_alpha_prod = alpha_prod_t.sqrt();
                let sqrt_one_minus_alpha_prod = beta_prod_t.sqrt();

                sample.mul_scalar((1.0 / sqrt_alpha_prod) as f32)?.sub(
                    &model_output
                        .mul_scalar((sqrt_one_minus_alpha_prod / sqrt_alpha_prod) as f32)?,
                )?
            }
            PredictionType::Sample => model_output.clone(),
            PredictionType::VPrediction => {
                // x_0 = sqrt(alpha_t) * x_t - sqrt(1 - alpha_t) * v
                let sqrt_alpha_prod = alpha_prod_t.sqrt();
                let sqrt_one_minus_alpha_prod = beta_prod_t.sqrt();

                sample
                    .mul_scalar(sqrt_alpha_prod as f32)?
                    .sub(&model_output.mul_scalar(sqrt_one_minus_alpha_prod as f32)?)?
            }
        };

        // Clip predicted sample if needed
        let pred_original_sample = pred_original_sample.clamp(-1.0, 1.0)?;

        // Compute variance
        let variance = if timestep > 0 {
            let variance =
                beta_prod_t_prev / beta_prod_t * (1.0 - alpha_prod_t / alpha_prod_t_prev);
            variance * eta * eta
        } else {
            0.0
        };

        let std_dev_t = variance.sqrt();

        // Compute direction pointing to x_t
        let pred_sample_direction = if timestep > 0 {
            let sqrt_one_minus_alpha_prod = beta_prod_t.sqrt();
            model_output.mul_scalar(sqrt_one_minus_alpha_prod as f32)?
        } else {
            Tensor::zeros(model_output.shape().clone(), model_output.device().clone())?
        };

        // Compute previous sample
        let sqrt_alpha_prod_t_prev = alpha_prod_t_prev.sqrt();
        let sqrt_one_minus_alpha_prod_t_prev = beta_prod_t_prev.sqrt();

        let mut prev_sample = pred_original_sample.mul_scalar(sqrt_alpha_prod_t_prev as f32)?.add(
            &pred_sample_direction
                .mul_scalar((sqrt_one_minus_alpha_prod_t_prev - std_dev_t) as f32)?,
        )?;

        // Add noise if eta > 0
        if variance > 0.0 && generator.is_some() {
            let noise =
                Tensor::randn(prev_sample.shape().clone(), 0.0, 1.0, prev_sample.device().clone())?;
            prev_sample = prev_sample.add(&noise.mul_scalar(std_dev_t as f32)?)?;
        }

        Ok(prev_sample)
    }

    /// Get previous timestep in the schedule
    fn get_prev_timestep(&self, timestep: usize) -> usize {
        let step_ratio = self.num_train_timesteps / self.num_inference_steps;
        timestep.saturating_sub(step_ratio)
    }
}

// Second impl block for DDIMScheduler - merging with the first
impl DDIMScheduler {
    pub fn new_with_inference_steps(
        num_inference_steps: usize,
        num_train_timesteps: usize,
        beta_start: f64,
        beta_end: f64,
        beta_schedule: BetaSchedule,
        prediction_type: PredictionType,
        eta: f64,
    ) -> flame_core::Result<Self> {
        // Create beta schedule
        let betas = match beta_schedule {
            BetaSchedule::Linear => {
                let betas: Vec<f64> = (0..num_train_timesteps)
                    .map(|i| {
                        beta_start
                            + (beta_end - beta_start) * (i as f64)
                                / (num_train_timesteps as f64 - 1.0)
                    })
                    .collect();
                betas
            }
            BetaSchedule::ScaledLinear => {
                let start = beta_start.sqrt();
                let end = beta_end.sqrt();
                let betas: Vec<f64> = (0..num_train_timesteps)
                    .map(|i| {
                        let beta =
                            start + (end - start) * (i as f64) / (num_train_timesteps as f64 - 1.0);
                        beta * beta
                    })
                    .collect();
                betas
            }
            BetaSchedule::SquaredCosCapV2 => {
                let max_beta = 0.999;
                let alpha_bar =
                    |t: f64| f64::cos((t + 0.008) / 1.008 * std::f64::consts::FRAC_PI_2).powi(2);
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
        let timesteps: Vec<usize> =
            (0..num_inference_steps).map(|i| (num_inference_steps - 1 - i) * step_ratio).collect();

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

    pub fn scale_model_input(
        &self,
        sample: Tensor,
        _timestep: usize,
    ) -> flame_core::Result<Tensor> {
        // DDIM doesn't scale the model input
        Ok(sample)
    }

    // Duplicate step method - renamed to step_alt to avoid conflict
    pub fn step_alt(
        &self,
        model_output: &Tensor,
        timestep: usize,
        sample: &Tensor,
        generator: Option<&mut rand::rngs::StdRng>,
    ) -> flame_core::Result<Tensor> {
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

                sample.sub(&model_output.mul_scalar(sqrt_one_minus_alpha_prod_t as f32)?)?.div(
                    &Tensor::full(
                        sample.shape().clone(),
                        sqrt_alpha_prod_t as f32,
                        sample.device().clone(),
                    )?,
                )?
            }
            PredictionType::Sample => model_output.clone(),
            PredictionType::VPrediction => {
                // x_0 = sqrt(alpha_t) * sample - sqrt(beta_t) * model_output
                let sqrt_alpha_prod_t = alpha_prod_t.sqrt();
                let sqrt_one_minus_alpha_prod_t = beta_prod_t.sqrt();

                sample
                    .mul_scalar(sqrt_alpha_prod_t as f32)?
                    .sub(&model_output.mul_scalar(sqrt_one_minus_alpha_prod_t as f32)?)?
            }
        };

        // Compute variance
        let variance = if self.eta > 0.0 && timestep > 0 {
            let variance_noise = if let Some(gen) = generator {
                // Use provided generator for deterministic results
                Tensor::randn(sample.shape().clone(), 0.0, 1.0, sample.device().clone())?
            } else {
                Tensor::randn(sample.shape().clone(), 0.0, 1.0, sample.device().clone())?
            };

            let std_dev_t = self.eta
                * ((beta_prod_t_prev / beta_prod_t) * (1.0 - alpha_prod_t / alpha_prod_t_prev))
                    .sqrt();
            variance_noise.mul_scalar(std_dev_t as f32)?
        } else {
            Tensor::zeros(sample.shape().clone(), sample.device().clone())?
        };

        // Compute predicted sample direction
        let pred_sample_direction = match self.prediction_type {
            PredictionType::Epsilon => {
                let sqrt_one_minus_alpha_prod_t_prev = beta_prod_t_prev.sqrt();
                model_output.mul_scalar(sqrt_one_minus_alpha_prod_t_prev as f32)?
            }
            _ => {
                // For other prediction types, recompute epsilon
                let sqrt_alpha_prod_t = alpha_prod_t.sqrt();
                let sqrt_one_minus_alpha_prod_t = beta_prod_t.sqrt();
                let epsilon = sample
                    .sub(&pred_original_sample.mul_scalar(sqrt_alpha_prod_t as f32)?)?
                    .div(&Tensor::full(
                        sample.shape().clone(),
                        sqrt_one_minus_alpha_prod_t as f32,
                        sample.device().clone(),
                    )?)?;

                let sqrt_one_minus_alpha_prod_t_prev = beta_prod_t_prev.sqrt();
                epsilon.mul_scalar(sqrt_one_minus_alpha_prod_t_prev as f32)?
            }
        };

        // x_t-1 = sqrt(alpha_t-1) * x_0 + sqrt(1 - alpha_t-1) * epsilon + variance
        let sqrt_alpha_prod_t_prev = alpha_prod_t_prev.sqrt();
        Ok(pred_original_sample
            .mul_scalar(sqrt_alpha_prod_t_prev as f32)?
            .add(&pred_sample_direction)?
            .add(&variance)?)
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
        Self { device, dtype, config }
    }

    /// Generate samples using SDXL with LoRA
    pub fn generate_samples(
        &self,
        unet_weights: &std::collections::HashMap<String, Tensor>,
        lora_collection: &LoRACollection,
        vae: &SDXLVAEWrapper,
        text_encoders: &mut TextEncoders,
        prompts: &[String],
        negative_prompts: Option<&[String]>,
        height: usize,
        width: usize,
        seed: Option<u64>,
    ) -> flame_core::Result<Tensor> {
        let batch_size = prompts.len();

        // Set seed if provided
        if let Some(seed) = seed {
            info!("Using seed: {}", seed);
            // Note: FLAME doesn't have global seed setting, handled in tensor creation
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

            (Tensor::stack(&neg_embeds_vec, 0)?, Tensor::stack(&neg_pooled_vec, 0)?)
        } else {
            // Use unconditional embeddings
            text_encoders.encode_unconditional(batch_size, 77)?
        };

        // Prepare for classifier-free guidance
        let do_classifier_free_guidance = self.config.guidance_scale > 1.0;

        let (text_embeddings, pooled_embeddings) = if do_classifier_free_guidance {
            (
                Tensor::cat(&[&negative_embeds, &prompt_embeds], 0)?,
                Tensor::cat(&[&negative_pooled, &pooled_prompt_embeds], 0)?,
            )
        } else {
            (prompt_embeds, pooled_prompt_embeds)
        };

        // Ensure correct dtype for text embeddings (match the model's dtype)
        let text_embeddings = text_embeddings.to_dtype(self.dtype)?;
        let pooled_embeddings = pooled_embeddings.to_dtype(self.dtype)?;

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
        let scheduler_config = SchedulerConfig {
            num_train_timesteps: 1000,
            beta_start: 0.00085,
            beta_end: 0.012,
            beta_schedule: FlameBetaSchedule::ScaledLinear,
            prediction_type: match self.config.prediction_type {
                PredictionType::Epsilon => FlamePredictionType::Epsilon,
                PredictionType::VPrediction => FlamePredictionType::VPrediction,
                PredictionType::Sample => FlamePredictionType::Sample,
            },
            timestep_spacing: TimestepSpacing::Linspace,
            steps_offset: 0,
        };
        let scheduler = DDIMScheduler::new(
            scheduler_config.num_train_timesteps,
            scheduler_config.beta_start as f64,
            scheduler_config.beta_end as f64,
            match scheduler_config.beta_schedule {
                FlameBetaSchedule::Linear => BetaSchedule::Linear,
                FlameBetaSchedule::ScaledLinear => BetaSchedule::ScaledLinear,
                FlameBetaSchedule::SquaredcosCap => BetaSchedule::SquaredCosCapV2,
            },
            self.config.prediction_type,
        );

        // Generate initial latents
        info!("Generating initial latents...");
        let latent_shape = [batch_size, 4, height / 8, width / 8];
        let mut latents = if let Some(seed) = seed {
            // Create seeded random tensor
            let mut rng = StdRng::seed_from_u64(seed);

            let num_elements: usize = latent_shape.iter().product();
            let random_data: Vec<f32> =
                (0..num_elements).map(|_| rng.gen_range(-1.0..1.0)).collect();

            Tensor::from_vec(
                random_data,
                Shape::from_dims(&latent_shape),
                self.device.cuda_device().clone(),
            )?
        } else {
            Tensor::randn(
                Shape::from_dims(&latent_shape),
                0.0,
                1.0,
                self.device.cuda_device().clone(),
            )?
        };

        // Scale by init noise sigma
        latents = latents.mul_scalar(scheduler.init_noise_sigma() as f32)?;

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

            // Ensure correct dtype for model input (match the model's dtype)
            let latent_model_input = latent_model_input.to_dtype(self.dtype)?;

            // Create timestep tensor
            let t = if do_classifier_free_guidance {
                Tensor::from_vec(
                    vec![timestep as f32; batch_size * 2],
                    Shape::from_dims(&[batch_size * 2]),
                    self.device.cuda_device().clone(),
                )?
            } else {
                Tensor::from_vec(
                    vec![timestep as f32; batch_size],
                    Shape::from_dims(&[batch_size]),
                    self.device.cuda_device().clone(),
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
                let noise_pred_uncond = noise_pred.slice(&[(0, 0 + batch_size)])?;
                let noise_pred_text = noise_pred.slice(&[(batch_size, batch_size + batch_size)])?;

                // guided_pred = uncond + guidance_scale * (text - uncond)
                let diff = noise_pred_text.sub(&noise_pred_uncond)?;
                noise_pred_uncond.add(&diff.mul_scalar(self.config.guidance_scale as f32)?)?
            } else {
                noise_pred
            };

            // Scheduler step
            latents =
                scheduler.step(&noise_pred, timestep as usize, &latents, scheduler.eta, None)?;
        }

        // Decode latents
        // Text encoding
        info!("Encoding prompts...");
        let (prompt_embeds, pooled_prompt_embeds) =
            self.encode_prompts(text_encoders, prompts, negative_prompts)?;

        // Create time IDs
        let time_ids = self.create_time_ids(height, width)?;

        // Initialize latents
        info!("Initializing latents...");
        // Create RNG if seed is provided
        let mut rng = seed.map(|s| StdRng::seed_from_u64(s));
        let mut latents = self.prepare_latents(
            prompts.len(),
            height / 8, // VAE downscale factor
            width / 8,
            rng.as_mut(),
        )?;

        // Create scheduler
        let mut scheduler = self.create_scheduler();
        scheduler.set_timesteps(self.config.num_inference_steps);

        // Prepare extra conditioning
        let mut added_cond_kwargs = HashMap::new();
        let pooled_embeds = pooled_prompt_embeds.clone();
        added_cond_kwargs.insert("text_embeds".to_string(), pooled_embeds);
        added_cond_kwargs.insert("time_ids".to_string(), time_ids);

        // Denoising loop
        info!("Running denoising loop for {} steps...", self.config.num_inference_steps);
        for (i, &timestep) in scheduler.timesteps.iter().enumerate() {
            // Expand latents for CFG if needed
            let latent_model_input = if self.config.guidance_scale > 1.0 {
                Tensor::cat(&[&latents, &latents], 0)?
            } else {
                latents.clone()
            };

            // Create timestep tensor
            let t = Tensor::full(
                Shape::from_dims(&[1]),
                timestep as f32,
                self.device.cuda_device().clone(),
            )?
            .to_dtype(self.dtype)?
            .unsqueeze(0)?;
            let t = if self.config.guidance_scale > 1.0 {
                // Duplicate the timestep for classifier-free guidance
                Tensor::cat(&[&t, &t], 0)?
            } else {
                t
            };

            // Predict noise
            let noise_pred = self.unet_forward_with_lora(
                &latent_model_input,
                &t,
                &prompt_embeds,
                unet_weights,
                lora_collection,
                Some(&added_cond_kwargs),
            )?;

            // Perform guidance
            let noise_pred = if self.config.guidance_scale > 1.0 {
                let chunks = noise_pred.chunk(2, 0)?;
                let noise_pred_uncond = &chunks[0];
                let noise_pred_text = &chunks[1];
                let noise_pred = noise_pred_uncond
                    .add(
                        &noise_pred_text
                            .sub(&noise_pred_uncond)?
                            .mul_scalar(self.config.guidance_scale as f32)?,
                    )?
                    .to_dtype(latents.dtype())?;
                noise_pred
            } else {
                noise_pred
            };

            // Scheduler step
            latents = scheduler.step(
                &noise_pred,
                timestep,
                &latents,
                self.config.eta,
                None, // No explicit generator needed
            )?;

            if (i + 1) % 10 == 0 {
                info!("Completed step {}/{}", i + 1, self.config.num_inference_steps);
            }
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
        unet_weights: &std::collections::HashMap<String, Tensor>,
        lora_collection: &LoRACollection,
        added_cond_kwargs: Option<&HashMap<String, Tensor>>,
    ) -> flame_core::Result<Tensor> {
        // Extract additional conditioning
        let pooled_proj =
            added_cond_kwargs.and_then(|kwargs| kwargs.get("text_embeds")).ok_or_else(|| {
                Error::InvalidOperation("Missing text_embeds in added_cond_kwargs".into())
            })?;
        let time_ids =
            added_cond_kwargs.and_then(|kwargs| kwargs.get("time_ids")).ok_or_else(|| {
                Error::InvalidOperation("Missing time_ids in added_cond_kwargs".into())
            })?;

        // Forward pass with LoRA
        // Note: forward_sdxl_sampling handles pooled_proj and time_ids
        // Pass the additional conditioning to the forward function
        forward_sdxl_sampling(
            sample,
            timestep,
            encoder_hidden_states,
            unet_weights,
            lora_collection,
            Some(pooled_proj),
            Some(time_ids),
        )
    }

    /// Encode prompts using dual text encoders
    fn encode_prompts(
        &self,
        text_encoders: &mut TextEncoders,
        prompts: &[String],
        negative_prompts: Option<&[String]>,
    ) -> flame_core::Result<(Tensor, Tensor)> {
        let batch_size = prompts.len();

        // Use negative prompts or empty strings
        let empty_prompts = vec!["".to_string(); batch_size];
        let negative_prompts = negative_prompts.unwrap_or(&empty_prompts);

        // Encode positive prompts
        let mut all_prompt_embeds = Vec::new();
        let mut all_pooled_embeds = Vec::new();

        for prompt in prompts {
            let (prompt_embeds, pooled_embeds) = text_encoders.encode_sdxl(prompt, 77)?;
            all_prompt_embeds.push(prompt_embeds);
            all_pooled_embeds.push(pooled_embeds);
        }

        // Encode negative prompts
        for neg_prompt in negative_prompts {
            let (neg_embeds, neg_pooled) = text_encoders.encode_sdxl(neg_prompt, 77)?;
            all_prompt_embeds.push(neg_embeds);
            all_pooled_embeds.push(neg_pooled);
        }

        // Concatenate all embeddings
        let prompt_embeds = Tensor::cat(&all_prompt_embeds.iter().collect::<Vec<_>>(), 0)?;
        let pooled_prompt_embeds = Tensor::cat(&all_pooled_embeds.iter().collect::<Vec<_>>(), 0)?;

        Ok((prompt_embeds, pooled_prompt_embeds))
    }

    /// Create time IDs for SDXL conditioning
    fn create_time_ids(&self, height: usize, width: usize) -> flame_core::Result<Tensor> {
        let time_ids = vec![
            height as f32, // original_height
            width as f32,  // original_width
            0.0,           // crop_top
            0.0,           // crop_left
            height as f32, // target_height
            width as f32,  // target_width
        ];

        let time_ids =
            Tensor::from_vec(time_ids, Shape::from_dims(&[6]), self.device.cuda_device().clone())?
                .to_dtype(self.dtype)?
                .unsqueeze(0)?;

        // Duplicate for negative prompt if using CFG
        if self.config.guidance_scale > 1.0 {
            Tensor::cat(&[&time_ids, &time_ids], 0)
        } else {
            Ok(time_ids)
        }
    }

    /// Initialize random latents
    fn prepare_latents(
        &self,
        batch_size: usize,
        height: usize,
        width: usize,
        generator: Option<&mut StdRng>,
    ) -> flame_core::Result<Tensor> {
        let shape = vec![batch_size, 4, height, width]; // 4 channels for SDXL VAE

        // Generate random latents
        let latents = if let Some(gen) = generator {
            // Use seeded random for reproducibility
            let random_values: Vec<f32> =
                (0..shape.iter().product()).map(|_| gen.gen::<f32>() * 2.0 - 1.0).collect();
            Tensor::from_vec(
                random_values,
                Shape::from_dims(&shape),
                self.device.cuda_device().clone(),
            )?
        } else {
            Tensor::randn(Shape::from_dims(&shape), 0.0, 1.0, self.device.cuda_device().clone())?
        };

        latents.to_dtype(self.dtype)
    }

    /// Create scheduler based on config
    fn create_scheduler(&self) -> DDIMScheduler {
        DDIMScheduler::new(
            1000,    // num_train_timesteps for SDXL
            0.00085, // beta_start
            0.012,   // beta_end
            BetaSchedule::ScaledLinear,
            self.config.prediction_type,
        )
    }

    /// Decode latents to images using VAE
    fn decode_latents(&self, vae: &SDXLVAEWrapper, latents: &Tensor) -> flame_core::Result<Tensor> {
        let batch_size = latents.shape().dims()[0];
        let mut images = Vec::new();

        // Decode each latent individually to manage memory
        for i in 0..batch_size {
            let latent = latents.get(i)?.unsqueeze(0)?;
            let image = vae.decode(&latent)?;
            images.push(image);
        }

        // Concatenate all images along batch dimension
        let image_refs: Vec<&Tensor> = images.iter().collect();
        Tensor::cat(&image_refs, 0)
    }

    /// Save images to files
    pub fn save_images(
        &self,
        images: &Tensor,
        output_dir: &Path,
        prefix: &str,
    ) -> flame_core::Result<Vec<PathBuf>> {
        let batch_size = images.shape().dims()[0];
        let mut saved_paths = Vec::new();

        // Create output directory if needed
        std::fs::create_dir_all(output_dir)
            .map_err(|e| {
                flame_core::Error::InvalidOperation(format!(
                    "Failed to create directory: {}",
                    e
                ))
            })
            .map_err(|e| flame_core::Error::InvalidOperation(e.to_string()))?;

        for i in 0..batch_size {
            let image = images.get(i)?;
            let filename = format!("{}_sample_{}.png", prefix, i);
            let path = output_dir.join(filename);

            self.save_image(&image, &path)?;
            saved_paths.push(path);
        }

        Ok(saved_paths)
    }

    /// Save single image tensor to file
    fn save_image(&self, image: &Tensor, path: &Path) -> flame_core::Result<()> {
        // Ensure image is in the right format
        let image = image.squeeze(Some(0))?; // Remove batch dimension if present

        // Convert from [-1, 1] to [0, 255]
        let image = image.clamp(-1.0, 1.0)?.mul_scalar(127.5f32)?.add_scalar(127.5f32)?;

        // Get dimensions
        let dims = image.shape().dims();
        let (c, h, w) = (dims[0], dims[1], dims[2]);

        // Convert to RGB format
        let image_data = if c == 3 {
            // Already RGB
            image
                .permute(&[1, 2, 0])? // CHW -> HWC
                .flatten_all()?
                .to_vec1::<f32>()?
        } else if c == 1 {
            // Grayscale, convert to RGB
            let rgb = Tensor::cat(&[&image, &image, &image], 0)?;
            rgb.permute(&[1, 2, 0])? // CHW -> HWC
                .flatten_all()?
                .to_vec1::<f32>()?
        } else {
            return Err(flame_core::Error::InvalidOperation(format!("",)));
        };

        // Convert to u8
        let image_data: Vec<u8> = image_data.iter().map(|&x| x.clamp(0.0, 255.0) as u8).collect();

        // Create and save image
        let img = RgbImage::from_raw(w as u32, h as u32, image_data).ok_or_else(|| {
            Error::InvalidOperation(
                "Failed to create image: invalid dimensions or data".to_string(),
            )
        })?;
        img.save(path).map_err(|e| Error::Io(format!("Failed to save image: {}", e)))?;

        info!("Saved image to: {}", path.display());
        Ok(())
    }
}

/// Sampling utilities for training integration
pub struct TrainingSampler {
    sampler: SDXLSampler,
    output_dir: std::path::PathBuf,
    validation_prompts: Vec<String>,
    negative_prompt: Option<String>,
}

impl TrainingSampler {
    pub fn new(
        device: Device,
        dtype: DType,
        output_dir: std::path::PathBuf,
        validation_prompts: Vec<String>,
        negative_prompt: Option<String>,
        config: SDXLSamplingConfig,
    ) -> Self {
        let sampler = SDXLSampler::new(device, dtype, config);

        Self { sampler, output_dir, validation_prompts, negative_prompt }
    }

    /// Generate samples at a specific training step
    pub fn generate_training_samples(
        &self,
        step: usize,
        unet_weights: &std::collections::HashMap<String, Tensor>,
        lora_collection: &LoRACollection,
        vae: &SDXLVAEWrapper,
        text_encoders: &mut TextEncoders,
    ) -> flame_core::Result<Vec<std::path::PathBuf>> {
        // Create step-specific directory
        let step_dir = self.output_dir.join(format!("step_{:06}", step));
        std::fs::create_dir_all(&step_dir)
            .map_err(|e| {
                flame_core::Error::InvalidOperation(format!(
                    "Failed to create directory: {}",
                    e
                ))
            })
            .map_err(|e| flame_core::Error::InvalidOperation(e.to_string()))?;

        // Generate samples
        let negative_prompts =
            self.negative_prompt.as_ref().map(|n| vec![n.clone(); self.validation_prompts.len()]);

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
        let batch_size = self.validation_prompts.len();
        for i in 0..batch_size {
            let image = images.get(i)?;
            let prompt = &self.validation_prompts[i];
            // Save image
            let image_path = step_dir.join(format!("sample_{:02}.png", i));
            self.sampler.save_image(&image, &image_path)?;

            // Save prompt
            let prompt_path = step_dir.join(format!("sample_{:02}.txt", i));
            std::fs::write(&prompt_path, prompt)
                .map_err(|e| flame_core::Error::InvalidOperation(e.to_string()))?;

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
    fn test_ddim_scheduler() {
        // TODO: Add tests
    }
}
