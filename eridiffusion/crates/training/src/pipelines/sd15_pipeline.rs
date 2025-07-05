//! SD 1.5/2.x specific training pipeline

use super::{TrainingPipeline, PipelineConfig, PreparedBatch, PromptEmbeds, PipelineUtils};
use eridiffusion_core::{Result, Error, ModelArchitecture, context, validation::TensorValidator};
use eridiffusion_models::{DiffusionModel, ModelInputs};
use eridiffusion_data::DataLoaderBatch;
use candle_core::{Tensor, DType};
use std::collections::HashMap;

pub struct SD15Pipeline {
    config: PipelineConfig,
    noise_scheduler: NoiseScheduler,
}

impl SD15Pipeline {
    pub fn new(config: PipelineConfig) -> Result<Self> {
        let noise_scheduler = NoiseScheduler::new(
            1000, // num_train_timesteps
            config.v_parameterization,
            config.snr_gamma,
        )?;
        
        Ok(Self {
            config,
            noise_scheduler,
        })
    }
}

impl TrainingPipeline for SD15Pipeline {
    fn architecture(&self) -> ModelArchitecture {
        ModelArchitecture::SD15
    }
    
    fn prepare_batch(&self, batch: &DataLoaderBatch) -> Result<PreparedBatch> {
        // Validate inputs
        TensorValidator::validate_finite(&batch.images, "batch.images")?;
        
        // Encode images to latents
        let latents = batch.images.clone(); // In real impl, would use VAE encoding
        
        // Apply input perturbation if configured
        let latents = PipelineUtils::apply_input_perturbation(&latents, self.config.input_perturbation)?;
        
        Ok(PreparedBatch {
            images: batch.images.clone(),
            latents: Some(latents),
            captions: batch.captions.clone(),
            metadata: HashMap::new(),
        })
    }
    
    fn encode_images(&self, images: &Tensor) -> Result<Tensor> {
        // VAE encoding should be handled by the caller
        // For now, assume images are already latents
        Ok(images.clone())
    }
    
    fn encode_prompts(&self, prompts: &[String], model: &dyn DiffusionModel) -> Result<PromptEmbeds> {
        // SD1.5 uses single CLIP text encoder
        // In real implementation, would tokenize and encode
        let batch_size = prompts.len();
        let device = model.device().to_candle()?;
        
        // Mock text embeddings [batch, 77, 768]
        let text_embeds = Tensor::randn(0.0f32, 1.0, (batch_size, 77, 768), &device)?;
        
        Ok(PromptEmbeds::new(text_embeds))
    }
    
    fn add_noise(
        &self,
        latents: &Tensor,
        noise: &Tensor,
        timesteps: &Tensor,
    ) -> Result<Tensor> {
        // Apply noise offset if configured
        let noise = PipelineUtils::apply_noise_offset(noise, self.config.noise_offset)?;
        
        // Standard forward diffusion process
        self.noise_scheduler.add_noise(latents, &noise, timesteps)
    }
    
    fn compute_loss(
        &self,
        model: &dyn DiffusionModel,
        noisy_latents: &Tensor,
        noise: &Tensor,
        timesteps: &Tensor,
        prompt_embeds: &PromptEmbeds,
        batch: &PreparedBatch,
    ) -> Result<Tensor> {
        // Get model prediction
        let inputs = self.get_model_inputs(noisy_latents, timesteps, prompt_embeds, batch)?;
        let output = model.forward(&inputs)?;
        let noise_pred = self.postprocess_outputs(&output)?;
        
        // Compute target based on parameterization
        let target = if self.config.v_parameterization {
            let latents = batch.latents.as_ref().ok_or_else(|| 
                Error::Training("Latents not found in batch".to_string()))?;
            self.noise_scheduler.get_velocity(latents, noise, timesteps)?
        } else {
            noise.clone()
        };
        
        // Compute loss
        let loss = match self.config.loss_type.as_str() {
            "mse" => (noise_pred.sub(&target)?.sqr()?.mean_all()?),
            "mae" => (noise_pred.sub(&target)?.abs()?.mean_all()?),
            "huber" => {
                let diff = noise_pred.sub(&target)?;
                let abs_diff = diff.abs()?;
                let quadratic = diff.sqr()?.affine(0.5, 0.0)?;
                let linear = abs_diff.sub(&Tensor::new(0.5f32, noisy_latents.device())?)?;
                let mask = abs_diff.le(1.0)?;
                quadratic.where_cond(&mask, &linear)?.mean_all()?
            }
            _ => return Err(Error::Config(format!("Unknown loss type: {}", self.config.loss_type))),
        };
        
        // Apply SNR weighting if configured
        let loss = self.apply_snr_weight(&loss, timesteps)?;
        
        // Add prior preservation loss if enabled
        if let Some(prior_loss) = self.compute_prior_loss(model, batch)? {
            let weighted_prior = prior_loss.affine(self.config.prior_loss_weight as f64, 0.0)?;
            Ok(loss.add(&weighted_prior)?)
        } else {
            Ok(loss)
        }
    }
    
    fn get_model_inputs(
        &self,
        noisy_latents: &Tensor,
        timesteps: &Tensor,
        prompt_embeds: &PromptEmbeds,
        batch: &PreparedBatch,
    ) -> Result<ModelInputs> {
        Ok(ModelInputs {
            latents: noisy_latents.clone(),
            timestep: timesteps.clone(),
            encoder_hidden_states: Some(prompt_embeds.encoder_hidden_states.clone()),
            attention_mask: prompt_embeds.attention_mask.clone(),
            guidance_scale: Some(1.0), // Default for training
            pooled_projections: prompt_embeds.pooled_projections.clone(),
            additional: HashMap::new(),
        })
    }
    
    fn apply_snr_weight(&self, loss: &Tensor, timesteps: &Tensor) -> Result<Tensor> {
        if let Some(snr_gamma) = self.config.snr_gamma {
            let snr = self.noise_scheduler.get_snr(timesteps)?;
            let min_snr_gamma = self.config.min_snr_gamma.unwrap_or(snr_gamma);
            
            // Compute SNR weight: min(snr, gamma) / snr
            let snr_weight = snr.minimum(min_snr_gamma)?.div(&snr)?;
            
            loss.mul(&snr_weight).map_err(|e| Error::Training(e.to_string()))
        } else {
            Ok(loss.clone())
        }
    }
    
    fn compute_prior_loss(
        &self,
        model: &dyn DiffusionModel,
        batch: &PreparedBatch,
    ) -> Result<Option<Tensor>> {
        if !self.config.prior_preservation {
            return Ok(None);
        }
        
        // Prior preservation for DreamBooth-style training
        // Would need class images and prompts
        // For now, return None
        Ok(None)
    }
}

/// Simple noise scheduler for SD1.5
struct NoiseScheduler {
    num_train_timesteps: usize,
    v_parameterization: bool,
    snr_gamma: Option<f32>,
    alphas_cumprod: Tensor,
}

impl NoiseScheduler {
    fn new(num_train_timesteps: usize, v_parameterization: bool, snr_gamma: Option<f32>) -> Result<Self> {
        // Linear beta schedule
        // Create linear space for betas
        let beta_start = 0.00085f32;
        let beta_end = 0.012f32;
        let step = (beta_end - beta_start) / (num_train_timesteps - 1) as f32;
        let betas_vec: Vec<f32> = (0..num_train_timesteps)
            .map(|i| beta_start + i as f32 * step)
            .collect();
        let betas = Tensor::new(betas_vec.as_slice(), &candle_core::Device::Cpu)?;
        let alphas = Tensor::new(1.0f32, &candle_core::Device::Cpu)?.broadcast_sub(&betas)?;
        // Manual cumulative product
        let alphas_vec = alphas.to_vec1::<f32>()?;
        let mut cumprod_vec = vec![1.0f32; alphas_vec.len()];
        cumprod_vec[0] = alphas_vec[0];
        for i in 1..alphas_vec.len() {
            cumprod_vec[i] = cumprod_vec[i-1] * alphas_vec[i];
        }
        let alphas_cumprod = Tensor::new(cumprod_vec.as_slice(), &candle_core::Device::Cpu)?;
        
        Ok(Self {
            num_train_timesteps,
            v_parameterization,
            snr_gamma,
            alphas_cumprod,
        })
    }
    
    fn add_noise(&self, latents: &Tensor, noise: &Tensor, timesteps: &Tensor) -> Result<Tensor> {
        let alphas_cumprod = self.alphas_cumprod.to_device(latents.device())?;
        let sqrt_alpha_prod = timesteps
            .to_dtype(DType::U32)?
            .gather(&alphas_cumprod, 0)?
            .sqrt()?
            .reshape(latents.shape())?;
        
        let sqrt_one_minus_alpha_prod = timesteps
            .to_dtype(DType::U32)?
            .gather(&alphas_cumprod, 0)?
            .affine(-1.0, 1.0)?
            .sqrt()?
            .reshape(latents.shape())?;
        
        // noisy_latents = sqrt_alpha_prod * latents + sqrt_one_minus_alpha_prod * noise
        let scaled_latents = latents.mul(&sqrt_alpha_prod)?;
        let scaled_noise = noise.mul(&sqrt_one_minus_alpha_prod)?;
        scaled_latents.add(&scaled_noise).map_err(|e| Error::Training(e.to_string()))
    }
    
    fn get_velocity(&self, latents: &Tensor, noise: &Tensor, timesteps: &Tensor) -> Result<Tensor> {
        let alphas_cumprod = self.alphas_cumprod.to_device(latents.device())?;
        let sqrt_alpha_prod = timesteps
            .to_dtype(DType::U32)?
            .gather(&alphas_cumprod, 0)?
            .sqrt()?
            .reshape(latents.shape())?;
        
        let sqrt_one_minus_alpha_prod = timesteps
            .to_dtype(DType::U32)?
            .gather(&alphas_cumprod, 0)?
            .affine(-1.0, 1.0)?
            .sqrt()?
            .reshape(latents.shape())?;
        
        // v = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * latents
        let v1 = sqrt_alpha_prod.mul(noise)?;
        let v2 = sqrt_one_minus_alpha_prod.mul(latents)?;
        v1.sub(&v2).map_err(|e| Error::Training(e.to_string()))
    }
    
    fn get_snr(&self, timesteps: &Tensor) -> Result<Tensor> {
        let alphas_cumprod = self.alphas_cumprod.to_device(timesteps.device())?;
        let alpha_prod = timesteps
            .to_dtype(DType::U32)?
            .gather(&alphas_cumprod, 0)?;
        
        // SNR = alpha_prod / (1 - alpha_prod)
        let one_minus_alpha = Tensor::new(1.0f32, timesteps.device())?.broadcast_sub(&alpha_prod)?;
        alpha_prod.div(&one_minus_alpha).map_err(|e| Error::Training(e.to_string()))
    }
}