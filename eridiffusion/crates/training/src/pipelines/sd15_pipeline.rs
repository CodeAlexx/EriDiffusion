//! SD 1.5/2.x specific training pipeline

use super::{TrainingPipeline, PipelineConfig, PreparedBatch, PromptEmbeds, PipelineUtils};
use eridiffusion_core::{Error, ModelArchitecture, validation::TensorValidator};
use eridiffusion_core::{DiffusionModel, ModelInputs};
use eridiffusion_data::DataLoaderBatch;
use std::collections::HashMap;
use crate::loss::{masked_eps_loss, masked_l1_loss};
use flame_core::{Tensor, DType, Shape};
use crate::tensor_utils::scalar_f32;

pub struct SD15Pipeline {
    config: PipelineConfig,
    noise_scheduler: NoiseScheduler,
}

impl SD15Pipeline {
    pub fn new(config: PipelineConfig) -> anyhow::Result<Self> {
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
    
    fn prepare_batch(&self, batch: &DataLoaderBatch) -> anyhow::Result<PreparedBatch> {
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
    
    fn encode_images(&self, images: &Tensor) -> anyhow::Result<Tensor> {
        // VAE encoding should be handled by the caller
        // For now, assume images are already latents
        Ok(images.clone())
    }
    
    fn encode_prompts(&self, prompts: &[String], model: &dyn DiffusionModel) -> anyhow::Result<PromptEmbeds> {
        // SD1.5 uses single CLIP text encoder
        // In real implementation, would tokenize and encode
        let batch_size = prompts.len();
        // Mock text embeddings [batch, 77, 768] on the model's device
        let dev_arc = model.device().to_flame_cuda()?;
        let mut text_embeds = Tensor::randn(Shape::from_dims(&[batch_size, 77, 768]), 0.0, 1.0, dev_arc)?;
        // Prefer BF16 compute tensors
        text_embeds = text_embeds.to_dtype(DType::BF16)?;
        
        Ok(PromptEmbeds::new(text_embeds))
    }
    
    fn add_noise(
        &self,
        latents: &Tensor,
        noise: &Tensor,
        timesteps: &Tensor,
    ) -> anyhow::Result<Tensor> {
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
    ) -> anyhow::Result<Tensor> {
        // Get model prediction
        let inputs = self.get_model_inputs(noisy_latents, timesteps, prompt_embeds, batch)?;
        let output = model.forward(&inputs)?;
        let noise_pred = self.postprocess_outputs(&output)?;
        
        // Compute target based on parameterization
        let target = if self.config.v_parameterization {
            let latents = batch.latents.as_ref().ok_or_else(|| 
                Error::Training("Latents not found in batch".into()))?;
            self.noise_scheduler.get_velocity(latents, noise, timesteps)?
        } else {
            noise.clone()
        };
        
        // Compute loss
        let loss = match self.config.loss_type.as_str() {
            "mse" => masked_eps_loss(&noise_pred, &target, None)?,
            "mae" => masked_l1_loss(&noise_pred, &target, None)?,
            "huber" => {
                // Huber with delta=1.0, reduce via policy helper
                let diff = noise_pred.sub(&target)?;
                let abs_diff = diff.abs()?;
                let quadratic = diff.square()?.affine(0.5, 0.0)?; // 0.5 * x^2
                let linear = abs_diff.affine(1.0, -0.5)?;         // |x| - 0.5
                let one = scalar_f32(1.0, abs_diff.device().clone())?;
                let mask = abs_diff.le(&one)?;                    // |x| <= 1
                let combined = mask.where_tensor(&quadratic, &linear)?;
                crate::policy::reduce_mean_fp32_keepdim(&combined)?
            }
            _ => return Err(Error::Config(format!("Unknown loss type: {}", self.config.loss_type)).into()),
        };
        
        // Apply SNR weighting if configured
        let loss = self.apply_snr_weight(&loss, timesteps)?;
        
        // Add prior preservation loss if enabled
        if let Some(prior_loss) = self.compute_prior_loss(model, batch)? {
            let weighted_prior = prior_loss.affine(self.config.prior_loss_weight, 0.0)?;
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
        _batch: &PreparedBatch,
    ) -> anyhow::Result<ModelInputs> {
        Ok(ModelInputs {
            latents: noisy_latents.clone(),
            timestep: timesteps.clone(),
            encoder_hidden_states: Some(prompt_embeds.encoder_hidden_states.clone()),
            attention_mask: match &prompt_embeds.attention_mask { Some(m) => Some(m.clone()), None => None },
            guidance_scale: Some(1.0), // Default for training
            pooled_projections: match &prompt_embeds.pooled_projections { Some(p) => Some(p.clone()), None => None },
            additional: HashMap::new(),
        })
    }
    
    fn apply_snr_weight(&self, loss: &Tensor, timesteps: &Tensor) -> anyhow::Result<Tensor> {
        if let Some(snr_gamma) = self.config.snr_gamma {
            let snr = self.noise_scheduler.get_snr(timesteps)?;
            let min_snr_gamma = self.config.min_snr_gamma.unwrap_or(snr_gamma);
            let cap = scalar_f32(min_snr_gamma, snr.device().clone())?;
            let snr_weight = snr.minimum(&cap)?.div(&snr)?;
            Ok(loss.mul(&snr_weight)?)
        } else {
            Ok(loss.clone())
        }
    }
    
    fn compute_prior_loss(
        &self,
        model: &dyn DiffusionModel,
        batch: &PreparedBatch,
    ) -> anyhow::Result<Option<Tensor>> {
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
    alphas_cumprod: Vec<f32>,
}

impl NoiseScheduler {
    fn new(num_train_timesteps: usize, v_parameterization: bool, snr_gamma: Option<f32>) -> anyhow::Result<Self> {
        // Linear beta schedule in pure Rust then upload as Tensor
        let beta_start = 0.00085f32;
        let beta_end = 0.012f32;
        let step = (beta_end - beta_start) / (num_train_timesteps.saturating_sub(1)) as f32;
        let mut alphas_cumprod_vec = Vec::with_capacity(num_train_timesteps);
        let mut running = 1.0f32;
        for i in 0..num_train_timesteps {
            let beta = beta_start + i as f32 * step;
            let alpha = 1.0f32 - beta;
            running *= alpha;
            alphas_cumprod_vec.push(running);
        }
        Ok(Self { num_train_timesteps, v_parameterization, snr_gamma, alphas_cumprod: alphas_cumprod_vec })
    }
    
    fn add_noise(&self, latents: &Tensor, noise: &Tensor, timesteps: &Tensor) -> anyhow::Result<Tensor> {
        // Build per-sample scalars on host and upload to the same device as latents
        let steps: Vec<f32> = timesteps.flatten_all()?.to_vec1::<f32>()?
            .into_iter().map(|t| t.max(0.0).min((self.num_train_timesteps - 1) as f32)).collect();
        let mut v_sqrt_alpha = Vec::with_capacity(steps.len());
        let mut v_sqrt_one_minus = Vec::with_capacity(steps.len());
        for t in &steps {
            let idx = *t as usize;
            let a = self.alphas_cumprod[idx].max(0.0).min(1.0);
            v_sqrt_alpha.push(a.sqrt());
            v_sqrt_one_minus.push((1.0 - a).sqrt());
        }
        let dev = latents.device().clone();
        let sqrt_alpha_prod = Tensor::from_slice(&v_sqrt_alpha, Shape::from_dims(&[steps.len()]), dev.clone())?
            .reshape(broadcast_shape(latents.shape().dims()).dims())?;
        let sqrt_one_minus_alpha_prod = Tensor::from_slice(&v_sqrt_one_minus, Shape::from_dims(&[steps.len()]), dev.clone())?
            .reshape(broadcast_shape(latents.shape().dims()).dims())?;
        
        // noisy_latents = sqrt_alpha_prod * latents + sqrt_one_minus_alpha_prod * noise
        let scaled_latents = latents.mul(&sqrt_alpha_prod)?;
        let scaled_noise = noise.mul(&sqrt_one_minus_alpha_prod)?;
        Ok(scaled_latents.add(&scaled_noise).map_err(|e| Error::Training(e.to_string()))?)
    }
    
    fn get_velocity(&self, latents: &Tensor, noise: &Tensor, timesteps: &Tensor) -> anyhow::Result<Tensor> {
        let steps: Vec<f32> = timesteps.flatten_all()?.to_vec1::<f32>()?
            .into_iter().map(|t| t.max(0.0).min((self.num_train_timesteps - 1) as f32)).collect();
        let mut v_sqrt_alpha = Vec::with_capacity(steps.len());
        let mut v_sqrt_one_minus = Vec::with_capacity(steps.len());
        for t in &steps {
            let idx = *t as usize;
            let a = self.alphas_cumprod[idx].max(0.0).min(1.0);
            v_sqrt_alpha.push(a.sqrt());
            v_sqrt_one_minus.push((1.0 - a).sqrt());
        }
        let dev = latents.device().clone();
        let sqrt_alpha_prod = Tensor::from_slice(&v_sqrt_alpha, Shape::from_dims(&[steps.len()]), dev.clone())?
            .reshape(broadcast_shape(latents.shape().dims()).dims())?;
        let sqrt_one_minus_alpha_prod = Tensor::from_slice(&v_sqrt_one_minus, Shape::from_dims(&[steps.len()]), dev.clone())?
            .reshape(broadcast_shape(latents.shape().dims()).dims())?;
        
        // v = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * latents
        let v1 = sqrt_alpha_prod.mul(noise)?;
        let v2 = sqrt_one_minus_alpha_prod.mul(latents)?;
        Ok(v1.sub(&v2).map_err(|e| Error::Training(e.to_string()))?)
    }
    
    fn get_snr(&self, timesteps: &Tensor) -> anyhow::Result<Tensor> {
        let steps: Vec<f32> = timesteps.flatten_all()?.to_vec1::<f32>()?
            .into_iter().map(|t| t.max(0.0).min((self.num_train_timesteps - 1) as f32)).collect();
        let v_alpha: Vec<f32> = steps.iter().map(|t| self.alphas_cumprod[*t as usize].max(0.0).min(1.0)).collect();
        let dev = timesteps.device().clone();
        let alpha_prod = Tensor::from_slice(&v_alpha, Shape::from_dims(&[steps.len()]), dev.clone())?;
        let one_minus: Vec<f32> = v_alpha.iter().map(|a| 1.0 - *a).collect();
        let one_minus_alpha = Tensor::from_slice(&one_minus, Shape::from_dims(&[steps.len()]), dev)?;
        Ok(alpha_prod.div(&one_minus_alpha).map_err(|e| Error::Training(e.to_string()))?)
    }
}

fn broadcast_shape(dims: &[usize]) -> Shape {
    let b = *dims.first().unwrap_or(&1);
    let mut s = vec![b];
    for _ in 1..dims.len() { s.push(1); }
    Shape::from_dims(&s)
}
