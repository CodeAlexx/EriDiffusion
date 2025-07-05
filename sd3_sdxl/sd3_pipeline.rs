//! SD3/SD3.5 training pipeline with MMDiT and flow matching

use super::{TrainingPipeline, PipelineConfig, PreparedBatch, PromptEmbeds, PipelineUtils};
use eridiffusion_core::{Result, Error, ModelArchitecture, context};
use eridiffusion_models::{DiffusionModel, ModelInputs, ModelOutput, VAE, TextEncoder};
use eridiffusion_data::DataLoaderBatch;
use candle_core::{Tensor, DType, Device};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{info, debug};

pub struct SD3Pipeline {
    config: PipelineConfig,
    flow_matching: FlowMatching,
    vae: Option<Arc<dyn VAE + Send + Sync>>,
    text_encoders: Option<TextEncoders>,
}

/// Text encoders for SD3
struct TextEncoders {
    clip_l: Arc<dyn TextEncoder + Send + Sync>,
    clip_g: Arc<dyn TextEncoder + Send + Sync>,
    t5_xxl: Arc<dyn TextEncoder + Send + Sync>,
}

impl SD3Pipeline {
    pub fn new(config: PipelineConfig) -> Result<Self> {
        let flow_matching = FlowMatching::new(
            config.training_config.learning_rate,
            1000, // num_train_timesteps
        )?;
        
        Ok(Self {
            config,
            flow_matching,
            vae: None,
            text_encoders: None,
        })
    }
    
    /// Set VAE for latent encoding
    pub fn with_vae(mut self, vae: Arc<dyn VAE + Send + Sync>) -> Self {
        self.vae = Some(vae);
        self
    }
    
    /// Set text encoders
    pub fn with_text_encoders(
        mut self,
        clip_l: Arc<dyn TextEncoder + Send + Sync>,
        clip_g: Arc<dyn TextEncoder + Send + Sync>,
        t5_xxl: Arc<dyn TextEncoder + Send + Sync>,
    ) -> Self {
        self.text_encoders = Some(TextEncoders {
            clip_l,
            clip_g,
            t5_xxl,
        });
        self
    }
}

impl TrainingPipeline for SD3Pipeline {
    fn architecture(&self) -> ModelArchitecture {
        ModelArchitecture::SD3
    }
    
    fn prepare_batch(&self, batch: &DataLoaderBatch) -> Result<PreparedBatch> {
        // Encode images to latents if VAE is available
        let latents = if let Some(vae) = &self.vae {
            debug!("Encoding batch of {} images to latents", batch.batch_size());
            vae.encode(&batch.images)?
        } else {
            // Assume pre-encoded latents are provided
            batch.images.clone()
        };
        
        // Verify latent shape for SD3 (16 channels)
        let latent_channels = latents.dims()[1];
        if latent_channels != 16 {
            return Err(Error::InvalidShape(format!(
                "Expected 16 latent channels for SD3, got {}",
                latent_channels
            )));
        }
        
        // Apply input perturbation
        let latents = PipelineUtils::apply_input_perturbation(&latents, self.config.input_perturbation)?;
        
        // Get batch size
        let batch_size = batch.images.dims()[0];
        
        // Extract metadata - use defaults for now
        let mut metadata = HashMap::new();
        metadata.insert("original_sizes".to_string(), 
            serde_json::json!(vec![(1024, 1024); batch_size]));
        metadata.insert("crop_coords".to_string(), 
            serde_json::json!(vec![(0, 0); batch_size]));
        
        Ok(PreparedBatch {
            images: batch.images.clone(),
            latents: Some(latents),
            captions: batch.captions.clone(),
            metadata,
        })
    }
    
    fn encode_images(&self, images: &Tensor) -> Result<Tensor> {
        if let Some(vae) = &self.vae {
            vae.encode(images)
        } else {
            // If no VAE, assume images are already latents
            Ok(images.clone())
        }
    }
    
    fn encode_prompts(&self, prompts: &[String], _model: &dyn DiffusionModel) -> Result<PromptEmbeds> {
        if let Some(encoders) = &self.text_encoders {
            debug!("Encoding {} prompts with SD3 text encoders", prompts.len());
            
            // Encode with CLIP-L
            let (clip_l_embeds, pooled_l) = encoders.clip_l.encode(prompts)?;
            
            // Encode with CLIP-G  
            let (clip_g_embeds, pooled_g) = encoders.clip_g.encode(prompts)?;
            
            // Encode with T5-XXL
            let (t5_embeds, _) = encoders.t5_xxl.encode(prompts)?;
            
            // Concatenate all embeddings [batch, seq_len, 768+1280+4096=6144]
            let combined_embeds = Tensor::cat(&[&clip_l_embeds, &clip_g_embeds, &t5_embeds], 2)?;
            
            // Concatenate pooled embeddings [batch, 768+1280=2048]
            let pooled_embeds = if let (Some(pl), Some(pg)) = (pooled_l.as_ref(), pooled_g.as_ref()) {
                Some(Tensor::cat(&[pl, pg], 1)?)
            } else {
                None
            };
            
            Ok(PromptEmbeds {
                encoder_hidden_states: combined_embeds,
                pooled_projections: pooled_embeds,
                attention_mask: None,
            })
        } else {
            Err(Error::Config("Text encoders not configured. SD3 requires CLIP-L, CLIP-G, and T5-XXL encoders.".to_string()))
        }
    }
    
    fn add_noise(
        &self,
        latents: &Tensor,
        noise: &Tensor,
        timesteps: &Tensor,
    ) -> Result<Tensor> {
        // SD3 uses flow matching, not traditional diffusion
        // Interpolate between data and noise
        let t = timesteps.unsqueeze(1)?.unsqueeze(2)?.unsqueeze(3)?
            .broadcast_div(&Tensor::new(1000.0f32, timesteps.device())?)?;
        
        // Apply noise offset
        let noise = PipelineUtils::apply_noise_offset(noise, self.config.noise_offset)?;
        
        // Linear interpolation: x_t = (1 - t) * x_0 + t * noise
        let one_minus_t = Tensor::new(1.0f32, t.device())?.broadcast_sub(&t)?;
        let noisy = latents.broadcast_mul(&one_minus_t)?.add(&noise.broadcast_mul(&t)?)?;
        
        Ok(noisy)
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
        let velocity_pred = self.postprocess_outputs(&output)?;
        
        // Flow matching target: velocity from data to noise
        let velocity_target = if let Some(ref latents) = batch.latents {
            noise.sub(latents)?
        } else {
            return Err(Error::Training("No latents found in batch".to_string()));
        };
        
        // Compute loss with flow matching weighting
        let loss = match self.config.loss_type.as_str() {
            "mse" => {
                let diff = velocity_pred.sub(&velocity_target)?;
                diff.sqr()?.mean_all()?
            }
            "mae" => {
                let diff = velocity_pred.sub(&velocity_target)?;
                diff.abs()?.mean_all()?
            }
            _ => return Err(Error::Config(format!("Unknown loss type: {}", self.config.loss_type))),
        };
        
        // Apply flow matching loss weighting
        let weighted_loss = self.flow_matching.weight_loss(&loss, timesteps)?;
        
        // Mean flow loss if enabled
        if self.config.mean_flow_loss {
            if let Some(ref latents) = batch.latents {
                self.compute_mean_flow_loss(
                    model,
                    latents,
                    noise,
                    timesteps,
                    prompt_embeds,
                    batch,
                )
            } else {
                Err(Error::Training("No latents found in batch for mean flow loss".to_string()))
            }
        } else {
            Ok(weighted_loss)
        }
    }
    
    fn get_model_inputs(
        &self,
        noisy_latents: &Tensor,
        timesteps: &Tensor,
        prompt_embeds: &PromptEmbeds,
        _batch: &PreparedBatch,
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
    
    fn apply_snr_weight(&self, loss: &Tensor, _timesteps: &Tensor) -> Result<Tensor> {
        // SD3 uses flow matching, not SNR weighting
        Ok(loss.clone())
    }
    
    fn compute_prior_loss(
        &self,
        _model: &dyn DiffusionModel,
        _batch: &PreparedBatch,
    ) -> Result<Option<Tensor>> {
        // SD3 doesn't use prior preservation
        Ok(None)
    }
}

impl SD3Pipeline {
    /// Compute mean flow loss for improved one-step generation
    fn compute_mean_flow_loss(
        &self,
        model: &dyn DiffusionModel,
        latents: &Tensor,
        noise: &Tensor,
        timesteps: &Tensor,
        prompt_embeds: &PromptEmbeds,
        batch: &PreparedBatch,
    ) -> Result<Tensor> {
        let device = latents.device();
        let batch_size = latents.dims()[0];
        let total_steps = 1000.0;
        
        // Sample random time pairs (t, r) where t < r
        let t1 = Tensor::rand(0.0, 1.0, &[batch_size], device)?;
        let t2 = Tensor::rand(0.0, 1.0, &[batch_size], device)?;
        
        let t_frac = t1.minimum(&t2)?;
        let r_frac = t1.maximum(&t2)?;
        
        // Ensure minimum time gap
        let min_gap = 0.01;
        let gap = r_frac.sub(&t_frac)?;
        let mask = gap.lt(min_gap)?;
        let r_frac = r_frac.where_cond(&mask, &t_frac.add(&Tensor::new(min_gap, device)?)?)?;
        
        // Interpolate at time t
        let t_expand = t_frac.unsqueeze(1)?.unsqueeze(2)?.unsqueeze(3)?;
        let lerp_t = latents.broadcast_mul(&Tensor::new(1.0f32, device)?.broadcast_sub(&t_expand)?)?
            .add(&noise.broadcast_mul(&t_expand)?)?;
        
        // Get velocity prediction at (z_t, r, t)
        let timesteps_t = t_frac.affine(total_steps, 0.0)?;
        let timesteps_r = r_frac.affine(total_steps, 0.0)?;
        let timesteps_cat = Tensor::cat(&[&timesteps_t, &timesteps_r], 0)?;
        
        let inputs_t = self.get_model_inputs(&lerp_t, &timesteps_cat, prompt_embeds, batch)?;
        let u_pred = model.forward(&inputs_t)?.sample;
        
        // Compute finite difference approximation of du/dt
        let eps = 1e-3;
        let t_plus_eps = t_frac.add(&Tensor::new(eps, device)?)?;
        let t_plus_expand = t_plus_eps.unsqueeze(1)?.unsqueeze(2)?.unsqueeze(3)?;
        let lerp_perturbed = latents.broadcast_mul(&Tensor::new(1.0f32, device)?.broadcast_sub(&t_plus_expand)?)?
            .add(&noise.broadcast_mul(&t_plus_expand)?)?;
        
        let timesteps_perturbed = t_plus_eps.affine(total_steps, 0.0)?;
        let timesteps_cat_pert = Tensor::cat(&[&timesteps_perturbed, &timesteps_r], 0)?;
        
        let inputs_pert = self.get_model_inputs(&lerp_perturbed, &timesteps_cat_pert, prompt_embeds, batch)?;
        let u_perturbed = model.forward(&inputs_pert)?.sample;
        
        // Approximate du/dt
        let du_dt = u_perturbed.sub(&u_pred)?.broadcast_div(&Tensor::new(eps, device)?)?;
        
        // Compute shifted prediction
        let time_gap = r_frac.sub(&t_frac)?
            .unsqueeze(1)?.unsqueeze(2)?.unsqueeze(3)?;
        let u_shifted = u_pred.add(&du_dt.broadcast_mul(&time_gap)?)?;
        
        // Target velocity (for flow matching, from data to noise)
        let v_target = noise.sub(latents)?.broadcast_div(&time_gap.maximum(1e-4)?)?;
        
        // MSE loss
        let loss = u_shifted.sub(&v_target)?.sqr()?.mean_all()?;
        
        Ok(loss)
    }
}

/// Flow matching utilities for SD3
struct FlowMatching {
    learning_rate: f32,
    num_train_timesteps: usize,
}

impl FlowMatching {
    fn new(learning_rate: f32, num_train_timesteps: usize) -> Result<Self> {
        Ok(Self {
            learning_rate,
            num_train_timesteps,
        })
    }
    
    fn weight_loss(&self, loss: &Tensor, _timesteps: &Tensor) -> Result<Tensor> {
        // Flow matching uses uniform weighting by default
        // Can implement logit-normal weighting as in the paper
        Ok(loss.clone())
    }
}
