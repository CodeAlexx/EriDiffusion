//! SDXL training pipeline with UNet and traditional DDPM

use super::{TrainingPipeline, PipelineConfig, PreparedBatch, PromptEmbeds, PipelineUtils};
use eridiffusion_core::{Result, Error, ModelArchitecture, Device};
use eridiffusion_models::{DiffusionModel, ModelInputs, ModelOutput, VAE, TextEncoder};
use eridiffusion_data::DataLoaderBatch;
use candle_core::{Tensor, DType};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{info, debug, warn};

/// SDXL training pipeline
pub struct SDXLPipeline {
    config: PipelineConfig,
    vae: Option<Arc<dyn VAE + Send + Sync>>,
    text_encoders: Option<SDXLTextEncoders>,
    noise_scheduler: DDPMScheduler,
    use_refiner: bool,
}

/// Text encoders for SDXL
struct SDXLTextEncoders {
    clip_l: Arc<dyn TextEncoder + Send + Sync>,      // CLIP-L: 768 dim
    clip_g: Arc<dyn TextEncoder + Send + Sync>,      // OpenCLIP-G: 1280 dim
}

/// DDPM noise scheduler for SDXL
#[derive(Debug, Clone)]
pub struct DDPMScheduler {
    num_train_timesteps: usize,
    beta_start: f32,
    beta_end: f32,
    beta_schedule: String,
    prediction_type: String,
    rescale_betas_zero_snr: bool,
    
    // Precomputed values
    betas: Option<Tensor>,
    alphas: Option<Tensor>,
    alphas_cumprod: Option<Tensor>,
}

impl Default for DDPMScheduler {
    fn default() -> Self {
        Self {
            num_train_timesteps: 1000,
            beta_start: 0.00085,
            beta_end: 0.012,
            beta_schedule: "scaled_linear".to_string(),
            prediction_type: "epsilon".to_string(),
            rescale_betas_zero_snr: false,
            betas: None,
            alphas: None,
            alphas_cumprod: None,
        }
    }
}

impl DDPMScheduler {
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Initialize the scheduler with precomputed values
    pub fn init(&mut self, device: &Device) -> Result<()> {
        let num_steps = self.num_train_timesteps;
        
        // Compute beta schedule
        let betas = match self.beta_schedule.as_str() {
            "linear" => {
                let step = (self.beta_end - self.beta_start) / (num_steps - 1) as f32;
                let betas: Vec<f32> = (0..num_steps)
                    .map(|i| self.beta_start + i as f32 * step)
                    .collect();
                Tensor::new(betas, device)?
            }
            "scaled_linear" => {
                let step = (self.beta_end.sqrt() - self.beta_start.sqrt()) / (num_steps - 1) as f32;
                let betas: Vec<f32> = (0..num_steps)
                    .map(|i| {
                        let beta_sqrt = self.beta_start.sqrt() + i as f32 * step;
                        beta_sqrt * beta_sqrt
                    })
                    .collect();
                Tensor::new(betas, device)?
            }
            _ => return Err(Error::Config(format!("Unknown beta schedule: {}", self.beta_schedule))),
        };
        
        // Rescale betas for zero SNR if enabled
        let betas = if self.rescale_betas_zero_snr {
            self.rescale_zero_terminal_snr(&betas)?
        } else {
            betas
        };
        
        // Compute alphas and cumulative products
        let ones = Tensor::ones(betas.shape(), betas.dtype(), device)?;
        let alphas = ones.sub(&betas)?;
        let alphas_cumprod = self.compute_cumprod(&alphas)?;
        
        self.betas = Some(betas);
        self.alphas = Some(alphas);
        self.alphas_cumprod = Some(alphas_cumprod);
        
        Ok(())
    }
    
    /// Sample random timesteps for training
    pub fn sample_timesteps(&self, batch_size: usize, device: &Device) -> Result<Tensor> {
        let timesteps: Vec<i64> = (0..batch_size)
            .map(|_| rand::random::<usize>() % self.num_train_timesteps)
            .map(|t| t as i64)
            .collect();
        
        Tensor::new(timesteps, device)
            .map_err(|e| Error::Training(format!("Failed to create timesteps: {}", e)))
    }
    
    /// Get alpha_cumprod for given timesteps
    pub fn get_alpha_cumprod(&self, timesteps: &Tensor) -> Result<Tensor> {
        let alphas_cumprod = self.alphas_cumprod.as_ref()
            .ok_or_else(|| Error::Training("Scheduler not initialized".to_string()))?;
        
        alphas_cumprod.gather(timesteps, 0)
    }
    
    /// Rescale betas for zero terminal SNR
    fn rescale_zero_terminal_snr(&self, betas: &Tensor) -> Result<Tensor> {
        // Convert betas to alphas
        let ones = Tensor::ones(betas.shape(), betas.dtype(), betas.device())?;
        let alphas = ones.sub(betas)?;
        
        // Compute cumulative product
        let alphas_cumprod = self.compute_cumprod(&alphas)?;
        
        // Get final alpha_cumprod
        let final_alpha_cumprod = alphas_cumprod.i((-1,))?;
        
        // Rescale to ensure zero terminal SNR
        let rescale_factor = final_alpha_cumprod.sqrt()?;
        let alphas_cumprod_rescaled = alphas_cumprod.div(&rescale_factor)?;
        
        // Convert back to betas
        let alphas_rescaled = alphas_cumprod_rescaled.div(&alphas_cumprod_rescaled.roll_dims(1, 0, 1)?)?;
        let betas_rescaled = ones.sub(&alphas_rescaled)?;
        
        Ok(betas_rescaled)
    }
    
    /// Compute cumulative product
    fn compute_cumprod(&self, alphas: &Tensor) -> Result<Tensor> {
        let mut cumprod = Vec::new();
        let alphas_vec = alphas.to_vec1::<f32>()?;
        
        let mut running_product = 1.0f32;
        for alpha in alphas_vec {
            running_product *= alpha;
            cumprod.push(running_product);
        }
        
        Tensor::new(cumprod, alphas.device())
    }
}

impl SDXLPipeline {
    pub fn new(config: PipelineConfig) -> Result<Self> {
        let mut noise_scheduler = DDPMScheduler::new();
        noise_scheduler.init(&config.device)?;
        
        Ok(Self {
            config,
            vae: None,
            text_encoders: None,
            noise_scheduler,
            use_refiner: false,
        })
    }
    
    /// Set VAE for latent encoding
    pub fn with_vae(mut self, vae: Arc<dyn VAE + Send + Sync>) -> Self {
        self.vae = Some(vae);
        self
    }
    
    /// Set text encoders for SDXL
    pub fn with_text_encoders(
        mut self,
        clip_l: Arc<dyn TextEncoder + Send + Sync>,
        clip_g: Arc<dyn TextEncoder + Send + Sync>,
    ) -> Self {
        self.text_encoders = Some(SDXLTextEncoders {
            clip_l,
            clip_g,
        });
        self
    }
    
    /// Enable refiner model
    pub fn with_refiner(mut self, use_refiner: bool) -> Self {
        self.use_refiner = use_refiner;
        self
    }
    
    /// Add noise using DDPM schedule
    fn add_noise_ddpm(
        &self,
        original_samples: &Tensor,
        noise: &Tensor,
        timesteps: &Tensor,
    ) -> Result<Tensor> {
        let alpha_cumprod = self.noise_scheduler.get_alpha_cumprod(timesteps)?;
        let sqrt_alpha_cumprod = alpha_cumprod.sqrt()?;
        let sqrt_one_minus_alpha_cumprod = (Tensor::new(1.0f32, alpha_cumprod.device())?.sub(&alpha_cumprod)?)?.sqrt()?;
        
        // Expand to match sample dimensions
        let sqrt_alpha_cumprod = sqrt_alpha_cumprod.unsqueeze(1)?.unsqueeze(2)?.unsqueeze(3)?;
        let sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod.unsqueeze(1)?.unsqueeze(2)?.unsqueeze(3)?;
        
        // Apply noise: sqrt(alpha_cumprod) * x + sqrt(1 - alpha_cumprod) * noise
        let noisy_samples = original_samples.broadcast_mul(&sqrt_alpha_cumprod)?
            .add(&noise.broadcast_mul(&sqrt_one_minus_alpha_cumprod)?)?;
        
        Ok(noisy_samples)
    }
    
    /// Get conditioning for SDXL (includes time embeddings and other conditioning)
    fn get_conditioning(
        &self,
        batch: &PreparedBatch,
        timesteps: &Tensor,
    ) -> Result<HashMap<String, Tensor>> {
        let mut conditioning = HashMap::new();
        
        // Time embeddings
        let time_embeds = PipelineUtils::get_timestep_embedding(timesteps, 320, 10000.0)?;
        conditioning.insert("time_embeds".to_string(), time_embeds);
        
        // Original size conditioning
        if let Some(original_sizes) = batch.metadata.get("original_sizes") {
            if let Ok(sizes) = serde_json::from_value::<Vec<(u32, u32)>>(original_sizes.clone()) {
                let size_embeds = self.encode_size_conditioning(&sizes, timesteps.device())?;
                conditioning.insert("original_size_embeds".to_string(), size_embeds);
            }
        }
        
        // Crop coordinates conditioning
        if let Some(crop_coords) = batch.metadata.get("crop_coords") {
            if let Ok(coords) = serde_json::from_value::<Vec<(u32, u32)>>(crop_coords.clone()) {
                let crop_embeds = self.encode_size_conditioning(&coords, timesteps.device())?;
                conditioning.insert("crop_coords_embeds".to_string(), crop_embeds);
            }
        }
        
        // Target size conditioning (usually same as original for training)
        if let Some(original_sizes) = batch.metadata.get("original_sizes") {
            if let Ok(sizes) = serde_json::from_value::<Vec<(u32, u32)>>(original_sizes.clone()) {
                let target_embeds = self.encode_size_conditioning(&sizes, timesteps.device())?;
                conditioning.insert("target_size_embeds".to_string(), target_embeds);
            }
        }
        
        Ok(conditioning)
    }
    
    /// Encode size conditioning for SDXL
    fn encode_size_conditioning(&self, sizes: &[(u32, u32)], device: &Device) -> Result<Tensor> {
        let flattened: Vec<f32> = sizes.iter()
            .flat_map(|(w, h)| vec![*w as f32, *h as f32])
            .collect();
        
        Tensor::new(flattened.as_slice(), device)?
            .reshape(&[sizes.len(), 2])
    }
}

impl TrainingPipeline for SDXLPipeline {
    fn architecture(&self) -> ModelArchitecture {
        ModelArchitecture::SDXL
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
        
        // Verify latent shape for SDXL (4 channels)
        let latent_channels = latents.dims()[1];
        if latent_channels != 4 {
            warn!("Expected 4 latent channels for SDXL, got {}. Continuing anyway.", latent_channels);
        }
        
        // Apply input perturbation
        let latents = PipelineUtils::apply_input_perturbation(&latents, self.config.input_perturbation)?;
        
        // Get batch size and image dimensions
        let batch_size = batch.images.dims()[0];
        let height = batch.images.dims()[2];
        let width = batch.images.dims()[3];
        
        // Extract metadata with proper SDXL conditioning
        let mut metadata = HashMap::new();
        metadata.insert("original_sizes".to_string(), 
            serde_json::json!(vec![(width, height); batch_size]));
        metadata.insert("crop_coords".to_string(), 
            serde_json::json!(vec![(0, 0); batch_size]));
        metadata.insert("target_sizes".to_string(), 
            serde_json::json!(vec![(width, height); batch_size]));
        
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
            Ok(images.clone())
        }
    }
    
    fn encode_prompts(&self, prompts: &[String], _model: &dyn DiffusionModel) -> Result<PromptEmbeds> {
        if let Some(encoders) = &self.text_encoders {
            debug!("Encoding {} prompts with SDXL text encoders", prompts.len());
            
            // Encode with CLIP-L (768 dim)
            let (clip_l_embeds, clip_l_pooled) = encoders.clip_l.encode(prompts)?;
            
            // Encode with OpenCLIP-G (1280 dim)
            let (clip_g_embeds, clip_g_pooled) = encoders.clip_g.encode(prompts)?;
            
            // For SDXL, we typically use only the CLIP-G embeddings for the main conditioning
            // and concatenate pooled embeddings for the added conditioning
            let encoder_hidden_states = clip_g_embeds;
            
            // Concatenate pooled embeddings if available
            let pooled_projections = match (clip_l_pooled.as_ref(), clip_g_pooled.as_ref()) {
                (Some(l_pooled), Some(g_pooled)) => {
                    Some(Tensor::cat(&[l_pooled, g_pooled], 1)?)
                }
                (None, Some(g_pooled)) => Some(g_pooled.clone()),
                (Some(l_pooled), None) => Some(l_pooled.clone()),
                (None, None) => None,
            };
            
            Ok(PromptEmbeds {
                encoder_hidden_states,
                pooled_projections,
                attention_mask: None,
            })
        } else {
            Err(Error::Config("Text encoders not configured. SDXL requires CLIP-L and OpenCLIP-G encoders.".to_string()))
        }
    }
    
    fn add_noise(
        &self,
        latents: &Tensor,
        noise: &Tensor,
        timesteps: &Tensor,
    ) -> Result<Tensor> {
        // Apply noise offset first
        let noise = PipelineUtils::apply_noise_offset(noise, self.config.noise_offset)?;
        
        // Use DDPM noise addition
        self.add_noise_ddpm(latents, &noise, timesteps)
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
        // Get model inputs with SDXL-specific conditioning
        let inputs = self.get_model_inputs(noisy_latents, timesteps, prompt_embeds, batch)?;
        let output = model.forward(&inputs)?;
        let model_pred = self.postprocess_outputs(&output)?;
        
        // Determine the target based on prediction type
        let target = match self.noise_scheduler.prediction_type.as_str() {
            "epsilon" => noise.clone(),
            "v_prediction" => {
                if let Some(ref latents) = batch.latents {
                    let alpha_cumprod = self.noise_scheduler.get_alpha_cumprod(timesteps)?;
                    let sqrt_alpha_cumprod = alpha_cumprod.sqrt()?
                        .unsqueeze(1)?.unsqueeze(2)?.unsqueeze(3)?;
                    let sqrt_one_minus_alpha_cumprod = (Tensor::new(1.0f32, alpha_cumprod.device())?.sub(&alpha_cumprod)?)?.sqrt()?
                        .unsqueeze(1)?.unsqueeze(2)?.unsqueeze(3)?;
                    
                    // v = sqrt(alpha_cumprod) * noise - sqrt(1 - alpha_cumprod) * latents
                    sqrt_alpha_cumprod.broadcast_mul(noise)?
                        .sub(&sqrt_one_minus_alpha_cumprod.broadcast_mul(latents)?)?
                } else {
                    return Err(Error::Training("No latents found for v_prediction".to_string()));
                }
            }
            "sample" => {
                if let Some(ref latents) = batch.latents {
                    latents.clone()
                } else {
                    return Err(Error::Training("No latents found for sample prediction".to_string()));
                }
            }
            _ => return Err(Error::Config(format!("Unknown prediction type: {}", self.noise_scheduler.prediction_type))),
        };
        
        // Compute loss
        let loss = match self.config.loss_type.as_str() {
            "mse" => {
                let diff = model_pred.sub(&target)?;
                diff.sqr()?.mean_all()?
            }
            "mae" | "l1" => {
                let diff = model_pred.sub(&target)?;
                diff.abs()?.mean_all()?
            }
            "huber" => {
                let diff = model_pred.sub(&target)?;
                let abs_diff = diff.abs()?;
                let quadratic = abs_diff.le(1.0)?.to_dtype(DType::F32)?;
                let linear = Tensor::new(1.0f32, diff.device())?.sub(&quadratic)?;
                
                let quadratic_loss = diff.sqr()?.broadcast_mul(&quadratic)?.mul(&Tensor::new(0.5f32, diff.device())?)?;
                let linear_loss = abs_diff.sub(&Tensor::new(0.5f32, diff.device())?)?.broadcast_mul(&linear)?;
                
                quadratic_loss.add(&linear_loss)?.mean_all()?
            }
            _ => return Err(Error::Config(format!("Unknown loss type: {}", self.config.loss_type))),
        };
        
        // Apply SNR weighting if configured
        let weighted_loss = self.apply_snr_weight(&loss, timesteps)?;
        
        Ok(weighted_loss)
    }
    
    fn get_model_inputs(
        &self,
        noisy_latents: &Tensor,
        timesteps: &Tensor,
        prompt_embeds: &PromptEmbeds,
        batch: &PreparedBatch,
    ) -> Result<ModelInputs> {
        // Get additional conditioning for SDXL
        let conditioning = self.get_conditioning(batch, timesteps)?;
        
        Ok(ModelInputs {
            latents: noisy_latents.clone(),
            timestep: timesteps.clone(),
            encoder_hidden_states: Some(prompt_embeds.encoder_hidden_states.clone()),
            attention_mask: prompt_embeds.attention_mask.clone(),
            guidance_scale: Some(1.0),
            pooled_projections: prompt_embeds.pooled_projections.clone(),
            additional: conditioning,
        })
    }
    
    fn apply_snr_weight(&self, loss: &Tensor, timesteps: &Tensor) -> Result<Tensor> {
        if let Some(snr_gamma) = self.config.snr_gamma {
            let alphas_cumprod = self.noise_scheduler.alphas_cumprod.as_ref()
                .ok_or_else(|| Error::Training("Scheduler not initialized".to_string()))?;
            
            if let Some(min_snr_gamma) = self.config.min_snr_gamma {
                // Use min-SNR weighting
                PipelineUtils::apply_min_snr_weighting(loss, timesteps, alphas_cumprod, min_snr_gamma)
            } else {
                // Use standard SNR weighting
                let snr = PipelineUtils::compute_snr(timesteps, alphas_cumprod)?;
                let weights = snr.minimum(&Tensor::new(snr_gamma, snr.device())?)?
                    .div(&snr)?;
                
                loss.broadcast_mul(&weights)
                    .map_err(|e| Error::Training(format!("Failed to apply SNR weighting: {}", e)))
            }
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
        
        // TODO: Implement prior preservation for SDXL if needed
        // This would involve generating samples with class prompts
        // and computing a regularization loss
        
        Ok(None)
    }
    
    fn get_scheduler_config(&self) -> HashMap<String, serde_json::Value> {
        let mut config = HashMap::new();
        config.insert("num_train_timesteps".to_string(), 
            serde_json::json!(self.noise_scheduler.num_train_timesteps));
        config.insert("beta_start".to_string(), 
            serde_json::json!(self.noise_scheduler.beta_start));
        config.insert("beta_end".to_string(), 
            serde_json::json!(self.noise_scheduler.beta_end));
        config.insert("beta_schedule".to_string(), 
            serde_json::json!(self.noise_scheduler.beta_schedule));
        config.insert("prediction_type".to_string(), 
            serde_json::json!(self.noise_scheduler.prediction_type));
        config
    }
}
