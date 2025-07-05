//! Flux Kontext training pipeline with flow matching and control conditioning

use super::{TrainingPipeline, PipelineConfig, PreparedBatch, PromptEmbeds, PipelineUtils};
use eridiffusion_core::{Result, Error, ModelArchitecture, context};
use eridiffusion_models::{DiffusionModel, ModelInputs, VAE, TextEncoder};
use eridiffusion_data::DataLoaderBatch;
use candle_core::{Tensor, DType, Device};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{info, debug};

pub struct FluxKontextPipeline {
    config: PipelineConfig,
    flow_matching: FlowMatching,
    vae: Option<Arc<dyn VAE + Send + Sync>>,
    text_encoders: Option<FluxTextEncoders>,
}

/// Text encoders for Flux (CLIP + T5)
struct FluxTextEncoders {
    clip: Arc<dyn TextEncoder + Send + Sync>,
    t5_xxl: Arc<dyn TextEncoder + Send + Sync>,
}

/// Flow matching scheduler configuration for Flux Kontext
struct FlowMatchingConfig {
    base_image_seq_len: usize,
    base_shift: f32,
    max_image_seq_len: usize,
    max_shift: f32,
    num_train_timesteps: usize,
    shift: f32,
    use_dynamic_shifting: bool,
}

impl Default for FlowMatchingConfig {
    fn default() -> Self {
        Self {
            base_image_seq_len: 256,
            base_shift: 0.5,
            max_image_seq_len: 4096,
            max_shift: 1.15,
            num_train_timesteps: 1000,
            shift: 3.0,
            use_dynamic_shifting: true,
        }
    }
}

impl FluxKontextPipeline {
    pub fn new(config: PipelineConfig) -> Result<Self> {
        let flow_config = FlowMatchingConfig::default();
        let flow_matching = FlowMatching::new(
            config.training_config.learning_rate,
            flow_config,
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
    
    /// Set text encoders for Flux (CLIP + T5)
    pub fn with_text_encoders(
        mut self,
        clip: Arc<dyn TextEncoder + Send + Sync>,
        t5_xxl: Arc<dyn TextEncoder + Send + Sync>,
    ) -> Self {
        self.text_encoders = Some(FluxTextEncoders {
            clip,
            t5_xxl,
        });
        self
    }
}

impl TrainingPipeline for FluxKontextPipeline {
    fn architecture(&self) -> ModelArchitecture {
        ModelArchitecture::FluxKontext
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
        
        // Verify latent shape for Flux (16 channels)
        let latent_channels = latents.dims()[1];
        if latent_channels != 16 {
            return Err(Error::InvalidShape(format!(
                "Expected 16 latent channels for Flux, got {}",
                latent_channels
            )));
        }
        
        // Apply input perturbation
        let latents = PipelineUtils::apply_input_perturbation(&latents, self.config.input_perturbation)?;
        
        // Handle control conditioning if present
        let control_latents = if let Some(control_images) = &batch.control_images {
            if let Some(vae) = &self.vae {
                debug!("Encoding control images to latents");
                // Control images should be normalized to [-1, 1] range
                let normalized_control = control_images.affine(2.0, -1.0)?;
                Some(vae.encode(&normalized_control)?)
            } else {
                Some(control_images.clone())
            }
        } else {
            None
        };
        
        // Concatenate control latents with main latents if present
        let combined_latents = if let Some(control) = control_latents {
            // Ensure control latents match target dimensions
            let target_h = latents.dims()[2];
            let target_w = latents.dims()[3];
            let control_resized = if control.dims()[2] != target_h || control.dims()[3] != target_w {
                // Bilinear interpolation for resizing
                resize_tensor(&control, target_h, target_w)?
            } else {
                control
            };
            
            // Concatenate along channel dimension: [batch, 32, h, w] (16 + 16)
            Tensor::cat(&[&latents, &control_resized], 1)?
        } else {
            latents
        };
        
        let mut metadata = HashMap::new();
        metadata.insert("has_control".to_string(), serde_json::Value::Bool(control_latents.is_some()));
        
        Ok(PreparedBatch {
            images: batch.images.clone(),
            latents: Some(combined_latents),
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
    
    fn encode_prompts(&self, prompts: &[String], model: &dyn DiffusionModel) -> Result<PromptEmbeds> {
        if let Some(encoders) = &self.text_encoders {
            debug!("Encoding {} prompts with Flux text encoders", prompts.len());
            
            // Encode with CLIP (768 dim)
            let (clip_embeds, pooled_clip) = encoders.clip.encode(prompts)?;
            
            // Encode with T5-XXL (4096 dim, max_length=512)
            let (t5_embeds, _) = encoders.t5_xxl.encode_with_max_length(prompts, 512)?;
            
            // For Flux, we use T5 embeddings as main encoder_hidden_states
            // and CLIP pooled embeddings as pooled_projections
            Ok(PromptEmbeds {
                encoder_hidden_states: t5_embeds,
                pooled_projections: pooled_clip,
                attention_mask: None,
            })
        } else {
            return Err(Error::Config("Text encoders not configured. Flux Kontext requires CLIP and T5-XXL encoders.".to_string()));
        }
    }
    
    fn add_noise(
        &self,
        latents: &Tensor,
        noise: &Tensor,
        timesteps: &Tensor,
    ) -> Result<Tensor> {
        // Flux uses flow matching with dynamic shifting
        let device = latents.device();
        let batch_size = latents.dims()[0];
        
        // Convert timesteps to [0, 1] range
        let t = timesteps.broadcast_div(&Tensor::new(1000.0f32, device)?)?;
        
        // Apply dynamic shifting if enabled
        let shifted_t = if self.flow_matching.config.use_dynamic_shifting {
            self.apply_dynamic_shifting(&t, latents)?
        } else {
            t
        };
        
        // Expand timesteps for broadcasting
        let t_expanded = shifted_t.unsqueeze(1)?.unsqueeze(2)?.unsqueeze(3)?;
        
        // Apply noise offset
        let noise = PipelineUtils::apply_noise_offset(noise, self.config.noise_offset)?;
        
        // Flow matching interpolation: x_t = (1 - t) * x_0 + t * noise
        let one_minus_t = Tensor::new(1.0f32, device)?.broadcast_sub(&t_expanded)?;
        let noisy = latents.mul(&one_minus_t)?.add(&noise.mul(&t_expanded)?)?;
        
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
        // Prepare model inputs with proper packing and IDs
        let inputs = self.get_model_inputs(noisy_latents, timesteps, prompt_embeds, batch)?;
        let output = model.forward(&inputs)?;
        let velocity_pred = self.postprocess_outputs(&output)?;
        
        // Extract original latents (without control conditioning)
        let original_latents = if let Some(ref latents) = batch.latents {
            let has_control = batch.metadata.get("has_control")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            
            if has_control && latents.dims()[1] == 32 {
                // Split concatenated latents to get original
                let chunks = latents.chunk(2, 1)?;
                chunks[0].clone()
            } else {
                latents.clone()
            }
        } else {
            return Err(Error::Training("No latents found in batch".to_string()));
        };
        
        // Flow matching target: velocity from data to noise
        let velocity_target = noise.sub(&original_latents)?;
        
        // Compute loss
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
        
        Ok(weighted_loss)
    }
    
    fn get_model_inputs(
        &self,
        noisy_latents: &Tensor,
        timesteps: &Tensor,
        prompt_embeds: &PromptEmbeds,
        batch: &PreparedBatch,
    ) -> Result<ModelInputs> {
        let device = noisy_latents.device();
        let batch_size = noisy_latents.dims()[0];
        
        // Handle control conditioning by splitting channels and reorganizing
        let (latent_input, has_control) = if noisy_latents.dims()[1] == 32 {
            // Split concatenated latents: [batch, 32, h, w] -> [batch, 16, h, w] + [batch, 16, h, w]
            let chunks = noisy_latents.chunk(2, 1)?;
            let main_latents = chunks[0].clone();
            let control_latents = chunks[1].clone();
            
            // Stack on batch dimension for packing: [2*batch, 16, h, w]
            let combined = Tensor::cat(&[&main_latents, &control_latents], 0)?;
            (combined, true)
        } else {
            (noisy_latents.clone(), false)
        };
        
        // Pack latents for transformer: [batch, seq_len, channels*patch_size]
        let packed_latents = self.pack_latents(&latent_input)?;
        
        // Generate image position IDs
        let img_ids = self.generate_image_ids(&latent_input, has_control)?;
        
        // Generate text position IDs
        let txt_ids = self.generate_text_ids(batch_size, &prompt_embeds.encoder_hidden_states)?;
        
        // Convert timesteps to [0, 1] range for the model
        let normalized_timesteps = timesteps.broadcast_div(&Tensor::new(1000.0f32, device)?)?;
        
        // Prepare guidance embeddings (use default scale of 1.0 for training)
        let guidance = Tensor::new(vec![1.0f32; latent_input.dims()[0]], device)?;
        
        let mut additional = HashMap::new();
        additional.insert("img_ids".to_string(), img_ids);
        additional.insert("txt_ids".to_string(), txt_ids);
        additional.insert("guidance".to_string(), guidance);
        additional.insert("latent_size".to_string(), 
            Tensor::new(packed_latents.dims()[1] / if has_control { 2 } else { 1 }, device)?);
        
        Ok(ModelInputs {
            latents: packed_latents,
            timestep: normalized_timesteps,
            encoder_hidden_states: Some(prompt_embeds.encoder_hidden_states.clone()),
            attention_mask: prompt_embeds.attention_mask.clone(),
            guidance_scale: Some(1.0),
            pooled_projections: prompt_embeds.pooled_projections.clone(),
            additional,
        })
    }
    
    fn apply_snr_weight(&self, loss: &Tensor, timesteps: &Tensor) -> Result<Tensor> {
        // Flux uses flow matching, not traditional SNR weighting
        Ok(loss.clone())
    }
    
    fn compute_prior_loss(
        &self,
        model: &dyn DiffusionModel,
        batch: &PreparedBatch,
    ) -> Result<Option<Tensor>> {
        // Flux Kontext doesn't use prior preservation
        Ok(None)
    }
    
    fn get_scheduler_config(&self) -> HashMap<String, serde_json::Value> {
        let mut config = HashMap::new();
        config.insert("base_image_seq_len".to_string(), 
            serde_json::Value::Number(self.flow_matching.config.base_image_seq_len.into()));
        config.insert("base_shift".to_string(), 
            serde_json::Value::Number(serde_json::Number::from_f64(self.flow_matching.config.base_shift as f64).unwrap()));
        config.insert("max_image_seq_len".to_string(), 
            serde_json::Value::Number(self.flow_matching.config.max_image_seq_len.into()));
        config.insert("max_shift".to_string(), 
            serde_json::Value::Number(serde_json::Number::from_f64(self.flow_matching.config.max_shift as f64).unwrap()));
        config.insert("num_train_timesteps".to_string(), 
            serde_json::Value::Number(self.flow_matching.config.num_train_timesteps.into()));
        config.insert("shift".to_string(), 
            serde_json::Value::Number(serde_json::Number::from_f64(self.flow_matching.config.shift as f64).unwrap()));
        config.insert("use_dynamic_shifting".to_string(), 
            serde_json::Value::Bool(self.flow_matching.config.use_dynamic_shifting));
        config
    }
}

impl FluxKontextPipeline {
    /// Pack latents for transformer input: [batch, channels, height, width] -> [batch, seq_len, packed_channels]
    fn pack_latents(&self, latents: &Tensor) -> Result<Tensor> {
        let (batch, channels, height, width) = latents.dims4()?;
        let patch_h = 2;
        let patch_w = 2;
        
        if height % patch_h != 0 || width % patch_w != 0 {
            return Err(Error::InvalidShape(format!(
                "Latent dimensions ({}, {}) must be divisible by patch size ({}, {})",
                height, width, patch_h, patch_w
            )));
        }
        
        // Rearrange: [batch, channels, height, width] -> [batch, seq_len, channels*patch_h*patch_w]
        let seq_h = height / patch_h;
        let seq_w = width / patch_w;
        let seq_len = seq_h * seq_w;
        let packed_channels = channels * patch_h * patch_w;
        
        let reshaped = latents
            .reshape(&[batch, channels, seq_h, patch_h, seq_w, patch_w])?
            .transpose(3, 4)?  // [batch, channels, seq_h, seq_w, patch_h, patch_w]
            .reshape(&[batch, seq_len, packed_channels])?;
        
        Ok(reshaped)
    }
    
    /// Generate image position IDs for the transformer
    fn generate_image_ids(&self, latents: &Tensor, has_control: bool) -> Result<Tensor> {
        let device = latents.device();
        let (batch_size, _, height, width) = latents.dims4()?;
        let seq_h = height / 2;
        let seq_w = width / 2;
        
        // Create base position IDs
        let mut ids = Vec::with_capacity(seq_h * seq_w * 3);
        for h in 0..seq_h {
            for w in 0..seq_w {
                ids.push(0.0f32); // type_id (0 for main image)
                ids.push(h as f32); // height_id
                ids.push(w as f32); // width_id
            }
        }
        
        let base_ids = Tensor::from_vec(ids, &[1, seq_h * seq_w, 3], device)?;
        let img_ids = base_ids.broadcast_as(&[batch_size, seq_h * seq_w, 3])?;
        
        if has_control {
            // Create control IDs (same positions but type_id = 1)
            let mut ctrl_ids_vec = Vec::with_capacity(seq_h * seq_w * 3);
            for h in 0..seq_h {
                for w in 0..seq_w {
                    ctrl_ids_vec.push(1.0f32); // type_id (1 for control)
                    ctrl_ids_vec.push(h as f32); // height_id
                    ctrl_ids_vec.push(w as f32); // width_id
                }
            }
            
            let ctrl_ids = Tensor::from_vec(ctrl_ids_vec, &[1, seq_h * seq_w, 3], device)?
                .broadcast_as(&[batch_size, seq_h * seq_w, 3])?;
            
            // Concatenate main and control IDs
            Tensor::cat(&[&img_ids, &ctrl_ids], 1)
        } else {
            Ok(img_ids)
        }
    }
    
    /// Generate text position IDs for the transformer
    fn generate_text_ids(&self, batch_size: usize, text_embeds: &Tensor) -> Result<Tensor> {
        let device = text_embeds.device();
        let seq_len = text_embeds.dims()[1];
        
        // Text IDs are zero (no positional encoding for text)
        let txt_ids = Tensor::zeros(&[batch_size, seq_len, 3], DType::F32, device)?;
        Ok(txt_ids)
    }
    
    /// Apply dynamic shifting for flow matching
    fn apply_dynamic_shifting(&self, t: &Tensor, latents: &Tensor) -> Result<Tensor> {
        let config = &self.flow_matching.config;
        let device = t.device();
        
        // Calculate image sequence length
        let (_, _, height, width) = latents.dims4()?;
        let img_seq_len = (height / 2) * (width / 2);
        
        if !config.use_dynamic_shifting {
            return Ok(t.clone());
        }
        
        // Linear interpolation between base_shift and max_shift based on sequence length
        let shift_factor = if img_seq_len <= config.base_image_seq_len {
            config.base_shift
        } else if img_seq_len >= config.max_image_seq_len {
            config.max_shift
        } else {
            let ratio = (img_seq_len - config.base_image_seq_len) as f32 / 
                       (config.max_image_seq_len - config.base_image_seq_len) as f32;
            config.base_shift + ratio * (config.max_shift - config.base_shift)
        };
        
        // Apply shifting: shifted_t = t + shift_factor * (1 - t) * t
        let one_minus_t = Tensor::new(1.0f32, device)?.broadcast_sub(t)?;
        let shift_term = t.mul(&one_minus_t)?.affine(shift_factor as f64, 0.0)?;
        let shifted_t = t.add(&shift_term)?;
        
        Ok(shifted_t)
    }
}

/// Flow matching utilities for Flux Kontext
struct FlowMatching {
    learning_rate: f32,
    config: FlowMatchingConfig,
}

impl FlowMatching {
    fn new(learning_rate: f32, config: FlowMatchingConfig) -> Result<Self> {
        Ok(Self {
            learning_rate,
            config,
        })
    }
    
    fn weight_loss(&self, loss: &Tensor, timesteps: &Tensor) -> Result<Tensor> {
        // Flow matching uses uniform weighting by default
        // Can implement additional weighting schemes if needed
        Ok(loss.clone())
    }
}

/// Helper function to resize tensor using bilinear interpolation
fn resize_tensor(tensor: &Tensor, target_h: usize, target_w: usize) -> Result<Tensor> {
    // This is a simplified version - in practice you'd want proper bilinear interpolation
    // For now, we'll use nearest neighbor by reshaping
    let (batch, channels, current_h, current_w) = tensor.dims4()?;
    
    if current_h == target_h && current_w == target_w {
        return Ok(tensor.clone());
    }
    
    // Simple nearest neighbor resize (replace with proper bilinear in production)
    let h_scale = target_h as f32 / current_h as f32;
    let w_scale = target_w as f32 / current_w as f32;
    
    // This is a placeholder - implement proper interpolation
    tensor.upsample_nearest2d(target_h, target_w)
        .map_err(|e| Error::Training(format!("Failed to resize tensor: {}", e)))
}
