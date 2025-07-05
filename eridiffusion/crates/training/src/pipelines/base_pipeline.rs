//! Base training pipeline trait and configuration

use eridiffusion_core::{Result, Error, Device, ModelArchitecture, context};
use eridiffusion_models::{DiffusionModel, ModelInputs, ModelOutput};
use eridiffusion_core::NetworkAdapter;
use eridiffusion_data::DataLoaderBatch;
use crate::{loss::Loss, optimizer::Optimizer, TrainingConfig};
use candle_core::{Tensor, DType};
use std::collections::HashMap;

/// Prepared batch data
#[derive(Debug, Clone)]
pub struct PreparedBatch {
    pub images: Tensor,
    pub latents: Option<Tensor>,
    pub captions: Vec<String>,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Text prompt embeddings
#[derive(Debug, Clone)]
pub struct PromptEmbeds {
    pub encoder_hidden_states: Tensor,
    pub pooled_projections: Option<Tensor>,
    pub attention_mask: Option<Tensor>,
}

/// Pipeline utilities
pub struct PipelineUtils;

/// Pipeline-specific configuration
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub training_config: TrainingConfig,
    pub device: Device,
    pub dtype: DType,
    
    // Model-specific settings
    pub use_ema: bool,
    pub ema_decay: f32,
    pub gradient_checkpointing: bool,
    
    // Loss settings
    pub loss_type: String,
    pub snr_gamma: Option<f32>,
    pub min_snr_gamma: Option<f32>,
    pub v_parameterization: bool,
    pub flow_matching: bool,
    
    // Noise settings
    pub noise_offset: f32,
    pub input_perturbation: f32,
    
    // Advanced features
    pub prior_preservation: bool,
    pub prior_loss_weight: f32,
    pub diff_output_preservation: bool,
    pub turbo_training: bool,
    pub mean_flow_loss: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            training_config: TrainingConfig::default(),
            device: Device::Cpu,
            dtype: DType::F32,
            use_ema: false,
            ema_decay: 0.9999,
            gradient_checkpointing: false,
            loss_type: "mse".to_string(),
            snr_gamma: None,
            min_snr_gamma: None,
            v_parameterization: false,
            flow_matching: false,
            noise_offset: 0.0,
            input_perturbation: 0.0,
            prior_preservation: false,
            prior_loss_weight: 1.0,
            diff_output_preservation: false,
            turbo_training: false,
            mean_flow_loss: false,
        }
    }
}

/// Base trait for model-specific training pipelines
pub trait TrainingPipeline: Send + Sync {
    /// Get the model architecture this pipeline handles
    fn architecture(&self) -> ModelArchitecture;
    
    /// Prepare a batch for training (model-specific preprocessing)
    fn prepare_batch(&self, batch: &DataLoaderBatch) -> Result<PreparedBatch>;
    
    /// Encode prompts into embeddings (model-specific)
    fn encode_prompts(&self, prompts: &[String], model: &dyn DiffusionModel) -> Result<PromptEmbeds>;
    
    /// Encode images to latents (model-specific)
    fn encode_images(&self, images: &Tensor) -> Result<Tensor>;
    
    /// Add noise to latents (handles both standard and flow matching)
    fn add_noise(
        &self,
        latents: &Tensor,
        noise: &Tensor,
        timesteps: &Tensor,
    ) -> Result<Tensor>;
    
    /// Compute the training loss
    fn compute_loss(
        &self,
        model: &dyn DiffusionModel,
        noisy_latents: &Tensor,
        noise: &Tensor,
        timesteps: &Tensor,
        prompt_embeds: &PromptEmbeds,
        batch: &PreparedBatch,
    ) -> Result<Tensor>;
    
    /// Get model inputs for forward pass
    fn get_model_inputs(
        &self,
        noisy_latents: &Tensor,
        timesteps: &Tensor,
        prompt_embeds: &PromptEmbeds,
        batch: &PreparedBatch,
    ) -> Result<ModelInputs>;
    
    /// Post-process model outputs if needed
    fn postprocess_outputs(&self, outputs: &ModelOutput) -> Result<Tensor> {
        Ok(outputs.sample.clone())
    }
    
    /// Apply SNR weighting if configured
    fn apply_snr_weight(&self, loss: &Tensor, timesteps: &Tensor) -> Result<Tensor>;
    
    /// Handle prior preservation loss if enabled
    fn compute_prior_loss(
        &self,
        model: &dyn DiffusionModel,
        batch: &PreparedBatch,
    ) -> Result<Option<Tensor>>;
    
    /// Get noise scheduler configuration
    fn get_scheduler_config(&self) -> HashMap<String, serde_json::Value> {
        HashMap::new()
    }
}

// PreparedBatch and PromptEmbeds already defined above

impl PromptEmbeds {
    pub fn new(encoder_hidden_states: Tensor) -> Self {
        Self {
            encoder_hidden_states,
            pooled_projections: None,
            attention_mask: None,
        }
    }
    
    pub fn with_pooled(encoder_hidden_states: Tensor, pooled_projections: Tensor) -> Self {
        Self {
            encoder_hidden_states,
            pooled_projections: Some(pooled_projections),
            attention_mask: None,
        }
    }
    
    pub fn with_mask(mut self, mask: Tensor) -> Self {
        self.attention_mask = Some(mask);
        self
    }
    
    pub fn concat(&self, other: &PromptEmbeds) -> Result<PromptEmbeds> {
        let encoder_hidden_states = Tensor::cat(&[&self.encoder_hidden_states, &other.encoder_hidden_states], 0)
            .map_err(|e| Error::Training(format!("Failed to concat encoder hidden states: {}", e)))?;
        
        let pooled_projections = match (&self.pooled_projections, &other.pooled_projections) {
            (Some(a), Some(b)) => Some(Tensor::cat(&[a, b], 0)?),
            _ => None,
        };
        
        let attention_mask = match (&self.attention_mask, &other.attention_mask) {
            (Some(a), Some(b)) => Some(Tensor::cat(&[a, b], 0)?),
            _ => None,
        };
        
        Ok(PromptEmbeds {
            encoder_hidden_states,
            pooled_projections,
            attention_mask,
        })
    }
}

// PipelineUtils already defined above

impl PipelineUtils {
    /// Apply noise offset augmentation
    pub fn apply_noise_offset(noise: &Tensor, offset: f32) -> Result<Tensor> {
        if offset == 0.0 {
            return Ok(noise.clone());
        }
        
        let offset_noise = Tensor::randn_like(noise, 0.0, 1.0)?;
        let scaled_offset = offset_noise.affine(offset as f64, 0.0)?;
        noise.add(&scaled_offset)
            .map_err(|e| Error::Training(e.to_string()))
    }
    
    /// Apply input perturbation
    pub fn apply_input_perturbation(latents: &Tensor, perturbation: f32) -> Result<Tensor> {
        if perturbation == 0.0 {
            return Ok(latents.clone());
        }
        
        let noise = Tensor::randn_like(latents, 0.0, 1.0)?;
        let perturbed = latents.add(&noise.affine(perturbation as f64, 0.0)?)?;
        Ok(perturbed)
    }
    
    /// Get timestep embeddings
    pub fn get_timestep_embedding(
        timesteps: &Tensor,
        dim: usize,
        max_period: f32,
    ) -> Result<Tensor> {
        let half_dim = dim / 2;
        let freqs = (0..half_dim)
            .map(|i| {
                let exponent = -(i as f32) / half_dim as f32;
                (exponent * max_period.ln()).exp()
            })
            .collect::<Vec<_>>();
        
        let freqs_tensor = Tensor::new(freqs.as_slice(), timesteps.device())?;
        let angles = timesteps.unsqueeze(1)?.matmul(&freqs_tensor.unsqueeze(0)?)?;
        
        let sin = angles.sin()?;
        let cos = angles.cos()?;
        
        Tensor::cat(&[&sin, &cos], 1)
            .map_err(|e| Error::Training(e.to_string()))
    }
}