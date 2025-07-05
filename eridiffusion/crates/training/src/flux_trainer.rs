//! Flux-specific training implementation

use eridiffusion_core::{Device, Result, Error, ModelInputs, TensorExt, FluxVariant};
use eridiffusion_models::{FluxModel, TextEncoder, DiffusionModel, VAE};
use candle_core::{Tensor, DType};
use candle_nn::{VarBuilder, VarMap, Optimizer, AdamW, ParamsAdamW};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use crate::pipelines::sampling::{TrainingSampler, SamplingConfig};
use crate::optimizer::OptimizerConfig;
use crate::mixed_precision::{MixedPrecisionConfig, GradScaler};
use crate::gradient_accumulator::GradientAccumulator;
use crate::metrics_logger::MetricsLogger;
use eridiffusion_data::{DataLoader, BatchProcessor};
use serde::{Serialize, Deserialize};

/// Main Flux trainer struct
pub struct FluxTrainer {
    model: Arc<dyn DiffusionModel>,
    vae: Arc<dyn VAE>,
    text_encoder_t5: Arc<dyn TextEncoder>,
    text_encoder_clip: Arc<dyn TextEncoder>,
    optimizer: AdamW,
    gradient_accumulator: GradientAccumulator,
    var_map: VarMap,
    config: FluxTrainingConfig,
    device: Device,
    global_step: usize,
    ema_model: Option<Arc<dyn DiffusionModel>>,
    mixed_precision: Option<GradScaler>,
}

impl FluxTrainer {
    /// Create new Flux trainer
    pub fn new(
        model: Arc<dyn DiffusionModel>,
        vae: Arc<dyn VAE>,
        text_encoder_t5: Arc<dyn TextEncoder>,
        text_encoder_clip: Arc<dyn TextEncoder>,
        config: FluxTrainingConfig,
        device: Device,
    ) -> Result<Self> {
        // Initialize optimizer
        let var_map = VarMap::new();
        // TODO: Replace with actual optimizer builder when var_map is populated
        let optimizer = AdamW::new(
            var_map.all_vars(),
            ParamsAdamW {
                lr: config.learning_rate,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
                weight_decay: 0.01,
            },
        )?;
        
        // Initialize gradient accumulator
        let gradient_accumulator = if config.mixed_precision {
            GradientAccumulator::new_with_mixed_precision(
                config.gradient_accumulation_steps,
                device.clone(),
                true,
            )?
        } else {
            GradientAccumulator::new(
                config.gradient_accumulation_steps,
                device.clone(),
            )?
        };
        
        // Initialize mixed precision if enabled
        let mixed_precision = if config.mixed_precision {
            Some(GradScaler::new(MixedPrecisionConfig::default()))
        } else {
            None
        };
        
        // Clone model for EMA if enabled
        let ema_model = if config.ema_decay > 0.0 {
            Some(model.clone())
        } else {
            None
        };
        
        Ok(Self {
            model,
            vae,
            text_encoder_t5,
            text_encoder_clip,
            optimizer,
            gradient_accumulator,
            var_map,
            config,
            device,
            global_step: 0,
            ema_model,
            mixed_precision,
        })
    }
    
    /// Main training step
    pub async fn train_step(
        &mut self,
        images: &Tensor,
        captions: &[String],
        negative_prompts: &[String],
    ) -> Result<f32> {
        let batch_size = images.dim(0)?;
        
        // 1. Encode images with VAE
        let latents = self.vae.encode(images)?;
        
        // 2. Encode text with both encoders
        let (text_embeddings_t5, pooled_text_embeddings) = self.encode_text(captions).await?;
        let (neg_text_embeddings_t5, neg_pooled_text_embeddings) = 
            self.encode_text(negative_prompts).await?;
        
        // 3. Apply guidance dropout
        let (text_embeddings_t5, pooled_text_embeddings, guidance_mask) = 
            self.apply_guidance_dropout(
                &text_embeddings_t5,
                &pooled_text_embeddings,
                &neg_text_embeddings_t5,
                &neg_pooled_text_embeddings,
            )?;
        
        // 4. Sample timesteps for rectified flow
        let timesteps = self.sample_flow_timesteps(batch_size)?;
        
        // 5. Add rectified flow noise
        let (noisy_latents, velocity_targets) = 
            self.add_rectified_flow_noise(&latents, &timesteps)?;
        
        // 6. Prepare model inputs
        let model_inputs = ModelInputs {
            latents: noisy_latents,
            timestep: timesteps.clone(),
            encoder_hidden_states: Some(text_embeddings_t5),
            pooled_projections: Some(pooled_text_embeddings),
            guidance_scale: Some(self.config.guidance_scale),
            attention_mask: None,
            additional: HashMap::new(),
        };
        
        // 7. Forward pass through model
        let model_output = if self.config.gradient_checkpointing {
            // Use gradient checkpointing for memory efficiency
            self.model.forward_with_gradient_checkpointing(&model_inputs)?
        } else {
            self.model.forward(&model_inputs)?
        };
        
        // 8. Compute rectified flow loss
        let loss_weights = self.compute_loss_weights(&timesteps)?;
        let loss = self.compute_rectified_flow_loss(
            &model_output.sample,
            &velocity_targets,
            &loss_weights,
            &guidance_mask,
        )?;
        
        // 9. Backward pass and optimization
        let loss_value = loss.to_scalar::<f32>()?;
        self.backward_and_optimize(loss)?;
        
        // 10. Update EMA model if enabled
        if self.ema_model.is_some() {
            self.update_ema()?;
        }
        
        self.global_step += 1;
        
        Ok(loss_value)
    }
    
    /// Prepare inputs for Flux training
    pub fn prepare_training_inputs(
        latents: &Tensor,
        text_embeddings: &Tensor,
        pooled_embeddings: &Tensor,
        timesteps: &Tensor,
        guidance_scale: Option<f32>,
    ) -> Result<ModelInputs> {
        // Ensure proper shapes
        if latents.rank() != 4 {
            return Err(Error::Training(format!(
                "Expected 4D latents, got {}D", latents.rank()
            )));
        }
        
        if text_embeddings.rank() != 3 {
            return Err(Error::Training(format!(
                "Expected 3D text embeddings, got {}D", text_embeddings.rank()
            )));
        }
        
        // Create ModelInputs
        let mut inputs = ModelInputs {
            latents: latents.clone(),
            timestep: timesteps.clone(),
            encoder_hidden_states: Some(text_embeddings.clone()),
            pooled_projections: Some(pooled_embeddings.clone()),
            guidance_scale: guidance_scale,
            attention_mask: None,
            additional: std::collections::HashMap::new(),
        };
        
        Ok(inputs)
    }
    
    /// Encode text with both T5 and CLIP encoders
    async fn encode_text(&self, prompts: &[String]) -> Result<(Tensor, Tensor)> {
        // Encode with T5
        let (t5_embeddings, _) = self.text_encoder_t5.encode(prompts)?;
        
        // Encode with CLIP and get pooled output
        let (_, pooled_embeddings) = self.text_encoder_clip.encode(prompts)?;
        let pooled_embeddings = pooled_embeddings
            .ok_or_else(|| Error::Training("CLIP pooled output not found".to_string()))?;
        
        Ok((t5_embeddings, pooled_embeddings))
    }
    
    /// Apply guidance dropout for classifier-free guidance
    fn apply_guidance_dropout(
        &self,
        text_embeddings: &Tensor,
        pooled_embeddings: &Tensor,
        neg_text_embeddings: &Tensor,
        neg_pooled_embeddings: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let batch_size = text_embeddings.dim(0)?;
        let device = self.to_candle_device()?;
        
        // Create dropout mask
        let dropout_mask = Tensor::rand(0., 1., &[batch_size], &device)?
            .lt(&Tensor::new(&[self.config.text_drop_prob], &device)?)?;
        
        // Apply dropout: use negative embeddings where mask is true
        let text_out = dropout_mask
            .unsqueeze(1)?
            .unsqueeze(2)?
            .broadcast_as(text_embeddings.shape())?
            .where_cond(neg_text_embeddings, text_embeddings)?;
            
        let pooled_out = dropout_mask
            .unsqueeze(1)?
            .broadcast_as(pooled_embeddings.shape())?
            .where_cond(neg_pooled_embeddings, pooled_embeddings)?;
        
        Ok((text_out, pooled_out, dropout_mask))
    }
    
    /// Sample timesteps for rectified flow training
    fn sample_flow_timesteps(&self, batch_size: usize) -> Result<Tensor> {
        let device = self.to_candle_device()?;
        // Sample uniform timesteps from 0 to 1 for rectified flow
        Tensor::rand(0., 1., (batch_size,), &device)
            .map_err(|e| Error::TensorOp(e))
    }
    
    /// Add rectified flow noise to latents
    fn add_rectified_flow_noise(
        &self,
        latents: &Tensor,
        timesteps: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let device = self.to_candle_device()?;
        let shape = latents.shape();
        
        // Sample noise
        let noise = Tensor::randn(0., 1., shape.dims(), &device)?;
        
        // Reshape timesteps for broadcasting
        let t = timesteps
            .unsqueeze(1)?
            .unsqueeze(2)?
            .unsqueeze(3)?;
        
        // Apply flow shift for better training stability
        let t_shifted = t.affine(1.0, self.config.flow_shift as f64)?
            .div(&Tensor::new(&[1.0 + self.config.flow_shift as f64], &device)?)?;
        
        // Interpolate: x_t = t * x_1 + (1 - t) * x_0
        // where x_1 is the data (latents) and x_0 is noise
        let noisy_latents = t_shifted
            .broadcast_as(shape)?
            .mul(latents)?
            .add(&t_shifted.neg()?.add_scalar(1.0)?.broadcast_as(shape)?.mul(&noise)?)?;
        
        // Velocity target: v = x_1 - x_0 = latents - noise
        let velocity_targets = latents.sub(&noise)?;
        
        Ok((noisy_latents, velocity_targets))
    }
    
    /// Compute loss weights with Min-SNR weighting
    fn compute_loss_weights(&self, timesteps: &Tensor) -> Result<Tensor> {
        let device = self.to_candle_device()?;
        
        // For rectified flow, we use a simplified Min-SNR weighting
        // SNR approximation: SNR(t) ≈ t / (1 - t)
        let t = timesteps;
        let snr = t.div(&t.neg()?.add_scalar(1.0)?)?;
        
        // Apply Min-SNR clipping
        let weights = snr.minimum(&Tensor::new(&[self.config.min_snr_gamma], &device)?)?;
        
        // Reshape for broadcasting
        let weights = weights
            .unsqueeze(1)?
            .unsqueeze(2)?
            .unsqueeze(3)?;
        
        Ok(weights)
    }
    
    /// Compute rectified flow loss
    fn compute_rectified_flow_loss(
        &self,
        model_output: &Tensor,
        velocity_targets: &Tensor,
        loss_weights: &Tensor,
        guidance_mask: &Tensor,
    ) -> Result<Tensor> {
        // MSE between predicted and target velocities
        let diff = model_output.sub(velocity_targets)?;
        let squared_diff = diff.sqr()?;
        
        // Apply loss weights
        let weighted_loss = squared_diff.mul(loss_weights)?;
        
        // Mean over all dimensions
        let loss = weighted_loss.mean_all()?;
        
        Ok(loss)
    }
    
    /// Backward pass with gradient accumulation
    fn backward_and_optimize(&mut self, loss: Tensor) -> Result<()> {
        // Scale loss for gradient accumulation
        let scaled_loss = loss.affine(
            1.0 / self.config.gradient_accumulation_steps as f64,
            0.0,
        )?;
        
        // Accumulate gradients
        self.gradient_accumulator.accumulate(&scaled_loss)?;
        
        // Step optimizer if ready
        if self.gradient_accumulator.step_optimizer(
            &self.var_map,
            &mut self.optimizer,
            Some(self.config.max_grad_norm),
        )? {
            // Update learning rate with warmup
            self.update_learning_rate()?;
        }
        
        Ok(())
    }
    
    /// Update learning rate with linear warmup and cosine decay
    fn update_learning_rate(&mut self) -> Result<()> {
        let current_lr = if self.global_step < self.config.warmup_steps {
            // Linear warmup
            self.config.learning_rate * (self.global_step as f64 / self.config.warmup_steps as f64)
        } else {
            // Cosine decay
            let progress = (self.global_step - self.config.warmup_steps) as f64
                / (self.config.max_steps - self.config.warmup_steps) as f64;
            let cosine_decay = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
            self.config.learning_rate * cosine_decay
        };
        
        self.optimizer.set_learning_rate(current_lr);
        Ok(())
    }
    
    /// Update EMA model weights
    fn update_ema(&mut self) -> Result<()> {
        // EMA update: ema_param = decay * ema_param + (1 - decay) * param
        // This is a simplified version - actual implementation would iterate
        // through all parameters and update them
        // TODO: Implement proper EMA update when we have access to model parameters
        Ok(())
    }
    
    /// Convert device to candle device
    fn to_candle_device(&self) -> Result<candle_core::Device> {
        match &self.device {
            Device::Cpu => Ok(candle_core::Device::Cpu),
            Device::Cuda(id) => candle_core::Device::new_cuda(*id)
                .map_err(|e| Error::Device(format!("Failed to create CUDA device: {}", e))),
        }
    }
    
    /// Save checkpoint
    pub fn save_checkpoint(&self, path: &Path) -> Result<()> {
        println!("Saving checkpoint to {:?}...", path);
        
        // Create checkpoint directory
        std::fs::create_dir_all(path)?;
        
        // Save model weights
        let model_path = path.join("model.safetensors");
        // TODO: Implement model state dict saving
        // self.model.save_pretrained(&model_path)?;
        
        // Save optimizer state
        let optimizer_path = path.join("optimizer.safetensors");
        // TODO: Implement optimizer state saving
        // self.optimizer.save(&optimizer_path)?;
        
        // Save training state
        let state = serde_json::json!({
            "global_step": self.global_step,
            "config": self.config,
        });
        let state_path = path.join("training_state.json");
        std::fs::write(&state_path, serde_json::to_string_pretty(&state)?)?;
        
        // Save EMA model if enabled
        if self.ema_model.is_some() {
            let ema_path = path.join("ema_model.safetensors");
            // TODO: Implement EMA model saving
            // self.ema_model.save_pretrained(&ema_path)?;
        }
        
        Ok(())
    }
    
    /// Load checkpoint
    pub fn load_checkpoint(&mut self, path: &Path) -> Result<()> {
        println!("Loading checkpoint from {:?}...", path);
        
        // Load training state
        let state_path = path.join("training_state.json");
        if state_path.exists() {
            let state_str = std::fs::read_to_string(&state_path)?;
            let state: serde_json::Value = serde_json::from_str(&state_str)?;
            self.global_step = state["global_step"].as_u64().unwrap_or(0) as usize;
        }
        
        // Load model weights
        let model_path = path.join("model.safetensors");
        if model_path.exists() {
            // TODO: Implement model state dict loading
            // self.model.load_pretrained(&model_path)?;
        }
        
        // Load optimizer state
        let optimizer_path = path.join("optimizer.safetensors");
        if optimizer_path.exists() {
            // TODO: Implement optimizer state loading
            // self.optimizer.load(&optimizer_path)?;
        }
        
        // Load EMA model if exists
        let ema_path = path.join("ema_model.safetensors");
        if ema_path.exists() && self.ema_model.is_some() {
            // TODO: Implement EMA model loading
            // self.ema_model.load_pretrained(&ema_path)?;
        }
        
        Ok(())
    }
    
    /// Generate samples during training
    pub async fn generate_samples(
        model: &dyn DiffusionModel,
        vae: &dyn VAE,
        text_encoder: &dyn TextEncoder,
        config: &FluxTrainingConfig,
        device: &Device,
        step: usize,
        output_dir: &Path,
    ) -> Result<Vec<PathBuf>> {
        // Create sampling configuration
        let sampling_config = SamplingConfig {
            num_inference_steps: config.num_inference_steps,
            guidance_scale: config.guidance_scale,
            eta: 0.0,
            generator_seed: Some(42), // Fixed seed for reproducibility
            output_dir: output_dir.to_path_buf(),
            sample_prompts: vec![
                "a majestic mountain landscape with snow-capped peaks at sunset".to_string(),
                "a futuristic city with flying cars and neon lights at night".to_string(),
                "a serene Japanese garden with cherry blossoms and a koi pond".to_string(),
                "a steampunk mechanical dragon breathing fire".to_string(),
                "an astronaut exploring an alien planet with bioluminescent plants".to_string(),
            ],
            negative_prompt: None, // Flux doesn't use negative prompts
            height: 1024,
            width: 1024,
        };
        
        // Create sampler
        let sampler = TrainingSampler::new(sampling_config, device.clone());
        
        // Generate samples using the Flux sampling method
        sampler.sample_flux(model, vae, text_encoder, step).await
    }
}

/// Flux training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FluxTrainingConfig {
    /// Learning rate
    pub learning_rate: f64,
    
    /// Warmup steps
    pub warmup_steps: usize,
    
    /// Total training steps
    pub max_steps: usize,
    
    /// Gradient accumulation steps
    pub gradient_accumulation_steps: usize,
    
    /// Whether to use gradient checkpointing
    pub gradient_checkpointing: bool,
    
    /// Whether to train text encoders
    pub train_text_encoder: bool,
    
    /// Text encoder learning rate multiplier
    pub text_encoder_lr_multiplier: f32,
    
    /// Guidance scale for training
    pub guidance_scale: f32,
    
    /// Probability of dropping text for CFG training
    pub text_drop_prob: f32,
    
    /// Min-SNR gamma for loss weighting
    pub min_snr_gamma: f32,
    
    /// EMA decay rate
    pub ema_decay: f32,
    
    /// Rectified flow shift parameter
    pub flow_shift: f32,
    
    /// Number of inference steps for sampling
    pub num_inference_steps: usize,
    
    /// Whether to use mixed precision
    pub mixed_precision: bool,
    
    /// Maximum gradient norm for clipping
    pub max_grad_norm: f64,
}

impl Default for FluxTrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-4,
            warmup_steps: 1000,
            max_steps: 100000,
            gradient_accumulation_steps: 4,
            gradient_checkpointing: true,
            train_text_encoder: false,
            text_encoder_lr_multiplier: 0.1,
            guidance_scale: 3.5,
            text_drop_prob: 0.1,
            min_snr_gamma: 5.0,
            ema_decay: 0.9999,
            flow_shift: 1.0,
            num_inference_steps: 28,
            mixed_precision: true,
            max_grad_norm: 1.0,
        }
    }
}

/// Create Flux trainer from config
pub async fn create_flux_trainer(
    model_path: &Path,
    vae_path: &Path,
    t5_path: &Path,
    clip_path: &Path,
    t5_tokenizer_path: &Path,
    clip_tokenizer_path: &Path,
    variant: FluxVariant,
    config: FluxTrainingConfig,
    device: Device,
) -> Result<FluxTrainer> {
    use crate::flux_model_loader::{load_flux_model, load_flux_vae, load_t5_encoder, load_clip_encoder};
    
    println!("Loading Flux model components...");
    
    // Load models
    let model = load_flux_model(model_path, variant, &device).await?;
    let vae = load_flux_vae(vae_path, &device).await?;
    let t5_encoder = load_t5_encoder(t5_path, t5_tokenizer_path, &device).await?;
    let clip_encoder = load_clip_encoder(clip_path, clip_tokenizer_path, &device).await?;
    
    // Create trainer
    FluxTrainer::new(
        Arc::from(model),
        Arc::from(vae),
        Arc::from(t5_encoder),
        Arc::from(clip_encoder),
        config,
        device,
    )
}