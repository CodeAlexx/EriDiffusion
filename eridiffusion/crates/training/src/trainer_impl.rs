//! Complete trainer implementation for diffusion models

use crate::{
    TrainerConfig, TrainingState, TrainingPipeline, 
    CheckpointManager, CheckpointMetadata,
    pipelines::{SamplingConfig, TrainingSampler},
};
use eridiffusion_core::{Result, Error, Device, ModelArchitecture};
use eridiffusion_models::{DiffusionModel, VAE, TextEncoder};
use eridiffusion_core::NetworkAdapter;
use eridiffusion_data::{DataLoaderBatch, LatentBatch};
use candle_core::Tensor;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn, error};

/// Complete trainer for diffusion models
pub struct DiffusionTrainer {
    config: TrainerConfig,
    state: Arc<RwLock<TrainingState>>,
    model: Arc<dyn DiffusionModel>,
    vae: Arc<dyn VAE>,
    text_encoder: Arc<dyn TextEncoder>,
    network_adapter: Arc<RwLock<dyn NetworkAdapter>>,
    pipeline: Arc<dyn TrainingPipeline>,
    checkpoint_manager: CheckpointManager,
    sampler: TrainingSampler,
    device: Device,
}

impl DiffusionTrainer {
    /// Create new trainer
    pub fn new(
        config: TrainerConfig,
        model: Box<dyn DiffusionModel>,
        vae: Box<dyn VAE>,
        text_encoder: Box<dyn TextEncoder>,
        network_adapter: Box<dyn NetworkAdapter>,
        pipeline: Box<dyn TrainingPipeline>,
    ) -> Result<Self> {
        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
        
        // Create checkpoint manager
        let checkpoint_manager = CheckpointManager::new(
            config.output_dir.clone(),
            5, // Keep 5 latest
            config.save_total_limit,
        )?;
        
        // Create sampler
        let sampling_config = SamplingConfig {
            output_dir: config.output_dir.join("samples"),
            ..Default::default()
        };
        let sampler = TrainingSampler::new(sampling_config, device.clone());
        
        // Initialize training state
        let state = TrainingState {
            global_step: 0,
            epoch: 0,
            best_loss: f32::INFINITY,
            current_loss: 0.0,
            learning_rate: config.learning_rate,
            should_stop: false,
        };
        
        Ok(Self {
            config,
            state: Arc::new(RwLock::new(state)),
            model: Arc::new(model),
            vae: Arc::new(vae),
            text_encoder: Arc::new(text_encoder),
            network_adapter: Arc::new(RwLock::new(network_adapter)),
            pipeline: Arc::new(pipeline),
            checkpoint_manager,
            sampler,
            device,
        })
    }
    
    /// Resume from checkpoint
    pub async fn resume_from_checkpoint(&mut self, checkpoint_path: PathBuf) -> Result<()> {
        info!("Resuming from checkpoint: {:?}", checkpoint_path);
        
        let (metadata, weights) = self.checkpoint_manager.load_checkpoint(&checkpoint_path)?;
        
        // Update state
        let mut state = self.state.write().await;
        state.global_step = metadata.step;
        state.epoch = metadata.epoch;
        state.current_loss = metadata.loss;
        state.learning_rate = metadata.learning_rate;
        drop(state);
        
        // Load adapter weights
        let mut adapter = self.network_adapter.write().await;
        adapter.load_weights(&checkpoint_path.join("adapter_model.safetensors")).await?;
        
        info!("Resumed training from step {}", metadata.step);
        Ok(())
    }
    
    /// Train for one epoch
    pub async fn train_epoch(
        &mut self,
        dataloader: &mut dyn AsyncIterator<Item = Result<DataLoaderBatch>>,
        optimizer: &mut dyn crate::Optimizer,
    ) -> Result<f32> {
        let mut epoch_loss = 0.0;
        let mut batch_count = 0;
        
        while let Some(batch_result) = dataloader.next().await {
            let batch = batch_result?;
            
            // Convert to latent batch
            let latent_batch = LatentBatch::from_batch(
                batch,
                self.vae.as_ref(),
                Some(self.text_encoder.as_ref()),
            ).await?;
            
            // Training step
            let loss = self.training_step(latent_batch, optimizer).await?;
            
            epoch_loss += loss;
            batch_count += 1;
            
            // Update state
            let mut state = self.state.write().await;
            state.global_step += 1;
            state.current_loss = loss;
            
            // Logging
            if state.global_step % self.config.logging_steps == 0 {
                info!(
                    "Step {}: loss = {:.4}, lr = {:.2e}",
                    state.global_step, loss, state.learning_rate
                );
            }
            
            // Validation/Sampling
            if state.global_step % self.config.validation_steps == 0 {
                let step = state.global_step;
                drop(state);
                
                self.validation_step(step).await?;
            }
            
            // Checkpointing
            let state = self.state.read().await;
            if state.global_step % self.config.checkpointing_steps == 0 {
                let step = state.global_step;
                let epoch = state.epoch;
                let loss = state.current_loss;
                let lr = state.learning_rate;
                drop(state);
                
                self.save_checkpoint(step, epoch, loss, lr).await?;
            }
            
            // Check if should stop
            let state = self.state.read().await;
            if state.should_stop {
                info!("Stopping training at step {}", state.global_step);
                break;
            }
        }
        
        Ok(if batch_count > 0 { epoch_loss / batch_count as f32 } else { 0.0 })
    }
    
    /// Single training step
    async fn training_step(
        &self,
        batch: LatentBatch,
        optimizer: &mut dyn crate::Optimizer,
    ) -> Result<f32> {
        // Prepare batch
        let prepared_batch = self.pipeline.prepare_batch(&DataLoaderBatch::new(
            batch.latents.clone(),
            batch.captions.clone(),
            None,
            Some(batch.loss_weights.clone()),
            batch.metadata.clone(),
        ))?;
        
        // Get prompt embeddings
        let prompt_embeds = crate::pipelines::base_pipeline::PromptEmbeds {
            text_embeds: batch.text_embeds.unwrap_or_else(|| {
                panic!("Text embeddings not pre-computed")
            }),
            pooled_embeds: batch.pooled_embeds,
            text_embeds_2: None,
            pooled_embeds_2: None,
            text_embeds_3: None,
        };
        
        // Sample noise
        let noise = Tensor::randn_like(&prepared_batch.latents)?;
        
        // Sample timesteps
        let batch_size = prepared_batch.latents.dims()[0];
        let timesteps = if self.pipeline.architecture() == ModelArchitecture::SD3 || 
                        self.pipeline.architecture() == ModelArchitecture::SD35 {
            // Flow matching: sample from [0, 1]
            Tensor::rand(0.0, 1.0, &[batch_size], &self.device)?.affine(1000.0, 0.0)?
        } else {
            // Traditional diffusion: sample from [0, 1000]
            Tensor::rand(0.0, 1000.0, &[batch_size], &self.device)?
        };
        
        // Add noise
        let noisy_latents = self.pipeline.add_noise(
            &prepared_batch.latents,
            &noise,
            &timesteps,
        )?;
        
        // Forward pass and compute loss
        let loss = self.pipeline.compute_loss(
            self.model.as_ref(),
            &noisy_latents,
            &noise,
            &timesteps,
            &prompt_embeds,
            &prepared_batch,
        )?;
        
        // Backward pass
        let grads = loss.backward()?;
        
        // Get adapter parameters
        let adapter = self.network_adapter.read().await;
        let params = adapter.trainable_parameters();
        
        // Optimizer step
        let state = self.state.read().await;
        optimizer.step(&params, &grads, state.learning_rate as f64)?;
        
        Ok(loss.to_scalar::<f32>()?)
    }
    
    /// Validation step with sampling
    async fn validation_step(&self, step: usize) -> Result<()> {
        info!("Running validation at step {}", step);
        
        // Generate samples
        match self.pipeline.architecture() {
            ModelArchitecture::SD3 | ModelArchitecture::SD35 => {
                self.sampler.sample_sd3(
                    self.model.as_ref(),
                    self.vae.as_ref(),
                    self.text_encoder.as_ref(),
                    step,
                ).await?;
            }
            ModelArchitecture::SDXL => {
                self.sampler.sample_sdxl(
                    self.model.as_ref(),
                    self.vae.as_ref(),
                    self.text_encoder.as_ref(),
                    step,
                ).await?;
            }
            ModelArchitecture::PixArt | ModelArchitecture::PixArtSigma => {
                // PixArt uses similar flow matching to SD3
                self.sample_sd3(
                    prompt,
                    negative_prompt,
                    num_steps,
                    guidance_scale,
                    seed,
                    step,
                ).await?;
            }
            ModelArchitecture::AuraFlow | ModelArchitecture::Lumina => {
                // These also use flow matching
                self.sample_sd3(
                    prompt,
                    negative_prompt,
                    num_steps,
                    guidance_scale,
                    seed,
                    step,
                ).await?;
            }
            arch => {
                info!(
                    "Sampling for {:?} uses flow matching similar to SD3/Flux. \
                    Using SD3 sampling method.", 
                    arch
                );
                self.sample_sd3(
                    prompt,
                    negative_prompt,
                    num_steps,
                    guidance_scale,
                    seed,
                    step,
                ).await?;
            }
        }
        
        Ok(())
    }
    
    /// Save checkpoint
    async fn save_checkpoint(
        &self,
        step: usize,
        epoch: usize,
        loss: f32,
        learning_rate: f32,
    ) -> Result<()> {
        info!("Saving checkpoint at step {}", step);
        
        let adapter = self.network_adapter.read().await;
        
        // Create training config for metadata
        let training_config = serde_json::json!({
            "learning_rate": self.config.learning_rate,
            "batch_size": self.config.batch_size,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "mixed_precision": self.config.mixed_precision,
            "max_grad_norm": self.config.max_grad_norm,
        });
        
        let checkpoint_path = self.checkpoint_manager.save_checkpoint(
            step,
            epoch,
            loss,
            learning_rate,
            adapter.as_ref(),
            Some(&optimizer), // Save optimizer state
            &training_config,
        )?;
        
        info!("Checkpoint saved to: {:?}", checkpoint_path);
        Ok(())
    }
}

/// Async iterator trait for DataLoader
#[async_trait::async_trait]
pub trait AsyncIterator: Send {
    type Item;
    
    async fn next(&mut self) -> Option<Self::Item>;
}

/// Training state that can be shared across threads
#[derive(Debug, Clone)]
pub struct TrainingState {
    pub global_step: usize,
    pub epoch: usize,
    pub best_loss: f32,
    pub current_loss: f32,
    pub learning_rate: f32,
    pub should_stop: bool,
}