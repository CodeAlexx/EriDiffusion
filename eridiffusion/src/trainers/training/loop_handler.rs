use anyhow::Context;
use flame_core::device::Device;
use flame_core::Result;
use flame_core::{CudaDevice, DType, Tensor};
use log::{debug, info, warn};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use crate::config::trainer_config::{ProcessConfig, SampleConfig};
use crate::trainers::enhanced_data_loader::{DataBatch, EnhancedDataLoader};
use crate::trainers::gradient_accumulator::GradientAccumulator;
use crate::trainers::lora::LoRACollection;

pub struct TrainingState {
    pub global_step: usize,
    pub epoch: usize,
    pub total_loss: f32,
    pub loss_history: Vec<f32>,
    pub learning_rate: f32,
    pub best_loss: f32,
    pub start_time: Instant,
}

impl TrainingState {
    pub fn new(initial_lr: f32) -> Self {
        Self {
            global_step: 0,
            epoch: 0,
            total_loss: 0.0,
            loss_history: Vec::new(),
            learning_rate: initial_lr,
            best_loss: f32::INFINITY,
            start_time: Instant::now(),
        }
    }

    pub fn update_loss(&mut self, loss: f32) {
        self.total_loss += loss;
        self.loss_history.push(loss);

        if loss < self.best_loss {
            self.best_loss = loss;
        }
    }

    pub fn average_loss(&self) -> f32 {
        if self.loss_history.is_empty() {
            0.0
        } else {
            self.total_loss / self.loss_history.len() as f32
        }
    }

    pub fn elapsed_time(&self, device: &CudaDevice) -> std::time::Duration {
        self.start_time.elapsed()
    }
}

pub struct TrainingLoop {
    pub state: TrainingState,
    pub config: ProcessConfig,
    pub device: Device,
    pub dtype: DType,
    gradient_accumulator: GradientAccumulator,
}

impl TrainingLoop {
    pub fn new(config: ProcessConfig, device: Device, dtype: DType) -> flame_core::Result<Self> {
        let initial_lr = config.train.lr as f32;

        let gradient_accumulator =
            GradientAccumulator::new(config.train.gradient_accumulation_steps, device.clone());

        Ok(Self {
            state: TrainingState::new(initial_lr),
            config,
            device,
            dtype,
            gradient_accumulator,
        })
    }

    /// Main training step
    pub fn training_step<F>(
        &mut self,
        batch: &DataBatch,
        forward_fn: F,
        lora_collection: &LoRACollection,
        optimizer: &mut impl OptimizerTrait,
    ) -> flame_core::Result<f32>
    where
        F: Fn(&DataBatch) -> flame_core::Result<Tensor>,
    {
        // Forward pass
        let loss = forward_fn(batch)?;

        // Backward pass
        let grad_map = loss.backward()?;

        // Accumulate gradients
        // TODO: Need to iterate over parameters and accumulate each gradient
        // For now, store the grad_map for later processing
        let _ = grad_map;

        // Check if we should update
        if self.gradient_accumulator.should_update() {
            // Scale gradients by accumulation steps
            let scale = 1.0 / self.config.train.gradient_accumulation_steps as f32;

            // Apply gradients to parameters
            for param in lora_collection.parameters() {
                if let Some(grad) = grad_map.get(param.id()) {
                    let scaled_grad = grad.mul_scalar(scale as f64 as f32)?;
                    optimizer.step(param, &scaled_grad)?;
                }
            }

            // Clear accumulated gradients
            self.gradient_accumulator.step();

            // Update global step
            self.state.global_step += 1;
        }

        // Get loss value
        let loss_val = loss.to_scalar::<f32>()?;
        self.state.update_loss(loss_val);

        Ok(loss_val)
    }

    /// Run full training loop
    pub fn run<F, S>(
        &mut self,
        dataloader: &mut EnhancedDataLoader,
        forward_fn: F,
        sample_fn: S,
        lora_collection: &LoRACollection,
        optimizer: &mut impl OptimizerTrait,
        output_dir: &PathBuf,
    ) -> flame_core::Result<()>
    where
        F: Fn(&DataBatch) -> flame_core::Result<Tensor>,
        S: Fn(usize) -> flame_core::Result<()>,
    {
        let total_steps = self.config.train.steps;

        let sample_every = self.config.sample.sample_every;

        let save_every = self.config.save.save_every;

        info!("Starting training for {} steps", total_steps);

        while self.state.global_step < total_steps {
            // Get batch
            let batch = dataloader.next_batch()?;

            // Training step
            let loss = self.training_step(&batch, |b| forward_fn(b), lora_collection, optimizer)?;

            // Logging
            if self.state.global_step % 10 == 0 {
                info!(
                    "Step {}/{}, Loss: {:.6}, Avg Loss: {:.6}, LR: {:.2e}, Time: {:?}",
                    self.state.global_step,
                    total_steps,
                    loss,
                    self.state.average_loss(),
                    self.state.learning_rate,
                    self.state.elapsed_time(self.device.cuda_device())
                );
            }

            // Sampling
            if self.state.global_step % sample_every == 0 {
                debug!("Generating samples at step {}", self.state.global_step);
                sample_fn(self.state.global_step)?;
            }

            // Checkpointing
            if self.state.global_step % save_every == 0 {
                debug!("Saving checkpoint at step {}", self.state.global_step);
                self.save_checkpoint(lora_collection, optimizer, output_dir)?;
            }
        }

        // Final checkpoint
        info!("Training complete! Saving final checkpoint...");
        self.save_checkpoint(lora_collection, optimizer, output_dir)?;

        Ok(())
    }

    /// Save training checkpoint
    fn save_checkpoint(
        &self,
        lora_collection: &LoRACollection,
        optimizer: &impl OptimizerTrait,
        output_dir: &PathBuf,
    ) -> flame_core::Result<()> {
        let checkpoint_dir = output_dir.join(format!("checkpoint-{}", self.state.global_step));
        std::fs::create_dir_all(&checkpoint_dir)
            .map_err(|e| {
                flame_core::Error::InvalidOperation(format!(
                    "Failed to create directory: {}",
                    e
                ))
            })
            .map_err(|e| flame_core::Error::InvalidOperation(e.to_string()))?;

        // Save LoRA weights
        let lora_path = checkpoint_dir.join("sdxl_lora.safetensors");
        lora_collection.save_weights(&lora_path)?;

        // Save training state
        let state_path = checkpoint_dir.join("training_state.json");
        let state_json = serde_json::json!({
            "global_step": self.state.global_step,
            "epoch": self.state.epoch,
            "average_loss": self.state.average_loss(),
            "best_loss": self.state.best_loss,
            "learning_rate": self.state.learning_rate,
            "elapsed_seconds": self.state.elapsed_time(self.device.cuda_device()).as_secs(),
        });
        std::fs::write(
            state_path,
            serde_json::to_string_pretty(&state_json).map_err(|e| {
                flame_core::Error::InvalidOperation(format!("Failed to serialize JSON: {}", e))
            })?,
        )
        .map_err(|e| flame_core::Error::InvalidOperation(e.to_string()))?;

        info!("Checkpoint saved to {:?}", checkpoint_dir);
        Ok(())
    }
}

// Use the OptimizerTrait from optimization module
use super::optimization::OptimizerTrait;
