use flame_core::Parameter;
/// FLAME training utilities
use flame_core::{Result, Tensor};

use std::collections::HashMap;

/// Trait for models that can be trained with FLAME
pub trait FLAMEModel {
    /// Get all trainable parameters
    fn parameters(&self) -> Vec<&Parameter>;

    /// Get named parameters for debugging
    fn named_parameters(&self) -> std::collections::HashMap<String, &Parameter>;

    /// Get parameter count
    fn num_parameters(&self) -> usize {
        self.parameters().iter().map(|p| p.shape().dims().iter().product::<usize>()).sum()
    }
}

/// Parameter collection for training
pub struct TrainingBatch {
    pub images: Tensor,
    pub prompts: Vec<String>,
    pub uncond_prompts: Vec<String>,
}

pub struct TrainingConfig {
    pub learning_rate: f32,
    pub batch_size: usize,
    pub num_epochs: usize,
    pub gradient_accumulation_steps: usize,
    pub max_grad_norm: f32,
    pub warmup_steps: usize,
    pub save_every: usize,
    pub sample_every: usize,
    pub use_ema: bool,
    pub ema_decay: f32,
    pub mixed_precision: bool,
    // Optimizer settings
    pub unet_lr: f32,
    pub text_encoder_lr: f32,
    pub vae_lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub train_text_encoder: bool,
    pub train_vae: bool,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-4,
            batch_size: 1,
            num_epochs: 100,
            gradient_accumulation_steps: 1,
            max_grad_norm: 1.0,
            warmup_steps: 500,
            save_every: 1000,
            sample_every: 500,
            use_ema: false,
            ema_decay: 0.9999,
            mixed_precision: false,
            // Optimizer defaults
            unet_lr: 1e-4,
            text_encoder_lr: 1e-5,
            vae_lr: 1e-5,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            train_text_encoder: false,
            train_vae: false,
        }
    }
}

pub struct ParameterCollection {
    params: Vec<flame_core::Tensor>,
    names: std::collections::HashMap<String, usize>,
}

impl ParameterCollection {
    pub fn new() -> Self {
        Self { params: Vec::new(), names: HashMap::new() }
    }

    pub fn add(&mut self, name: &str, param: flame_core::Tensor) {
        let idx = self.params.len();
        self.params.push(param);
        self.names.insert(name.to_string(), idx);
    }

    pub fn get(&self, name: &str) -> Option<&flame_core::Tensor> {
        self.names.get(name).map(|&idx| &self.params[idx])
    }

    pub fn parameters(&self) -> &[flame_core::Tensor] {
        &self.params
    }

    pub fn named_parameters(&self) -> std::collections::HashMap<String, &flame_core::Tensor> {
        self.names.iter().map(|(name, &idx)| (name.clone(), &self.params[idx])).collect()
    }
}

/// Multi-optimizer for different model components
pub struct MultiOptimizer {
    unet_optimizer: flame_core::optimizers::Adam,
    text_encoder_optimizer: Option<flame_core::optimizers::Adam>,
    vae_optimizer: Option<flame_core::optimizers::Adam>,
}

impl MultiOptimizer {
    pub fn new(config: &TrainingConfig) -> flame_core::Result<Self> {
        let unet_optimizer =
            flame_core::optimizers::Adam::new(flame_core::optimizers::AdamConfig {
                lr: config.unet_lr,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
                weight_decay: 0.0,
            });

        let text_encoder_optimizer = if config.train_text_encoder {
            Some(flame_core::optimizers::Adam::new(flame_core::optimizers::AdamConfig {
                lr: config.text_encoder_lr,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
                weight_decay: 0.0,
            }))
        } else {
            None
        };

        let vae_optimizer = if config.train_vae {
            Some(flame_core::optimizers::Adam::new(flame_core::optimizers::AdamConfig {
                lr: config.vae_lr,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
                weight_decay: 0.0,
            }))
        } else {
            None
        };

        Ok(Self { unet_optimizer, text_encoder_optimizer, vae_optimizer })
    }

    pub fn step(
        &mut self,
        unet_params: &mut Vec<(usize, &mut Tensor, &Tensor)>,
        text_encoder_params: Option<&mut Vec<(usize, &mut Tensor, &Tensor)>>,
        vae_params: Option<&mut Vec<(usize, &mut Tensor, &Tensor)>>,
    ) -> flame_core::Result<()> {
        // Update UNet parameters
        self.unet_optimizer.step(unet_params)?;

        // Update text encoder if training
        if let (Some(optimizer), Some(params)) =
            (&mut self.text_encoder_optimizer, text_encoder_params)
        {
            optimizer.step(params)?;
        }

        // Update VAE if training
        if let (Some(optimizer), Some(params)) = (&mut self.vae_optimizer, vae_params) {
            optimizer.step(params)?;
        }

        Ok(())
    }

    pub fn zero_grad(&mut self) {
        // self.unet_// Gradients handled by FLAME removed - handle gradients manually
        if let Some(opt) = &mut self.text_encoder_optimizer {
            // opt.zero_grad() removed - handle gradients manually
        }
        if let Some(opt) = &mut self.vae_optimizer {
            // opt.zero_grad() removed - handle gradients manually
        }
    }
}

/// Generic training loop with FLAME
pub fn train_with_flame<M, D, F>(
    model: &mut M,
    mut dataloader: D,
    optimizer: &mut flame_core::optimizers::Adam,
    config: &TrainingConfig,
    compute_loss_fn: F,
) -> flame_core::Result<Vec<f32>>
where
    M: FLAMEModel,
    D: Iterator<Item = TrainingBatch>,
    F: Fn(&M, &TrainingBatch) -> flame_core::Result<Tensor>,
{
    let mut losses = Vec::new();
    let parameters = model.parameters();

    for epoch in 0..config.num_epochs {
        println!("Epoch {}/{}", epoch + 1, config.num_epochs);

        for (step, batch) in (&mut dataloader).enumerate() {
            // Forward pass and compute loss
            let loss = compute_loss_fn(model, &batch)?;

            // Store loss value before backward
            losses.push(loss.item()?);

            // Backward pass - FLAME mutable gradients!
            let mut grad_map = loss.backward()?;

            // Gradient clipping if needed
            if config.max_grad_norm > 0.0 {
                let param_refs: Vec<&Parameter> = parameters.iter().map(|p| *p).collect();
                clip_gradients(&param_refs, config.max_grad_norm, &mut grad_map)?;
            }

            // Update parameters - convert to optimizer format
            // TODO: Fix optimizer API mismatch
            // FLAME's optimizer expects (usize, &mut Tensor, &Tensor) tuples
            // but we have Parameters and a GradientMap
            // This needs a proper solution to convert between the two APIs

            // optimizer.step(&mut param_grads)?; // TODO: Fix parameter mutability
            // // Gradients handled by FLAME removed - handle gradients manually

            // Logging
            if step % 100 == 0 {
                println!("  Step {}: Loss = {:.6}", step, losses.last().unwrap());
            }
        }
    }

    Ok(losses)
}

/// Gradient clipping utility
pub fn clip_gradients(
    parameters: &[&Parameter],
    max_norm: f32,
    grad_map: &mut flame_core::GradientMap,
) -> flame_core::Result<()> {
    // Calculate total gradient norm
    let mut total_norm = 0.0;

    for param in parameters {
        if let Some(grad) = grad_map.get(param.id()) {
            total_norm += grad.pow(2.0)?.sum()?.item()?;
        }
    }

    total_norm = total_norm.sqrt();

    // Clip if needed
    if total_norm > max_norm {
        let scale = max_norm / total_norm;

        for param in parameters {
            if let Some(grad) = grad_map.get_mut(param.id()) {
                *grad = grad.mul_scalar(scale)?;
            }
        }
    }

    Ok(())
}
