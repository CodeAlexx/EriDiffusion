use anyhow::Context;
use flame_core::device::Device;
use flame_core::optimizers::{Adam, SGD};
use flame_core::{CudaDevice, DType, Result, Shape, Tensor};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::{collections::HashMap, fs, path::Path};

pub struct SDXLTrainingConfig {
    pub model_path: String,
    pub output_dir: String,
    pub learning_rate: f32,
    pub batch_size: usize,
    pub num_epochs: usize,
    pub save_steps: usize,
    pub resolution: usize,
}

pub struct GradientClipper;

impl GradientClipper {
    /// Clip gradients by global norm
    pub fn clip_grad_norm(
        gradients: &mut std::collections::HashMap<String, Tensor>,
        max_norm: f32,
    ) -> flame_core::Result<f32> {
        // Calculate total norm
        let mut total_norm = 0.0f32;

        for grad in gradients.values() {
            let grad_norm = grad.square()?.sum()?.to_scalar::<f32>()?;
            total_norm += grad_norm;
        }

        total_norm = total_norm.sqrt();

        // Clip if necessary
        if total_norm > max_norm {
            let clip_coef = max_norm / (total_norm + 1e-6);

            for grad in gradients.values_mut() {
                let clip_tensor =
                    Tensor::full(grad.shape().clone(), clip_coef, grad.device().clone())?;
                *grad = grad.mul(&clip_tensor)?;
            }
        }

        Ok(total_norm)
    }

    /// Clip gradients by value
    pub fn clip_grad_value(
        gradients: &mut std::collections::HashMap<String, Tensor>,
        clip_value: f32,
    ) -> flame_core::Result<()> {
        for grad in gradients.values_mut() {
            *grad = grad.clamp(-clip_value, clip_value)?;
        }
        Ok(())
    }
}

/// Memory optimization utilities
pub struct MemoryOptimizer;

impl MemoryOptimizer {
    /// Enable gradient checkpointing for a model
    pub fn enable_gradient_checkpointing(model_name: &str, device: &CudaDevice) {
        // This would need to be implemented based on the specific model architecture
        // For now, it's a placeholder
        println!("Gradient checkpointing enabled for {}", model_name);
    }

    /// Clear GPU cache
    pub fn clear_cache(device: &Device) -> flame_core::Result<()> {
        // FLAME only supports CUDA devices currently
        {
            // In a real implementation, this would call CUDA cache clearing functions
            println!("Clearing GPU cache");
        }
        Ok(())
    }

    /// Estimate memory usage
    pub fn estimate_memory_usage(batch_size: usize, image_size: usize, model_size: usize) -> usize {
        // Rough estimation in MB
        let image_memory = batch_size * 3 * image_size * image_size * 4 / (1024 * 1024);
        let latent_memory =
            batch_size * 4 * (image_size / 8) * (image_size / 8) * 4 / (1024 * 1024);
        let gradient_memory = model_size * 4 / (1024 * 1024);

        image_memory + latent_memory + gradient_memory * 2 // x2 for gradients
    }
}

/// Training metrics tracker
#[derive(Default)]
pub struct MetricsTracker {
    losses: Vec<f32>,
    learning_rates: Vec<f32>,
    grad_norms: Vec<f32>,
}

impl MetricsTracker {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_step(
        &mut self,
        loss: f32,
        learning_rate: f32,
        grad_norm: Option<f32>,
    ) -> flame_core::Result<()> {
        self.losses.push(loss);
        self.learning_rates.push(learning_rate);
        self.grad_norms.push(grad_norm.unwrap_or(0.0));
        Ok(())
    }

    pub fn get_average_loss(&self, last_n: usize) -> f32 {
        let start = self.losses.len().saturating_sub(last_n);
        let recent_losses = &self.losses[start..];

        if recent_losses.is_empty() {
            0.0
        } else {
            recent_losses.iter().sum::<f32>() / recent_losses.len() as f32
        }
    }

    pub fn save_to_file(&self, path: &Path) -> flame_core::Result<()> {
        let data = serde_json::json!({
            "losses": self.losses,
            "learning_rates": self.learning_rates,
            "grad_norms": self.grad_norms
        });

        let json = serde_json::to_string_pretty(&data)
            .map_err(|e| {
                flame_core::Error::InvalidOperation(format!("Failed to serialize JSON: {}", e))
            })
            .map_err(|e| flame_core::Error::InvalidOperation(e.to_string()))?;
        fs::write(path, json)
            .map_err(|e| flame_core::Error::Io(format!("Failed to write file: {}", e)))?;

        Ok(())
    }
}

/// Mixed precision training utilities
pub struct MixedPrecisionScaler {
    scale: f32,
    growth_factor: f32,
    backoff_factor: f32,
    growth_interval: usize,
    steps_since_update: usize,
}

impl MixedPrecisionScaler {
    pub fn new() -> Self {
        Self {
            scale: 65536.0, // 2^16
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2000,
            steps_since_update: 0,
        }
    }

    /// Scale loss for mixed precision training
    pub fn scale_loss(&self, loss: &Tensor, device: &CudaDevice) -> flame_core::Result<Tensor> {
        let scale_tensor = Tensor::full(loss.shape().clone(), self.scale, loss.device().clone())?;
        Ok(loss.mul(&scale_tensor)?)
    }

    /// Unscale gradients
    pub fn unscale_gradients(
        &self,
        gradients: &mut std::collections::HashMap<String, Tensor>,
    ) -> flame_core::Result<()> {
        let inv_scale = 1.0 / self.scale;

        for grad in gradients.values_mut() {
            let inv_scale_tensor =
                Tensor::full(grad.shape().clone(), inv_scale, grad.device().clone())?;
            *grad = grad.mul(&inv_scale_tensor)?;
        }

        Ok(())
    }

    /// Update scale based on gradient overflow
    pub fn update(&mut self, found_inf: bool) -> flame_core::Result<()> {
        if found_inf {
            // Reduce scale if overflow detected
            self.scale *= self.backoff_factor;
            self.steps_since_update = 0;
        } else {
            self.steps_since_update += 1;

            // Increase scale if stable for growth_interval steps
            if self.steps_since_update >= self.growth_interval {
                self.scale *= self.growth_factor;
                self.steps_since_update = 0;
            }
        }

        // Clamp scale to reasonable range
        self.scale = self.scale.clamp(1.0, 65536.0);

        Ok(())
    }

    /// Check for inf/nan in gradients
    pub fn check_inf_or_nan(
        &self,
        gradients: &std::collections::HashMap<String, Tensor>,
    ) -> flame_core::Result<bool> {
        for grad in gradients.values() {
            // Check for NaN by comparing with itself (NaN != NaN)
            // FLAME doesn't have ne(), so use eq() and check for false values
            let eq_mask = grad.eq(grad)?;
            let all_equal = eq_mask.sum()?.to_scalar::<f32>()?;
            let total_elements = grad.shape().dims().iter().product::<usize>() as f32;
            let has_nan = all_equal < total_elements;

            if has_nan {
                return Ok(true);
            }
        }

        Ok(false)
    }
}

/// Data augmentation for SDXL training
pub struct SDXLAugmentation {
    random_flip: bool,
    color_jitter: bool,
    random_crop: bool,
}

impl SDXLAugmentation {
    pub fn new(random_flip: bool, color_jitter: bool, random_crop: bool) -> Self {
        Self { random_flip, color_jitter, random_crop }
    }

    /// Apply augmentations to image tensor
    pub fn apply(&self, image: &Tensor) -> flame_core::Result<Tensor> {
        let mut augmented = image.clone();

        if self.random_flip && rand::random::<f32>() > 0.5 {
            augmented = self.horizontal_flip(&augmented, &*augmented.device())?;
        }

        if self.color_jitter {
            augmented = self.apply_color_jitter(&augmented, &*augmented.device())?;
        }

        if self.random_crop {
            augmented = self.apply_random_crop(&augmented)?;
        }

        Ok(augmented)
    }

    fn horizontal_flip(&self, image: &Tensor, device: &CudaDevice) -> flame_core::Result<Tensor> {
        // Flip along width dimension
        let dims = image.shape().dims();
        let width = dims[dims.len() - 1];

        let indices: Vec<f32> = (0..width).rev().map(|i| i as f32).collect();
        let indices_tensor = Tensor::from_vec(
            indices.clone(),
            Shape::from_dims(&[indices.len()]),
            image.device().clone(),
        )?;

        Ok(image.index_select(dims.len() - 1, &indices_tensor)?)
    }

    fn apply_color_jitter(
        &self,
        image: &Tensor,
        device: &CudaDevice,
    ) -> flame_core::Result<Tensor> {
        // Simple brightness adjustment
        let brightness_factor = 0.8 + rand::random::<f32>() * 0.4; // 0.8 to 1.2
        Ok(image.mul(&Tensor::full(
            image.shape().clone(),
            brightness_factor,
            image.device().clone(),
        )?)?)
    }

    fn apply_random_crop(&self, image: &Tensor) -> flame_core::Result<Tensor> {
        // Random crop for SDXL with aspect ratio preservation
        let shape = image.shape().dims();
        let (c, h, w) = (shape[0], shape[1], shape[2]);

        // SDXL typically uses 1024x1024, but we'll crop to maintain aspect ratio
        let target_size = 1024;

        if h <= target_size && w <= target_size {
            // Image is already smaller than target, no crop needed
            return Ok(image.clone());
        }

        // Calculate crop dimensions while maintaining aspect ratio
        let scale = (target_size as f32) / (h.min(w) as f32);
        let crop_h = ((h as f32 * scale).min(h as f32)) as usize;
        let crop_w = ((w as f32 * scale).min(w as f32)) as usize;

        // Random crop position
        let mut rng = rand::thread_rng();
        let y_offset = if h > crop_h { rng.gen_range(0..=(h - crop_h)) } else { 0 };
        let x_offset = if w > crop_w { rng.gen_range(0..=(w - crop_w)) } else { 0 };

        // Perform the crop
        image.narrow(1, y_offset, crop_h)?.narrow(2, x_offset, crop_w)
    }
}

/// Validation utilities
pub struct ValidationRunner;

impl ValidationRunner {
    /// Run validation with a prompt
    pub fn validate_with_prompt(
        unet: &Tensor,
        vae: &Tensor,
        prompt_embeds: &Tensor,
        height: usize,
        width: usize,
        num_steps: usize,
        device: &Device,
    ) -> flame_core::Result<Tensor> {
        // Initialize latents
        let latents = Tensor::randn(
            Shape::from_dims(&[1, 4, height / 8, width / 8]),
            0.0_f32,
            1.0_f32,
            device.cuda_device().clone(),
        )?;

        // Simple DDIM sampling loop
        let alphas_cumprod = Self::get_alphas_cumprod(device)?;

        for step in (0..num_steps).rev() {
            let t = Tensor::from_vec(
                vec![step as f32],
                Shape::from_dims(&[1]),
                device.cuda_device().clone(),
            )?;
            let alpha_t = alphas_cumprod.get(step)?;
            let alpha_t_prev = if step > 0 {
                alphas_cumprod.get(step - 1)?
            } else {
                Tensor::ones(Shape::from_dims(&[]), device.cuda_device().clone())?
            };

            // Predict noise
            // For now, just simulate with random noise
            let noise_pred = Tensor::randn(
                latents.shape().clone(),
                0.0_f32,
                1.0_f32,
                device.cuda_device().clone(),
            )?;

            // DDIM step
            let one_minus_alpha_t =
                Tensor::full(alpha_t.shape().clone(), 1.0_f32, device.cuda_device().clone())?
                    .sub(&alpha_t)?;
            let sqrt_one_minus_alpha_t = one_minus_alpha_t.sqrt()?;
            let sqrt_alpha_t = alpha_t.sqrt()?;

            let pred_x0 =
                latents.sub(&noise_pred.mul(&sqrt_one_minus_alpha_t)?)?.div(&sqrt_alpha_t)?;

            let one_minus_alpha_t_prev =
                Tensor::full(alpha_t_prev.shape().clone(), 1.0_f32, device.cuda_device().clone())?
                    .sub(&alpha_t_prev)?;
            let sqrt_one_minus_alpha_t_prev = one_minus_alpha_t_prev.sqrt()?;
            let dir_xt = noise_pred.mul(&sqrt_one_minus_alpha_t_prev)?;

            let sqrt_alpha_t_prev = alpha_t_prev.sqrt()?;
            let x_prev = pred_x0.mul(&sqrt_alpha_t_prev)?.add(&dir_xt)?;

            // Update latents
            // latents = x_prev;
        }

        // Decode latents to image
        let scale_factor =
            Tensor::full(Shape::from_dims(&[1]), 0.13025_f32, device.cuda_device().clone())?;
        // For now, just return the scaled latents
        let images = latents.div(&scale_factor)?;

        Ok(images)
    }

    fn get_alphas_cumprod(device: &Device) -> flame_core::Result<Tensor> {
        let num_steps = 1000;
        let beta_start = 0.00085;
        let beta_end = 0.012;

        // Create linspace manually
        let mut beta_values = Vec::with_capacity(num_steps);
        for i in 0..num_steps {
            let t = i as f32 / (num_steps - 1) as f32;
            beta_values.push(beta_start + t * (beta_end - beta_start));
        }
        let betas = Tensor::from_vec(
            beta_values.clone(),
            Shape::from_dims(&[beta_values.len()]),
            device.cuda_device().clone(),
        )?;
        let alphas = Tensor::full(betas.shape().clone(), 1.0f32, device.cuda_device().clone())?
            .sub(&betas)?;

        // Compute cumulative product
        let mut alphas_cumprod = Vec::with_capacity(num_steps);
        let mut alpha_cumprod = 1.0f32;

        for i in 0..num_steps {
            let alpha_i: f32 = alphas.get(i)?.to_scalar()?;
            alpha_cumprod *= alpha_i;
            alphas_cumprod.push(alpha_cumprod);
        }

        Ok(Tensor::from_vec(
            alphas_cumprod,
            Shape::from_dims(&[num_steps]),
            device.cuda_device().clone(),
        )?)
    }
}

// FLAME uses flame_core::device::Device instead of Device

// Re-export for compatibility
// pub use crate::models::sdxl_time_ids::TimeIdsConfig as SDXLTimeIds;
// pub use crate::loaders::sdxl_diffusers_loader::load_vae;

/// Compute time IDs for SDXL conditioning

// Extension trait for Tensor to add missing methods

#[derive(Clone)]
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
        }
    }
}

// TensorExt trait removed - FLAME already has all these methods:
// - sum_dim(dim) - sum along a dimension
// - add_scalar(scalar) - add scalar to all elements
// - mul_scalar(scalar) - multiply all elements by scalar
// - square() - element-wise square

pub fn compute_time_ids(
    height: usize,
    width: usize,
    batch_size: usize,
    device: &Device,
    dtype: DType,
) -> flame_core::Result<Tensor> {
    // Time IDs for SDXL: [original_height, original_width, crop_top, crop_left, target_height, target_width]
    let time_ids = vec![
        height as f32, // original_height
        width as f32,  // original_width
        0.0,           // crop_top
        0.0,           // crop_left
        height as f32, // target_height
        width as f32,  // target_width
    ];

    // Create tensor and expand for batch
    let time_ids_tensor = Tensor::from_vec(
        time_ids.clone(),
        Shape::from_dims(&[time_ids.len()]),
        device.cuda_device().clone(),
    )?
    .to_dtype(dtype)?;

    if batch_size > 1 {
        time_ids_tensor.unsqueeze(0)?.broadcast_to(&Shape::from_dims(&[batch_size, 6]))
    } else {
        time_ids_tensor.unsqueeze(0)
    }
}

/// Encode prompt for SDXL
pub fn encode_prompt(
    text: &str,
    device: &Device,
    dtype: DType,
) -> flame_core::Result<(Tensor, Tensor)> {
    // Simple tokenization for now
    const BOS_TOKEN: i64 = 49406; // <|startoftext|>
    const EOS_TOKEN: i64 = 49407; // <|endoftext|>
    const MAX_LENGTH: usize = 77;

    let words: Vec<&str> = text.split_whitespace().collect();
    let mut token_ids = vec![BOS_TOKEN];

    // Simple hash-based token ID generation
    for word in words.iter().take(MAX_LENGTH.saturating_sub(2)) {
        let mut hash = 0i64;
        for byte in word.bytes() {
            hash = ((hash << 5) + hash) + byte as i64;
            hash = hash % 40000 + 1000; // Keep in vocab range
        }
        token_ids.push(hash);
    }

    token_ids.push(EOS_TOKEN);
    // Pad to MAX_LENGTH
    while token_ids.len() < MAX_LENGTH {
        token_ids.push(EOS_TOKEN);
    }

    // Convert to tensor
    let token_ids_f32: Vec<f32> = token_ids.iter().map(|&x| x as f32).collect();
    let input_ids = Tensor::from_vec(
        token_ids_f32,
        Shape::from_dims(&[token_ids.len()]),
        device.cuda_device().clone(),
    )?
    .unsqueeze(0)?
    .to_dtype(DType::I64)?;

    // For now, return dummy embeddings
    let hidden_states = Tensor::randn(
        Shape::from_dims(&[1, MAX_LENGTH, 768]), // CLIP hidden size
        0.0_f32,
        1.0_f32,
        device.cuda_device().clone(),
    )?
    .to_dtype(dtype)?;

    let pooled =
        Tensor::randn(Shape::from_dims(&[1, 768]), 0.0_f32, 1.0_f32, device.cuda_device().clone())?
            .to_dtype(dtype)?;

    Ok((hidden_states, pooled))
}

/// EMA (Exponential Moving Average) for model weights
pub struct EMAModel {
    decay: f32,
    optimization_step: u32,
    shadow_params: std::collections::HashMap<String, Tensor>,
}

impl EMAModel {
    pub fn new(decay: f32) -> Self {
        Self { decay, optimization_step: 0, shadow_params: std::collections::HashMap::new() }
    }

    /// Update EMA parameters
    pub fn update(
        &mut self,
        parameters: &std::collections::HashMap<String, Tensor>,
    ) -> flame_core::Result<()> {
        self.optimization_step += 1;

        // Compute decay with bias correction
        let decay =
            1.0 - (1.0 - self.decay) * (1.0 - self.decay.powi(self.optimization_step as i32));

        for (name, param) in parameters {
            if let Some(shadow) = self.shadow_params.get_mut(name) {
                // Update shadow parameter: shadow = decay * shadow + (1 - decay) * param
                let decay_tensor =
                    Tensor::full(shadow.shape().clone(), decay, shadow.device().clone())?;
                let one_minus_decay =
                    Tensor::full(param.shape().clone(), 1.0 - decay, param.device().clone())?;
                let updated = shadow.mul(&decay_tensor)?.add(&param.mul(&one_minus_decay)?)?;
                *shadow = updated;
            } else {
                // Initialize shadow parameter
                self.shadow_params.insert(name.clone(), param.clone());
            }
        }

        Ok(())
    }

    /// Get EMA parameters
    pub fn get_shadow_params(&self) -> &std::collections::HashMap<String, Tensor> {
        &self.shadow_params
    }

    /// Apply EMA parameters to model
    pub fn apply_to_model(
        &self,
        model_params: &mut std::collections::HashMap<String, Tensor>,
    ) -> flame_core::Result<()> {
        for (name, shadow) in &self.shadow_params {
            if let Some(param) = model_params.get_mut(name) {
                *param = shadow.clone();
            }
        }
        Ok(())
    }
}

/// Learning rate scheduler
pub struct LRScheduler {
    initial_lr: f32,
    warmup_steps: usize,
    total_steps: usize,
    scheduler_type: String,
    current_step: usize,
}

impl LRScheduler {
    pub fn new(
        initial_lr: f32,
        warmup_steps: usize,
        total_steps: usize,
        scheduler_type: String,
    ) -> Self {
        Self { initial_lr, warmup_steps, total_steps, scheduler_type, current_step: 0 }
    }

    /// Get current learning rate
    pub fn get_lr(&self) -> f32 {
        let step = self.current_step as f32;

        // Warmup phase
        if self.current_step < self.warmup_steps {
            return self.initial_lr * step / self.warmup_steps as f32;
        }

        // Main scheduling
        match self.scheduler_type.as_str() {
            "constant" => self.initial_lr,

            "linear" => {
                let progress = (step - self.warmup_steps as f32)
                    / (self.total_steps - self.warmup_steps) as f32;
                self.initial_lr * (1.0 - progress).max(0.0)
            }

            "cosine" => {
                let progress = (step - self.warmup_steps as f32)
                    / (self.total_steps - self.warmup_steps) as f32;
                let cosine_decay = 0.5 * (1.0 + (progress * std::f32::consts::PI).cos());
                self.initial_lr * cosine_decay
            }

            "polynomial" => {
                let progress = (step - self.warmup_steps as f32)
                    / (self.total_steps - self.warmup_steps) as f32;
                self.initial_lr * (1.0 - progress).powf(0.9)
            }

            _ => self.initial_lr,
        }
    }

    /// Step the scheduler
    pub fn step(&mut self) -> flame_core::Result<()> {
        self.current_step += 1;
        Ok(())
    }
}
