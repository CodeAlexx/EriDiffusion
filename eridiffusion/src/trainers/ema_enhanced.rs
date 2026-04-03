//! Exponential Moving Average (EMA) for model weights
//!
//! Provides stable model weights during training by maintaining
//! a moving average of parameters, commonly used in diffusion models
//! for better sample quality during inference.

use flame_core::device::Device;
use flame_core::{CudaDevice, DType, Error, Parameter, Result, Shape, Tensor};
use std::collections::HashMap;
use std::sync::Arc;

/// Exponential Moving Average tracker for model parameters
pub struct EMAModel {
    /// Decay rate for EMA (typically 0.999 or 0.9999)
    decay: f32,

    /// Current step counter for bias correction
    step: usize,

    /// Shadow parameters (EMA weights)
    shadow_params: HashMap<String, Tensor>,

    /// Device for tensor operations
    device: Arc<CudaDevice>,

    /// Whether to use bias correction for early steps
    use_bias_correction: bool,

    /// Power for EMA decay schedule (1.0 = constant, >1 = slower start)
    power: f32,
}

impl EMAModel {
    /// Create new EMA model with given decay rate
    pub fn new(decay: f32, device: Arc<CudaDevice>) -> flame_core::Result<Self> {
        if !(0.0 < decay && decay <= 1.0) {
            return Err(flame_core::Error::InvalidOperation(format!(
                "EMA decay must be in (0, 1], got {}",
                decay
            )));
        }

        Ok(Self {
            decay,
            step: 0,
            shadow_params: HashMap::new(),
            device,
            use_bias_correction: true,
            power: 1.0,
        })
    }

    /// Create EMA with custom configuration
    pub fn with_config(
        decay: f32,
        device: Arc<CudaDevice>,
        use_bias_correction: bool,
        power: f32,
    ) -> flame_core::Result<Self> {
        let mut ema = Self::new(decay, device)?;
        ema.use_bias_correction = use_bias_correction;
        ema.power = power;
        Ok(ema)
    }

    /// Initialize shadow parameters from model
    pub fn init_from_params(
        &mut self,
        params: &HashMap<String, &Parameter>,
    ) -> flame_core::Result<()> {
        for (name, param) in params {
            let shadow = param.tensor()?;
            self.shadow_params.insert(name.to_string(), shadow);
        }
        Ok(())
    }

    /// Update EMA weights with current model parameters
    pub fn update(&mut self, params: &HashMap<String, &Parameter>) -> flame_core::Result<()> {
        self.step += 1;

        // Compute effective decay based on step and power
        let effective_decay = self.compute_effective_decay();

        for (name, param) in params {
            let current_weight = param.tensor()?;

            // Update shadow: shadow = decay * shadow + (1 - decay) * current
            if let Some(shadow) = self.shadow_params.get(name) {
                let decay_scalar =
                    Tensor::full(shadow.shape().clone(), effective_decay, shadow.device().clone())?;
                let one_minus_decay = Tensor::full(
                    current_weight.shape().clone(),
                    1.0 - effective_decay,
                    current_weight.device().clone(),
                )?;
                let updated =
                    shadow.mul(&decay_scalar)?.add(&current_weight.mul(&one_minus_decay)?)?;

                self.shadow_params.insert(name.to_string(), updated);
            } else {
                // First time seeing this parameter
                self.shadow_params.insert(name.to_string(), current_weight);
            }
        }

        Ok(())
    }

    /// Compute effective decay rate considering bias correction and power
    fn compute_effective_decay(&self) -> f32 {
        let mut decay = self.decay;

        // Apply power schedule
        if self.power != 1.0 {
            let progress = (self.step as f32 / 1000.0).min(1.0); // Normalize to [0, 1]
            decay = decay.powf(self.power * (1.0 - progress) + progress);
        }

        // Apply bias correction
        if self.use_bias_correction && self.step < 1000 {
            // Bias correction formula: decay_corrected = decay * (1 - decay^(step-1)) / (1 - decay^step)
            let numerator = 1.0 - decay.powi((self.step - 1) as i32);
            let denominator = 1.0 - decay.powi(self.step as i32);
            decay = decay * numerator / denominator.max(1e-8);
        }

        decay
    }

    /// Copy EMA weights to model (for evaluation/inference)
    pub fn copy_to(&self, params: &mut HashMap<String, &mut Parameter>) -> flame_core::Result<()> {
        for (name, param) in params {
            if let Some(shadow) = self.shadow_params.get(name) {
                param.set_data(shadow.clone())?;
            } else {
                return Err(flame_core::Error::InvalidOperation(format!(
                    "No EMA weights found for parameter: {}",
                    name
                )));
            }
        }
        Ok(())
    }

    /// Store current model weights (before copying EMA weights)
    pub fn store_params(
        &self,
        params: &HashMap<String, &Parameter>,
    ) -> flame_core::Result<HashMap<String, Tensor>> {
        let mut stored = HashMap::new();
        for (name, param) in params {
            stored.insert(name.to_string(), param.as_tensor()?);
        }
        Ok(stored)
    }

    /// Restore original model weights (after evaluation)
    pub fn restore_params(
        &self,
        params: &mut HashMap<String, &mut Parameter>,
        stored: &HashMap<String, Tensor>,
    ) -> flame_core::Result<()> {
        for (name, param) in params {
            if let Some(original) = stored.get(name) {
                param.set_data(original.clone())?;
            }
        }
        Ok(())
    }

    /// Get shadow parameters
    pub fn shadow_params(&self) -> &HashMap<String, Tensor> {
        &self.shadow_params
    }

    /// Get current step
    pub fn step(&self) -> usize {
        self.step
    }

    /// Set step (for resuming training)
    pub fn set_step(&mut self, step: usize) {
        self.step = step;
    }

    /// Get current effective decay rate
    pub fn current_decay(&self) -> f32 {
        self.compute_effective_decay()
    }

    /// State dict for checkpointing
    pub fn state_dict(&self) -> EMAState {
        EMAState {
            decay: self.decay,
            step: self.step,
            shadow_params: self.shadow_params.clone(),
            use_bias_correction: self.use_bias_correction,
            power: self.power,
        }
    }

    /// Load from state dict
    pub fn load_state_dict(&mut self, state: EMAState) -> flame_core::Result<()> {
        self.decay = state.decay;
        self.step = state.step;
        self.shadow_params = state.shadow_params;
        self.use_bias_correction = state.use_bias_correction;
        self.power = state.power;
        Ok(())
    }
}

/// EMA state for serialization
pub struct EMAState {
    pub decay: f32,
    pub step: usize,
    pub shadow_params: HashMap<String, Tensor>,
    pub use_bias_correction: bool,
    pub power: f32,
}

/// Context manager for EMA evaluation
pub struct EMAContext<'a> {
    ema_model: &'a EMAModel,
    params: &'a mut HashMap<String, &'a mut Parameter>,
    stored_params: Option<HashMap<String, Tensor>>,
}

impl<'a> EMAContext<'a> {
    pub fn new(
        ema_model: &'a EMAModel,
        params: &'a mut HashMap<String, &'a mut Parameter>,
    ) -> flame_core::Result<Self> {
        Ok(Self { ema_model, params, stored_params: None })
    }

    /// Enter context (copy EMA weights to model)
    pub fn enter(&mut self) -> flame_core::Result<()> {
        // Store original parameters
        let stored = self.ema_model.store_params(
            &self.params.iter().map(|(k, v)| (k.clone(), *v as &Parameter)).collect(),
        )?;
        self.stored_params = Some(stored);

        // Copy EMA weights to model
        self.ema_model.copy_to(self.params)?;
        Ok(())
    }

    /// Exit context (restore original weights)
    pub fn exit(&mut self) -> flame_core::Result<()> {
        if let Some(stored) = &self.stored_params {
            self.ema_model.restore_params(self.params, stored)?;
        }
        Ok(())
    }
}

/// Helper function to create EMA-averaged model for inference
pub fn apply_ema_weights(
    ema_model: &EMAModel,
    params: &mut HashMap<String, &mut Parameter>,
) -> flame_core::Result<()> {
    ema_model.copy_to(params)
}

/// Helper to compute EMA decay based on desired half-life
pub fn decay_from_half_life(half_life_steps: usize) -> f32 {
    // decay^half_life = 0.5
    // => decay = 0.5^(1/half_life)
    0.5f32.powf(1.0 / half_life_steps as f32)
}

/// Helper to compute EMA decay for a target smoothing window
pub fn decay_from_window(window_size: usize) -> f32 {
    // Standard formula: decay = 1 (-2i64) as usize/(N+1) where N is window size
    1.0 - 2.0 / (window_size as f32 + 1.0)
}

#[cfg(all(test, feature = "dev-bins"))]
mod tests {
    use super::*;

    #[test]
    fn test_ema_decay_computation() {
        // Test half-life computation
        let decay = decay_from_half_life(1000);
        assert!((decay.powi(1000) -.5).abs() < 0.001);

        // Test window computation
        let decay = decay_from_window(99);
        assert!((decay -.98).abs() < 0.001);
    }

    #[test]
    fn test_ema_update(device: &CudaDevice) -> flame_core::Result<()> {
        let device = device;
        let mut ema = EMAModel::new(0.999, device.clone())?;

        // Create dummy parameters
        let param1 = Parameter::randn(Shape::new(vec![10]), 0.0, 1.0, DType::F32, device)?;

        let mut params = HashMap::new();
        params.insert("param1".to_string(), &param1);

        // Initialize EMA
        ema.init_from_params(&params)?;

        // Update EMA
        ema.update(&params)?;

        assert_eq!(ema.step(), 1);
        assert!(ema.shadow_params().contains_key("param1"));

        Ok(())
    }
}
