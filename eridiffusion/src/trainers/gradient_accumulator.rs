use flame_core::device::Device;
use flame_core::optimizers::{Adam, SGD};
use flame_core::{DType, Result, Shape, Tensor};
use std::collections::HashMap;

// Proper gradient accumulation for FLAME
// Accumulates gradients across multiple steps before applying updates

// FLAME uses flame_core::device::Device instead of Device

/// Gradient accumulator that properly sums gradients across steps
pub struct GradientAccumulator {
    accumulated_grads: std::collections::HashMap<String, Tensor>,
    accumulation_steps: usize,
    current_step: usize,
    device: Device,
}

impl GradientAccumulator {
    pub fn new(accumulation_steps: usize, device: Device) -> Self {
        Self { accumulated_grads: HashMap::new(), accumulation_steps, current_step: 0, device }
    }

    /// Accumulate gradients from current step
    pub fn accumulate(&mut self, param_name: &str, grad: &Tensor) -> flame_core::Result<()> {
        // Scale gradient by accumulation steps to get correct average
        let divisor = Tensor::full(
            grad.shape().clone(),
            self.accumulation_steps as f32,
            grad.device().clone(),
        )?;
        let scaled_grad = grad.div(&divisor)?;

        if let Some(existing) = self.accumulated_grads.get_mut(param_name) {
            // Add to existing accumulated gradient
            let sum = existing.add(&scaled_grad)?;
            *existing = sum;
        } else {
            // First accumulation for this parameter
            self.accumulated_grads.insert(param_name.to_string(), scaled_grad);
        }

        Ok(())
    }

    /// Check if accumulation is complete
    pub fn should_update(&self) -> bool {
        self.current_step >= self.accumulation_steps - 1
    }

    /// Step the accumulator
    pub fn step(&mut self) {
        self.current_step += 1;
        if self.current_step >= self.accumulation_steps {
            self.current_step = 0;
        }
    }

    /// Get accumulated gradient for a parameter
    pub fn get_gradient(&self, param_name: &str) -> Option<&Tensor> {
        self.accumulated_grads.get(param_name)
    }

    /// Clear accumulated gradients after update
    pub fn clear(&mut self) {
        self.accumulated_grads.clear();
        self.current_step = 0;
    }

    /// Get all accumulated gradients
    pub fn get_accumulated_grads(&self) -> &HashMap<String, Tensor> {
        &self.accumulated_grads
    }
}

/// Extension for LoRA collection to support gradient accumulation
pub trait LoRAGradientAccumulation {
    /// Accumulate gradients for all LoRA parameters
    fn accumulate_gradients(&self, accumulator: &mut GradientAccumulator)
        -> flame_core::Result<()>;

    /// Apply accumulated gradients
    fn apply_accumulated_gradients(
        &self,
        accumulator: &GradientAccumulator,
        learning_rate: f32,
    ) -> flame_core::Result<()>;
}
