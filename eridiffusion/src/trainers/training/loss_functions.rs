use flame_core::device::Device;
use flame_core::Result;
use flame_core::{DType, Tensor};
use std::collections::HashMap;
// use super::super::snr_weighting::compute_snr_weight; // Function doesn't exist in snr_weighting module

#[derive(Debug, Clone, Copy)]
pub enum LossType {
    MSE,
    L1,
    Huber(f32),
    SmoothL1,
}

/// Compute diffusion training loss
pub fn compute_loss(
    predicted: &Tensor,
    target: &Tensor,
    loss_type: LossType,
    timesteps: Option<&Tensor>,
    snr_gamma: Option<f32>,
) -> flame_core::Result<Tensor> {
    // Base loss computation
    let base_loss = match loss_type {
        LossType::MSE => {
            let diff = predicted.sub(target)?;
            diff.square()?
        }
        LossType::L1 => predicted.sub(target)?.abs()?,
        LossType::Huber(delta) => {
            compute_huber_loss(predicted, target, delta, &Device::from(predicted.device().clone()))?
        }
        LossType::SmoothL1 => compute_smooth_l1_loss(predicted, target)?,
    };

    // Apply SNR weighting if provided
    let weighted_loss = if let (Some(_timesteps), Some(_gamma)) = (timesteps, snr_gamma) {
        // TODO: Implement SNR weighting
        // let snr_weights = compute_snr_weight(timesteps, gamma)?;
        // base_loss.mul(&snr_weights)?
        base_loss
    } else {
        base_loss
    };

    // Reduce to scalar
    weighted_loss.mean()
}

/// Huber loss (smooth L1)
fn compute_huber_loss(
    pred: &Tensor,
    target: &Tensor,
    delta: f32,
    device: &Device,
) -> flame_core::Result<Tensor> {
    let diff = pred.sub(target)?;

    // For now, just return L2 loss as FLAME doesn't support comparisons
    // TODO: Implement proper Huber loss when comparison ops are available
    // Proper Huber loss would be:
    // where |diff| <= delta: 0.5 * diff^2
    // where |diff| > delta: delta * (|diff| - 0.5 * delta)

    // Using L2 loss as approximation
    Ok(diff.square()?.mul_scalar(0.5 as f32)?)
}

/// Smooth L1 loss (used in some diffusion models)
fn compute_smooth_l1_loss(pred: &Tensor, target: &Tensor) -> flame_core::Result<Tensor> {
    compute_huber_loss(pred, target, 1.0, &Device::from(pred.device().clone()))
}

/// Compute velocity prediction loss (for v-parameterization)
pub fn compute_velocity_loss(
    model_output: &Tensor,
    target_velocity: &Tensor,
    loss_type: LossType,
) -> flame_core::Result<Tensor> {
    compute_loss(model_output, target_velocity, loss_type, None, None)
}

/// Compute epsilon prediction loss (standard diffusion)
pub fn compute_epsilon_loss(
    model_output: &Tensor,
    noise: &Tensor,
    loss_type: LossType,
    timesteps: Option<&Tensor>,
    snr_gamma: Option<f32>,
) -> flame_core::Result<Tensor> {
    compute_loss(model_output, noise, loss_type, timesteps, snr_gamma)
}

/// Compute min-SNR weighted loss for better training stability
pub fn compute_min_snr_loss(
    predicted: &Tensor,
    target: &Tensor,
    timesteps: &Tensor,
    gamma: f32,
    base_loss_type: LossType,
) -> flame_core::Result<Tensor> {
    // Compute base loss
    let base_loss = match base_loss_type {
        LossType::MSE => {
            let diff = predicted.sub(target)?;
            diff.square()?
        }
        _ => compute_loss(predicted, target, base_loss_type, None, None)?,
    };

    // Compute SNR weights
    // TODO: Implement SNR weighting
    // let snr_weights = compute_snr_weight(timesteps, gamma)?;

    // Apply weights and reduce
    // base_loss.mul(&snr_weights)?.mean()
    base_loss.mean()
}

/// Loss scaling utilities
pub struct LossScaler {
    scale: f32,
    growth_factor: f32,
    backoff_factor: f32,
    growth_interval: usize,
    counter: usize,
}

impl LossScaler {
    pub fn new() -> Self {
        Self {
            scale: 65536.0, // 2^16
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2000,
            counter: 0,
        }
    }

    pub fn scale_loss(&self, loss: &Tensor) -> flame_core::Result<Tensor> {
        loss.mul_scalar(self.scale as f64 as f32)
    }

    pub fn unscale_gradients(&self, grads: &mut HashMap<String, Tensor>) -> flame_core::Result<()> {
        let inv_scale = 1.0 / self.scale;
        for grad in grads.values_mut() {
            *grad = grad.mul_scalar(inv_scale as f64 as f32)?;
        }
        Ok(())
    }

    pub fn update(&mut self, found_inf: bool) {
        if found_inf {
            // Reduce scale on overflow
            self.scale *= self.backoff_factor;
            self.counter = 0;
        } else {
            // Increase scale after growth_interval successful steps
            self.counter += 1;
            if self.counter >= self.growth_interval {
                self.scale *= self.growth_factor;
                self.counter = 0;
            }
        }
    }
}
