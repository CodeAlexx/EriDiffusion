//! Ernie sampling — FlowMatchEulerDiscreteScheduler for inference/preview.
use flame_core::{Result, Tensor};

const SHIFT: f32 = 3.0;
const NUM_TRAIN_TIMESTEPS: f32 = 1000.0;

/// Build sigma schedule: num_steps+1 values from 1.0 to 0.0 with exponential shift.
pub fn schedule(num_steps: usize) -> Vec<f32> {
    let mut sigmas = Vec::with_capacity(num_steps + 1);
    for i in 0..=num_steps {
        let sigma = 1.0 - i as f32 / num_steps as f32;
        let shifted = if sigma <= 0.0 || sigma >= 1.0 {
            sigma
        } else {
            SHIFT / (SHIFT + (1.0 / sigma - 1.0))
        };
        sigmas.push(shifted);
    }
    sigmas
}

/// Convert sigma to model timestep (sigma * 1000).
pub fn sigma_to_timestep(sigma: f32) -> f32 { sigma * NUM_TRAIN_TIMESTEPS }

/// One Euler ODE step: x_next = x + (sigma_next - sigma) * pred
pub fn euler_step(x: &Tensor, pred: &Tensor, sigma: f32, sigma_next: f32) -> Result<Tensor> {
    let dt = sigma_next - sigma;
    x.add(&pred.mul_scalar(dt)?)
}
