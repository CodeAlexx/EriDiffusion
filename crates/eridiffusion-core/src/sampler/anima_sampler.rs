//! Anima sampler — rectified-flow Euler step with optional shift schedule.
//!
//! Reference: kohya `library/anima_train_utils.py:do_sample` (lines 304-382)
//!   sigmas = linspace(1.0, 0.0, steps + 1)
//!   if flow_shift != 1.0:
//!     sigmas = (sigmas * flow_shift) / (1 + (flow_shift - 1) * sigmas)
//!   for i in range(steps):
//!     pred = model(x, t=sigmas[i] * 1000)  # CFG applied externally
//!     dt = sigmas[i+1] - sigmas[i]         # negative → x moves toward clean
//!     x = x + dt * pred                     # pred = noise - latents (rect-flow target)
//!
//! Default `flow_shift = 3.0` per do_sample's signature; training default
//! (`--discrete_flow_shift`) is `1.0`.

use flame_core::{Result, Tensor};

/// Build a rectified-flow sigma schedule with optional shift.
/// Returns `num_steps + 1` values from 1.0 down to 0.0 (pre-shift).
pub fn schedule(num_steps: usize, shift: f32) -> Vec<f32> {
    let mut s: Vec<f32> = (0..=num_steps)
        .map(|i| 1.0 - i as f32 / num_steps as f32)
        .collect();
    if (shift - 1.0).abs() > f32::EPSILON {
        for v in s.iter_mut() {
            *v = shift * *v / (1.0 + (shift - 1.0) * *v);
        }
    }
    s
}

/// Convert a sigma in [0, 1] to the timestep the DiT expects.
/// kohya scales sigmas by 1000 then divides by 1000 in the noise pred path
/// (`anima_train_network.py:279` / `do_sample` uses raw sigma * 1000).
/// We pass the scaled value to model.forward; the model is expected to handle
/// the [0, 1] convention internally (timesteps / 1000.0 in get_noise_pred_and_target).
pub fn sigma_to_timestep(sigma: f32) -> f32 {
    sigma
}

/// Single Euler step: x_next = x + (sigma_next - sigma) * pred.
/// `pred` is the rectified-flow target (noise - latents).
pub fn euler_step(x: &Tensor, pred: &Tensor, sigma: f32, sigma_next: f32) -> Result<Tensor> {
    let dt = sigma_next - sigma; // negative when going from 1.0 → 0.0
    x.add(&pred.mul_scalar(dt)?)
}
