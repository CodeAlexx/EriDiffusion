//! LTX-2 rectified-flow Euler sampler with sequence-length-dependent
//! shift schedule. Mirrors `CustomFlowMatchEulerDiscreteScheduler`
//! (ai-toolkit) and `LTX2Scheduler` (musubi).
//!
//! ## Math (T2V):
//! - Forward noising at train time: `noisy = (1 - sigma) * clean + sigma * noise`
//!   (`sample_timestep_logit_normal` style).
//! - Velocity target: `noise - clean`.
//! - At sample time: Euler step `x_next = x + (sigma_next - sigma) * pred`,
//!   identical to ERNIE/Z-Image schedulers.
//!
//! ## Shift schedule
//! The sigma curve is "shifted" by an LTX-2-specific factor that interpolates
//! between `base_shift = 0.95` at 1024 tokens and `max_shift = 2.05` at
//! 4096 tokens. Token count = `F * H * W` (latent units).
//!
//! Formula (mu: shift in log-sigma space):
//!   `mu = base_shift + (max_shift - base_shift) * (n_tokens - 1024) / (4096 - 1024)`
//!   `mu = mu.clamp(base_shift, max_shift)`
//!   then `sigma_shifted = exp(mu) * sigma / (1 + (exp(mu) - 1) * sigma)`
//!
//! The `time_shift_type=exponential` mode in ai-toolkit is the same;
//! `dynamic_shifting=True` means we recompute `mu` per call from the
//! actual token count.

use flame_core::{Result, Tensor};

const NUM_TRAIN_TIMESTEPS: f32 = 1000.0;
const BASE_SHIFT: f32 = 0.95;
const MAX_SHIFT: f32 = 2.05;
const BASE_TOKEN_COUNT: f32 = 1024.0;
const MAX_TOKEN_COUNT: f32 = 4096.0;

/// Compute `mu` (the shift exponent) for a given video token count.
/// Token count = `n_video_tokens` after patchify (B=1 case: F * H_lat * W_lat).
pub fn shift_for_token_count(n_tokens: usize) -> f32 {
    let nt = n_tokens as f32;
    let raw = BASE_SHIFT
        + (MAX_SHIFT - BASE_SHIFT) * (nt - BASE_TOKEN_COUNT) / (MAX_TOKEN_COUNT - BASE_TOKEN_COUNT);
    raw.clamp(BASE_SHIFT, MAX_SHIFT)
}

/// Apply the LTX-2 exponential time shift to a raw sigma in [0, 1].
fn apply_shift(sigma: f32, mu: f32) -> f32 {
    if sigma <= 0.0 || sigma >= 1.0 {
        return sigma;
    }
    let e = mu.exp();
    e * sigma / (1.0 + (e - 1.0) * sigma)
}

/// Build a shifted sigma schedule: `num_steps + 1` values from 1.0 down to 0.0.
/// `n_tokens` = number of patchified video tokens (used to pick the shift).
pub fn schedule(num_steps: usize, n_tokens: usize) -> Vec<f32> {
    let mu = shift_for_token_count(n_tokens);
    let mut sigmas = Vec::with_capacity(num_steps + 1);
    for i in 0..=num_steps {
        let raw = 1.0 - i as f32 / num_steps as f32;
        sigmas.push(apply_shift(raw, mu));
    }
    sigmas
}

/// Convert sigma in [0, 1] to the timestep value the DiT consumes.
/// LTX-2 multiplies by `timestep_scale_multiplier = 1000`.
pub fn sigma_to_timestep(sigma: f32) -> f32 {
    sigma * NUM_TRAIN_TIMESTEPS
}

/// Single Euler ODE step: `x_next = x + (sigma_next - sigma) * pred`.
pub fn euler_step(x: &Tensor, pred: &Tensor, sigma: f32, sigma_next: f32) -> Result<Tensor> {
    let dt = sigma_next - sigma;
    x.add(&pred.mul_scalar(dt)?)
}

/// Logit-normal timestep sampler used at training (matches musubi's
/// `shifted_logit_normal` and ai-toolkit's flow scheduler training mode).
/// Returns a continuous timestep in [0, NUM_TRAIN_TIMESTEPS).
pub fn sample_timestep_logit_normal(rng: &mut rand::rngs::StdRng, mu: f32) -> f32 {
    use rand_distr::{Distribution, Normal};
    let normal = Normal::new(0.0f32, 1.0f32).unwrap();
    let z = normal.sample(rng);
    let sigma_raw = 1.0 / (1.0 + (-z).exp());
    let sigma = apply_shift(sigma_raw, mu);
    sigma * NUM_TRAIN_TIMESTEPS
}
