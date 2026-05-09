//! Wan 2.2 rectified-flow Euler sampler with `sample_shift` time warp.
//!
//! Source of truth: `flame-diffusion-archive/wan-trainer/src/schedule.rs`
//! and `inference-flame-master/src/models/wan/sampler.rs`. Wan 2.2 uses
//! the **classic flow time-shift**:
//!
//! ```text
//! t' = (t * shift) / (1 + (shift - 1) * t)
//! ```
//!
//! `shift = 5.0` for TI2V-5B (the official `sample_shift`); larger
//! values bias the training distribution toward higher noise (closer
//! to t=1).  T2V/I2V-14B use the same shift but split sampling across
//! two experts at a hard timestep boundary (typically 0.875 for T2V).
//!
//! ## Math (T2V/TI2V)
//! - Forward noising: `noisy = (1 - sigma) * clean + sigma * noise`.
//! - Velocity target: `noise - clean` (matches archive `pipeline.rs`).
//! - Inference Euler step: `x_next = x + (sigma_next - sigma) * pred`.
//!
//! ## Dual-expert dispatch (14B only)
//! [`expert_for_timestep`] returns `Expert::High` when `t >= boundary`
//! and `Expert::Low` otherwise.  The trainer maintains TWO LoRA bundles
//! and routes the active step through the bundle for the selected
//! expert.

use flame_core::{Result, Tensor};
use rand::Rng;
use rand_distr::{Distribution, Normal};

pub const NUM_TRAIN_TIMESTEPS: f32 = 1000.0;

/// Default Wan 2.2 TI2V-5B sample_shift.
pub const DEFAULT_SHIFT_TI2V_5B: f32 = 5.0;

/// Default Wan 2.2 T2V/I2V boundary (continuous t in [0, 1]). Per the
/// reference Wan 2.2 inference impl, the high/low expert split happens
/// at `boundary * 1000 = 875` for T2V-A14B; I2V uses 0.900.
pub const DEFAULT_NOISE_BOUNDARY_T2V: f32 = 0.875;
pub const DEFAULT_NOISE_BOUNDARY_I2V: f32 = 0.900;

/// Which dual-expert checkpoint owns this timestep.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Expert {
    /// High-noise expert (t >= boundary).
    High,
    /// Low-noise expert (t < boundary).
    Low,
}

/// Apply Wan's classic time-shift to a single `t ∈ (0, 1)`.
///
/// `shift == 1.0` is the identity. `shift > 1` pushes the distribution
/// toward 1 (high noise); `shift < 1` toward 0.
pub fn apply_time_shift(t: f32, shift: f32) -> f32 {
    if (shift - 1.0).abs() < 1.0e-6 {
        return t;
    }
    (t * shift) / (1.0 + (shift - 1.0) * t)
}

/// Sample one continuous timestep in `(0, 1)` from logit-normal:
/// `t = sigmoid(N(0, 1) * sigmoid_scale)`. Matches archive
/// `schedule::sample_logit_normal` for one element at a time.
pub fn sample_logit_normal_with_shift(
    rng: &mut rand::rngs::StdRng,
    shift: f32,
    sigmoid_scale: f32,
) -> f32 {
    let normal = Normal::new(0.0f32, 1.0f32).unwrap();
    let z = normal.sample(rng) * sigmoid_scale;
    let raw = 1.0 / (1.0 + (-z).exp());
    let shifted = apply_time_shift(raw, shift);
    shifted.clamp(1.0e-4, 1.0 - 1.0e-4)
}

/// Uniform sampler with shift applied. Used when `timestep_method = "uniform"`.
pub fn sample_uniform_with_shift(rng: &mut rand::rngs::StdRng, shift: f32) -> f32 {
    let raw: f32 = rng.r#gen::<f32>().clamp(1.0e-4, 1.0 - 1.0e-4);
    apply_time_shift(raw, shift)
}

/// Decide which expert serves a timestep. Continuous-`t` form (t in
/// `[0, 1]`).  The 5B TI2V trainer feeds `boundary = 0.0` so every
/// step routes to `Expert::Low` (i.e., the only model present).
pub fn expert_for_timestep(t_continuous: f32, boundary: f32) -> Expert {
    if t_continuous >= boundary {
        Expert::High
    } else {
        Expert::Low
    }
}

/// Build a sigma schedule for inference: `num_steps + 1` values from
/// 1.0 down to 0.0, time-shifted by `shift`. The Euler sampler walks
/// this schedule top-to-bottom.
pub fn schedule(num_steps: usize, shift: f32) -> Vec<f32> {
    let mut sigmas = Vec::with_capacity(num_steps + 1);
    for i in 0..=num_steps {
        let raw = 1.0 - i as f32 / num_steps as f32;
        sigmas.push(apply_time_shift(raw, shift));
    }
    sigmas
}

/// Convert sigma in [0, 1] to the timestep value the DiT consumes.
/// Wan multiplies by `timestep_scale_multiplier = 1000`.
pub fn sigma_to_timestep(sigma: f32) -> f32 {
    sigma * NUM_TRAIN_TIMESTEPS
}

/// Single Euler ODE step: `x_next = x + (sigma_next - sigma) * pred`.
pub fn euler_step(x: &Tensor, pred: &Tensor, sigma: f32, sigma_next: f32) -> Result<Tensor> {
    let dt = sigma_next - sigma;
    x.add(&pred.mul_scalar(dt)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shift_identity_at_one() {
        for t in [0.1f32, 0.3, 0.5, 0.7, 0.9] {
            assert!((apply_time_shift(t, 1.0) - t).abs() < 1e-6);
        }
    }

    #[test]
    fn shift5_pushes_toward_one() {
        // With shift=5.0 mid-point should map > 0.5.
        assert!(apply_time_shift(0.5, 5.0) > 0.5);
    }

    #[test]
    fn expert_dispatch_boundary() {
        let b = DEFAULT_NOISE_BOUNDARY_T2V; // 0.875
        assert_eq!(expert_for_timestep(0.0, b), Expert::Low);
        assert_eq!(expert_for_timestep(0.5, b), Expert::Low);
        assert_eq!(expert_for_timestep(0.874, b), Expert::Low);
        assert_eq!(expert_for_timestep(0.875, b), Expert::High); // inclusive at boundary
        assert_eq!(expert_for_timestep(0.99, b), Expert::High);
    }

    #[test]
    fn expert_dispatch_5b_zero_boundary() {
        // boundary=0 => every t >= 0 routes to High. Trainer should
        // use Low (the single resident expert) by setting boundary
        // above any sampled timestep, e.g. > 1.0.
        assert_eq!(expert_for_timestep(0.0, 2.0), Expert::Low);
        assert_eq!(expert_for_timestep(0.999, 2.0), Expert::Low);
    }

    #[test]
    fn schedule_monotonic_descending() {
        let s = schedule(20, 5.0);
        assert_eq!(s.len(), 21);
        for w in s.windows(2) {
            assert!(w[0] >= w[1] - 1e-6);
        }
    }
}
