//! Klein / Flux-2 sampling — direct velocity Euler ODE with dynamic mu schedule.
//!
//! Mirrors `inference-flame/src/sampling/klein_sampling.rs` (BFL reference port).

use flame_core::{Result, Tensor};

/// BFL `compute_empirical_mu` — image_seq_len + num_steps → schedule mu.
pub fn compute_empirical_mu(image_seq_len: usize, num_steps: usize) -> f64 {
    let a1 = 8.73809524e-05f64;
    let b1 = 1.89833333f64;
    let a2 = 0.00016927f64;
    let b2 = 0.45666666f64;

    let seq = image_seq_len as f64;
    if image_seq_len > 4300 {
        return a2 * seq + b2;
    }
    let m_200 = a2 * seq + b2;
    let m_10 = a1 * seq + b1;
    let a = (m_200 - m_10) / 190.0;
    let b = m_200 - 200.0 * a;
    a * num_steps as f64 + b
}

fn time_snr_shift(t: f64, mu: f64, sigma: f64) -> f64 {
    let exp_mu = mu.exp();
    exp_mu / (exp_mu + (1.0 / t - 1.0).powf(sigma))
}

/// `num_steps + 1` sigma values from ~1.0 down to 0.0 with dynamic mu shift.
pub fn get_schedule(num_steps: usize, image_seq_len: usize) -> Vec<f32> {
    let mu = compute_empirical_mu(image_seq_len, num_steps);
    let mut timesteps = Vec::with_capacity(num_steps + 1);
    for i in 0..=num_steps {
        let t = 1.0 - i as f64 / num_steps as f64;
        let shifted = if t <= 0.0 || t >= 1.0 {
            t
        } else {
            time_snr_shift(t, mu, 1.0)
        };
        timesteps.push(shifted as f32);
    }
    timesteps
}

/// One Euler step: x_next = x + (sigma_next - sigma) * pred (direct velocity).
pub fn euler_step(x: &Tensor, pred: &Tensor, sigma: f32, sigma_next: f32) -> Result<Tensor> {
    let dt = sigma_next - sigma;
    x.add(&pred.mul_scalar(dt)?)
}

/// Sigma → model timestep.
///
/// Returns sigma directly (in `[0, 1]`). The `klein.rs::timestep_embedding`
/// function multiplies by `time_factor=1000.0` internally, so callers must
/// pass the raw sigma here. Matches upstream Python `Flux2Sampler.py:122`
/// (`expanded_timestep / 1000`) and inference-flame's `klein_sampling.rs`
/// `euler_denoise` (passes `t_curr` from `get_schedule()` directly).
///
/// Audit fix (KLEIN_VERIFY.md §H3 / KLEIN_SKEPTIC.md §H1): previously
/// returned `sigma * 1000`, which combined with the `*1000` inside
/// `timestep_embedding` produced `sigma * 1e6` in sin/cos arguments —
/// 1000× out-of-distribution vs trained checkpoint.
pub fn sigma_to_timestep(sigma: f32) -> f32 { sigma }
