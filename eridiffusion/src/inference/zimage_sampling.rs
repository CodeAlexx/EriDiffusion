//! Z-Image / NextDiT sampling — Euler ODE sampler with SNR time shift.
//!
//! Z-Image uses linear sigma schedule with SNR shift (not exponential like Flux).
//! Sampling formula is identical to Klein: denoised = x - model_output * sigma.
//! The -v convention is baked into the velocity field — no sign flip needed.

use flame_core::{Result, Tensor};

/// Build Z-Image sigma schedule with SNR time shift.
///
/// Returns `num_steps + 1` values from ~1.0 (high noise) to 0.0 (clean).
/// Shift > 1.0 pushes more budget to high-noise region (default 3.0).
pub fn build_sigma_schedule(num_steps: usize, shift: f32) -> Vec<f32> {
    let mut sigmas = Vec::with_capacity(num_steps + 1);
    for i in 0..=num_steps {
        let t = 1.0 - (i as f32 / num_steps as f32);
        let shifted = if (shift - 1.0).abs() < 1e-6 {
            t
        } else {
            shift * t / (1.0 + (shift - 1.0) * t)
        };
        sigmas.push(shifted);
    }
    sigmas
}

/// Euler ODE sampler for Z-Image.
///
/// `model_fn(x, sigma) -> denoised` must return the denoised prediction.
/// `sigmas` should have N+1 entries from `build_sigma_schedule`.
pub fn euler_denoise<F>(model_fn: F, noise: Tensor, sigmas: &[f32]) -> Result<Tensor>
where
    F: Fn(&Tensor, f32) -> Result<Tensor>,
{
    let mut x = noise;
    let total = sigmas.len() - 1;

    for i in 0..total {
        let sigma = sigmas[i];
        let sigma_next = sigmas[i + 1];

        let denoised = model_fn(&x, sigma)?;

        // Euler step: d = (x - denoised) / sigma,  x += d * dt
        let d = x.sub(&denoised)?.mul_scalar(1.0 / sigma)?;
        let dt = sigma_next - sigma;
        x = x.add(&d.mul_scalar(dt)?)?;

        if i == 0 || (i + 1) % 5 == 0 || i + 1 == total {
            println!("  Step {}/{} sigma={:.4}", i + 1, total, sigma);
        }
    }

    Ok(x)
}
