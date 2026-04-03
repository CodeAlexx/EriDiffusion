//! LTX-2.3 video sampling — Euler velocity step with distilled sigma schedule.
//!
//! Distilled model uses 8 fixed steps, no CFG (guidance_scale=1.0).
//! Flow matching: denoised = x - model_output * sigma.

use flame_core::{Result, Tensor};

/// Distilled 8-step sigma schedule (hardcoded from Lightricks).
pub const LTX2_DISTILLED_SIGMAS: [f32; 9] = [
    1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0,
];

/// Build token-dependent sigma schedule for dev model (25 steps).
///
/// Shifts sigmas based on latent token count using Flux-style exponential shift.
pub fn build_dev_sigma_schedule(
    num_steps: usize,
    num_latent_tokens: usize,
    base_shift: f32,
    max_shift: f32,
    terminal: f32,
) -> Vec<f32> {
    // Linear interpolation of shift based on token count
    let m = (max_shift - base_shift) / (4096.0 - 1024.0);
    let b = base_shift - m * 1024.0;
    let shift = (num_latent_tokens as f32 * m + b).clamp(base_shift, max_shift);

    let mut sigmas = Vec::with_capacity(num_steps + 1);
    for i in 0..=num_steps {
        let t = 1.0 - (i as f32 / num_steps as f32);
        // Exponential shift: exp(shift) / (exp(shift) + (1/t - 1))
        let shifted = if t <= 0.0 {
            0.0
        } else if t >= 1.0 {
            1.0
        } else {
            let es = shift.exp();
            es / (es + (1.0 / t - 1.0))
        };
        sigmas.push(shifted);
    }

    // Stretch to terminal value
    if terminal > 0.0 && terminal < 1.0 {
        for s in sigmas.iter_mut() {
            if *s > 0.0 {
                *s = *s * (1.0 - terminal) + terminal;
            }
        }
        // Last sigma stays 0
        if let Some(last) = sigmas.last_mut() {
            *last = 0.0;
        }
    }

    sigmas
}

/// Euler velocity sampler for LTX-2.
///
/// `model_fn(x, sigma) -> velocity` returns the model's velocity prediction.
/// Step: `denoised = x - velocity * sigma`, then Euler update.
pub fn euler_denoise_ltx2<F>(
    model_fn: F,
    noise: Tensor,
    sigmas: &[f32],
) -> Result<Tensor>
where
    F: Fn(&Tensor, f32) -> Result<Tensor>,
{
    let mut x = noise;
    let total = sigmas.len() - 1;

    for i in 0..total {
        let sigma = sigmas[i];
        let sigma_next = sigmas[i + 1];

        let velocity = model_fn(&x, sigma)?;

        if sigma_next == 0.0 {
            // Final step: return denoised directly
            let denoised = x.sub(&velocity.mul_scalar(sigma)?)?;
            x = denoised;
        } else {
            // Euler step: d = velocity, x += d * dt
            let dt = sigma_next - sigma;
            x = x.add(&velocity.mul_scalar(dt)?)?;
        }

        println!("  Step {}/{} sigma={:.4}", i + 1, total, sigma);
    }

    Ok(x)
}
