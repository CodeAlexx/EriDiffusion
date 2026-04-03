//! Z-Image / NextDiT sampling — Euler and DPM++ 2M flow samplers.
//!
//! Z-Image uses linear sigma schedule with SNR time shift.
//! Sampling formula: denoised = x - model_output * sigma.

use flame_core::{DType, Result, Tensor};

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

/// Euler velocity step (musubi convention).
///
/// `model_fn(x, sigma) -> velocity` returns the velocity directly.
/// Step: `x = x + dt * velocity` where `dt = sigma_next - sigma`.
pub fn euler_denoise<F>(model_fn: F, noise: Tensor, sigmas: &[f32]) -> Result<Tensor>
where
    F: Fn(&Tensor, f32) -> Result<Tensor>,
{
    let mut x = noise;
    let total = sigmas.len() - 1;

    for i in 0..total {
        let sigma = sigmas[i];
        let sigma_next = sigmas[i + 1];

        let velocity = model_fn(&x, sigma)?;

        // Direct velocity step: x += velocity * dt
        let dt = sigma_next - sigma;
        x = x.add(&velocity.mul_scalar(dt)?)?;

        if i == 0 || (i + 1) % 5 == 0 || i + 1 == total {
            println!("  Step {}/{} sigma={:.4}", i + 1, total, sigma);
        }
    }

    Ok(x)
}

/// Flow-matching log-SNR: lambda = log((1 - t) / t) where t = sigma.
fn sigma_to_lambda(sigma: f32) -> f32 {
    let t = sigma.clamp(1e-8, 1.0 - 1e-8);
    ((1.0 - t) / t).ln()
}

/// DPM++ 2M sampler for flow-matching models (second-order, deterministic).
///
/// Uses lambda_t = log((1-t)/t) as the time variable. First step uses
/// first-order update; subsequent steps use second-order multistep
/// correction from the previous denoised prediction.
///
/// Port of serenity-inference `flow_dpm_pp_2m_sample`.
///
/// Reference: Lu et al. 2022 "DPM-Solver++: Fast Solver for Guided
/// Sampling of Diffusion Probabilistic Models"
/// DPM++ 2M sampler for flow-matching models (second-order, deterministic).
///
/// Accumulation runs in F32 to avoid BF16 quantization artifacts.
/// Model forward pass stays BF16 (x is converted on the fly).
///
/// Port of serenity-inference `flow_dpm_pp_2m_sample`.
pub fn dpm_pp_2m_denoise<F>(model_fn: F, noise: Tensor, sigmas: &[f32]) -> Result<Tensor>
where
    F: Fn(&Tensor, f32) -> Result<Tensor>,
{
    // Accumulate in F32 to avoid BF16 quantization grid
    let mut x = noise.to_dtype(DType::F32)?;
    let total = sigmas.len() - 1;
    let mut old_denoised: Option<Tensor> = None;

    for i in 0..total {
        let sigma = sigmas[i];
        let sigma_next = sigmas[i + 1];

        // Model expects BF16 input, returns BF16 — convert on the fly
        let x_bf16 = x.to_dtype(DType::BF16)?;
        let denoised_bf16 = model_fn(&x_bf16, sigma)?;
        let denoised = denoised_bf16.to_dtype(DType::F32)?;

        println!("  Step {}/{} sigma={:.4}", i + 1, total, sigma);

        // Final step: return denoised directly
        if sigma_next == 0.0 {
            x = denoised;
            break;
        }

        // Flow parameterization: alpha = 1-t, sig = t (where t = sigma)
        let sig_s = sigma.clamp(1e-8, 1.0 - 1e-8);
        let sig_t = sigma_next.clamp(1e-8, 1.0 - 1e-8);
        let alpha_t = 1.0 - sig_t;

        let lam_s = sigma_to_lambda(sigma);
        let lam_t = sigma_to_lambda(sigma_next);
        let h = lam_t - lam_s;

        let neg_h_expm1 = (-h as f64).exp_m1() as f32;

        if old_denoised.is_none() {
            let ratio = sig_t / sig_s;
            let coeff = alpha_t * neg_h_expm1;
            x = x.mul_scalar(ratio)?.sub(&denoised.mul_scalar(coeff)?)?;
        } else {
            let lam_prev = sigma_to_lambda(sigmas[i - 1]);
            let h_last = lam_s - lam_prev;
            let r = h_last / h;

            let c1 = 1.0 + 0.5 / r;
            let c2 = 0.5 / r;
            let d = denoised.mul_scalar(c1)?.sub(&old_denoised.as_ref().unwrap().mul_scalar(c2)?)?;

            let ratio = sig_t / sig_s;
            let coeff = alpha_t * neg_h_expm1;
            x = x.mul_scalar(ratio)?.sub(&d.mul_scalar(coeff)?)?;
        }

        old_denoised = Some(denoised);
    }

    // Return as BF16 for VAE decode
    x.to_dtype(DType::BF16)
}
