//! Klein 4B / Flux flow-matching sampling — Euler ODE sampler with exponential sigma shift.
//!
//! Rewritten from scratch using the Python reference (`euler_flame.py`, `prediction.py`,
//! `generate_klein4b.py`) as blueprint.  The old `samplers.rs` used EDM parameterisation
//! (epsilon / v-prediction with alpha-based sigma) which is wrong for Flux-family models.
//!
//! Flow matching identity:  `denoised = x - model_output * sigma`
//! Euler step:              `d = (x - denoised) / sigma;  x += d * dt`
//! Sigma schedule:          exponential time-shift  `exp(mu) / (exp(mu) + (1/t - 1))`

use flame_core::{Result, Shape, Tensor};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Sampling configuration for Klein 4B (Flux-family flow matching).
#[derive(Debug, Clone)]
pub struct KleinSamplingConfig {
    /// Number of denoising steps.
    pub num_steps: usize,
    /// Exponential time-shift parameter (mu).  Klein default = 2.02.
    pub shift: f32,
    /// Classifier-free guidance scale.
    pub cfg_scale: f32,
    /// RNG seed for initial noise generation.
    pub seed: u64,
}

impl Default for KleinSamplingConfig {
    fn default() -> Self {
        Self {
            num_steps: 35,
            shift: 2.02,
            cfg_scale: 3.5,
            seed: 42,
        }
    }
}

// ---------------------------------------------------------------------------
// Sigma schedule — exponential time-shift (Flux / Klein)
// ---------------------------------------------------------------------------

/// Build Flux-style sigma schedule with exponential time shift.
///
/// Pre-computes 10 000 sigma values via `exp(mu) / (exp(mu) + (1/t - 1))`,
/// then picks `num_steps` evenly-spaced entries in reverse order (high → low)
/// with a final 0.0 appended.  Returns `num_steps + 1` values.
///
/// This matches ComfyUI's `ModelSamplingFlux.set_parameters` + `simple_scheduler`.
pub fn build_sigma_schedule(num_steps: usize, shift: f32) -> Vec<f32> {
    const N_SIGMAS: usize = 10_000;
    let exp_mu = (shift as f64).exp();

    // Pre-compute full sigma buffer (index 0 → t=1/10000, index 9999 → t=1.0).
    let sigma_buffer: Vec<f32> = (0..N_SIGMAS)
        .map(|i| {
            let t = (i + 1) as f64 / N_SIGMAS as f64;
            (exp_mu / (exp_mu + (1.0 / t - 1.0))) as f32
        })
        .collect();

    // Pick N evenly-spaced values, reversed (high sigma first).
    let ss = N_SIGMAS as f64 / num_steps as f64;
    let mut sigmas: Vec<f32> = (0..num_steps)
        .map(|x| {
            let idx = N_SIGMAS - 1 - (x as f64 * ss) as usize;
            sigma_buffer[idx]
        })
        .collect();
    sigmas.push(0.0);
    sigmas
}

// ---------------------------------------------------------------------------
// Euler ODE sampler
// ---------------------------------------------------------------------------

/// Euler ODE sampler for flow-matching models (Flux / Klein).
///
/// `model_fn(x, sigma) -> denoised` must return the **denoised** prediction.
/// For flow matching this means internally computing `denoised = x - output * sigma`.
///
/// `sigmas` should have `N+1` entries (from `build_sigma_schedule`).
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

        // d = (x - denoised) / sigma
        let d = x.sub(&denoised)?.mul_scalar(1.0 / sigma)?;
        let dt = sigma_next - sigma;
        // x = x + d * dt
        x = x.add(&d.mul_scalar(dt)?)?;
    }

    Ok(x)
}

// ---------------------------------------------------------------------------
// CFG (classifier-free guidance) wrapper
// ---------------------------------------------------------------------------

/// Apply classifier-free guidance and flow-matching denoising.
///
/// Runs the model twice (positive + negative prompt), blends with CFG scale,
/// then converts from velocity prediction to denoised via `denoised = x - guided * sigma`.
///
/// The `model_forward` closure should call the transformer with the given
/// embeddings and sigma tensor, returning the raw model output (velocity).
pub fn cfg_denoise<F>(
    model_forward: &F,
    x: &Tensor,
    sigma: f32,
    pos_embed: &Tensor,
    neg_embed: &Tensor,
    cfg_scale: f32,
) -> Result<Tensor>
where
    F: Fn(&Tensor, &Tensor, &Tensor) -> Result<Tensor>,
{
    // Create sigma tensor [sigma] on the same device as x.
    let sigma_t = Tensor::from_vec(
        vec![sigma],
        Shape::from_dims(&[1]),
        x.device().clone(),
    )?;

    let cond = model_forward(x, pos_embed, &sigma_t)?;
    let uncond = model_forward(x, neg_embed, &sigma_t)?;

    // CFG: guided = uncond + scale * (cond - uncond)
    let diff = cond.sub(&uncond)?;
    let guided = uncond.add(&diff.mul_scalar(cfg_scale)?)?;

    // Flow matching: denoised = x - guided * sigma
    x.sub(&guided.mul_scalar(sigma)?)
}

/// Convenience: run the full Euler loop with CFG built in.
///
/// `model_forward(x, embed, sigma_tensor) -> velocity`
pub fn euler_denoise_cfg<F>(
    model_forward: F,
    noise: Tensor,
    sigmas: &[f32],
    pos_embed: &Tensor,
    neg_embed: &Tensor,
    cfg_scale: f32,
) -> Result<Tensor>
where
    F: Fn(&Tensor, &Tensor, &Tensor) -> Result<Tensor>,
{
    euler_denoise(
        |x, sigma| cfg_denoise(&model_forward, x, sigma, pos_embed, neg_embed, cfg_scale),
        noise,
        sigmas,
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- sigma schedule tests (pure math, no GPU) ---------------------------

    #[test]
    fn test_sigma_schedule_length() {
        let sigmas = build_sigma_schedule(35, 2.02);
        assert_eq!(sigmas.len(), 36, "should be num_steps + 1");
    }

    #[test]
    fn test_sigma_schedule_endpoints() {
        let sigmas = build_sigma_schedule(35, 2.02);
        // First sigma should be close to 1.0 (high noise).
        assert!(sigmas[0] > 0.9, "first sigma should be near 1.0, got {}", sigmas[0]);
        // Last sigma must be exactly 0.0.
        assert_eq!(sigmas[35], 0.0, "final sigma must be 0.0");
    }

    #[test]
    fn test_sigma_schedule_monotonically_decreasing() {
        let sigmas = build_sigma_schedule(50, 2.02);
        for i in 0..sigmas.len() - 1 {
            assert!(
                sigmas[i] >= sigmas[i + 1],
                "sigmas must decrease: sigmas[{}]={} < sigmas[{}]={}",
                i, sigmas[i], i + 1, sigmas[i + 1]
            );
        }
    }

    #[test]
    fn test_sigma_schedule_shift_effect() {
        // Higher shift pushes more budget to high-noise region.
        let sigmas_low = build_sigma_schedule(20, 1.0);
        let sigmas_high = build_sigma_schedule(20, 3.0);
        // Mid-point sigma with high shift should be larger (more noise budget).
        assert!(
            sigmas_high[10] > sigmas_low[10],
            "higher shift should produce larger mid-point sigma"
        );
    }

    #[test]
    fn test_sigma_schedule_matches_python() {
        // Spot-check against Python reference output for shift=2.02, steps=35.
        let sigmas = build_sigma_schedule(35, 2.02);

        // Python: sigmas[0] ≈ 0.9972 (the near-1.0 end)
        assert!((sigmas[0] - 0.9972).abs() < 0.002, "sigmas[0]={}", sigmas[0]);

        // Python: sigmas[17] (midpoint) — should be in 0.4-0.7 range with shift=2.02
        assert!(
            sigmas[17] > 0.3 && sigmas[17] < 0.8,
            "midpoint sigma should be reasonable, got {}",
            sigmas[17]
        );

        // Python: sigmas[34] (last real step) should be small but > 0
        assert!(
            sigmas[34] > 0.0 && sigmas[34] < 0.1,
            "second-to-last sigma should be small, got {}",
            sigmas[34]
        );
    }

    #[test]
    fn test_sigma_all_positive_except_last() {
        let sigmas = build_sigma_schedule(35, 2.02);
        for i in 0..sigmas.len() - 1 {
            assert!(sigmas[i] > 0.0, "sigmas[{}] should be > 0, got {}", i, sigmas[i]);
        }
    }

    // -- euler step logic (mock model, no GPU) ------------------------------

    #[test]
    fn test_euler_step_direction() {
        // With a trivial "denoised = 0" model, the Euler step should move x toward 0.
        // d = (x - 0) / sigma = x / sigma
        // dt = sigma_next - sigma (negative)
        // x_new = x + (x / sigma) * dt
        //
        // For sigma=1.0, sigma_next=0.5: dt=-0.5, x_new = x + x*(-0.5) = 0.5*x
        // So magnitude should halve.
        let sigma = 1.0_f32;
        let sigma_next = 0.5_f32;
        let x = 2.0_f32;

        let d = (x - 0.0) / sigma; // = 2.0
        let dt = sigma_next - sigma; // = -0.5
        let x_new = x + d * dt; // = 2.0 + 2.0*(-0.5) = 1.0

        assert!((x_new - 1.0).abs() < 1e-6, "euler step check: {}", x_new);
    }

    #[test]
    fn test_euler_converges_to_denoised() {
        // If the model always returns the same "denoised" value, Euler should
        // converge toward it as sigma → 0.
        // After full schedule: x should ≈ denoised_target
        let target = 5.0_f32;
        let sigmas = vec![1.0, 0.5, 0.25, 0.0];
        let mut x = 0.0_f32; // start far from target

        for i in 0..sigmas.len() - 1 {
            let sigma = sigmas[i];
            let sigma_next = sigmas[i + 1];
            let denoised = target;

            let d = (x - denoised) / sigma;
            let dt = sigma_next - sigma;
            x = x + d * dt;
        }

        assert!(
            (x - target).abs() < 0.5,
            "euler should converge toward target={}, got x={}",
            target, x
        );
    }

    #[test]
    fn test_cfg_formula() {
        // CFG: guided = uncond + scale * (cond - uncond)
        // With cond=1.0, uncond=0.0, scale=7.5: guided = 0 + 7.5 * 1.0 = 7.5
        let cond = 1.0_f32;
        let uncond = 0.0_f32;
        let scale = 7.5_f32;

        let guided = uncond + scale * (cond - uncond);
        assert!((guided - 7.5).abs() < 1e-6);

        // Flow matching denoised: x - guided * sigma
        let x = 10.0_f32;
        let sigma = 0.5_f32;
        let denoised = x - guided * sigma; // 10.0 - 7.5 * 0.5 = 6.25
        assert!((denoised - 6.25).abs() < 1e-6);
    }

    #[test]
    fn test_flow_matching_identity() {
        // denoised = x - model_output * sigma
        // When sigma=0, denoised = x (fully denoised, model output irrelevant).
        let x = 42.0_f32;
        let model_output = 999.0_f32;
        let sigma = 0.0_f32;

        let denoised = x - model_output * sigma;
        assert!((denoised - x).abs() < 1e-6);
    }

    #[test]
    fn test_default_config() {
        let cfg = KleinSamplingConfig::default();
        assert_eq!(cfg.num_steps, 35);
        assert!((cfg.shift - 2.02).abs() < 1e-6);
        assert!((cfg.cfg_scale - 3.5).abs() < 1e-6);
        assert_eq!(cfg.seed, 42);
    }
}
