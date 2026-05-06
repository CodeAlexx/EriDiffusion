//! Learning-rate schedules and flow-matching timestep sampling.
//!
//! Ported verbatim from flame-diffusion/src/schedule.rs (2026-05-05). Pure
//! CPU math — no flame-core deps. Used by Flux/SD3/Klein/Wan-family trainers.

use rand::Rng;

/// Cosine learning-rate decay from `base_lr` over `total_steps`.
pub fn cosine_lr(base_lr: f32, step: usize, total_steps: usize) -> f32 {
    if total_steps <= 1 {
        return base_lr;
    }

    let progress = step as f32 / (total_steps.saturating_sub(1)) as f32;
    let cosine = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
    base_lr * cosine
}

/// Constant learning rate with linear warmup.
pub fn constant_with_warmup(base_lr: f32, step: usize, warmup_steps: usize) -> f32 {
    if warmup_steps == 0 || step >= warmup_steps {
        base_lr
    } else {
        base_lr * (step as f32 + 1.0) / warmup_steps as f32
    }
}

/// Logit-normal timestep sample in (0, 1).
///
/// `t = sigmoid(N(mean, std))`. Defaults `mean=0.0, std=1.0`.
pub fn sample_timestep_logit_normal<R: Rng>(rng: &mut R, mean: f32, std: f32) -> f32 {
    let u1 = rng.r#gen::<f32>().max(1.0e-6);
    let u2 = rng.r#gen::<f32>();
    let mag = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * std::f32::consts::PI * u2;
    let z = mag * theta.cos();
    let u = z * std + mean;
    1.0 / (1.0 + (-u).exp())
}

/// Apply Flux-family resolution-dependent shift to a normalized timestep
/// in (0, 1). With `shift = 1.0` this is identity.
///
/// `t' = shift * t / ((shift - 1) * t + 1)`
pub fn apply_resolution_shift(t: f32, shift: f32) -> f32 {
    if (shift - 1.0).abs() < 1.0e-6 {
        return t;
    }
    shift * t / ((shift - 1.0) * t + 1.0)
}

/// Dynamic resolution shift `mu` from image token count, then `shift = exp(mu)`.
/// Flux defaults: base_seq_len=256, max_seq_len=4096, base_shift=0.5, max_shift=1.15.
pub fn dynamic_shift(
    image_seq_len: usize,
    base_seq_len: usize,
    max_seq_len: usize,
    base_shift: f32,
    max_shift: f32,
) -> f32 {
    let m = (max_shift - base_shift) / (max_seq_len as f32 - base_seq_len as f32);
    let b = base_shift - m * base_seq_len as f32;
    let mu = image_seq_len as f32 * m + b;
    mu.exp()
}

/// Sample a flow-matching training timestep: logit-normal then resolution shift.
///
/// Pass `image_seq_len = (latent_h/patch) * (latent_w/patch)`.
/// `dynamic=false` uses the constant `shift`.
pub fn sample_flow_timestep<R: Rng>(
    rng: &mut R,
    mean: f32,
    std: f32,
    shift: f32,
    image_seq_len: Option<usize>,
    dynamic: bool,
) -> f32 {
    let t_raw = sample_timestep_logit_normal(rng, mean, std);
    let actual_shift = if dynamic {
        dynamic_shift(image_seq_len.unwrap_or(1024), 256, 4096, 0.5, 1.15)
    } else {
        shift
    };
    let t = apply_resolution_shift(t_raw, actual_shift);
    t.clamp(1.0e-4, 1.0 - 1.0e-4)
}
