//! Shared utilities: timestep sampling, noise schedules

/// Logit-normal timestep sample in (0, 1)
pub fn sample_timestep_logit_normal(seed: &mut u64) -> f32 {
    use rand::{Rng, SeedableRng};
    use rand::rngs::StdRng;
    let mut rng = StdRng::seed_from_u64(*seed);
    *seed = seed.wrapping_add(1);
    let u1 = rng.gen::<f32>().max(1e-6);
    let u2 = rng.gen::<f32>();
    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
    1.0 / (1.0 + (-z).exp())
}

/// Uniform timestep in [0, max)
pub fn sample_timestep_uniform(seed: &mut u64, max: f32) -> f32 {
    use rand::{Rng, SeedableRng};
    use rand::rngs::StdRng;
    let mut rng = StdRng::seed_from_u64(*seed);
    *seed = seed.wrapping_add(1);
    rng.gen::<f32>() * max
}

/// Flux-family resolution-dependent shift
pub fn apply_resolution_shift(t: f32, shift: f32) -> f32 {
    if (shift - 1.0).abs() < 1e-6 { return t; }
    shift * t / ((shift - 1.0) * t + 1.0)
}

/// Cosine learning-rate decay
pub fn cosine_lr(base_lr: f32, step: usize, total_steps: usize) -> f32 {
    if total_steps <= 1 { return base_lr; }
    let progress = step as f32 / (total_steps.saturating_sub(1)) as f32;
    base_lr * 0.5 * (1.0 + (std::f32::consts::PI * progress).cos())
}

/// Linear warmup
pub fn lr_with_warmup(base_lr: f32, step: usize, warmup: usize) -> f32 {
    if warmup == 0 || step >= warmup { base_lr }
    else { base_lr * (step as f32 + 1.0) / warmup as f32 }
}
