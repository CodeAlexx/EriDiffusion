//! Learning-rate scheduler dispatch — central entry point for all LR schedules.
//!
//! Phase 5 wires the full enum surface:
//!   - LrScheduler::Constant            → linear warmup → flat base_lr
//!   - LrScheduler::Linear              → linear decay base→base*min_factor
//!   - LrScheduler::Cosine              → cosine decay with floor
//!   - LrScheduler::CosineWithRestarts  → cosine with hard restarts every `cycles`
//!   - LrScheduler::Polynomial          → polynomial decay (default power=2)
//!   - LrScheduler::Rex                 → REX (Mishra & Sarawagi 2019)
//!
//! Default-off byte invariance: when scheduler == Constant, dispatch_lr is
//! bit-equivalent to the legacy `constant_with_warmup` path Klein/ACE-Step
//! used pre-Phase 5 for the same (step, warmup_steps) inputs.
//!
//! Config flags:
//!   - `learning_rate_scheduler: LrScheduler` (existing)
//!   - `learning_rate_warmup_steps: f64` (existing)
//!   - `learning_rate_cycles: f64` (existing, only meaningful for restarts)
//!   - `lr_min_factor: f32` (default 0.0)
//!
//! Reference: SimpleTuner `helpers/training/scheduler.py`,
//!            OneTrainer `modules/util/lr_scheduler_util.py`.

use crate::config::LrScheduler;

/// Constant LR with linear warmup. Byte-equivalent to legacy
/// `crate::training::schedule::constant_with_warmup` for the same inputs.
pub fn constant_lr(base_lr: f32, step: usize, warmup_steps: usize) -> f32 {
    if warmup_steps == 0 || step >= warmup_steps {
        base_lr
    } else {
        base_lr * (step as f32 + 1.0) / warmup_steps as f32
    }
}

/// Linear decay from `base_lr` at `step == warmup_steps` to
/// `min_factor * base_lr` at `step == total_steps`.
pub fn linear_lr(
    base_lr: f32,
    step: usize,
    total_steps: usize,
    warmup_steps: usize,
    min_factor: f32,
) -> f32 {
    if step < warmup_steps {
        return constant_lr(base_lr, step, warmup_steps);
    }
    let denom = total_steps.saturating_sub(warmup_steps).max(1) as f32;
    let progress = ((step - warmup_steps) as f32 / denom).clamp(0.0, 1.0);
    base_lr * (1.0 - (1.0 - min_factor) * progress)
}

/// Cosine decay from `base_lr` to `min_factor * base_lr` over the
/// post-warmup horizon.
pub fn cosine_lr(
    base_lr: f32,
    step: usize,
    total_steps: usize,
    warmup_steps: usize,
    min_factor: f32,
) -> f32 {
    if step < warmup_steps {
        return constant_lr(base_lr, step, warmup_steps);
    }
    let denom = total_steps.saturating_sub(warmup_steps).max(1) as f32;
    let progress = ((step - warmup_steps) as f32 / denom).clamp(0.0, 1.0);
    let cos_factor = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
    base_lr * (min_factor + (1.0 - min_factor) * cos_factor)
}

/// Cosine with `cycles` hard restarts. Each cycle decays from base_lr to
/// `min_factor * base_lr`. Cycles is f32 (matches `learning_rate_cycles` field).
pub fn cosine_restarts_lr(
    base_lr: f32,
    step: usize,
    total_steps: usize,
    warmup_steps: usize,
    min_factor: f32,
    cycles: f32,
) -> f32 {
    if step < warmup_steps {
        return constant_lr(base_lr, step, warmup_steps);
    }
    let denom = total_steps.saturating_sub(warmup_steps).max(1) as f32;
    let progress = ((step - warmup_steps) as f32 / denom).clamp(0.0, 1.0);
    let cycle_progress = (progress * cycles.max(1.0)).fract();
    let cos_factor = 0.5 * (1.0 + (std::f32::consts::PI * cycle_progress).cos());
    base_lr * (min_factor + (1.0 - min_factor) * cos_factor)
}

/// Polynomial decay with given `power` (default 2.0). Linear is `power=1.0`.
pub fn polynomial_lr(
    base_lr: f32,
    step: usize,
    total_steps: usize,
    warmup_steps: usize,
    min_factor: f32,
    power: f32,
) -> f32 {
    if step < warmup_steps {
        return constant_lr(base_lr, step, warmup_steps);
    }
    let denom = total_steps.saturating_sub(warmup_steps).max(1) as f32;
    let progress = ((step - warmup_steps) as f32 / denom).clamp(0.0, 1.0);
    let factor = (1.0 - progress).powf(power);
    base_lr * (min_factor + (1.0 - min_factor) * factor)
}

/// REX (reflected exponential) schedule from Mishra & Sarawagi 2019,
/// "REX: Revisiting Budgeted Training with an Improved Schedule".
/// Decays smoother than cosine in the tail.
///
/// `f(p) = (1 - p) / (1 - 0.5 * p)` gives `f(0) = 1`, `f(1) = 0`,
/// monotone decreasing. Lerped with `min_factor` floor.
pub fn rex_lr(
    base_lr: f32,
    step: usize,
    total_steps: usize,
    warmup_steps: usize,
    min_factor: f32,
) -> f32 {
    if step < warmup_steps {
        return constant_lr(base_lr, step, warmup_steps);
    }
    let denom = total_steps.saturating_sub(warmup_steps).max(1) as f32;
    let progress = ((step - warmup_steps) as f32 / denom).clamp(0.0, 1.0);
    let factor = (1.0 - progress) / (1.0 - 0.5 * progress);
    base_lr * (min_factor + (1.0 - min_factor) * factor.max(0.0))
}

/// Dispatch a learning-rate value for `step` based on the scheduler kind.
///
/// **Default-off invariance**: when `sched == Constant`, the result is
/// byte-identical to `crate::training::schedule::constant_with_warmup` for
/// the same `(base_lr, step, warmup_steps)`.
pub fn dispatch_lr(
    sched: &LrScheduler,
    base_lr: f32,
    step: usize,
    total_steps: usize,
    warmup_steps: usize,
    min_factor: f32,
    cycles: f32,
) -> f32 {
    match sched {
        LrScheduler::Constant => constant_lr(base_lr, step, warmup_steps),
        LrScheduler::Linear => linear_lr(base_lr, step, total_steps, warmup_steps, min_factor),
        LrScheduler::Cosine => cosine_lr(base_lr, step, total_steps, warmup_steps, min_factor),
        LrScheduler::CosineWithRestarts => cosine_restarts_lr(
            base_lr, step, total_steps, warmup_steps, min_factor, cycles,
        ),
        LrScheduler::Polynomial => {
            polynomial_lr(base_lr, step, total_steps, warmup_steps, min_factor, 2.0)
        }
        LrScheduler::Rex => rex_lr(base_lr, step, total_steps, warmup_steps, min_factor),
    }
}

/// Back-compat alias for the Phase 0 dispatch fn name (callers used `dispatch`).
#[deprecated(note = "Use dispatch_lr instead")]
pub fn dispatch(
    scheduler: LrScheduler,
    base_lr: f32,
    step: usize,
    total_steps: usize,
    warmup: usize,
    min_factor: f32,
    cycles: f32,
) -> f32 {
    dispatch_lr(&scheduler, base_lr, step, total_steps, warmup, min_factor, cycles)
}

/// Parse a CLI string like "constant" / "cosine" into [`LrScheduler`], with
/// a logged fallback to Constant on bad input.
pub fn parse_cli_scheduler(s: &str) -> LrScheduler {
    s.parse().unwrap_or_else(|e: String| {
        log::warn!("[lr_scheduler] {e} — falling back to Constant");
        LrScheduler::Constant
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::schedule::constant_with_warmup;

    #[test]
    fn constant_lr_matches_legacy_constant_with_warmup() {
        // Sweep representative (step, warmup) tuples — Klein-style runs.
        for &warmup in &[0usize, 1, 50, 100, 200] {
            for &step in &[0usize, 1, 5, 49, 99, 100, 199, 200, 1000] {
                let base = 3e-5_f32;
                let new_v = constant_lr(base, step, warmup);
                let old_v = constant_with_warmup(base, step, warmup);
                assert_eq!(
                    new_v.to_bits(),
                    old_v.to_bits(),
                    "constant_lr({base}, {step}, {warmup}) = {new_v} != {old_v}"
                );
            }
        }
    }

    #[test]
    fn dispatch_constant_matches_legacy() {
        // Default Constant variant must be byte-equivalent.
        for &warmup in &[0usize, 100, 200] {
            for &step in &[0usize, 5, 99, 200, 1500] {
                let base = 3e-5_f32;
                let new_v = dispatch_lr(
                    &LrScheduler::Constant,
                    base, step, /*total*/ 3000, warmup,
                    /*min_factor*/ 0.0, /*cycles*/ 1.0,
                );
                let old_v = constant_with_warmup(base, step, warmup);
                assert_eq!(new_v.to_bits(), old_v.to_bits());
            }
        }
    }

    #[test]
    fn cosine_endpoints() {
        let base = 1.0_f32;
        let total = 1000;
        let warmup = 0;
        // At step == warmup, progress=0 → factor=1.0 → base_lr.
        let v0 = cosine_lr(base, warmup, total, warmup, 0.0);
        assert!((v0 - base).abs() < 1e-6, "cosine at progress=0: {v0}");
        // At step == total, progress=1 → factor=0 → min_factor*base.
        let v1 = cosine_lr(base, total, total, warmup, 0.1);
        assert!((v1 - 0.1 * base).abs() < 1e-6, "cosine at progress=1: {v1}");
    }

    #[test]
    fn linear_is_monotonic_decreasing() {
        let base = 1.0_f32;
        let total = 1000;
        let warmup = 0;
        let mut prev = f32::INFINITY;
        for step in (0..=total).step_by(50) {
            let v = linear_lr(base, step, total, warmup, 0.0);
            assert!(v <= prev + 1e-6, "linear not monotone at {step}: {v} > {prev}");
            prev = v;
        }
    }

    #[test]
    fn cosine_restarts_two_minima() {
        // With cycles=2: cycle 0 covers progress [0, 0.5), cycle 1 covers
        // [0.5, 1.0]. Each cycle's cosine factor decays 1→0 across its span.
        // We expect mid-cycle samples to be < base, near-end samples ≈ 0.
        let base = 1.0_f32;
        let total = 1000;
        let warmup = 0;
        // 1/4 progress = mid of cycle 0, cos(pi*0.5) = 0 → factor 0.5.
        let v_quarter = cosine_restarts_lr(base, 250, total, warmup, 0.0, 2.0);
        assert!((v_quarter - 0.5).abs() < 1e-4, "v_quarter ≈ 0.5: got {v_quarter}");
        // 3/4 progress = mid of cycle 1, factor 0.5 again.
        let v_three_q = cosine_restarts_lr(base, 750, total, warmup, 0.0, 2.0);
        assert!((v_three_q - 0.5).abs() < 1e-4, "v_three_q ≈ 0.5: got {v_three_q}");
        // Just before each cycle boundary, factor ≈ 0 (deep minimum).
        let v_pre_half = cosine_restarts_lr(base, 499, total, warmup, 0.0, 2.0);
        assert!(v_pre_half < 0.05, "v_pre_half should be near 0: {v_pre_half}");
    }

    #[test]
    fn polynomial_endpoints() {
        let base = 1.0_f32;
        let total = 1000;
        let warmup = 0;
        let v0 = polynomial_lr(base, 0, total, warmup, 0.0, 2.0);
        let v1 = polynomial_lr(base, total, total, warmup, 0.0, 2.0);
        assert!((v0 - base).abs() < 1e-6);
        assert!(v1.abs() < 1e-6);
    }

    #[test]
    fn rex_endpoints() {
        let base = 1.0_f32;
        let total = 1000;
        let warmup = 0;
        let v0 = rex_lr(base, 0, total, warmup, 0.0);
        let v1 = rex_lr(base, total, total, warmup, 0.0);
        assert!((v0 - base).abs() < 1e-6);
        assert!(v1.abs() < 1e-6);
    }

    #[test]
    fn warmup_ramp_all_variants() {
        // During warmup, every variant must use the constant linear ramp.
        let base = 3e-5_f32;
        let warmup = 100;
        let total = 1000;
        for &step in &[0usize, 1, 50, 99] {
            let expected = base * (step as f32 + 1.0) / warmup as f32;
            for sched in &[
                LrScheduler::Constant,
                LrScheduler::Linear,
                LrScheduler::Cosine,
                LrScheduler::CosineWithRestarts,
                LrScheduler::Polynomial,
                LrScheduler::Rex,
            ] {
                let v = dispatch_lr(sched, base, step, total, warmup, 0.0, 1.0);
                assert!(
                    (v - expected).abs() < 1e-9,
                    "warmup mismatch {sched:?} step={step}: {v} != {expected}"
                );
            }
        }
    }
}
