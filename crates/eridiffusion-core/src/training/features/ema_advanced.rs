//! EMA (advanced) — power-decay schedule with warmup, min/max decay clamps,
//! and validation-time parameter swap.
//!
//! Phase 3.
//!
//! Decay formula (matches diffusers `EMAModel` and SimpleTuner
//! `helpers/training/ema.py`):
//!
//! ```text
//!   step_eff = step - update_after_step
//!   decay    = 1 - (1 + step_eff / inv_gamma)^(-power)
//!   decay    = clamp(decay, min_decay, max_decay)
//!   decay    = 0.0   if step <= update_after_step
//! ```
//!
//! `decay = 0.0` means "skip this update" (the consumer short-circuits when
//! the schedule returns 0).
//!
//! Default-off invariance: leaving `inv_gamma=1.0`, `power=0.6667`,
//! `update_after_step=0`, `min_decay=0.0`, `max_decay=0.999` yields the
//! standard warmup curve. Existing trainers that don't construct EMA at all
//! are unaffected.
//!
//! Reference: diffusers `EMAModel`, SimpleTuner `helpers/training/ema.py`.

/// Knobs for EMA decay schedule.
#[derive(Debug, Clone, Copy)]
pub struct EmaConfig {
    pub inv_gamma: f32,
    pub power: f32,
    pub update_after_step: u64,
    pub min_decay: f32,
    pub max_decay: f32,
}

impl Default for EmaConfig {
    fn default() -> Self {
        Self {
            inv_gamma: 1.0,
            power: 0.6667,
            update_after_step: 0,
            min_decay: 0.0,
            max_decay: 0.9999,
        }
    }
}

/// Decay multiplier `α(step)` for the EMA update
/// `shadow = α·shadow + (1-α)·param`.
///
/// Returns `0.0` when `step <= update_after_step` (caller treats this as
/// "skip"). After warmup, returns the diffusers-compatible power-decay
/// curve clamped to `[min_decay, max_decay]`.
pub fn decay_at_step(cfg: &EmaConfig, step: u64) -> f32 {
    if step <= cfg.update_after_step {
        return 0.0;
    }
    let effective_step = (step - cfg.update_after_step) as f32;
    let inv_gamma = cfg.inv_gamma.max(1e-8);
    let value = 1.0 - (1.0 + effective_step / inv_gamma).powf(-cfg.power);
    value.max(cfg.min_decay).min(cfg.max_decay)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn pre_warmup_returns_zero() {
        let cfg = EmaConfig {
            update_after_step: 100,
            ..Default::default()
        };
        assert_eq!(decay_at_step(&cfg, 0), 0.0);
        assert_eq!(decay_at_step(&cfg, 50), 0.0);
        assert_eq!(decay_at_step(&cfg, 100), 0.0);
        assert!(decay_at_step(&cfg, 101) > 0.0);
    }

    #[test]
    fn diffusers_formula_step1() {
        // With defaults inv_gamma=1.0, power=0.6667, update_after_step=0:
        // step=1: 1 - (1 + 1/1)^(-0.6667) = 1 - 2^(-0.6667) ≈ 1 - 0.6300 = 0.3700
        let cfg = EmaConfig::default();
        let d = decay_at_step(&cfg, 1);
        let expected = 1.0 - 2.0_f32.powf(-0.6667);
        assert!(approx(d, expected, 1e-5), "got {} expected {}", d, expected);
    }

    #[test]
    fn approaches_max_decay_at_high_step() {
        let cfg = EmaConfig::default();
        let d = decay_at_step(&cfg, 1_000_000);
        // Should be very close to max_decay = 0.9999.
        assert!(d <= cfg.max_decay + 1e-7);
        assert!(d > 0.999, "expected near max_decay, got {}", d);
    }

    #[test]
    fn min_decay_floors_early_warmup() {
        let cfg = EmaConfig {
            min_decay: 0.5,
            ..Default::default()
        };
        // step=1 raw value ≈ 0.37, should be floored to 0.5.
        let d = decay_at_step(&cfg, 1);
        assert!(approx(d, 0.5, 1e-6), "got {}", d);
    }

    #[test]
    fn max_decay_clamps_late_curve() {
        let cfg = EmaConfig {
            max_decay: 0.95,
            ..Default::default()
        };
        let d = decay_at_step(&cfg, 1_000_000);
        assert!(approx(d, 0.95, 1e-6), "got {}", d);
    }

    #[test]
    fn update_after_step_offset() {
        // With update_after_step=10, step=11 should have effective_step=1
        // and return the same value as step=1 with update_after_step=0.
        let cfg_a = EmaConfig::default();
        let cfg_b = EmaConfig {
            update_after_step: 10,
            ..Default::default()
        };
        let a = decay_at_step(&cfg_a, 1);
        let b = decay_at_step(&cfg_b, 11);
        assert!(approx(a, b, 1e-7), "{} != {}", a, b);
    }
}
