//! Timestep biasing — reshape the per-step training timestep distribution.
//!
//! Sampled timesteps come from `sample_timestep_logit_normal` (or analogous
//! samplers in other trainers) on `[0, NUM_TRAIN_TIMESTEPS)`. This module
//! takes the raw sampled `t` and remaps it according to a strategy, so the
//! caller can up-weight high-noise / low-noise regimes or restrict training
//! to a sub-range without changing the base sampler.
//!
//! Default-off invariance: `Strategy::None` returns `t` unchanged.
//!
//! Reference: SimpleTuner `--timestep-bias-strategy {later,earlier,range,custom}`.

/// Bias strategy for the sampled training timestep.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strategy {
    /// No biasing — return `t` unchanged.
    None,
    /// Pull `t` toward `NUM_TRAIN_TIMESTEPS` (high-noise regime). With
    /// `multiplier = 0.0` this is a no-op; at `1.0` every sample is at
    /// the maximum timestep. Linear blend in between.
    Later,
    /// Pull `t` toward 0 (low-noise regime). Same blend semantics as
    /// `Later`.
    Earlier,
    /// Remap `t` from `[0, total)` linearly into `[range_min*total, range_max*total)`.
    /// Preserves the *shape* of the input distribution while clipping it
    /// to the chosen sub-range. `range_min < range_max`, both in `[0, 1]`.
    Range,
}

impl Strategy {
    /// Parse from CLI string. Case-insensitive. Unknown → error.
    pub fn parse(s: &str) -> Result<Self, String> {
        Ok(match s.to_ascii_lowercase().as_str() {
            "" | "none" | "off" => Self::None,
            "later" | "later_timesteps" => Self::Later,
            "earlier" | "earlier_timesteps" => Self::Earlier,
            "range" | "ranged" => Self::Range,
            other => {
                return Err(format!(
                    "--timestep-bias-strategy must be one of \
                     none|later|earlier|range, got `{other}`"
                ));
            }
        })
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Later => "later",
            Self::Earlier => "earlier",
            Self::Range => "range",
        }
    }
}

/// Knobs for the bias.
#[derive(Debug, Clone, Copy)]
pub struct BiasConfig {
    pub strategy: Strategy,
    /// Strength for `Later` / `Earlier`. Clamped to `[0.0, 1.0]`. Ignored
    /// for `None` and `Range`.
    pub multiplier: f32,
    /// Lower bound for `Range`, fraction of `NUM_TRAIN_TIMESTEPS` in
    /// `[0, 1]`. Clamped + ordered against `range_max` at apply time.
    pub range_min: f32,
    /// Upper bound for `Range`, fraction in `[0, 1]`.
    pub range_max: f32,
}

impl Default for BiasConfig {
    fn default() -> Self {
        Self {
            strategy: Strategy::None,
            multiplier: 0.0,
            range_min: 0.0,
            range_max: 1.0,
        }
    }
}

/// Apply the bias to one sampled timestep `t ∈ [0, total)`.
///
/// `total` is the trainer's `NUM_TRAIN_TIMESTEPS` (typically 1000).
/// Output is in `[0, total)` for `None`/`Earlier`/`Later`/`Range`. The
/// `Range` variant maps the input range linearly into the configured
/// fractional bounds; if the bounds collapse (`range_min == range_max`)
/// every sample lands at exactly that value.
pub fn apply_bias(t: f32, total: f32, cfg: &BiasConfig) -> f32 {
    match cfg.strategy {
        Strategy::None => t,
        Strategy::Later => {
            let m = cfg.multiplier.clamp(0.0, 1.0);
            // Linear blend toward the upper bound: t' = t + m·(total − t).
            t + m * (total - t)
        }
        Strategy::Earlier => {
            let m = cfg.multiplier.clamp(0.0, 1.0);
            // Linear blend toward 0: t' = t · (1 − m).
            t * (1.0 - m)
        }
        Strategy::Range => {
            let lo = cfg.range_min.clamp(0.0, 1.0);
            let hi = cfg.range_max.clamp(0.0, 1.0).max(lo);
            let frac = (t / total).clamp(0.0, 1.0);
            (lo + frac * (hi - lo)) * total
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn none_is_identity() {
        let cfg = BiasConfig::default();
        assert_eq!(apply_bias(123.0, 1000.0, &cfg), 123.0);
        assert_eq!(apply_bias(999.0, 1000.0, &cfg), 999.0);
        assert_eq!(apply_bias(0.0, 1000.0, &cfg), 0.0);
    }

    #[test]
    fn later_pulls_toward_total() {
        let cfg = BiasConfig {
            strategy: Strategy::Later,
            multiplier: 0.5,
            ..Default::default()
        };
        // t=200, total=1000, m=0.5 → 200 + 0.5*(1000-200) = 600
        assert!(approx(apply_bias(200.0, 1000.0, &cfg), 600.0, 1e-4));
        // m=1.0 fully pulls to total
        let cfg2 = BiasConfig { multiplier: 1.0, ..cfg };
        assert!(approx(apply_bias(0.0, 1000.0, &cfg2), 1000.0, 1e-4));
    }

    #[test]
    fn earlier_pulls_toward_zero() {
        let cfg = BiasConfig {
            strategy: Strategy::Earlier,
            multiplier: 0.5,
            ..Default::default()
        };
        // t=800, m=0.5 → 800*0.5 = 400
        assert!(approx(apply_bias(800.0, 1000.0, &cfg), 400.0, 1e-4));
        let cfg2 = BiasConfig { multiplier: 1.0, ..cfg };
        assert!(approx(apply_bias(999.0, 1000.0, &cfg2), 0.0, 1e-4));
    }

    #[test]
    fn range_remaps_linearly() {
        let cfg = BiasConfig {
            strategy: Strategy::Range,
            range_min: 0.2,
            range_max: 0.8,
            ..Default::default()
        };
        // t=0 → 0.2*1000 = 200; t=1000 → 0.8*1000 = 800; t=500 → 500
        assert!(approx(apply_bias(0.0, 1000.0, &cfg), 200.0, 1e-4));
        assert!(approx(apply_bias(1000.0, 1000.0, &cfg), 800.0, 1e-4));
        assert!(approx(apply_bias(500.0, 1000.0, &cfg), 500.0, 1e-4));
    }

    #[test]
    fn range_collapsed_pins_to_value() {
        let cfg = BiasConfig {
            strategy: Strategy::Range,
            range_min: 0.5,
            range_max: 0.5,
            ..Default::default()
        };
        for t in [0.0, 250.0, 500.0, 750.0, 999.0] {
            assert!(approx(apply_bias(t, 1000.0, &cfg), 500.0, 1e-4));
        }
    }

    #[test]
    fn parse_strategies() {
        assert_eq!(Strategy::parse("none"), Ok(Strategy::None));
        assert_eq!(Strategy::parse(""), Ok(Strategy::None));
        assert_eq!(Strategy::parse("Later"), Ok(Strategy::Later));
        assert_eq!(Strategy::parse("EARLIER"), Ok(Strategy::Earlier));
        assert_eq!(Strategy::parse("range"), Ok(Strategy::Range));
        assert!(Strategy::parse("forwards").is_err());
    }
}
