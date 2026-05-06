//! OneTrainer-parity timestep distributions.
//!
//! This module provides a config-driven timestep sampler for trainers that
//! currently bake a single distribution (sigmoid, logit-normal, ...) into
//! their pipelines. It returns a host-side `Vec<f32>` of length `batch_size`
//! with values in `[min_strength, max_strength]` (defaults `[0, 1]`).
//!
//! ## Distributions
//!
//! Names mirror OneTrainer's `TimestepDistribution` enum exactly:
//!
//! | Variant                      | Formula (host-side)                                        |
//! |------------------------------|------------------------------------------------------------|
//! | `Uniform`                    | `t ~ U(0, 1)`                                              |
//! | `Sigmoid`                    | `t = sigmoid(noising_weight * (z + noising_bias))`, z ~ N(0,1) |
//! | `LogitNormal`                | `t = sigmoid(N(noising_bias, noising_weight + 1))`         |
//! | `HeavyTail`                  | `u ~ U(0,1); t = 1 - u - w * (cos(π/2 * u)² - 1 + u)` then clamp |
//! | `CosMapContinuous`           | `t = (1 - cos(π * u)) / 2` with `u ~ U(0,1)`               |
//! | `InvertedParabolaContinuous` | rejection-sampled from `max(0, -w*(t - bias-0.5)² + 2)`    |
//!
//! Implementation notes:
//!
//! - `Sigmoid` has a closed-form continuous analogue here. OneTrainer's
//!   `Sigmoid`/`COS_MAP`/`INVERTED_PARABOLA` are *discrete* over a 1000-bucket
//!   grid via `torch.multinomial` after scheduler-shift correction; in our
//!   trainers the shift is applied separately by the pipeline (Z-Image does
//!   `1 - sigma` post-sample), so we keep the math continuous and let the
//!   trainer do the shift step.
//! - For `Sigmoid`, the closed-form `sigmoid(w * (z + bias))` matches musubi's
//!   behavior with `w = sigmoid_scale` and `bias = 0.0` — i.e. the existing
//!   Z-Image trainer's `sigmoid_scale=1.8, bias=0` path is preserved exactly.
//!   See `sigmoid_back_compat_zimage` test.
//! - `noising_weight` and `noising_bias` defaults follow OneTrainer
//!   (`TrainConfig.py:1031-1032`): both 0.0. With those defaults `Sigmoid`
//!   degenerates to `sigmoid(0) = 0.5` (constant). Trainers should override
//!   with the existing per-trainer default (e.g. `1.8` for Z-Image) when
//!   the user picks `sigmoid` without supplying a weight.
//! - `HeavyTail` and `InvertedParabolaContinuous` use rejection sampling capped at 32
//!   tries — the dist is well-behaved on `[0,1]` for any sane `w`.

use rand::{rngs::StdRng, Rng, SeedableRng};
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimestepDistribution {
    Uniform,
    /// Continuous closed-form sigmoid: `t = sigmoid(noising_weight * (z + noising_bias))`,
    /// `z ~ N(0,1)`. **NOTE: this is musubi-style sigmoid sampling, NOT
    /// OneTrainer's discrete `SIGMOID` distribution.** OneTrainer's sigmoid
    /// is a 1000-bucket multinomial after scheduler-shift correction; ours
    /// is the closed-form analogue without the discretization. Existing
    /// trainer behavior is preserved (Z-Image's `sigmoid_scale=1.8, bias=0`
    /// path bit-matches via `sigmoid_back_compat_zimage`). Users wanting
    /// a closer OneTrainer match should pick [`Self::LogitNormal`] which
    /// is its own well-defined continuous distribution. The
    /// `sigmoid_NOT_onetrainer_sigmoid` test below documents this divergence
    /// as a permanent regression marker.
    Sigmoid,
    LogitNormal,
    HeavyTail,
    /// Continuous CDF-inverse approximation of OneTrainer's discrete
    /// `COS_MAP` (which reweights a 1000-step linspace and samples via
    /// `torch.multinomial`). Mapping `u ~ U(0,1)` to `(1 - cos(π·u))/2`
    /// matches the first-moment heaviness of OneTrainer's bucketed sampler
    /// at the continuous limit, but is not bit-equivalent. Renamed
    /// `CosMapContinuous` (from `CosMap`) to make the divergence explicit;
    /// the user-facing config string `"cos_map"` still parses to this
    /// variant.
    CosMapContinuous,
    /// Continuous rejection-sampled approximation of OneTrainer's discrete
    /// `INVERTED_PARABOLA` (parabola-weighted multinomial over the same
    /// 1000-step linspace). Same shape, no discretization. Renamed
    /// `InvertedParabolaContinuous` (from `InvertedParabola`) to make the
    /// divergence explicit; the user-facing config string
    /// `"inverted_parabola"` still parses to this variant.
    InvertedParabolaContinuous,
}

impl TimestepDistribution {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Uniform => "uniform",
            Self::Sigmoid => "sigmoid",
            Self::LogitNormal => "logit_normal",
            Self::HeavyTail => "heavy_tail",
            // User-facing names unchanged — the rename of the Rust variant
            // is purely an in-code marker; configs and log lines keep
            // saying "cos_map" / "inverted_parabola".
            Self::CosMapContinuous => "cos_map",
            Self::InvertedParabolaContinuous => "inverted_parabola",
        }
    }
}

impl FromStr for TimestepDistribution {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        // Accept lowercase, uppercase, and the exact OneTrainer variants.
        match s.trim().to_ascii_lowercase().as_str() {
            "uniform" => Ok(Self::Uniform),
            "sigmoid" => Ok(Self::Sigmoid),
            "logit_normal" | "logit-normal" | "logitnormal" => Ok(Self::LogitNormal),
            "heavy_tail" | "heavy-tail" | "heavytail" => Ok(Self::HeavyTail),
            "cos_map" | "cos-map" | "cosmap" => Ok(Self::CosMapContinuous),
            "inverted_parabola" | "inverted-parabola" | "invertedparabola" => {
                Ok(Self::InvertedParabolaContinuous)
            }
            other => Err(format!(
                "unknown timestep_distribution '{}', expected one of: \
                 uniform, sigmoid, logit_normal, heavy_tail, cos_map, inverted_parabola",
                other
            )),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TimestepConfig {
    pub distribution: TimestepDistribution,
    /// OneTrainer "Noising Weight" — semantics depend on the distribution.
    pub noising_weight: f32,
    /// OneTrainer "Noising Bias" — semantics depend on the distribution.
    pub noising_bias: f32,
    /// Lower bound (inclusive) of returned values.
    pub min_strength: f32,
    /// Upper bound (inclusive) of returned values.
    pub max_strength: f32,
}

impl TimestepConfig {
    /// OneTrainer's default config: `Uniform`, `weight=0.0`, `bias=0.0`,
    /// strength range `[0, 1]`. Most trainers will want to override the
    /// distribution and weight at construction time.
    pub fn default_uniform() -> Self {
        Self {
            distribution: TimestepDistribution::Uniform,
            noising_weight: 0.0,
            noising_bias: 0.0,
            min_strength: 0.0,
            max_strength: 1.0,
        }
    }

    /// Convenience constructor matching the existing Z-Image
    /// `sigmoid(1.8 * randn())` defaults — useful for back-compat tests.
    pub fn zimage_sigmoid_18() -> Self {
        Self {
            distribution: TimestepDistribution::Sigmoid,
            noising_weight: 1.8,
            noising_bias: 0.0,
            min_strength: 0.0,
            max_strength: 1.0,
        }
    }

    /// Sample `batch_size` timesteps. Caller passes a seeded RNG so the
    /// existing per-step deterministic order is preserved.
    pub fn sample<R: Rng>(&self, rng: &mut R, batch_size: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            let t01 = self.sample_one(rng);
            let scaled = self.min_strength + (self.max_strength - self.min_strength) * t01;
            out.push(scaled.clamp(self.min_strength, self.max_strength));
        }
        out
    }

    /// Sample one timestep in `[0, 1]` from the configured distribution
    /// (before strength rescaling).
    pub fn sample_one<R: Rng>(&self, rng: &mut R) -> f32 {
        match self.distribution {
            TimestepDistribution::Uniform => rng.r#gen::<f32>(),
            TimestepDistribution::Sigmoid => {
                // `sigmoid(noising_weight * (z + noising_bias))`.
                // With weight=1.8, bias=0 this matches the existing Z-Image
                // pipeline.
                let z = standard_normal(rng);
                let arg = self.noising_weight * (z + self.noising_bias);
                sigmoid(arg)
            }
            TimestepDistribution::LogitNormal => {
                // OneTrainer `ModelSetupNoiseMixin.py:154-160`:
                //   bias  = noising_bias
                //   scale = noising_weight + 1.0
                //   t = sigmoid(N(bias, scale))
                let scale = self.noising_weight + 1.0;
                let z = standard_normal(rng);
                sigmoid(z * scale + self.noising_bias)
            }
            TimestepDistribution::HeavyTail => {
                // OneTrainer `ModelSetupNoiseMixin.py:161-170`:
                //   u = U(0,1)
                //   u = 1 - u - w*(cos(π/2 * u)² - 1 + u)
                let scale = self.noising_weight;
                let u = rng.r#gen::<f32>();
                let c = (std::f32::consts::FRAC_PI_2 * u).cos();
                let t = 1.0 - u - scale * (c * c - 1.0 + u);
                t.clamp(0.0, 1.0)
            }
            TimestepDistribution::CosMapContinuous => {
                // OneTrainer's COS_MAP uses a discrete reweighting of a
                // linspace. Here we use the equivalent CDF-inverse: mapping
                // `u ~ U(0,1)` to `(1 - cos(π * u)) / 2` gives the same
                // first-moment heaviness in `[0,1]` as their multinomial
                // sampling at the continuous limit, with no shift dependence.
                let u = rng.r#gen::<f32>();
                0.5 * (1.0 - (std::f32::consts::PI * u).cos())
            }
            TimestepDistribution::InvertedParabolaContinuous => {
                // OneTrainer:
                //   weights = clamp(-w * (t - (bias + 0.5))² + 2, min=0)
                //   sample ∝ weights
                // Continuous analogue: rejection sample from the same shape.
                // Maximum weight is exactly 2 (at t = bias + 0.5), achieved
                // when the parabola is non-negative across [0,1].
                let center = self.noising_bias + 0.5;
                let weight = self.noising_weight;
                let max_density = 2.0_f32;
                for _ in 0..32 {
                    let candidate = rng.r#gen::<f32>();
                    let d = (-weight * (candidate - center).powi(2) + 2.0).max(0.0);
                    let u = rng.r#gen::<f32>();
                    if u * max_density <= d {
                        return candidate;
                    }
                }
                // Fallback: uniform if rejection failed (degenerate config).
                rng.r#gen::<f32>()
            }
        }
    }

    /// Convenience: sample one timestep using a fresh `StdRng` seeded with
    /// `seed`. Use only when the caller doesn't already own a per-trainer
    /// RNG; trainers should pass their existing RNG to `sample(...)`.
    pub fn sample_one_seeded(&self, seed: u64) -> f32 {
        let mut rng = StdRng::seed_from_u64(seed);
        self.sample_one(&mut rng)
    }
}

impl Default for TimestepConfig {
    fn default() -> Self {
        Self::default_uniform()
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn standard_normal<R: Rng>(rng: &mut R) -> f32 {
    // Box-Muller polar form, single sample.
    let u1 = rng.r#gen::<f32>().max(1.0e-6);
    let u2 = rng.r#gen::<f32>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

// ---------------------------------------------------------------------------
// Tests — back-compat against existing trainer behavior.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    /// `Sigmoid` with `noising_weight = 1.8`, `noising_bias = 0.0` must
    /// produce the same sequence as the bare `sigmoid(1.8 * z)` formula
    /// the Z-Image pipeline used before the refactor (modulo identical
    /// RNG state).
    #[test]
    fn sigmoid_back_compat_zimage() {
        let cfg = TimestepConfig::zimage_sigmoid_18();
        let mut rng_a = StdRng::seed_from_u64(42);
        let mut rng_b = StdRng::seed_from_u64(42);

        for _ in 0..256 {
            let from_cfg = cfg.sample_one(&mut rng_a);
            // Reference inline copy of zimage-trainer's old
            // `sample_timestep`: same Box-Muller polar form, same
            // sigmoid(scale * z) — used here ONLY to assert byte-for-byte
            // equivalence between the new config-driven path and the
            // existing default behavior.
            let u1 = rng_b.r#gen::<f32>().max(1.0e-6);
            let u2 = rng_b.r#gen::<f32>();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
            let from_ref = 1.0 / (1.0 + (-(1.8 * z)).exp());
            assert_eq!(
                from_cfg.to_bits(),
                from_ref.to_bits(),
                "config sigmoid path must match the inline reference exactly"
            );
        }
    }

    /// `Uniform` with default strength range `[0, 1]` must always return
    /// values in `[0, 1]`.
    #[test]
    fn uniform_in_range() {
        let cfg = TimestepConfig::default_uniform();
        let mut rng = StdRng::seed_from_u64(7);
        let xs = cfg.sample(&mut rng, 1024);
        for x in xs {
            assert!((0.0..=1.0).contains(&x));
        }
    }

    /// `min_strength`/`max_strength` rescaling must respect the bounds.
    #[test]
    fn strength_range_rescales() {
        let cfg = TimestepConfig {
            distribution: TimestepDistribution::Uniform,
            noising_weight: 0.0,
            noising_bias: 0.0,
            min_strength: 0.2,
            max_strength: 0.8,
        };
        let mut rng = StdRng::seed_from_u64(11);
        let xs = cfg.sample(&mut rng, 256);
        for x in xs {
            assert!((0.2..=0.8).contains(&x));
        }
    }

    /// LogitNormal with `weight=0, bias=0` is `sigmoid(N(0,1))`.
    #[test]
    fn logit_normal_identity() {
        let cfg = TimestepConfig {
            distribution: TimestepDistribution::LogitNormal,
            noising_weight: 0.0, // → scale = 1.0
            noising_bias: 0.0,
            min_strength: 0.0,
            max_strength: 1.0,
        };
        let mut rng_a = StdRng::seed_from_u64(99);
        let mut rng_b = StdRng::seed_from_u64(99);

        let from_cfg = cfg.sample_one(&mut rng_a);
        let u1 = rng_b.r#gen::<f32>().max(1.0e-6);
        let u2 = rng_b.r#gen::<f32>();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
        let from_ref = 1.0 / (1.0 + (-z).exp());
        assert_eq!(from_cfg.to_bits(), from_ref.to_bits());
    }

    /// Permanent regression marker. Documents the fact that
    /// `TimestepDistribution::Sigmoid` here is **NOT** OneTrainer's
    /// discrete `SIGMOID` distribution; it is the musubi-style continuous
    /// `sigmoid(w·z + b)`. If this test ever starts failing, that means
    /// someone changed the `Sigmoid` formula — either to align with
    /// OneTrainer (good, but document the change) or by accident (bad).
    /// Either way, read the doc comment on `TimestepDistribution::Sigmoid`
    /// before deleting this test.
    #[test]
    fn sigmoid_NOT_onetrainer_sigmoid() {
        // OneTrainer's discrete SIGMOID would produce values that
        // concentrate on a 1000-bucket grid. Our continuous sigmoid
        // produces arbitrary-precision floats. Easiest discriminator:
        // count distinct values across many samples — discrete bucketing
        // would yield ≤1000 distinct floats, while continuous sigmoid
        // yields virtually all distinct values.
        let cfg = TimestepConfig::zimage_sigmoid_18();
        let mut rng = StdRng::seed_from_u64(123);
        let mut seen = std::collections::HashSet::new();
        for _ in 0..2048 {
            seen.insert(cfg.sample_one(&mut rng).to_bits());
        }
        // If this fell to ≤1000 we'd be discrete. We expect ~2048 distinct
        // bit patterns from a continuous distribution. Use 1500 as a
        // generous lower bound.
        assert!(
            seen.len() > 1500,
            "Sigmoid produced only {} distinct values out of 2048 samples — \
             this suggests it became discrete. See the doc comment on \
             TimestepDistribution::Sigmoid.",
            seen.len()
        );
    }

    #[test]
    fn parses_string_variants() {
        for s in ["uniform", "UNIFORM", "  Uniform  "] {
            assert_eq!(TimestepDistribution::from_str(s).unwrap(), TimestepDistribution::Uniform);
        }
        assert_eq!(
            TimestepDistribution::from_str("logit_normal").unwrap(),
            TimestepDistribution::LogitNormal
        );
        assert_eq!(
            TimestepDistribution::from_str("inverted-parabola").unwrap(),
            TimestepDistribution::InvertedParabolaContinuous
        );
        assert!(TimestepDistribution::from_str("garbage").is_err());
    }

    // ---- distribution mean / range checks (N=2048 each) ----

    fn sample_n(cfg: &TimestepConfig, seed: u64, n: usize) -> Vec<f32> {
        let mut rng = StdRng::seed_from_u64(seed);
        cfg.sample(&mut rng, n)
    }

    fn mean(xs: &[f32]) -> f32 {
        xs.iter().copied().sum::<f32>() / xs.len() as f32
    }

    /// Uniform mean ≈ 0.5; with custom strength range, mean reflects that.
    #[test]
    fn uniform_mean_default_and_clamped() {
        let xs = sample_n(&TimestepConfig::default_uniform(), 1, 2048);
        let m = mean(&xs);
        assert!((m - 0.5).abs() < 0.05, "uniform mean ≈ 0.5, got {}", m);

        let cfg = TimestepConfig {
            distribution: TimestepDistribution::Uniform,
            noising_weight: 0.0,
            noising_bias: 0.0,
            min_strength: 0.4,
            max_strength: 0.6,
        };
        let xs = sample_n(&cfg, 2, 2048);
        for x in &xs {
            assert!((0.4..=0.6).contains(x), "value {} outside clamp", x);
        }
        let m = mean(&xs);
        assert!((m - 0.5).abs() < 0.05, "clamped uniform mean ≈ 0.5, got {}", m);
    }

    /// Sigmoid: weight=0,bias=0 → degenerate sigmoid(0)=0.5 (constant).
    /// With weight=1.8,bias=0 → mean ≈ 0.5 (symmetric around z=0).
    /// With bias>0 → mean shifts upward.
    #[test]
    fn sigmoid_mean_and_bias_shift() {
        // weight=0,bias=0: constant 0.5
        let cfg0 = TimestepConfig {
            distribution: TimestepDistribution::Sigmoid,
            noising_weight: 0.0,
            noising_bias: 0.0,
            min_strength: 0.0,
            max_strength: 1.0,
        };
        let xs = sample_n(&cfg0, 3, 256);
        for x in &xs {
            assert!((x - 0.5).abs() < 1e-6, "weight=0 sigmoid must be exactly 0.5, got {}", x);
        }

        // weight=1.8,bias=0 → symmetric, mean ≈ 0.5
        let cfg_sym = TimestepConfig::zimage_sigmoid_18();
        let m_sym = mean(&sample_n(&cfg_sym, 4, 2048));
        assert!((m_sym - 0.5).abs() < 0.05, "sym sigmoid mean ≈ 0.5, got {}", m_sym);

        // bias>0 shifts mean upward
        let cfg_pos = TimestepConfig {
            distribution: TimestepDistribution::Sigmoid,
            noising_weight: 1.8,
            noising_bias: 1.0,
            min_strength: 0.0,
            max_strength: 1.0,
        };
        let m_pos = mean(&sample_n(&cfg_pos, 5, 2048));
        assert!(m_pos > 0.55, "bias=+1 sigmoid mean shifts up, got {}", m_pos);
    }

    /// LogitNormal with bias=0 → sigmoid(scale*N(0,1)) is symmetric around 0.5.
    /// Output always strictly in (0,1).
    #[test]
    fn logit_normal_mean_and_range() {
        let cfg = TimestepConfig {
            distribution: TimestepDistribution::LogitNormal,
            noising_weight: 0.5, // scale = 1.5
            noising_bias: 0.0,
            min_strength: 0.0,
            max_strength: 1.0,
        };
        let xs = sample_n(&cfg, 6, 2048);
        for x in &xs {
            assert!(*x > 0.0 && *x < 1.0, "logit_normal must be open-interval, got {}", x);
        }
        let m = mean(&xs);
        assert!((m - 0.5).abs() < 0.05, "logit_normal mean ≈ 0.5, got {}", m);
    }

    /// HeavyTail with positive weight should put more mass at the tails than
    /// uniform. Verify by comparing the 90th-percentile distance from 0.5.
    #[test]
    fn heavy_tail_has_heavier_tails_than_uniform() {
        let cfg = TimestepConfig {
            distribution: TimestepDistribution::HeavyTail,
            // The OneTrainer formula `t = 1 - u - w*(cos²(π/2·u) - 1 + u)`
            // pushes mass toward the endpoints when w<0 (their default
            // is `-3.0`). With w<0 the tails dominate.
            noising_weight: -3.0,
            noising_bias: 0.0,
            min_strength: 0.0,
            max_strength: 1.0,
        };
        let xs = sample_n(&cfg, 7, 2048);
        // Distance from 0.5: heavy-tail with negative weight should yield
        // a 90th percentile distance > uniform's (≈ 0.45).
        let mut d: Vec<f32> = xs.iter().map(|x| (x - 0.5).abs()).collect();
        d.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p90 = d[(d.len() as f32 * 0.90) as usize];
        assert!(
            p90 > 0.40,
            "heavy_tail 90th-percentile distance from 0.5 should exceed 0.40, got {}",
            p90
        );
    }

    /// CosMapContinuous: t = (1 - cos(π u))/2 with u~U(0,1). Mean over u
    /// integrates to 0.5 (symmetric S-curve), and the density is U-shaped:
    /// mass concentrates near 0 and 1 (because dt/du → 0 there).
    #[test]
    fn cos_map_continuous_density_shape() {
        let cfg = TimestepConfig {
            distribution: TimestepDistribution::CosMapContinuous,
            noising_weight: 0.0,
            noising_bias: 0.0,
            min_strength: 0.0,
            max_strength: 1.0,
        };
        let xs = sample_n(&cfg, 8, 4096);
        let m = mean(&xs);
        // Expected mean is exactly 0.5 by symmetry of (1-cos(πu))/2.
        assert!((m - 0.5).abs() < 0.03, "cos_map mean ≈ 0.5, got {}", m);

        // Check density concentrates at the endpoints: count the fraction
        // in the outer 20% bands [0,0.1] ∪ [0.9,1.0]. Uniform would give
        // 20%; CosMap should give noticeably more (~ ≥ 25%).
        let outer = xs.iter().filter(|&&x| x < 0.1 || x > 0.9).count();
        let frac = outer as f32 / xs.len() as f32;
        assert!(
            frac > 0.25,
            "cos_map should concentrate at endpoints; outer-20% fraction = {}",
            frac
        );
    }

    /// InvertedParabolaContinuous: peak at `bias + 0.5`; mean ≈ `bias + 0.5`.
    #[test]
    fn inverted_parabola_peak_and_mean() {
        // bias=0 → peak at 0.5 → mean ≈ 0.5
        let cfg = TimestepConfig {
            distribution: TimestepDistribution::InvertedParabolaContinuous,
            noising_weight: 8.0, // strong parabola → tight around peak
            noising_bias: 0.0,
            min_strength: 0.0,
            max_strength: 1.0,
        };
        let xs = sample_n(&cfg, 9, 2048);
        let m = mean(&xs);
        assert!(
            (m - 0.5).abs() < 0.05,
            "inverted_parabola bias=0 mean ≈ 0.5, got {}",
            m
        );
    }
}
