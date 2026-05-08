//! TREAD — **T**oken **R**outing for **E**fficient **A**ttention **D**ispatch.
//!
//! Per training step, a random subset of patch tokens is "routed": passed
//! through only some transformer-block ranges and reinjected at the routed-out
//! point via a residual skip. Acts as a regularizer + small wall-time speedup.
//!
//! Phase 4 ships **infrastructure only** — the routing primitives + CLI
//! surface + tests live here, but model `forward` paths are NOT wired yet.
//! Phase 4.5 lands the model integration with the user awake (default-off
//! Klein convergence at step 1000 has been verified; touching the model
//! forward today is high-risk, low-reward).
//!
//! Reference: SimpleTuner `helpers/training/tread.py` and the auxiliary
//! distillation hooks in `helpers/distillation/tread/`.
//!
//! Phase target: 4 (infrastructure) → 4.5 (model integration)
//! Config flags: `tread_route_pattern: Option<String>` (Phase 0),
//!               `tread_keep_ratio: f32` (Phase 4, default 1.0)

use flame_core::{Shape, Tensor};
use rand::rngs::StdRng;
use rand::Rng;
use std::sync::Arc;

use crate::{EriDiffusionError, Result};

/// User-facing configuration for TREAD. Lives next to the
/// `tread_route_pattern` field in `TrainConfig`.
#[derive(Clone, Debug)]
pub struct TreadConfig {
    /// Block-index ranges eligible for routing. Comma-separated, each range
    /// `lo-hi` (half-open: tokens skip blocks `[lo, hi)`). Example: `"12-23"`
    /// or `"12-23,30-35"`.
    pub route_pattern: String,
    /// Fraction of tokens to keep in the routed-block range. `1.0` = no
    /// routing (byte-identical to the un-routed forward); `0.5` = half the
    /// tokens skip the routed blocks.
    pub keep_ratio: f32,
}

impl TreadConfig {
    /// Parse the route pattern into a vector of `(start, end)` block-index
    /// ranges. Half-open: a token IS routed when its block index satisfies
    /// `start <= idx < end`.
    pub fn parse(s: &str) -> Result<Vec<(usize, usize)>> {
        let mut ranges = Vec::new();
        for part in s.split(',') {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }
            let (lo_s, hi_s) = part.split_once('-').ok_or_else(|| {
                EriDiffusionError::Training(format!(
                    "tread parse: expected `lo-hi`, got `{part}`"
                ))
            })?;
            let lo: usize = lo_s
                .trim()
                .parse()
                .map_err(|e| EriDiffusionError::Training(format!("tread parse lo: {e}")))?;
            let hi: usize = hi_s
                .trim()
                .parse()
                .map_err(|e| EriDiffusionError::Training(format!("tread parse hi: {e}")))?;
            if hi <= lo {
                return Err(EriDiffusionError::Training(format!(
                    "tread parse: empty/inverted range `{part}` (need hi > lo)"
                )));
            }
            ranges.push((lo, hi));
        }
        Ok(ranges)
    }
}

/// Per-step routing state. Constructed once at the top of a training step and
/// passed into the model's `forward` so each block can consult `block_routes`
/// + `gather_routed` / `scatter_routed` at the boundary it cares about.
///
/// This is opaque to the model in Phase 4 (no model code consumes it yet).
/// Phase 4.5 will wire each model.
pub struct TreadStep {
    /// Boolean per-token: `true` = route through (kept in routed range);
    /// `false` = skip routed blocks (residual passthrough).
    pub keep_mask: Vec<bool>,
    /// Indices of tokens that route through. Sorted ascending. I32 device
    /// tensors derived from this are what `gather_routed` feeds to
    /// `Tensor::index_select(1, &idx)`.
    pub keep_indices: Vec<i64>,
    /// Indices of tokens that skip the routed blocks.
    pub skip_indices: Vec<i64>,
    /// Block range eligible for routing: a block `b` is "routed" iff
    /// `route_block_start <= b < route_block_end`. Outside this range, all
    /// tokens flow normally.
    pub route_block_start: usize,
    pub route_block_end: usize,
    /// Total token count along the sequence dim (safety asserts in the
    /// model integration).
    pub total_tokens: usize,
}

impl TreadStep {
    /// Build a routing step. `keep_ratio` clamped to `[0, 1]`. Determinism:
    /// the caller is responsible for the RNG seeding policy — pass a
    /// `StdRng` whose state matches the desired byte-invariance contract.
    /// At `keep_ratio=1.0`, every token is kept and every index list is
    /// dense `[0, total_tokens)`.
    pub fn new(
        total_tokens: usize,
        keep_ratio: f32,
        route_block_range: (usize, usize),
        rng: &mut StdRng,
    ) -> Self {
        let r = keep_ratio.clamp(0.0, 1.0);
        // Match SimpleTuner's "ceil"-style keep count for fractional ratios so
        // keep_ratio=1.0 trivially keeps everything.
        let keep_count = ((total_tokens as f32) * r).ceil() as usize;
        let keep_count = keep_count.min(total_tokens);

        // Fisher-Yates partial shuffle: only the first `keep_count` slots
        // need to be drawn from the full population.
        let mut indices: Vec<usize> = (0..total_tokens).collect();
        if total_tokens > 1 {
            for i in (1..total_tokens).rev() {
                let j = rng.gen_range(0..=i);
                indices.swap(i, j);
            }
        }

        // Sort the kept / skipped sets ascending so downstream gather/scatter
        // is reproducible regardless of the shuffle's internal ordering.
        let mut keep_sorted: Vec<usize> = indices[..keep_count].to_vec();
        keep_sorted.sort_unstable();
        let mut skip_sorted: Vec<usize> = indices[keep_count..].to_vec();
        skip_sorted.sort_unstable();

        let keep_indices: Vec<i64> = keep_sorted.iter().map(|&i| i as i64).collect();
        let skip_indices: Vec<i64> = skip_sorted.iter().map(|&i| i as i64).collect();

        let mut keep_mask = vec![false; total_tokens];
        for &i in &keep_sorted {
            keep_mask[i] = true;
        }

        Self {
            keep_mask,
            keep_indices,
            skip_indices,
            route_block_start: route_block_range.0,
            route_block_end: route_block_range.1,
            total_tokens,
        }
    }

    /// Returns true if the given block index participates in routing (token
    /// gather/scatter happens at this block's boundary).
    pub fn block_routes(&self, block_idx: usize) -> bool {
        block_idx >= self.route_block_start && block_idx < self.route_block_end
    }

    /// Number of tokens that route through (i.e. flow through the routed
    /// block range).
    pub fn keep_count(&self) -> usize {
        self.keep_indices.len()
    }

    /// Number of tokens that skip the routed block range.
    pub fn skip_count(&self) -> usize {
        self.skip_indices.len()
    }

    /// Build an I32 device tensor of `keep_indices`, ready to feed into
    /// `Tensor::index_select(dim=1, &idx)` to gather routed tokens from
    /// a `[B, T, D]` activation.
    ///
    /// Phase 4: returns the index tensor only — actual gather is a
    /// model-side call that has to happen inside the autograd-recording
    /// forward path (we don't want to clone the activation here).
    pub fn keep_index_tensor(&self, device: &Arc<cudarc::driver::CudaDevice>) -> Result<Tensor> {
        let n = self.keep_indices.len();
        let data: Vec<f32> = self.keep_indices.iter().map(|&i| i as f32).collect();
        let f32_idx = Tensor::from_vec(data, Shape::from_dims(&[n]), device.clone())
            .map_err(EriDiffusionError::from)?;
        // Convert to I32 — flame-core's index_select bf16 path requires I32.
        f32_idx
            .to_dtype(flame_core::DType::I32)
            .map_err(EriDiffusionError::from)
    }

    /// Build an I32 device tensor of `skip_indices` (companion to
    /// `keep_index_tensor`). Used by the scatter/reinject side of routing.
    pub fn skip_index_tensor(&self, device: &Arc<cudarc::driver::CudaDevice>) -> Result<Tensor> {
        let n = self.skip_indices.len();
        let data: Vec<f32> = self.skip_indices.iter().map(|&i| i as f32).collect();
        let f32_idx = Tensor::from_vec(data, Shape::from_dims(&[n]), device.clone())
            .map_err(EriDiffusionError::from)?;
        f32_idx
            .to_dtype(flame_core::DType::I32)
            .map_err(EriDiffusionError::from)
    }

    /// Gather routed tokens from a `[B, T, D]` tensor along the T (sequence)
    /// dim. Returns a `[B, T_keep, D]` tensor.
    ///
    /// Phase 4: works (uses `Tensor::index_select`, which exists in flame-core
    /// for both BF16 and F32 paths). Used by the Phase 4.5 model integration.
    pub fn gather_routed(&self, x: &Tensor) -> Result<Tensor> {
        if x.shape().dims().len() < 2 {
            return Err(EriDiffusionError::Training(format!(
                "tread gather_routed: tensor rank < 2 ({:?})",
                x.shape().dims()
            )));
        }
        // Dim 1 is the sequence axis for `[B, T, D]`.
        let device = x.device();
        let idx = self.keep_index_tensor(device)?;
        x.index_select(1, &idx).map_err(EriDiffusionError::from)
    }

    /// Re-inject the routed tokens back into the un-routed sequence.
    ///
    /// The intended math: at the END of the routed block range, the routed
    /// activation is `[B, T_keep, D]`, and the residual `skip_source` from
    /// BEFORE the routed range is `[B, T, D]`. We scatter the processed
    /// `routed` rows back into their original sequence positions inside
    /// `skip_source`, leaving the skipped positions untouched.
    ///
    /// At `keep_ratio=1.0` (the default) `keep_indices` covers `[0, T)`
    /// densely and `routed.shape() == skip_source.shape()`, so we
    /// short-circuit and return `routed` directly; this is bit-exact with
    /// the non-routed forward, which is what default-off needs.
    ///
    /// Phase 4.5: backed by `Tensor::index_assign` (BF16 + F32) — the
    /// previous "PLACEHOLDER returning skip_source.clone()" is gone.
    pub fn scatter_routed(&self, routed: &Tensor, skip_source: &Tensor) -> Result<Tensor> {
        if self.keep_indices.len() == self.total_tokens {
            // keep_ratio=1.0 → dense keep, no actual routing happened, the
            // routed tensor IS the full-sequence activation.
            return Ok(routed.clone());
        }
        let device = skip_source.device();
        let idx = self.keep_index_tensor(device)?;
        // skip_source: [B, T, D] (or rank ≥2 with sequence at dim=1).
        // routed:      [B, T_keep, D]
        // index_assign(dim=1, idx, routed) → result with kept rows replaced
        // by routed-output and skipped rows preserved from skip_source.
        skip_source
            .index_assign(1, &idx, routed)
            .map_err(EriDiffusionError::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn parse_single_range() {
        let r = TreadConfig::parse("12-23").unwrap();
        assert_eq!(r, vec![(12, 23)]);
    }

    #[test]
    fn parse_multi_range_with_whitespace() {
        let r = TreadConfig::parse("12-23, 30-35").unwrap();
        assert_eq!(r, vec![(12, 23), (30, 35)]);
    }

    #[test]
    fn parse_empty_string_yields_empty() {
        let r = TreadConfig::parse("").unwrap();
        assert!(r.is_empty());
    }

    #[test]
    fn parse_rejects_inverted_range() {
        let err = TreadConfig::parse("23-12").unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("inverted") || msg.contains("hi > lo"), "{msg}");
    }

    #[test]
    fn parse_rejects_garbage() {
        assert!(TreadConfig::parse("twelve-thirteen").is_err());
        assert!(TreadConfig::parse("12,13").is_err());
    }

    #[test]
    fn keep_ratio_one_keeps_everything() {
        let mut rng = StdRng::seed_from_u64(42);
        let step = TreadStep::new(64, 1.0, (12, 23), &mut rng);
        assert_eq!(step.keep_count(), 64);
        assert_eq!(step.skip_count(), 0);
        assert!(step.keep_mask.iter().all(|&b| b));
        // Indices form 0..64 sorted ascending.
        let expected: Vec<i64> = (0..64).collect();
        assert_eq!(step.keep_indices, expected);
    }

    #[test]
    fn keep_ratio_half_keeps_half_rounded_up() {
        let mut rng = StdRng::seed_from_u64(7);
        let step = TreadStep::new(64, 0.5, (12, 23), &mut rng);
        // ceil(64 * 0.5) = 32
        assert_eq!(step.keep_count(), 32);
        assert_eq!(step.skip_count(), 32);
        assert_eq!(step.keep_count() + step.skip_count(), 64);
        let kept: usize = step.keep_mask.iter().filter(|&&b| b).count();
        assert_eq!(kept, 32);
    }

    #[test]
    fn keep_ratio_zero_keeps_nothing() {
        let mut rng = StdRng::seed_from_u64(7);
        let step = TreadStep::new(64, 0.0, (12, 23), &mut rng);
        assert_eq!(step.keep_count(), 0);
        assert_eq!(step.skip_count(), 64);
        assert!(step.keep_mask.iter().all(|&b| !b));
    }

    #[test]
    fn block_routes_inclusive_lo_exclusive_hi() {
        let mut rng = StdRng::seed_from_u64(0);
        let step = TreadStep::new(16, 0.5, (12, 23), &mut rng);
        assert!(!step.block_routes(11));
        assert!(step.block_routes(12));
        assert!(step.block_routes(22));
        assert!(!step.block_routes(23));
        assert!(!step.block_routes(0));
    }

    #[test]
    fn scatter_then_gather_round_trip_real_tensor() {
        // Real CUDA tensor round-trip: gather subset → scatter back into the
        // same `skip_source` should reproduce `skip_source` exactly. Confirms
        // index_assign + index_select are exact inverses for this op.
        use cudarc::driver::CudaDevice;
        use flame_core::{DType, Shape, Tensor};
        let device = match CudaDevice::new(0) {
            Ok(d) => d,
            Err(_) => return, // No CUDA on this host; skip silently.
        };
        let b = 1;
        let t_total = 8;
        let d = 4;
        let total = b * t_total * d;
        let data: Vec<f32> = (0..total).map(|i| i as f32).collect();
        let x = Tensor::from_vec(
            data.clone(),
            Shape::from_dims(&[b, t_total, d]),
            device.clone(),
        )
        .unwrap();

        let mut rng = StdRng::seed_from_u64(123);
        // 50% keep ratio — non-trivial routing.
        let step = TreadStep::new(t_total, 0.5, (0, 1), &mut rng);
        assert_eq!(step.keep_count(), 4);

        let routed = step.gather_routed(&x).expect("gather");
        let restored = step.scatter_routed(&routed, &x).expect("scatter");
        // Restoration must equal the original `skip_source` because we
        // round-tripped without modification.
        let v = restored.to_vec().unwrap();
        assert_eq!(v, data, "gather + scatter round-trip must reproduce input");

        // BF16 path — confirm dtype dispatches correctly too.
        let xb = x.to_dtype(DType::BF16).unwrap();
        let routed_bf16 = step.gather_routed(&xb).unwrap();
        let restored_bf16 = step.scatter_routed(&routed_bf16, &xb).unwrap();
        let vb: Vec<f32> = restored_bf16.to_dtype(DType::F32).unwrap().to_vec().unwrap();
        // Small ints exactly representable in BF16.
        assert_eq!(vb, data);
    }

    #[test]
    fn keep_indices_are_sorted_and_disjoint_from_skip() {
        let mut rng = StdRng::seed_from_u64(1234);
        let step = TreadStep::new(128, 0.7, (0, 5), &mut rng);
        // Sorted ascending.
        assert!(step.keep_indices.windows(2).all(|w| w[0] < w[1]));
        assert!(step.skip_indices.windows(2).all(|w| w[0] < w[1]));
        // Disjoint and union covers [0, 128).
        let mut all: Vec<i64> = step
            .keep_indices
            .iter()
            .chain(step.skip_indices.iter())
            .copied()
            .collect();
        all.sort_unstable();
        let expected: Vec<i64> = (0..128).collect();
        assert_eq!(all, expected);
    }
}
