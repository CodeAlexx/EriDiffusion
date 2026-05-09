//! Grad-coverage diagnostic — flag the chroma-pattern bug at step 1
//! instead of at step 500.
//!
//! Audit H10 (2026-05-09 wan2.2 audit). The chroma trainer had two
//! gradient-blocking bugs that produced a partial-coverage optimizer step
//! (some LoRA-Bs trained, others stayed at zeros init) and the bug was
//! only diagnosed by running `python -c "audit zero LoRA-B counts in
//! saved safetensors"` after a 50-step run. This utility runs the same
//! check INSIDE the training loop after `loss.backward()`, so a partial
//! gradient surfaces as a `log::warn!` at step 1.
//!
//! Cost: O(P) host roundtrips per call (one `to_vec` per parameter to
//! compute |grad|.sum()). At ~320 LoRA params for a Wan 2.2 expert this
//! is ~50-150 ms — too expensive every step. Recommended cadence:
//! step 1 + every checkpoint (or `--grad-coverage-every-steps 0` to
//! skip entirely, default).
//!
//! Usage:
//!
//! ```ignore
//! use eridiffusion_core::training::grad_coverage::GradCoverage;
//!
//! let cov = GradCoverage::measure(&active_params, &grads)?;
//! cov.report_warn_below(0.95, "wan22-low");
//! ```

use flame_core::{parameter::Parameter, tensor::Tensor, GradientMap, Result};

/// Snapshot of how many parameters in `active_params` have a non-zero
/// gradient after `loss.backward()`.
pub struct GradCoverage {
    /// Total params in the active set.
    pub total: usize,
    /// Params present in the gradient map (gradient was computed).
    pub with_grad: usize,
    /// Params whose gradient has at least one non-zero element.
    pub nonzero_grad: usize,
}

impl GradCoverage {
    /// Walk `params` and check each against the gradient map. Returns
    /// counts; does not log.
    pub fn measure(params: &[Parameter], grads: &GradientMap) -> Result<Self> {
        let mut with_grad = 0usize;
        let mut nonzero_grad = 0usize;
        for p in params {
            if let Some(g) = grads.get(p.id()) {
                with_grad += 1;
                if tensor_has_nonzero(g)? {
                    nonzero_grad += 1;
                }
            }
        }
        Ok(Self {
            total: params.len(),
            with_grad,
            nonzero_grad,
        })
    }

    /// Coverage as a fraction of `total` with non-zero gradient.
    pub fn coverage_pct(&self) -> f32 {
        if self.total == 0 {
            return 1.0;
        }
        self.nonzero_grad as f32 / self.total as f32
    }

    /// Log at info level. Use as routine reporting.
    pub fn report_info(&self, label: &str) {
        log::info!(
            "[grad-coverage:{label}] {}/{} params with non-zero grad ({:.1}%) — {} present, {} missing",
            self.nonzero_grad,
            self.total,
            self.coverage_pct() * 100.0,
            self.with_grad,
            self.total.saturating_sub(self.with_grad),
        );
    }

    /// If coverage is below `threshold` (e.g. 0.95) emit a `log::warn!` —
    /// the chroma-pattern signature. Otherwise log at info.
    pub fn report_warn_below(&self, threshold: f32, label: &str) {
        let pct = self.coverage_pct();
        if pct < threshold {
            log::warn!(
                "[grad-coverage:{label}] PARTIAL COVERAGE — {}/{} params with non-zero grad ({:.1}% < {:.1}%). \
                 This is the chroma-bug signature: a fused inference-only kernel or no_grad guard in the model \
                 forward is severing the gradient chain to a subset of LoRA params. \
                 See PARITY_SIMPLETUNER.md and project_chroma_lora_broken_2026-05-09 for the diagnosis playbook.",
                self.nonzero_grad,
                self.total,
                pct * 100.0,
                threshold * 100.0,
            );
        } else {
            self.report_info(label);
        }
    }
}

/// Returns true if `t` has at least one element with magnitude > 0.
/// Roundtrip via `to_vec` (F32). Cheap for small tensors (LoRA factors are
/// `[rank, dim]` → ~tens of KB), expensive for large weight tensors.
fn tensor_has_nonzero(t: &Tensor) -> Result<bool> {
    let v = t.to_vec()?;
    Ok(v.iter().any(|x| *x != 0.0))
}
