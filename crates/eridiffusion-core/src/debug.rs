//! Targeted debug instrumentation for trainers.
//!
//! All probes here are **env-gated** — they don't fire in production runs.
//! Add `ERNIE_DEBUG_GRADS=1` (or the equivalent for other trainers) to enable.
//!
//! Designed to catch the convergence-killer class of bugs: silent gradient
//! starvation in specific module classes (e.g. Q/K after rms_norm), wildly
//! mismatched magnitudes between LoRA-A and LoRA-B, or forward outputs whose
//! distribution doesn't match the training target.
//!
//! Use `enabled(env_var)` once at trainer setup so the env check isn't on
//! the hot path.

use crate::lora::LoRALinear;
use flame_core::gradient::GradientMap;
use flame_core::{DType, Tensor};

/// Returns true if the named env var is set to a truthy value (anything not
/// "0", "false", or empty). Cache the result at trainer setup — don't call
/// per-step.
pub fn enabled(env_var: &str) -> bool {
    match std::env::var(env_var) {
        Ok(v) => !matches!(v.as_str(), "" | "0" | "false" | "FALSE"),
        Err(_) => false,
    }
}

/// L2 norm of a tensor in F32 (cast for precision). Returns 0.0 on any error
/// so debug prints never crash training.
pub fn l2_norm(t: &Tensor) -> f32 {
    let try_norm = || -> flame_core::Result<f32> {
        let f32 = if t.dtype() == DType::F32 { t.clone() } else { t.to_dtype(DType::F32)? };
        let sq = f32.square()?.mean()?;
        let n = f32.shape().dims().iter().product::<usize>() as f32;
        Ok((sq.to_vec()?[0] * n).sqrt())
    };
    try_norm().unwrap_or(0.0)
}

/// Tensor stats: count of NaN, count of Inf, mean, std, abs-max.
/// Cheap when called sparsely (only at probe sites).
pub fn stats(t: &Tensor) -> TensorStats {
    let try_stats = || -> flame_core::Result<TensorStats> {
        let f32 = if t.dtype() == DType::F32 { t.clone() } else { t.to_dtype(DType::F32)? };
        let v = f32.to_vec()?;
        let n = v.len();
        let mut nan = 0usize; let mut inf = 0usize;
        let mut sum = 0f64; let mut sum_sq = 0f64;
        let mut abs_max = 0f32;
        for &x in &v {
            if x.is_nan() { nan += 1; continue; }
            if x.is_infinite() { inf += 1; continue; }
            sum += x as f64; sum_sq += (x as f64) * (x as f64);
            if x.abs() > abs_max { abs_max = x.abs(); }
        }
        let valid = (n - nan - inf) as f64;
        let mean = if valid > 0.0 { (sum / valid) as f32 } else { f32::NAN };
        let var = if valid > 1.0 { (sum_sq / valid - (sum / valid).powi(2)).max(0.0) as f32 } else { 0.0 };
        Ok(TensorStats { count: n, nan, inf, mean, std: var.sqrt(), abs_max })
    };
    try_stats().unwrap_or(TensorStats::error())
}

#[derive(Debug, Clone, Copy)]
pub struct TensorStats {
    pub count: usize,
    pub nan: usize,
    pub inf: usize,
    pub mean: f32,
    pub std: f32,
    pub abs_max: f32,
}

impl TensorStats {
    fn error() -> Self {
        Self { count: 0, nan: 0, inf: 0, mean: f32::NAN, std: 0.0, abs_max: 0.0 }
    }
    pub fn fmt_compact(&self) -> String {
        format!("n={} nan={} inf={} mean={:+.3e} std={:.3e} max|·|={:.3e}",
            self.count, self.nan, self.inf, self.mean, self.std, self.abs_max)
    }
}

/// Build a per-class summary of LoRA gradient norms. `class_keys` maps from
/// adapter index in the flat list → human-readable class name (e.g. "to_q",
/// "to_k", "ffn_gate"). Aggregates A and B separately.
///
/// Use after `loss.backward()` returns a `grads` HashMap, before applying
/// to params. Catches: certain classes silently getting zero grad (= autograd
/// path broken there), grad magnitude wildly different from B vs A (= scale
/// or init issue), grads with NaN/Inf (= loss blowup or numerical issue).
pub fn lora_grad_summary(
    adapters: &[LoRALinear],
    class_keys: &[&str],
    grads: &GradientMap,
) -> Vec<(String, ClassGradSummary)> {
    let mut by_class: std::collections::HashMap<String, ClassGradSummary> =
        std::collections::HashMap::new();

    for (i, adapter) in adapters.iter().enumerate() {
        let class = class_keys[i % class_keys.len()];

        // Lora A
        let a_id = adapter.lora_a().id();
        if let Some(g) = grads.get(a_id) {
            let key = format!("{class}.A");
            let entry = by_class.entry(key).or_insert_with(ClassGradSummary::new);
            entry.add(l2_norm(g), &stats(g));
        } else {
            let key = format!("{class}.A");
            by_class.entry(key).or_insert_with(ClassGradSummary::new).add_missing();
        }

        // Lora B
        let b_id = adapter.lora_b().id();
        if let Some(g) = grads.get(b_id) {
            let key = format!("{class}.B");
            let entry = by_class.entry(key).or_insert_with(ClassGradSummary::new);
            entry.add(l2_norm(g), &stats(g));
        } else {
            let key = format!("{class}.B");
            by_class.entry(key).or_insert_with(ClassGradSummary::new).add_missing();
        }
    }

    let mut out: Vec<_> = by_class.into_iter().collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

#[derive(Debug, Clone)]
pub struct ClassGradSummary {
    pub n_adapters: usize,
    pub n_missing: usize,
    pub norm_min: f32,
    pub norm_max: f32,
    pub norm_mean: f32,
    pub total_nan: usize,
    pub total_inf: usize,
    pub abs_max_overall: f32,
}

impl ClassGradSummary {
    fn new() -> Self {
        Self {
            n_adapters: 0, n_missing: 0,
            norm_min: f32::INFINITY, norm_max: 0.0, norm_mean: 0.0,
            total_nan: 0, total_inf: 0, abs_max_overall: 0.0,
        }
    }
    fn add(&mut self, norm: f32, st: &TensorStats) {
        self.n_adapters += 1;
        if norm < self.norm_min { self.norm_min = norm; }
        if norm > self.norm_max { self.norm_max = norm; }
        // Online mean: (old_mean * (n-1) + new) / n
        let n = self.n_adapters as f32;
        self.norm_mean = self.norm_mean * (n - 1.0) / n + norm / n;
        self.total_nan += st.nan; self.total_inf += st.inf;
        if st.abs_max > self.abs_max_overall { self.abs_max_overall = st.abs_max; }
    }
    fn add_missing(&mut self) {
        self.n_missing += 1;
    }
    pub fn fmt_compact(&self) -> String {
        format!(
            "n={:>3} miss={} norm[min={:.2e} mean={:.2e} max={:.2e}] nan={} inf={} max|·|={:.2e}",
            self.n_adapters, self.n_missing,
            self.norm_min, self.norm_mean, self.norm_max,
            self.total_nan, self.total_inf, self.abs_max_overall,
        )
    }
}

/// Print a one-shot snapshot — call sparsely (step 0, then every N steps).
pub fn print_lora_grad_summary(
    step: usize,
    adapters: &[LoRALinear],
    class_keys: &[&str],
    grads: &GradientMap,
) {
    let summary = lora_grad_summary(adapters, class_keys, grads);
    eprintln!("--- [debug step {}] LoRA grad summary ({} adapters across {} classes) ---",
        step, adapters.len(), class_keys.len());
    for (class, s) in &summary {
        eprintln!("  {:<20} {}", class, s.fmt_compact());
    }
    eprintln!("--- end grad summary ---");
}
