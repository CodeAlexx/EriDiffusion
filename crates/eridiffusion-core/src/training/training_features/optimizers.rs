//! Adafactor / AdamW8bit / Prodigy / Lion alongside the existing AdamW.
//!
//! API surface matches `flame_core::adam::AdamW` so callers just swap the
//! type — `step(params)` and `zero_grad(params)` take `&[Parameter]` and
//! the constructors take scalars (no `&[Parameter]` at construction time;
//! state is allocated lazily on first `step`).
//!
//! Tensor-level implementations (no fused CUDA kernels) — same path quality
//! as the pre-fused PyTorch references. AdamW8bit is a partial port: state
//! is quantized between steps to reduce host memory, but the math runs in
//! F32 (no custom 8-bit kernel). Documented at the AdamW8bit struct.
//!
//! All implementations follow the algorithms verbatim; references inline.

use flame_core::{parameter::Parameter, DType, Result, Tensor, TensorId};
use std::collections::{hash_map::Entry, HashMap};

// ---------------------------------------------------------------------------
// Dispatch enum
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizerKind {
    AdamW,
    Adafactor,
    AdamW8bit,
    Prodigy,
    Lion,
}

impl OptimizerKind {
    pub fn parse(s: &str) -> std::result::Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "adamw" | "adam_w" | "adam-w" => Ok(Self::AdamW),
            "adafactor" => Ok(Self::Adafactor),
            "adamw8bit" | "adamw_8bit" | "adamw-8bit" | "adam8bit" => Ok(Self::AdamW8bit),
            "prodigy" => Ok(Self::Prodigy),
            "lion" => Ok(Self::Lion),
            other => Err(format!(
                "unknown optimizer '{}' (expected one of: adamw, adafactor, adamw8bit, prodigy, lion)",
                other
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::AdamW => "adamw",
            Self::Adafactor => "adafactor",
            Self::AdamW8bit => "adamw8bit",
            Self::Prodigy => "prodigy",
            Self::Lion => "lion",
        }
    }

    /// Recommended `(beta1, beta2)` defaults for this optimizer family.
    ///
    /// AdamW / Adafactor / AdamW8bit / Prodigy all share the standard
    /// Adam-style `(0.9, 0.999)`.
    ///
    /// Lion (Chen et al., 2023) uses **`(0.9, 0.99)`** — `beta2 = 0.999`
    /// would slow the EMA momentum update enough to skew the sign-update
    /// direction. Trainers should call this when constructing the optimizer
    /// unless their config explicitly overrides the betas.
    pub fn default_betas(self) -> (f32, f32) {
        match self {
            Self::Lion => (0.9, 0.99),
            _ => (0.9, 0.999),
        }
    }
}

/// Dispatch wrapper. Select kind via [`Optimizer::new`]; downstream code calls
/// `step` / `zero_grad` regardless of which algorithm is active.
pub enum Optimizer {
    AdamW(flame_core::adam::AdamW),
    Adafactor(Adafactor),
    AdamW8bit(AdamW8bit),
    Prodigy(Prodigy),
    Lion(Lion),
}

impl Optimizer {
    /// Constructor that takes the same scalars as `AdamW::new` and a kind tag.
    /// Algorithms that ignore certain knobs document that explicitly:
    /// - Lion uses (beta1, beta2) but not eps.
    /// - Adafactor uses (eps, weight_decay), ignores beta1/beta2.
    /// - Prodigy uses (beta1, beta2, eps, weight_decay) — `lr` is a
    ///   multiplicative scaling on the adapted step size (reference
    ///   recommends 1.0). The initial D estimate is hardcoded to `1e-6`
    ///   (matches the upstream Prodigy default).
    pub fn new(
        kind: OptimizerKind,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
    ) -> Self {
        match kind {
            OptimizerKind::AdamW => {
                Self::AdamW(flame_core::adam::AdamW::new(lr, beta1, beta2, eps, weight_decay))
            }
            OptimizerKind::Adafactor => Self::Adafactor(Adafactor::new(lr, eps, weight_decay)),
            OptimizerKind::AdamW8bit => {
                Self::AdamW8bit(AdamW8bit::new(lr, beta1, beta2, eps, weight_decay))
            }
            OptimizerKind::Prodigy => {
                Self::Prodigy(Prodigy::new(lr, beta1, beta2, eps, weight_decay))
            }
            OptimizerKind::Lion => Self::Lion(Lion::new(lr, beta1, beta2, weight_decay)),
        }
    }

    pub fn kind(&self) -> OptimizerKind {
        match self {
            Self::AdamW(_) => OptimizerKind::AdamW,
            Self::Adafactor(_) => OptimizerKind::Adafactor,
            Self::AdamW8bit(_) => OptimizerKind::AdamW8bit,
            Self::Prodigy(_) => OptimizerKind::Prodigy,
            Self::Lion(_) => OptimizerKind::Lion,
        }
    }

    pub fn step(&mut self, params: &[Parameter]) -> Result<()> {
        match self {
            Self::AdamW(o) => o.step(params),
            Self::Adafactor(o) => o.step(params),
            Self::AdamW8bit(o) => o.step(params),
            Self::Prodigy(o) => o.step(params),
            Self::Lion(o) => o.step(params),
        }
    }

    pub fn zero_grad(&self, params: &[Parameter]) {
        match self {
            Self::AdamW(o) => o.zero_grad(params),
            Self::Adafactor(o) => o.zero_grad(params),
            Self::AdamW8bit(o) => o.zero_grad(params),
            Self::Prodigy(o) => o.zero_grad(params),
            Self::Lion(o) => o.zero_grad(params),
        }
    }

    pub fn set_lr(&mut self, lr: f32) {
        match self {
            Self::AdamW(o) => o.set_lr(lr),
            Self::Adafactor(o) => o.lr = lr,
            Self::AdamW8bit(o) => o.lr = lr,
            Self::Prodigy(o) => o.lr = lr,
            Self::Lion(o) => o.lr = lr,
        }
    }
}

// ---------------------------------------------------------------------------
// Adafactor
// ---------------------------------------------------------------------------

/// Adafactor (Shazeer & Stern, 2018) — factored second moment for 2D+ params,
/// per-element second moment for ≤1D. No first moment in this configuration
/// (matches `transformers.Adafactor` defaults `beta1=None`).
///
/// Hyperparameters mirror the upstream defaults documented in
/// `transformers/optimization.py:688`:
/// - `lr` — manual learning rate. Set to a small constant (e.g. 1e-3).
///   `relative_step=True` mode (auto-LR via `1/sqrt(step)`) is **not**
///   supported here; trainers wanting that should ship a separate constructor.
/// - `eps` — `(eps_grad, eps_param) = (1e-30, 1e-3)`. We accept a single
///   `eps` arg for API symmetry; `eps_grad` is hardcoded to `1e-30` and
///   `eps_param` is taken from the constructor `eps` (default 1e-3 if 0).
/// - `weight_decay` — decoupled (applied to param directly).
/// - `decay_rate = -0.8`, `clip_threshold = 1.0` — fixed.
/// - `scale_parameter` — when `true`, the effective per-step learning rate is
///   `max(eps_param, RMS(param)) * lr` (matches transformers' default). When
///   `false`, the lr is used directly. Default `false` for backward
///   compatibility with the original `Adafactor::new` constructor.
pub struct Adafactor {
    pub lr: f32,
    eps_grad: f32,
    /// Lower bound on the param-RMS used when `scale_parameter = true`. With
    /// `scale_parameter = false` this field is ignored (kept on the struct
    /// so existing serialized configs / constructors don't need to change).
    eps_param: f32,
    weight_decay: f32,
    decay_rate: f32,
    clip_threshold: f32,
    /// When `true`, scale the effective learning rate per param by
    /// `max(eps_param, RMS(param))` before applying. Mirrors transformers'
    /// `scale_parameter=True` default.
    scale_parameter: bool,
    /// Per-param step counter.
    step_count: HashMap<TensorId, u32>,
    /// Factored row second moment for rank ≥ 2 params.
    /// Shape is `param_shape[..ndim-1]`.
    exp_avg_sq_row: HashMap<TensorId, Tensor>,
    /// Factored col second moment for rank ≥ 2 params.
    /// Shape is `param_shape[..ndim-2] + param_shape[-1:]`.
    exp_avg_sq_col: HashMap<TensorId, Tensor>,
    /// Full second moment for rank ≤ 1 params.
    exp_avg_sq: HashMap<TensorId, Tensor>,
}

impl Adafactor {
    /// Backward-compat constructor: `scale_parameter = false`.
    pub fn new(lr: f32, eps: f32, weight_decay: f32) -> Self {
        Self::with_options(lr, eps, weight_decay, false)
    }

    /// Full constructor. With `scale_parameter = true`, the effective per-step
    /// LR is `max(eps_param, RMS(param)) * lr`.
    pub fn with_options(lr: f32, eps: f32, weight_decay: f32, scale_parameter: bool) -> Self {
        let eps_param = if eps == 0.0 { 1.0e-3 } else { eps };
        Self {
            lr,
            eps_grad: 1.0e-30,
            eps_param,
            weight_decay,
            decay_rate: -0.8,
            clip_threshold: 1.0,
            scale_parameter,
            step_count: HashMap::new(),
            exp_avg_sq_row: HashMap::new(),
            exp_avg_sq_col: HashMap::new(),
            exp_avg_sq: HashMap::new(),
        }
    }

    pub fn step(&mut self, params: &[Parameter]) -> Result<()> {
        for p in params {
            let Some(grad) = p.grad() else { continue };
            // Algorithm runs in F32; cast grad up if it isn't already.
            let grad_f32 = if grad.dtype() == DType::F32 {
                grad
            } else {
                grad.to_dtype(DType::F32)?
            };

            let id = p.id();
            let step = {
                let entry = self.step_count.entry(id).or_insert(0);
                *entry += 1;
                *entry
            };
            // beta2_t = 1 - step^decay_rate (decay_rate is negative, so step grows → beta2_t → 1)
            let beta2t = 1.0 - (step as f32).powf(self.decay_rate);
            let one_minus_beta2t = 1.0 - beta2t;

            let g_sq = grad_f32.square()?.add_scalar(self.eps_grad)?;
            let dims = grad_f32.shape().dims().to_vec();
            let factored = dims.len() >= 2;

            // Compute the unscaled update from the second moment.
            let mut update = if factored {
                let last = dims.len() - 1;
                let second = dims.len() - 2;

                // mean over last dim: shape = grad_shape[..-1]
                let grad_mean_last = g_sq.mean_dim(&[last], false)?;
                // mean over second-to-last: shape = grad_shape[..-2] + [last_dim]
                let grad_mean_second = g_sq.mean_dim(&[second], false)?;

                // Update factored row & col estimators in place.
                let row = match self.exp_avg_sq_row.entry(id) {
                    Entry::Occupied(e) => e.into_mut(),
                    Entry::Vacant(e) => {
                        let zeros = Tensor::zeros_dtype(
                            grad_mean_last.shape().clone(),
                            DType::F32,
                            grad_f32.device().clone(),
                        )?;
                        e.insert(zeros)
                    }
                };
                let new_row = row
                    .mul_scalar(beta2t)?
                    .add(&grad_mean_last.mul_scalar(one_minus_beta2t)?)?;
                *row = new_row.detach()?;

                let col = match self.exp_avg_sq_col.entry(id) {
                    Entry::Occupied(e) => e.into_mut(),
                    Entry::Vacant(e) => {
                        let zeros = Tensor::zeros_dtype(
                            grad_mean_second.shape().clone(),
                            DType::F32,
                            grad_f32.device().clone(),
                        )?;
                        e.insert(zeros)
                    }
                };
                let new_col = col
                    .mul_scalar(beta2t)?
                    .add(&grad_mean_second.mul_scalar(one_minus_beta2t)?)?;
                *col = new_col.detach()?;

                // approx_sq_grad: r_factor * c_factor outer-product style
                //   r_factor = rsqrt(row / row.mean(-1)) [unsqueeze -1]
                //   c_factor = rsqrt(col)               [unsqueeze -2]
                let row = self.exp_avg_sq_row.get(&id).unwrap();
                let col = self.exp_avg_sq_col.get(&id).unwrap();

                let row_dims = row.shape().dims().len();
                let row_mean = row.mean_dim(&[row_dims - 1], true)?;
                let r_factor = row.div(&row_mean)?.rsqrt()?.unsqueeze(row_dims)?; // append last
                let c_factor = col.unsqueeze(col.shape().dims().len() - 1)?.rsqrt()?;
                // Broadcast multiply: r is [..-1, 1], c is [..-2, 1, last]
                // Their product broadcasts across the last two dims of grad.
                let approx = r_factor.mul(&c_factor)?;
                approx.mul(&grad_f32)?
            } else {
                // Per-element second moment.
                let v = match self.exp_avg_sq.entry(id) {
                    Entry::Occupied(e) => e.into_mut(),
                    Entry::Vacant(e) => {
                        let zeros = Tensor::zeros_dtype(
                            grad_f32.shape().clone(),
                            DType::F32,
                            grad_f32.device().clone(),
                        )?;
                        e.insert(zeros)
                    }
                };
                let new_v = v.mul_scalar(beta2t)?.add(&g_sq.mul_scalar(one_minus_beta2t)?)?;
                *v = new_v.detach()?;
                let v = self.exp_avg_sq.get(&id).unwrap();
                v.rsqrt()?.mul(&grad_f32)?
            };

            // Clip update by RMS / clip_threshold (RMS normalize, then re-scale).
            let update_rms = rms_scalar(&update)?;
            let scale_div = (update_rms / self.clip_threshold).max(1.0);
            update = update.div_scalar(scale_div)?;

            // Apply decoupled weight decay then subtract update.
            let p_data = p.tensor()?;
            let p_f32 = if p_data.dtype() == DType::F32 {
                p_data
            } else {
                p_data.to_dtype(DType::F32)?
            };

            // Effective learning rate for this param. With scale_parameter=true,
            // multiply by max(eps_param, RMS(param)) — matches transformers'
            // `Adafactor` `scale_parameter=True, relative_step=False` mode.
            let lr_eff = if self.scale_parameter {
                let p_rms = rms_scalar(&p_f32)?.max(self.eps_param);
                self.lr * p_rms
            } else {
                self.lr
            };

            // Multiply by effective lr.
            update = update.mul_scalar(lr_eff)?;

            let mut new_p = p_f32;
            if self.weight_decay != 0.0 {
                // Decoupled WD scales by lr_eff (not raw lr) so that
                // scale_parameter respects the same per-param rescaling.
                let scale = 1.0 - self.weight_decay * lr_eff;
                new_p = new_p.mul_scalar(scale)?;
            }
            new_p = new_p.sub(&update)?;

            // Cast back to param dtype.
            let target_dtype = p.dtype()?;
            let cast_back = if target_dtype == DType::F32 {
                new_p
            } else {
                new_p.to_dtype(target_dtype)?
            };
            p.set_data(cast_back.detach()?)?;
        }
        Ok(())
    }

    pub fn zero_grad(&self, params: &[Parameter]) {
        for p in params {
            p.zero_grad();
        }
    }
}

/// Scalar RMS = sqrt(mean(x²)).
fn rms_scalar(x: &Tensor) -> Result<f32> {
    let m = x.square()?.mean_all()?;
    let v = m.to_vec1::<f32>()?;
    Ok(v.first().copied().unwrap_or(0.0).sqrt())
}

// ---------------------------------------------------------------------------
// AdamW8bit (partial port)
// ---------------------------------------------------------------------------

/// Host-RAM-frugal AdamW with 8-bit-quantized optimizer state in host memory.
///
/// **HONEST LABEL — read before using.** This implementation is *not* a
/// performance optimization. With the current design it is **slower than
/// `flame_core::adam::AdamW`** because every step:
///
/// 1. Dequantizes `m` and `v` from a host `Vec<i8>` back to a fresh device
///    `Tensor` (one PCIe H→D round-trip per param).
/// 2. Runs the AdamW math at full F32 on-device.
/// 3. Re-quantizes `m` and `v` back to host int8 (one PCIe D→H round-trip).
///
/// VRAM behavior is identical to `AdamW` during the step (full F32 m/v
/// are materialized) — the savings are entirely in **host RAM** between
/// steps. The intended use case is trainers with multi-billion-parameter
/// frozen blocks paged through pinned host memory: keeping optimizer
/// state quantized stops it from fighting the block-offload cache for
/// system RAM. **If you don't have that pressure, prefer `AdamW`.**
///
/// Real bitsandbytes 8-bit AdamW does block-wise dynamic 8-bit quantization
/// inside a custom CUDA kernel that never round-trips through host. We
/// don't have that kernel. Implementing it requires writing an NVRTC or
/// `.cu` kernel (per `flame-core/CLAUDE.md` conventions); when that exists
/// this struct should be rewritten to use it and this docstring's
/// "slower than AdamW" warning can be removed.
pub struct AdamW8bit {
    pub lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    t: u32,
    m_quant: HashMap<TensorId, QuantizedState>,
    v_quant: HashMap<TensorId, QuantizedState>,
}

struct QuantizedState {
    /// Host int8 storage (`i8` for signed first moment, `u8` for unsigned
    /// second moment — but stored as `i8` because the max-absolute-value
    /// quantization handles both cases with a single sign-aware scale).
    data: Vec<i8>,
    scale: f32,
    shape: Vec<usize>,
}

impl QuantizedState {
    fn quantize(t: &Tensor) -> Result<Self> {
        let host = if t.dtype() == DType::F32 {
            t.to_vec()?
        } else {
            t.to_dtype(DType::F32)?.to_vec()?
        };
        let absmax = host.iter().fold(0.0f32, |a, &b| a.max(b.abs())).max(1.0e-30);
        let scale = absmax / 127.0;
        let data: Vec<i8> = host
            .iter()
            .map(|&x| (x / scale).round().clamp(-127.0, 127.0) as i8)
            .collect();
        Ok(Self {
            data,
            scale,
            shape: t.shape().dims().to_vec(),
        })
    }

    fn dequantize(
        &self,
        device: std::sync::Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Tensor> {
        let host: Vec<f32> = self.data.iter().map(|&q| q as f32 * self.scale).collect();
        Tensor::from_vec(host, flame_core::Shape::from_dims(&self.shape), device)
    }
}

impl AdamW8bit {
    pub fn new(lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32) -> Self {
        Self {
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            t: 0,
            m_quant: HashMap::new(),
            v_quant: HashMap::new(),
        }
    }

    pub fn step(&mut self, params: &[Parameter]) -> Result<()> {
        self.t += 1;
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);

        for p in params {
            let Some(grad) = p.grad() else { continue };
            let grad_f32 = if grad.dtype() == DType::F32 {
                grad
            } else {
                grad.to_dtype(DType::F32)?
            };

            let id = p.id();
            // Dequantize moments from host int8 into device F32, or zero-init.
            let mut m = match self.m_quant.get(&id) {
                Some(q) => q.dequantize(grad_f32.device().clone())?,
                None => Tensor::zeros_dtype(
                    grad_f32.shape().clone(),
                    DType::F32,
                    grad_f32.device().clone(),
                )?,
            };
            let mut v = match self.v_quant.get(&id) {
                Some(q) => q.dequantize(grad_f32.device().clone())?,
                None => Tensor::zeros_dtype(
                    grad_f32.shape().clone(),
                    DType::F32,
                    grad_f32.device().clone(),
                )?,
            };

            // m = beta1 * m + (1 - beta1) * g
            m = m.mul_scalar(self.beta1)?
                .add(&grad_f32.mul_scalar(1.0 - self.beta1)?)?;
            // v = beta2 * v + (1 - beta2) * g²
            v = v.mul_scalar(self.beta2)?
                .add(&grad_f32.square()?.mul_scalar(1.0 - self.beta2)?)?;

            let m_hat = m.div_scalar(bc1)?;
            let v_hat = v.div_scalar(bc2)?;
            let denom = v_hat.sqrt()?.add_scalar(self.eps)?;
            let update = m_hat.div(&denom)?.mul_scalar(self.lr)?;

            let p_data = p.tensor()?;
            let p_f32 = if p_data.dtype() == DType::F32 {
                p_data
            } else {
                p_data.to_dtype(DType::F32)?
            };
            let mut new_p = p_f32;
            if self.weight_decay != 0.0 {
                let scale = 1.0 - self.weight_decay * self.lr;
                new_p = new_p.mul_scalar(scale)?;
            }
            new_p = new_p.sub(&update)?;
            let target_dtype = p.dtype()?;
            let cast_back = if target_dtype == DType::F32 {
                new_p
            } else {
                new_p.to_dtype(target_dtype)?
            };
            p.set_data(cast_back.detach()?)?;

            // Re-quantize state for next step.
            self.m_quant.insert(id, QuantizedState::quantize(&m)?);
            self.v_quant.insert(id, QuantizedState::quantize(&v)?);
        }
        Ok(())
    }

    pub fn zero_grad(&self, params: &[Parameter]) {
        for p in params {
            p.zero_grad();
        }
    }
}

// ---------------------------------------------------------------------------
// Prodigy
// ---------------------------------------------------------------------------

/// Prodigy (Mishchenko & Defazio, 2023, https://arxiv.org/abs/2306.06101) —
/// D-adaptation auto-tuning of the AdamW step size. Mirrors the reference
/// implementation at https://github.com/konstmish/prodigy/blob/main/prodigyopt/prodigy.py
/// (`decouple=True`, `safeguard_warmup=False`, `slice_p=1`,
/// `use_bias_correction=False`).
///
/// Hyperparameters:
///
/// - `lr` — multiplicative scaling on the adapted step size (typically 1.0).
///   The reference docs say "leave LR set to 1 unless you encounter
///   instability."
/// - `beta1`, `beta2`, `eps`, `weight_decay` — as for AdamW.
/// - `d_coef` (1.0) and `growth_rate` (∞ → unrestricted growth).
/// - `d0 = 1e-6` (small constant; reference default). Initial estimate of
///   the adapted step size.
///
/// Algorithm (per step, single param group):
/// ```text
///   beta3 = sqrt(beta2)
///   dlr   = d * lr * bias_correction       // bias_correction = 1 here
///
///   # numerator and denominator accumulators
///   d_numerator *= beta3
///   for p in params:
///     delta_numerator += (d/d0) * dlr * <grad, p0 - p>
///     m   = beta1 * m + d * (1 - beta1) * grad
///     v   = beta2 * v + d² * (1 - beta2) * grad²
///     s   = beta3 * s + ((d/d0) * dlr) * grad
///     d_denom += |s|.sum()                 // L1 norm
///
///   d_numerator += delta_numerator
///   d_hat = d_coef * d_numerator / d_denom
///   if d == d0: d = max(d, d_hat)
///   d_max = max(d_max, d_hat)
///   d = min(d_max, d * growth_rate)
///
///   # param update (decoupled wd)
///   denom  = sqrt(v) + d * eps
///   p     *= 1 - weight_decay * dlr
///   p     -= dlr * m / denom
/// ```
///
/// Notes: tensor-level Prodigy keeps per-param `m`, `v`, `s`, `p0` tensors.
/// `d`, `d_max`, `d_numerator` are scalars across all params.
pub struct Prodigy {
    /// Multiplicative scaling on the adapted step size. Reference recommends
    /// leaving this at 1.0.
    pub lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    d_coef: f32,
    growth_rate: f32,
    /// Initial D estimate (reference default 1e-6, fixed).
    d0: f32,
    /// Current adapted step size (scalar across all params).
    d: f32,
    /// Running max of `d_hat`.
    d_max: f32,
    /// Beta3-decayed running numerator (scalar across all params).
    d_numerator: f64,
    /// Optimizer step counter (= reference's `k+1` after step()).
    t: u32,
    m: HashMap<TensorId, Tensor>,
    v: HashMap<TensorId, Tensor>,
    /// Initial parameter snapshot (p₀).
    p0: HashMap<TensorId, Tensor>,
    /// Per-param numerator accumulator s.
    s: HashMap<TensorId, Tensor>,
}

impl Prodigy {
    pub fn new(lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32) -> Self {
        // Reference: d0 = 1e-6 (constant), NOT derived from lr. The user's
        // `lr` is a separate multiplicative scaling factor.
        let d0 = 1.0e-6_f32;
        Self {
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            d_coef: 1.0,
            growth_rate: f32::INFINITY,
            d0,
            d: d0,
            d_max: d0,
            d_numerator: 0.0,
            t: 0,
            m: HashMap::new(),
            v: HashMap::new(),
            p0: HashMap::new(),
            s: HashMap::new(),
        }
    }

    pub fn step(&mut self, params: &[Parameter]) -> Result<()> {
        self.t += 1;
        let beta3 = self.beta2.sqrt();
        let d = self.d;
        let d0 = self.d0;
        let lr = self.lr;
        // Reference: bias_correction = 1 when use_bias_correction = False.
        let bias_correction = 1.0_f32;
        let dlr = d * lr * bias_correction;

        // Pre-decay the running numerator.
        self.d_numerator *= beta3 as f64;

        let mut delta_numerator: f64 = 0.0;
        let mut d_denom: f64 = 0.0;

        // Phase 1: accumulate numerator/denom contributions and update m/v/s.
        for p in params {
            let Some(grad) = p.grad() else { continue };
            let grad_f32 = if grad.dtype() == DType::F32 {
                grad
            } else {
                grad.to_dtype(DType::F32)?
            };

            let id = p.id();
            let p_data = p.tensor()?;
            let p_f32 = if p_data.dtype() == DType::F32 {
                p_data
            } else {
                p_data.to_dtype(DType::F32)?
            };

            // Snapshot p0 the first time we see this param.
            if !self.p0.contains_key(&id) {
                self.p0.insert(id, p_f32.detach()?);
            }
            let p0 = self.p0.get(&id).unwrap().clone();

            // delta_numerator += (d/d0) * dlr * <g, p0 - p>
            let diff = p0.sub(&p_f32)?;
            let inner = grad_f32.mul(&diff)?.sum_all()?.to_vec1::<f32>()?;
            let inner_v = inner.first().copied().unwrap_or(0.0) as f64;
            delta_numerator += (d as f64 / d0 as f64) * (dlr as f64) * inner_v;

            // m = beta1 * m + d * (1 - beta1) * g
            let m_entry = match self.m.entry(id) {
                Entry::Occupied(e) => e.into_mut(),
                Entry::Vacant(e) => {
                    let zeros = Tensor::zeros_dtype(
                        grad_f32.shape().clone(),
                        DType::F32,
                        grad_f32.device().clone(),
                    )?;
                    e.insert(zeros)
                }
            };
            let m_new = m_entry
                .mul_scalar(self.beta1)?
                .add(&grad_f32.mul_scalar((1.0 - self.beta1) * d)?)?;
            *m_entry = m_new.detach()?;

            // v = beta2 * v + d² * (1 - beta2) * g²
            let v_entry = match self.v.entry(id) {
                Entry::Occupied(e) => e.into_mut(),
                Entry::Vacant(e) => {
                    let zeros = Tensor::zeros_dtype(
                        grad_f32.shape().clone(),
                        DType::F32,
                        grad_f32.device().clone(),
                    )?;
                    e.insert(zeros)
                }
            };
            let v_new = v_entry
                .mul_scalar(self.beta2)?
                .add(&grad_f32.square()?.mul_scalar((1.0 - self.beta2) * d * d)?)?;
            *v_entry = v_new.detach()?;

            // s = beta3 * s + ((d/d0) * dlr) * g     (safeguard_warmup=False)
            let s_entry = match self.s.entry(id) {
                Entry::Occupied(e) => e.into_mut(),
                Entry::Vacant(e) => {
                    let zeros = Tensor::zeros_dtype(
                        grad_f32.shape().clone(),
                        DType::F32,
                        grad_f32.device().clone(),
                    )?;
                    e.insert(zeros)
                }
            };
            let s_alpha = (d / d0) * dlr;
            let s_new = s_entry
                .mul_scalar(beta3)?
                .add(&grad_f32.mul_scalar(s_alpha)?)?;
            *s_entry = s_new.detach()?;

            // d_denom += |s|.sum()  (L1 norm)
            let s_abs_sum = self.s.get(&id).unwrap().abs()?.sum_all()?.to_vec1::<f32>()?;
            d_denom += s_abs_sum.first().copied().unwrap_or(0.0) as f64;
        }

        // Update scalar D estimate.
        if d_denom > 0.0 && lr > 0.0 {
            self.d_numerator += delta_numerator;
            let d_hat = (self.d_coef as f64) * self.d_numerator / d_denom;
            let d_hat_f32 = d_hat as f32;
            // Only grow on the first step (when d == d0).
            if (self.d - self.d0).abs() < f32::EPSILON {
                self.d = self.d.max(d_hat_f32);
            }
            self.d_max = self.d_max.max(d_hat_f32);
            // Bound by growth_rate * d, NOT d_hat directly.
            let grown = if self.growth_rate.is_finite() {
                self.d * self.growth_rate
            } else {
                f32::INFINITY
            };
            self.d = self.d_max.min(grown);
        } else {
            // No gradients on this step → keep d unchanged but undo the
            // d_numerator pre-decay so it survives to the next step.
            // (Reference returns early before the decay matters, but we
            // already mutated; restore by dividing back.)
            if beta3 > 0.0 {
                self.d_numerator /= beta3 as f64;
            }
        }

        // Phase 2: param update using the (possibly-updated) d. Reference
        // recomputes dlr here using the new d.
        let d = self.d;
        let dlr = d * lr * bias_correction;

        for p in params {
            let Some(_grad) = p.grad() else { continue };
            let id = p.id();

            // denom = sqrt(v) + d * eps    (raw v, not bias-corrected)
            let v_t = self.v.get(&id).unwrap().clone();
            let denom = v_t.sqrt()?.add_scalar(d * self.eps)?;

            // m_t (raw, not bias-corrected) — we always have beta1 > 0
            // because Prodigy::new accepts whatever the user passes; the
            // reference's beta1 == 0 fallback path is not exposed here.
            let m_t = self.m.get(&id).unwrap().clone();

            let p_data = p.tensor()?;
            let p_f32 = if p_data.dtype() == DType::F32 {
                p_data
            } else {
                p_data.to_dtype(DType::F32)?
            };
            let mut new_p = p_f32;
            // Decoupled weight decay scaled by dlr (NOT lr).
            if self.weight_decay != 0.0 {
                let scale = 1.0 - self.weight_decay * dlr;
                new_p = new_p.mul_scalar(scale)?;
            }
            // p -= dlr * m / denom
            let update = m_t.div(&denom)?.mul_scalar(dlr)?;
            new_p = new_p.sub(&update)?;

            let target_dtype = p.dtype()?;
            let cast_back = if target_dtype == DType::F32 {
                new_p
            } else {
                new_p.to_dtype(target_dtype)?
            };
            p.set_data(cast_back.detach()?)?;
        }
        Ok(())
    }

    pub fn zero_grad(&self, params: &[Parameter]) {
        for p in params {
            p.zero_grad();
        }
    }

    /// Current adapted step size — useful for logging.
    pub fn d(&self) -> f32 {
        self.d
    }
}

// ---------------------------------------------------------------------------
// Lion
// ---------------------------------------------------------------------------

/// Lion (Chen et al., 2023) — sign-of-momentum update. 1-tensor state per
/// parameter, no eps. Hyperparameter shape:
///
/// ```text
///   m   = momentum tensor (1st moment)
///   c_t = beta1 * m + (1 - beta1) * g       // interpolated update direction
///   p   = p - lr * sign(c_t) - lr * wd * p
///   m   = beta2 * m + (1 - beta2) * g       // EMA momentum
/// ```
///
/// Defaults from the Lion paper: `lr ~ adam_lr / 3 to / 10`, `beta1 = 0.9`,
/// `beta2 = 0.99` (NOT 0.999 — `0.999` slows the EMA momentum enough to
/// skew the sign-update direction), `wd ~ adam_wd * 3 to 10`. Trainers
/// using `Optimizer::new` with a `Lion` kind should pull betas from
/// [`OptimizerKind::default_betas`], which special-cases Lion correctly.
pub struct Lion {
    pub lr: f32,
    beta1: f32,
    beta2: f32,
    weight_decay: f32,
    m: HashMap<TensorId, Tensor>,
}

impl Lion {
    pub fn new(lr: f32, beta1: f32, beta2: f32, weight_decay: f32) -> Self {
        Self {
            lr,
            beta1,
            beta2,
            weight_decay,
            m: HashMap::new(),
        }
    }

    pub fn step(&mut self, params: &[Parameter]) -> Result<()> {
        for p in params {
            let Some(grad) = p.grad() else { continue };
            let grad_f32 = if grad.dtype() == DType::F32 {
                grad
            } else {
                grad.to_dtype(DType::F32)?
            };

            let id = p.id();
            let m_entry = match self.m.entry(id) {
                Entry::Occupied(e) => e.into_mut(),
                Entry::Vacant(e) => {
                    let zeros = Tensor::zeros_dtype(
                        grad_f32.shape().clone(),
                        DType::F32,
                        grad_f32.device().clone(),
                    )?;
                    e.insert(zeros)
                }
            };

            // c = beta1 * m + (1 - beta1) * g  → sign(c) is the update direction.
            let c = m_entry
                .mul_scalar(self.beta1)?
                .add(&grad_f32.mul_scalar(1.0 - self.beta1)?)?;
            let direction = c.sign()?;

            // m = beta2 * m + (1 - beta2) * g
            let m_new = m_entry
                .mul_scalar(self.beta2)?
                .add(&grad_f32.mul_scalar(1.0 - self.beta2)?)?;
            *m_entry = m_new.detach()?;

            let p_data = p.tensor()?;
            let p_f32 = if p_data.dtype() == DType::F32 {
                p_data
            } else {
                p_data.to_dtype(DType::F32)?
            };
            let mut new_p = p_f32;
            if self.weight_decay != 0.0 {
                let scale = 1.0 - self.weight_decay * self.lr;
                new_p = new_p.mul_scalar(scale)?;
            }
            new_p = new_p.sub(&direction.mul_scalar(self.lr)?)?;

            let target_dtype = p.dtype()?;
            let cast_back = if target_dtype == DType::F32 {
                new_p
            } else {
                new_p.to_dtype(target_dtype)?
            };
            p.set_data(cast_back.detach()?)?;
        }
        Ok(())
    }

    pub fn zero_grad(&self, params: &[Parameter]) {
        for p in params {
            p.zero_grad();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cudarc::driver::CudaDevice;
    use flame_core::Shape;

    #[test]
    fn parses_kind_strings() {
        assert_eq!(OptimizerKind::parse("AdamW").unwrap(), OptimizerKind::AdamW);
        assert_eq!(OptimizerKind::parse("adafactor").unwrap(), OptimizerKind::Adafactor);
        assert_eq!(OptimizerKind::parse("adamw8bit").unwrap(), OptimizerKind::AdamW8bit);
        assert_eq!(OptimizerKind::parse("Prodigy").unwrap(), OptimizerKind::Prodigy);
        assert_eq!(OptimizerKind::parse("LION").unwrap(), OptimizerKind::Lion);
        assert!(OptimizerKind::parse("garbage").is_err());
    }

    /// Lion's recommended default β₂ is 0.99, NOT the standard Adam 0.999.
    /// `(0.9, 0.99)` keeps the EMA momentum responsive enough that the
    /// sign-update direction is still meaningful (Chen et al. 2023 §3).
    #[test]
    fn default_betas_special_cases_lion() {
        assert_eq!(OptimizerKind::Lion.default_betas(), (0.9, 0.99));
        assert_eq!(OptimizerKind::AdamW.default_betas(), (0.9, 0.999));
        assert_eq!(OptimizerKind::Adafactor.default_betas(), (0.9, 0.999));
        assert_eq!(OptimizerKind::AdamW8bit.default_betas(), (0.9, 0.999));
        assert_eq!(OptimizerKind::Prodigy.default_betas(), (0.9, 0.999));
    }

    /// Each variant of `Optimizer::new` is constructible and step()/zero_grad()
    /// don't panic on an empty param list.
    #[test]
    fn dispatch_constructs_and_no_panic_on_empty() {
        // No CUDA needed — empty param list short-circuits before any
        // tensor work.
        for &kind in &[
            OptimizerKind::AdamW,
            OptimizerKind::Adafactor,
            OptimizerKind::AdamW8bit,
            OptimizerKind::Prodigy,
            OptimizerKind::Lion,
        ] {
            let mut opt = Optimizer::new(kind, 1e-3, 0.9, 0.999, 1e-8, 0.01);
            assert_eq!(opt.kind(), kind);
            opt.zero_grad(&[]);
            opt.step(&[]).expect("step on empty params is a no-op");
            opt.set_lr(2e-3); // shouldn't panic
        }
    }

    // ---- CUDA tests below ---------------------------------------------------
    fn cuda_or_skip() -> Option<std::sync::Arc<CudaDevice>> {
        match CudaDevice::new(0) {
            Ok(d) => Some(d),
            Err(_) => {
                eprintln!("skipping: no CUDA device");
                None
            }
        }
    }

    fn make_param(
        device: std::sync::Arc<CudaDevice>,
        data: Vec<f32>,
        shape: Vec<usize>,
    ) -> Parameter {
        let t = Tensor::from_vec(data, Shape::from_dims(&shape), device).unwrap();
        Parameter::new(t.requires_grad_(true))
    }

    fn set_grad(p: &Parameter, data: Vec<f32>, shape: Vec<usize>, device: std::sync::Arc<CudaDevice>) {
        let g = Tensor::from_vec(data, Shape::from_dims(&shape), device).unwrap();
        p.set_grad(g).unwrap();
    }

    fn vec_close(a: &[f32], b: &[f32], eps: f32) -> bool {
        if a.len() != b.len() {
            return false;
        }
        a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < eps)
    }

    /// AdamW (our dispatch into flame-core::AdamW): 5 steps with the same
    /// fixed grad on a 4-element param. We compare against an inline F32
    /// reference implementing the textbook AdamW update — *not* against
    /// flame-core itself (would be circular). This catches dispatch bugs
    /// (wrong betas, wrong order of WD, ...).
    #[test]
    fn adamw_dispatch_5_steps_matches_reference() {
        let Some(device) = cuda_or_skip() else { return };
        let lr = 1e-2_f32;
        let beta1 = 0.9_f32;
        let beta2 = 0.999_f32;
        let eps = 1e-8_f32;
        let wd = 0.01_f32;

        // Param: [1.0, 2.0, 3.0, 4.0]; constant grad: [0.1, -0.2, 0.3, -0.4].
        let init = vec![1.0, 2.0, 3.0, 4.0];
        let grad = vec![0.1, -0.2, 0.3, -0.4];
        let p = make_param(device.clone(), init.clone(), vec![4]);
        let mut opt = Optimizer::new(OptimizerKind::AdamW, lr, beta1, beta2, eps, wd);

        // Reference: same scalars, run inline.
        let mut ref_p = init.clone();
        let mut ref_m = vec![0.0_f32; 4];
        let mut ref_v = vec![0.0_f32; 4];

        for t in 1..=5 {
            set_grad(&p, grad.clone(), vec![4], device.clone());
            opt.step(&[p.clone()]).unwrap();

            // Reference update (textbook AdamW with decoupled WD).
            let bc1 = 1.0 - beta1.powi(t);
            let bc2 = 1.0 - beta2.powi(t);
            for i in 0..4 {
                ref_m[i] = beta1 * ref_m[i] + (1.0 - beta1) * grad[i];
                ref_v[i] = beta2 * ref_v[i] + (1.0 - beta2) * grad[i] * grad[i];
                let m_hat = ref_m[i] / bc1;
                let v_hat = ref_v[i] / bc2;
                // Decoupled WD: param *= (1 - lr*wd) BEFORE the step.
                ref_p[i] *= 1.0 - lr * wd;
                ref_p[i] -= lr * m_hat / (v_hat.sqrt() + eps);
            }
        }
        let got = p.tensor().unwrap().to_vec().unwrap();
        assert!(
            vec_close(&got, &ref_p, 5e-5),
            "AdamW 5-step trajectory mismatch:\n  got: {:?}\n  ref: {:?}",
            got,
            ref_p
        );
    }

    /// Adafactor on a 4×4 weight, fixed grad, 5 steps. We replicate the
    /// algorithm inline (factored second moment, RMS clipping, decoupled
    /// WD) and check for ≤1e-4 absolute deviation.
    #[test]
    fn adafactor_5_steps_matches_inline_reference() {
        let Some(device) = cuda_or_skip() else { return };
        let lr = 1e-3_f32;
        let eps = 1e-3_f32;
        let wd = 0.0_f32;
        let decay_rate = -0.8_f32;
        let clip = 1.0_f32;
        let eps_grad = 1e-30_f32;

        let init: Vec<f32> = (0..16).map(|i| 0.1 + i as f32 * 0.01).collect();
        // Non-trivial grad so factored row/col differ.
        let grad: Vec<f32> = (0..16).map(|i| 0.05 - i as f32 * 0.003).collect();
        let p = make_param(device.clone(), init.clone(), vec![4, 4]);
        let mut opt = Adafactor::new(lr, eps, wd);

        // Reference.
        let mut ref_p = init.clone();
        let mut row = vec![0.0_f32; 4]; // mean over last dim → shape [4]
        let mut col = vec![0.0_f32; 4]; // mean over second-to-last dim → shape [4]

        for t in 1..=5 {
            set_grad(&p, grad.clone(), vec![4, 4], device.clone());
            opt.step(&[p.clone()]).unwrap();

            let beta2t = 1.0 - (t as f32).powf(decay_rate);
            let one_m = 1.0 - beta2t;
            let g_sq: Vec<f32> = grad.iter().map(|g| g * g + eps_grad).collect();
            // Mean over last dim (per row): rows[r] = mean of g_sq[r,:]
            let mean_last: Vec<f32> = (0..4)
                .map(|r| (0..4).map(|c| g_sq[r * 4 + c]).sum::<f32>() / 4.0)
                .collect();
            // Mean over second-to-last (per col): cols[c] = mean of g_sq[:,c]
            let mean_second: Vec<f32> = (0..4)
                .map(|c| (0..4).map(|r| g_sq[r * 4 + c]).sum::<f32>() / 4.0)
                .collect();
            for r in 0..4 {
                row[r] = beta2t * row[r] + one_m * mean_last[r];
            }
            for c in 0..4 {
                col[c] = beta2t * col[c] + one_m * mean_second[c];
            }

            // r_factor = rsqrt(row / row.mean(-1, keepdim=True))  — shape [4,1]
            // c_factor = rsqrt(col)                              — shape [1,4] (after unsqueeze -1 then transpose-style broadcast)
            let row_mean: f32 = row.iter().sum::<f32>() / 4.0;
            let r_factor: Vec<f32> = row.iter().map(|x| 1.0 / (x / row_mean).sqrt()).collect();
            let c_factor: Vec<f32> = col.iter().map(|x| 1.0 / x.sqrt()).collect();

            // approx[r,c] = r_factor[r] * c_factor[c]
            let mut update: Vec<f32> = (0..16)
                .map(|i| {
                    let r = i / 4;
                    let c = i % 4;
                    r_factor[r] * c_factor[c] * grad[i]
                })
                .collect();

            // RMS clipping.
            let rms = (update.iter().map(|x| x * x).sum::<f32>() / 16.0).sqrt();
            let scale_div = (rms / clip).max(1.0);
            for u in &mut update {
                *u /= scale_div;
            }
            // lr scaling (scale_parameter=false → lr_eff = lr).
            for u in &mut update {
                *u *= lr;
            }
            // Decoupled WD then sub.
            for i in 0..16 {
                if wd != 0.0 {
                    ref_p[i] *= 1.0 - wd * lr;
                }
                ref_p[i] -= update[i];
            }
        }

        let got = p.tensor().unwrap().to_vec().unwrap();
        assert!(
            vec_close(&got, &ref_p, 1e-4),
            "Adafactor 5-step trajectory mismatch (max diff {})",
            got.iter()
                .zip(ref_p.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max)
        );
    }

    /// Adafactor with `scale_parameter = true` modifies the per-step lr:
    /// lr_eff = max(eps_param, RMS(p)) * lr. With our 4×4 init, RMS ≈ 0.21
    /// >> eps_param=1e-3, so the effective LR is much smaller than the raw
    /// `lr`. Verify the parameter moves *less* per step than scale_parameter=false.
    #[test]
    fn adafactor_scale_parameter_reduces_step() {
        let Some(device) = cuda_or_skip() else { return };
        let lr = 1e-3_f32;
        let init: Vec<f32> = (0..16).map(|i| 0.1 + i as f32 * 0.01).collect();
        let grad: Vec<f32> = vec![0.5; 16];

        // Without scale_parameter.
        let p_a = make_param(device.clone(), init.clone(), vec![4, 4]);
        let mut opt_a = Adafactor::with_options(lr, 1e-3, 0.0, false);
        set_grad(&p_a, grad.clone(), vec![4, 4], device.clone());
        opt_a.step(&[p_a.clone()]).unwrap();

        // With scale_parameter.
        let p_b = make_param(device.clone(), init.clone(), vec![4, 4]);
        let mut opt_b = Adafactor::with_options(lr, 1e-3, 0.0, true);
        set_grad(&p_b, grad.clone(), vec![4, 4], device.clone());
        opt_b.step(&[p_b.clone()]).unwrap();

        let a = p_a.tensor().unwrap().to_vec().unwrap();
        let b = p_b.tensor().unwrap().to_vec().unwrap();

        // RMS of init ≈ sqrt(mean(p²)) ≈ 0.196
        let init_rms = (init.iter().map(|x| x * x).sum::<f32>() / 16.0).sqrt();
        assert!(init_rms > 1e-3, "guard: init_rms must exceed eps_param");

        // A's deltas should be larger than B's (B uses lr * RMS(p) ≈ lr * 0.196).
        let delta_a: f32 = a
            .iter()
            .zip(init.iter())
            .map(|(x, i)| (x - i).abs())
            .sum();
        let delta_b: f32 = b
            .iter()
            .zip(init.iter())
            .map(|(x, i)| (x - i).abs())
            .sum();
        assert!(
            delta_a > delta_b * 2.0,
            "scale_parameter=true should reduce step size; |Δa|={}, |Δb|={}",
            delta_a,
            delta_b
        );
    }

    /// AdamW8bit ≈ AdamW within a quantization-noise tolerance.
    ///
    /// **Empirical finding from this test**: with N=4 params, mixed-sign
    /// `[0.1, -0.2, 0.3, -0.4]` grads, and 3 steps, the worst-case
    /// per-element drift is ≈ 1.3e-4 (quantizing a 4-element `v` tensor
    /// where absmax is dominated by one element forces the others to
    /// quantize at low effective precision). We allow up to 5e-4 to leave
    /// headroom — the test still catches algorithmic divergence (which
    /// would be O(lr · grad) ≈ O(1e-3) per step).
    #[test]
    fn adamw8bit_close_to_adamw() {
        let Some(device) = cuda_or_skip() else { return };
        let lr = 1e-2_f32;
        let beta1 = 0.9_f32;
        let beta2 = 0.999_f32;
        let eps = 1e-8_f32;
        let wd = 0.0_f32;

        let init: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let grad: Vec<f32> = vec![0.1, -0.2, 0.3, -0.4];

        let p_a = make_param(device.clone(), init.clone(), vec![4]);
        let p_b = make_param(device.clone(), init.clone(), vec![4]);

        let mut adam = flame_core::adam::AdamW::new(lr, beta1, beta2, eps, wd);
        let mut adam8 = AdamW8bit::new(lr, beta1, beta2, eps, wd);

        for _ in 0..3 {
            set_grad(&p_a, grad.clone(), vec![4], device.clone());
            set_grad(&p_b, grad.clone(), vec![4], device.clone());
            adam.step(&[p_a.clone()]).unwrap();
            adam8.step(&[p_b.clone()]).unwrap();
        }

        let a = p_a.tensor().unwrap().to_vec().unwrap();
        let b = p_b.tensor().unwrap().to_vec().unwrap();
        assert!(
            vec_close(&a, &b, 5e-4),
            "AdamW8bit drifted from AdamW > 5e-4:\n  AdamW: {:?}\n  8bit:  {:?}",
            a,
            b
        );
    }

    /// Prodigy on a strongly-convex quadratic `f(x) = 0.5 * x^T A x` with
    /// A = diag(2,3,5). Starting from a random-ish init, 200 steps should
    /// drive `||x||` close to zero.
    #[test]
    fn prodigy_minimizes_quadratic() {
        let Some(device) = cuda_or_skip() else { return };
        // 3-D PSD diag(2, 3, 5). Gradient is A @ x = (2 x0, 3 x1, 5 x2).
        let init: Vec<f32> = vec![1.0, -0.7, 0.4];
        let p = make_param(device.clone(), init.clone(), vec![3]);

        // Prodigy reference recommends `lr = 1.0`. Use standard betas/eps.
        let mut opt = Prodigy::new(1.0, 0.9, 0.999, 1e-8, 0.0);

        let a_diag = [2.0_f32, 3.0_f32, 5.0_f32];
        for _ in 0..200 {
            // Compute grad = diag(A) * x.
            let cur = p.tensor().unwrap().to_vec().unwrap();
            let grad: Vec<f32> = (0..3).map(|i| a_diag[i] * cur[i]).collect();
            set_grad(&p, grad, vec![3], device.clone());
            opt.step(&[p.clone()]).unwrap();
        }

        let final_x = p.tensor().unwrap().to_vec().unwrap();
        let norm = (final_x.iter().map(|x| x * x).sum::<f32>()).sqrt();
        // d starts at 1e-6 and grows; 200 steps on a well-conditioned
        // quadratic should pull ||x|| well below 0.1.
        assert!(
            norm < 0.1,
            "Prodigy failed to converge: ||x|| = {}, x = {:?}, d = {}",
            norm,
            final_x,
            opt.d()
        );
    }

    /// Lion sign update — single step, hand-checked.
    /// Initial: p=[1.0], m=[0.0]. Grad: g=[0.5].
    /// β₁=0.9, β₂=0.99, lr=0.01, wd=0.
    ///   c = β₁·m + (1-β₁)·g = 0.9·0 + 0.1·0.5 = 0.05
    ///   sign(c) = 1.0
    ///   m' = β₂·m + (1-β₂)·g = 0.99·0 + 0.01·0.5 = 0.005
    ///   p' = p - lr · sign(c) = 1.0 - 0.01·1.0 = 0.99
    #[test]
    fn lion_sign_update_hand_checked() {
        let Some(device) = cuda_or_skip() else { return };
        let p = make_param(device.clone(), vec![1.0], vec![1]);
        let mut opt = Lion::new(0.01, 0.9, 0.99, 0.0);
        set_grad(&p, vec![0.5], vec![1], device.clone());
        opt.step(&[p.clone()]).unwrap();
        let got = p.tensor().unwrap().to_vec().unwrap();
        assert!(
            (got[0] - 0.99).abs() < 1e-6,
            "Lion expected 0.99, got {}",
            got[0]
        );
    }

    /// Lion: a negative interpolated direction subtracts -lr (i.e. param goes UP).
    /// p=[1.0], g=[-0.5], β₁=0.9, m=0 ⇒ c = -0.05, sign(c) = -1
    ///   p' = 1.0 - 0.01·(-1) = 1.01
    #[test]
    fn lion_negative_grad_increases_param() {
        let Some(device) = cuda_or_skip() else { return };
        let p = make_param(device.clone(), vec![1.0], vec![1]);
        let mut opt = Lion::new(0.01, 0.9, 0.99, 0.0);
        set_grad(&p, vec![-0.5], vec![1], device.clone());
        opt.step(&[p.clone()]).unwrap();
        let got = p.tensor().unwrap().to_vec().unwrap();
        assert!((got[0] - 1.01).abs() < 1e-6, "Lion expected 1.01, got {}", got[0]);
    }
}

