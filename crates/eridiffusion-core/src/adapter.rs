//! Per-target adapter trait — the kohya / SimpleTuner pattern.
//!
//! `AdapterModule` is the unified Rust trait that lets `LoRALinear` (legacy
//! plain LoRA) and `LycorisLinear` (LoCon / LoHa / LoKr / Full / OFT, plus
//! optional DoRA) co-exist in the same `Vec<Box<dyn AdapterModule>>`. Each
//! adapter is a per-target wrapper, so model-side per-call-site code stays
//! IDENTICAL regardless of algo:
//!
//! ```rust,ignore
//! let delta = self.adapters[idx].forward_delta(input)?;
//! out = base_proj_out.add(&delta)?;
//! ```
//!
//! Same semantic as SimpleTuner's `lycoris/wrapper.py` monkey-patching, but
//! a Rust trait dispatched through a `Box<dyn>` vtable. Multi-arm match
//! statements at every LoRA call site are no longer needed.
//!
//! # Why owned `Tensor` returns instead of `&Tensor`?
//!
//! `LoRALinear` stores its leaves as [`flame_core::parameter::Parameter`],
//! which holds the underlying tensor behind `Arc<Mutex<Tensor>>` and only
//! exposes a clone via `Parameter::tensor()`. There is no `&Tensor`
//! accessor, so a `Vec<&Tensor>` return type can't represent the LoRA
//! variant without re-architecting `Parameter`. `Tensor::clone` is cheap
//! (the underlying GPU storage is `Arc`-shared via `TensorStorage`), so
//! returning `Vec<Tensor>` is the right trade. The `LycorisLinear` variant
//! similarly clones from its bare-`Tensor` fields — same cost.
//!
//! # Algos that don't fit `forward_delta` cleanly
//!
//! * **OFT** is multiplicative (`R^T·W·x`), not additive. Use
//!   [`AdapterModule::is_input_rotation`] to detect, then call
//!   [`AdapterModule::apply_input`] to rotate the input *before* the base
//!   linear. The default `forward_delta` impl on `LycorisLinear` returns
//!   `R·x − x` so a plain `base + delta` call site still works for the
//!   identity-base case, but the rotation path is the recommended one.
//! * **DoRA** wraps `WP = W_orig + ΔW` with `m * WP/||WP||`, which can't be
//!   written as an additive delta on `x` alone. For DoRA-on adapters the
//!   trainer needs to reconstruct `WP` and call
//!   [`AdapterModule::apply_dora`]. See `lycoris-rs/src/dora.rs` for the
//!   math; this trait just exposes the magnitude tensor and the
//!   `apply_weight_decompose` wrapper.

use flame_core::{parameter::Parameter, Error as FlameError, Result as FlameResult, Tensor};

use lycoris_rs::{LycorisAdapter, LycorisModule, StorageDtype};

use crate::lora::LoRALinear;

/// One target Linear/Conv2d wrapped with a trainable adapter.
///
/// Per-call-site code is identical regardless of algo:
///
/// ```rust,ignore
/// let delta = self.adapters[idx].forward_delta(input)?;
/// out = base + delta;
/// ```
///
/// Trait methods return `flame_core::Result<Tensor>` (not the `lycoris_rs`
/// `Result`) so the trait composes cleanly with the broader EDv2 forward
/// path (which threads `flame_core::Result`).
pub trait AdapterModule: Send + Sync {
    /// Delta to ADD to base output. Required for LoRA, LoCon, LoHa, LoKr,
    /// Full. For OFT this returns `R·x − x` so a plain
    /// `base = base + delta` call site still works for the identity-base
    /// case, but callers SHOULD prefer the
    /// [`apply_input`](Self::apply_input) path when
    /// [`is_input_rotation`](Self::is_input_rotation) returns `true` —
    /// it avoids one subtraction per step and is the only correct path for
    /// non-identity-base linears.
    fn forward_delta(&self, input: &Tensor) -> FlameResult<Tensor>;

    /// Returns `true` for multiplicative-on-input adapters (OFT). Caller
    /// should rotate the input via [`apply_input`](Self::apply_input)
    /// *before* the base linear, and skip the base + delta combine.
    fn is_input_rotation(&self) -> bool {
        false
    }

    /// Apply the input rotation directly. Errors on non-OFT adapters.
    fn apply_input(&self, _input: &Tensor) -> FlameResult<Tensor> {
        Err(FlameError::InvalidOperation(
            "apply_input is only valid on input-rotation adapters (OFT)".into(),
        ))
    }

    /// Returns `true` when DoRA magnitude is registered for this adapter.
    fn is_dora_active(&self) -> bool {
        false
    }

    /// Apply DoRA's `m * WP/||WP||` rescaling to a fully-reconstructed
    /// weight `WP = W_orig + ΔW`. Errors on adapters without an active
    /// magnitude.
    fn apply_dora(&self, _wp_out: &Tensor) -> FlameResult<Tensor> {
        Err(FlameError::InvalidOperation(
            "apply_dora is only valid on DoRA-on adapters".into(),
        ))
    }

    /// Owned leaf tensors, in stable per-algo order. The trainer wraps each
    /// into [`Parameter`] (or registers them with its optimizer) for the
    /// AdamW step. Order contracts come from
    /// [`lycoris_rs::LycorisModule::parameters`]; for `LoRALinear` the
    /// order is `[lora_a, lora_b]`.
    ///
    /// `Tensor::clone` is cheap (`TensorStorage` is `Arc`-shared), so
    /// returning owned tensors instead of `&Tensor` does not duplicate
    /// device memory.
    fn parameters(&self) -> Vec<Tensor>;

    /// Wrap each owned leaf in [`Parameter`]. Convenience for callers that
    /// want a flat parameter list to hand to the optimizer.
    fn to_parameters(&self) -> Vec<Parameter>;

    /// Stable string identifier for the algo: `"lora" | "locon" | "loha" |
    /// "lokr" | "full" | "oft"`.
    fn kind(&self) -> &'static str;

    /// Per-adapter (suffix, tensor) pairs used for save/load. Combined as
    /// `<prefix>.<adapter_name>.<suffix>`.
    ///
    /// Suffix convention (matches `lycoris-upstream` exactly):
    ///
    /// | Algo  | Suffixes                                                   |
    /// | ----- | ---------------------------------------------------------- |
    /// | LoRA  | `lora_A.weight`, `lora_B.weight`                           |
    /// | LoCon | `lora_down.weight`, `lora_up.weight` (+ `lora_mid.weight`) |
    /// | LoHa  | `hada_w1_a/b`, `hada_w2_a/b` (+ `hada_t1`, `hada_t2`)      |
    /// | LoKr  | `lokr_w1[_a/b]`, `lokr_w2[_a/b]` (+ `lokr_t2`)             |
    /// | Full  | `diff.weight` (+ `diff_b`)                                 |
    /// | OFT   | `oft_blocks`                                               |
    /// | DoRA  | `dora_scale` appended to any of the above                  |
    fn named_tensors(&self) -> Vec<(&'static str, Tensor)>;
}

// ---------------------------------------------------------------------------
// LoRALinear impl
// ---------------------------------------------------------------------------

impl AdapterModule for LoRALinear {
    fn forward_delta(&self, input: &Tensor) -> FlameResult<Tensor> {
        // Delegate to the inherent method on `LoRALinear`. The inherent
        // signature returns `crate::Result` (= `Result<_, EriDiffusionError>`),
        // so we re-surface any error as `FlameError::InvalidOperation` to
        // keep the trait's contract a pure flame_core type.
        LoRALinear::forward_delta(self, input)
            .map_err(|e| FlameError::InvalidOperation(format!("LoRALinear::forward_delta: {e}")))
    }

    fn parameters(&self) -> Vec<Tensor> {
        // `Parameter::tensor()` clones the inner tensor (cheap — Arc-shared
        // storage). Order: `[lora_a, lora_b]` to match
        // `LoRALinear::parameters()` and `LycorisModule::parameters()` for
        // LoCon-Linear, so optimizer-state index ↔ checkpoint index is
        // consistent across algos.
        let a = self
            .lora_a
            .tensor()
            .expect("LoRALinear: lora_a parameter mutex poisoned");
        let b = self
            .lora_b
            .tensor()
            .expect("LoRALinear: lora_b parameter mutex poisoned");
        vec![a, b]
    }

    fn to_parameters(&self) -> Vec<Parameter> {
        vec![self.lora_a.clone(), self.lora_b.clone()]
    }

    fn kind(&self) -> &'static str {
        "lora"
    }

    fn named_tensors(&self) -> Vec<(&'static str, Tensor)> {
        let a = self
            .lora_a
            .tensor()
            .expect("LoRALinear: lora_a parameter mutex poisoned");
        let b = self
            .lora_b
            .tensor()
            .expect("LoRALinear: lora_b parameter mutex poisoned");
        vec![("lora_A.weight", a), ("lora_B.weight", b)]
    }
}

// ---------------------------------------------------------------------------
// LycorisLinear wrapper
// ---------------------------------------------------------------------------

/// Wraps a [`LycorisAdapter`] (LoCon / LoHa / LoKr / Full / OFT) plus optional
/// DoRA magnitude into a single per-target adapter. Constructed via
/// [`AdapterStore::build_and_push_linear`](crate::lycoris::AdapterStore::build_and_push_linear)
/// or [`build_and_push_conv2d`](crate::lycoris::AdapterStore::build_and_push_conv2d).
pub struct LycorisLinear {
    pub adapter: LycorisAdapter,
    /// Trainable DoRA magnitude (`requires_grad=true`). `None` when DoRA
    /// disabled or when the algo doesn't support it (OFT).
    pub dora_magnitude: Option<Tensor>,
    /// Axis convention for DoRA. `true` = lycoris-upstream (`||·||` over
    /// input dims). `false` = OneTrainer (`||·||` over output dim).
    pub dora_wd_on_out: bool,
    /// Epsilon added to `||WP||_2` denominator in [`apply_dora`].
    pub dora_eps: f32,
    /// Storage dtype of leaf tensors.
    pub storage: StorageDtype,
}

impl LycorisLinear {
    /// Convenience constructor — used by the AdapterStore builders. External
    /// callers should go through [`AdapterStore::build_and_push_linear`].
    pub fn new(
        adapter: LycorisAdapter,
        dora_magnitude: Option<Tensor>,
        dora_wd_on_out: bool,
        dora_eps: f32,
        storage: StorageDtype,
    ) -> Self {
        Self {
            adapter,
            dora_magnitude,
            dora_wd_on_out,
            dora_eps,
            storage,
        }
    }

    /// Cast `input` to the trainer storage dtype if needed (typical pattern
    /// is BF16 model input → F32 LoRA leaves), run the upstream forward,
    /// cast the output back to the input dtype. Mirrors the cast pattern
    /// inside [`LoRALinear::forward_delta`].
    fn forward_delta_cast(&self, input: &Tensor) -> FlameResult<Tensor> {
        let storage_dt = self.storage.to_dtype();
        let input_dt = input.dtype();
        let cast_in = if input_dt != storage_dt {
            input.to_dtype(storage_dt)?
        } else {
            input.clone()
        };
        let out = match &self.adapter {
            LycorisAdapter::LoCon(m) => m.forward(&cast_in),
            LycorisAdapter::LoHa(m) => m.forward(&cast_in),
            LycorisAdapter::LoKr(m) => m.forward(&cast_in),
            LycorisAdapter::Full(_) => {
                // FullAdapter is a pure weight delta — the trainer must
                // either merge `delta_weight()` into the base weight
                // before the linear / conv, or compute `x @ ΔW.T`
                // explicitly with knowledge of the layer kind. There is
                // no generic `x → ΔW(x)` we can run from inside the
                // wrapper without dragging the layer kind in.
                return Err(FlameError::InvalidOperation(
                    "LycorisLinear::forward_delta: Full adapter has no residual forward — \
                     use `delta_weight()` and merge into the base weight instead"
                        .into(),
                ));
            }
            LycorisAdapter::OFT(m) => {
                // OFT is multiplicative: return `R·x − x` so the call site's
                // `base + delta` collapses to `base + (R−I)·x`. For the
                // identity-base case (`base ≡ x`) this gives `R·x` ≡ correct
                // OFT output. For a non-identity base this is not what OFT
                // means; callers should branch on [`is_input_rotation`] and
                // use [`apply_input`] instead.
                m.apply_to_input(&cast_in)
                    .and_then(|rx| rx.sub(&cast_in).map_err(lycoris_rs::Error::Flame))
            }
        }
        .map_err(|e| FlameError::InvalidOperation(format!("LycorisLinear::forward: {e}")))?;
        if out.dtype() != input_dt {
            Ok(out.to_dtype(input_dt)?)
        } else {
            Ok(out)
        }
    }
}

impl AdapterModule for LycorisLinear {
    fn forward_delta(&self, input: &Tensor) -> FlameResult<Tensor> {
        self.forward_delta_cast(input)
    }

    fn is_input_rotation(&self) -> bool {
        matches!(self.adapter, LycorisAdapter::OFT(_))
    }

    fn apply_input(&self, input: &Tensor) -> FlameResult<Tensor> {
        match &self.adapter {
            LycorisAdapter::OFT(m) => {
                let storage_dt = self.storage.to_dtype();
                let input_dt = input.dtype();
                let cast_in = if input_dt != storage_dt {
                    input.to_dtype(storage_dt)?
                } else {
                    input.clone()
                };
                let rx = m
                    .apply_to_input(&cast_in)
                    .map_err(|e| FlameError::InvalidOperation(format!("OFT::apply_to_input: {e}")))?;
                if rx.dtype() != input_dt {
                    Ok(rx.to_dtype(input_dt)?)
                } else {
                    Ok(rx)
                }
            }
            _ => Err(FlameError::InvalidOperation(
                "apply_input only valid on OFT adapters".into(),
            )),
        }
    }

    fn is_dora_active(&self) -> bool {
        self.dora_magnitude.is_some()
    }

    fn apply_dora(&self, wp_out: &Tensor) -> FlameResult<Tensor> {
        let m = self.dora_magnitude.as_ref().ok_or_else(|| {
            FlameError::InvalidOperation("apply_dora called on adapter without magnitude".into())
        })?;
        lycoris_rs::dora::apply_weight_decompose(wp_out, m, self.dora_wd_on_out, self.dora_eps)
            .map_err(|e| FlameError::InvalidOperation(format!("apply_weight_decompose: {e}")))
    }

    fn parameters(&self) -> Vec<Tensor> {
        // `LycorisAdapter::parameters()` already returns owned tensor clones
        // sourced from the live `Parameter` storage (see
        // `LycorisModule::parameters` contract in `lycoris-rs/src/lib.rs`).
        // Each clone preserves `TensorId` and `requires_grad`, so autograd
        // grads route back to the live handle returned by
        // `to_parameters` / `parameters_handles`.
        let mut p: Vec<Tensor> = self.adapter.parameters();
        if let Some(ref m) = self.dora_magnitude {
            p.push(m.clone());
        }
        p
    }

    fn to_parameters(&self) -> Vec<Parameter> {
        // Return live `Parameter` handles rather than wrapping plain tensor
        // clones. `Parameter::new(tensor)` would create a fresh
        // `Arc<Mutex<Tensor>>` whose `set_data` mutations are invisible to
        // the algorithm's internal `Parameter` storage (the gradient-
        // isolation bug). Cloning the existing handle is cheap (one Arc
        // bump) and the optimizer's mutations land directly on the
        // adapter's internal leaves, which `forward` reads via
        // `param.tensor()?`.
        let mut out: Vec<Parameter> = self.adapter.parameters_handles();
        if let Some(ref m) = self.dora_magnitude {
            // DoRA magnitude is currently a bare `Tensor`. If/when it
            // becomes a trainable leaf with optimizer updates, switch the
            // field to `Parameter` and clone the handle here. For now wrap
            // a fresh Parameter — current code paths don't optimize this.
            out.push(Parameter::new(m.clone()));
        }
        out
    }

    fn kind(&self) -> &'static str {
        match &self.adapter {
            LycorisAdapter::LoCon(_) => "locon",
            LycorisAdapter::LoHa(_) => "loha",
            LycorisAdapter::LoKr(_) => "lokr",
            LycorisAdapter::Full(_) => "full",
            LycorisAdapter::OFT(_) => "oft",
        }
    }

    fn named_tensors(&self) -> Vec<(&'static str, Tensor)> {
        // Helper: read the live tensor out of a `Parameter`, surfacing the
        // mutex-poisoned case as a panic (the only way it fires is a
        // previously-poisoned trainer state, which is a hard bug anyway).
        fn pt(p: &Parameter) -> Tensor {
            p.tensor().expect("LycorisLinear::named_tensors: parameter mutex poisoned")
        }
        let mut out: Vec<(&'static str, Tensor)> = Vec::new();
        match &self.adapter {
            LycorisAdapter::LoCon(m) => {
                // Suffix convention matches `lycoris-upstream` `locon.py`:
                // `lora_down.weight` / `lora_up.weight` (+ `lora_mid.weight`
                // when Tucker is on for conv variants).
                out.push(("lora_down.weight", pt(&m.down)));
                out.push(("lora_up.weight", pt(&m.up)));
                if let Some(ref mid) = m.mid {
                    out.push(("lora_mid.weight", pt(mid)));
                }
            }
            LycorisAdapter::LoHa(m) => {
                out.push(("hada_w1_a", pt(&m.w1a)));
                out.push(("hada_w1_b", pt(&m.w1b)));
                out.push(("hada_w2_a", pt(&m.w2a)));
                out.push(("hada_w2_b", pt(&m.w2b)));
                if let Some(ref t) = m.t1 {
                    out.push(("hada_t1", pt(t)));
                }
                if let Some(ref t) = m.t2 {
                    out.push(("hada_t2", pt(t)));
                }
            }
            LycorisAdapter::LoKr(m) => {
                if let Some(ref w) = m.w1 {
                    out.push(("lokr_w1", pt(w)));
                }
                if let Some(ref w) = m.w1a {
                    out.push(("lokr_w1_a", pt(w)));
                }
                if let Some(ref w) = m.w1b {
                    out.push(("lokr_w1_b", pt(w)));
                }
                if let Some(ref w) = m.w2 {
                    out.push(("lokr_w2", pt(w)));
                }
                if let Some(ref w) = m.w2a {
                    out.push(("lokr_w2_a", pt(w)));
                }
                if let Some(ref w) = m.w2b {
                    out.push(("lokr_w2_b", pt(w)));
                }
                if let Some(ref t) = m.t2 {
                    out.push(("lokr_t2", pt(t)));
                }
            }
            LycorisAdapter::Full(m) => {
                out.push(("diff.weight", pt(&m.diff)));
                if let Some(ref b) = m.diff_b {
                    out.push(("diff_b", pt(b)));
                }
            }
            LycorisAdapter::OFT(m) => {
                out.push(("oft_blocks", pt(&m.blocks)));
            }
        }
        if let Some(ref m) = self.dora_magnitude {
            out.push(("dora_scale", m.clone()));
        }
        out
    }
}
