//! LoRA adapters for SenseNova-U1 `_mot_gen` modules.
//!
//! Port of `train_u1/model/lora.py` from the upstream Python trainer,
//! adapted to flame-core tensor ops and EDv2's `Parameter` infrastructure.
//! Does **not** depend on `lycoris-rs` (per session direction: that crate may
//! not work for U1's structurally-different gen path).
//!
//! ## Targets
//!
//! Per-layer (× 42 transformer layers):
//!   * `<L>.self_attn.q_proj_mot_gen`, `.k_proj_mot_gen`,
//!     `.v_proj_mot_gen`, `.o_proj_mot_gen`
//!   * `<L>.mlp_mot_gen.gate_proj`, `.up_proj`, `.down_proj`
//!
//! Shared (× 2):
//!   * `fm_modules.fm_head.0`, `.2`
//!
//! where `<L> = language_model.model.layers.{i}`.
//!
//! ## Storage convention
//!
//! `<adapter_key>.lora_down.weight` shape `(r, in_features)` F32 master
//! `<adapter_key>.lora_up.weight`   shape `(out_features, r)` F32 master
//! `<adapter_key>.alpha`             scalar f32
//!
//! Save format matches Python's upstream PEFT-style emission so checkpoints
//! interop with the upstream inference + 8-step distill stacking.
//!
//! ## Forward
//!
//! `y = base(x) + (alpha/r) * lora_up(lora_down(x))`
//!
//! No dropout in the smoke build — easy to add later by wrapping the
//! `lora_down` input in a Bernoulli mask.

use std::collections::HashMap;
use std::sync::Arc;

use flame_core::{parameter::Parameter, CudaDevice, DType, Error, Result, Shape, Tensor};

// ---------------------------------------------------------------------------
// Target taxonomy
// ---------------------------------------------------------------------------

pub const ATTN_TARGETS: &[&str] = &[
    "q_proj_mot_gen",
    "k_proj_mot_gen",
    "v_proj_mot_gen",
    "o_proj_mot_gen",
];

pub const MLP_TARGETS: &[&str] = &[
    "mlp_mot_gen.gate_proj",
    "mlp_mot_gen.up_proj",
    "mlp_mot_gen.down_proj",
];

pub const FM_HEAD_TARGETS: &[&str] = &["fm_modules.fm_head.0", "fm_modules.fm_head.2"];

pub fn all_known_targets() -> Vec<&'static str> {
    let mut v: Vec<&'static str> = Vec::with_capacity(9);
    v.extend_from_slice(ATTN_TARGETS);
    v.extend_from_slice(MLP_TARGETS);
    v.extend_from_slice(FM_HEAD_TARGETS);
    v
}

/// Expand a group token (`attn`/`mlp`/`fm_head`/`all`) to its member targets.
/// Non-group tokens pass through as a single-element list.
pub fn expand_group(token: &str) -> Vec<&'static str> {
    match token {
        "attn" => ATTN_TARGETS.to_vec(),
        "mlp" => MLP_TARGETS.to_vec(),
        "fm_head" => FM_HEAD_TARGETS.to_vec(),
        "all" => all_known_targets(),
        _ => vec![],
    }
}

// ---------------------------------------------------------------------------
// LoraSpec
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct LoraSpec {
    /// One of the strings in `all_known_targets()`.
    pub target: &'static str,
    pub r: usize,
    pub alpha: f32,
    pub dropout: f32,
    pub enabled: bool,
}

impl LoraSpec {
    pub fn validate(&self) -> Result<()> {
        if !all_known_targets().contains(&self.target) {
            return Err(Error::InvalidInput(format!(
                "unknown LoRA target {:?}",
                self.target
            )));
        }
        if self.enabled && self.r == 0 {
            return Err(Error::InvalidInput(format!(
                "LoRA rank must be positive for {} (got 0)",
                self.target
            )));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Spec parsing — supports `target=rNaM` / `target=rN` / `target=off`
// ---------------------------------------------------------------------------

fn lookup_known(name: &str) -> Option<&'static str> {
    all_known_targets().into_iter().find(|t| *t == name)
}

fn parse_body(body: &str) -> Result<(usize, f32)> {
    let body = body.replace(' ', "");
    // Compact form: rNaM or rN
    if let Some(rest) = body.strip_prefix('r') {
        // Find optional 'a' separator
        if let Some(a_pos) = rest.find('a') {
            let r: usize = rest[..a_pos].parse().map_err(|e| {
                Error::InvalidInput(format!(
                    "LoRA spec body {body:?}: bad rank: {e}"
                ))
            })?;
            let alpha: f32 = rest[a_pos + 1..].parse().map_err(|e| {
                Error::InvalidInput(format!(
                    "LoRA spec body {body:?}: bad alpha: {e}"
                ))
            })?;
            return Ok((r, alpha));
        } else {
            let r: usize = rest.parse().map_err(|e| {
                Error::InvalidInput(format!(
                    "LoRA spec body {body:?}: bad rank: {e}"
                ))
            })?;
            return Ok((r, r as f32));
        }
    }
    Err(Error::InvalidInput(format!(
        "cannot parse LoRA spec body {body:?}; expected rNaM or rN"
    )))
}

pub fn parse_lora_spec_str(s: &str) -> Result<Vec<LoraSpec>> {
    let mut specs: HashMap<&'static str, LoraSpec> = HashMap::new();
    for raw in s.split(';') {
        let tok = raw.trim();
        if tok.is_empty() {
            continue;
        }
        let (target_str, body) = match tok.find('=') {
            Some(idx) => (&tok[..idx], tok[idx + 1..].trim()),
            None => (tok, ""),
        };
        let targets: Vec<&'static str> = {
            let expanded = expand_group(target_str);
            if !expanded.is_empty() {
                expanded
            } else {
                vec![lookup_known(target_str).ok_or_else(|| {
                    Error::InvalidInput(format!(
                        "unknown LoRA target {target_str:?}; valid: {:?} or groups {{attn,mlp,fm_head,all}}",
                        all_known_targets()
                    ))
                })?]
            }
        };
        for t in targets {
            match body {
                "" | "on" | "enable" => {
                    specs.insert(t, LoraSpec {
                        target: t, r: 64, alpha: 64.0, dropout: 0.0, enabled: true,
                    });
                }
                "off" | "disable" => {
                    specs.insert(t, LoraSpec {
                        target: t, r: 1, alpha: 1.0, dropout: 0.0, enabled: false,
                    });
                }
                _ => {
                    let (r, alpha) = parse_body(body)?;
                    specs.insert(t, LoraSpec {
                        target: t, r, alpha, dropout: 0.0, enabled: true,
                    });
                }
            }
        }
    }
    let out: Vec<LoraSpec> = specs.into_values().collect();
    for s in &out {
        s.validate()?;
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Presets
// ---------------------------------------------------------------------------

pub fn resolve_preset(name: &str) -> Result<Vec<LoraSpec>> {
    let spec_str = match name {
        // Default: matches the official 8-step LoRA module coverage at rank 64.
        "default" => "attn=r64a64;mlp=r64a64;fm_head=r64a64",
        "attn_only" => "attn=r64a64",
        "attn_mlp" => "attn=r64a64;mlp=r64a64",
        // Exact upstream 8-step distill LoRA shape.
        "official_r128" => "attn=r128a128;mlp=r128a128;fm_head=r128a128",
        other => {
            return Err(Error::InvalidInput(format!(
                "unknown preset {other:?}; valid: default | attn_only | attn_mlp | official_r128"
            )));
        }
    };
    parse_lora_spec_str(spec_str)
}

// ---------------------------------------------------------------------------
// U1LoraAdapter — paired down/up Parameters + scaling
// ---------------------------------------------------------------------------

/// LoRA adapter for one linear projection. F32 master Parameters; the
/// `linear_with_lora` helper casts to BF16 with autograd recording each call.
#[derive(Clone)]
pub struct U1LoraAdapter {
    pub r: usize,
    pub alpha: f32,
    pub scaling: f32,
    /// `(r, in_features)` F32, requires_grad=true. Init: kaiming-uniform.
    pub down: Parameter,
    /// `(out_features, r)` F32, requires_grad=true. Init: zeros (so adapter
    /// contributes nothing at step 0; loss is identical to base forward).
    pub up: Parameter,
}

impl U1LoraAdapter {
    /// Create a new adapter with kaiming-uniform `down` and zero `up`.
    pub fn new(
        in_features: usize,
        out_features: usize,
        r: usize,
        alpha: f32,
        seed: u64,
        device: Arc<CudaDevice>,
    ) -> Result<Self> {
        if r == 0 {
            return Err(Error::InvalidInput(
                "U1LoraAdapter::new: rank must be positive".into(),
            ));
        }

        // Kaiming-uniform fan_in = in_features, gain = sqrt(5)→bound formula
        // `sqrt(6/fan_in)` (PyTorch's default for Linear weights).
        use rand::{Rng, SeedableRng};
        let bound = (6.0_f32 / in_features as f32).sqrt();
        let mut rng_d = rand::rngs::StdRng::seed_from_u64(seed);
        let mut down_data = Vec::with_capacity(r * in_features);
        for _ in 0..(r * in_features) {
            down_data.push(rng_d.gen_range(-bound..bound));
        }
        let down_t = Tensor::from_vec(
            down_data,
            Shape::from_dims(&[r, in_features]),
            device.clone(),
        )?
        .to_dtype(DType::F32)?
        .requires_grad_(true);
        let down = Parameter::new(down_t);

        let up_data = vec![0.0_f32; out_features * r];
        let up_t = Tensor::from_vec(
            up_data,
            Shape::from_dims(&[out_features, r]),
            device.clone(),
        )?
        .to_dtype(DType::F32)?
        .requires_grad_(true);
        let up = Parameter::new(up_t);

        let scaling = alpha / r as f32;
        Ok(Self { r, alpha, scaling, down, up })
    }
}

// ---------------------------------------------------------------------------
// Key construction
// ---------------------------------------------------------------------------

/// Convert a `LoraSpec.target` + optional layer index to the full module path
/// used as the HashMap key in `lora_adapters`. Layer-scoped targets (attn,
/// mlp) require a layer index; fm_head targets ignore it.
pub fn target_to_key(target: &str, layer_idx: Option<usize>) -> Result<String> {
    if ATTN_TARGETS.contains(&target) {
        let i = layer_idx.ok_or_else(|| {
            Error::InvalidInput(format!(
                "target_to_key: attn target {target:?} requires layer_idx"
            ))
        })?;
        Ok(format!(
            "language_model.model.layers.{i}.self_attn.{target}"
        ))
    } else if MLP_TARGETS.contains(&target) {
        let i = layer_idx.ok_or_else(|| {
            Error::InvalidInput(format!(
                "target_to_key: mlp target {target:?} requires layer_idx"
            ))
        })?;
        Ok(format!("language_model.model.layers.{i}.{target}"))
    } else if FM_HEAD_TARGETS.contains(&target) {
        Ok(target.to_string())
    } else {
        Err(Error::InvalidInput(format!(
            "target_to_key: unknown target {target:?}"
        )))
    }
}

// ---------------------------------------------------------------------------
// Build adapters from a spec list (called by `load_for_training_lora`)
// ---------------------------------------------------------------------------

/// Hyperparameters needed by `build_lora_adapters` to size each adapter.
#[derive(Clone, Copy, Debug)]
pub struct LoraDims {
    pub num_layers: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub fm_head_hidden: usize, // 4096 for U1
    pub fm_head_out: usize,    // 3072 for U1
}

/// Build per-target adapters from a spec list. Returns a HashMap keyed by
/// the full module path (see `target_to_key`).
///
/// `seed_base` is offset per-target so identical-shape adapters at different
/// targets still get distinct init.
pub fn build_lora_adapters(
    specs: &[LoraSpec],
    dims: LoraDims,
    seed_base: u64,
    device: Arc<CudaDevice>,
) -> Result<HashMap<String, U1LoraAdapter>> {
    let mut out: HashMap<String, U1LoraAdapter> = HashMap::new();
    let mut seed_counter: u64 = 0;
    for spec in specs.iter() {
        if !spec.enabled {
            continue;
        }
        let (in_f, out_f, has_per_layer) = match spec.target {
            // q/k/v/o all project hidden → hidden in U1 (num_heads*head_dim =
            // 32*128 = 4096 = hidden_size; same for k/v post-merge).
            // Actually k/v project to num_kv_heads*head_dim = 8*128 = 1024.
            // Reflecting that:
            "q_proj_mot_gen" => (dims.hidden_size, dims.hidden_size, true),
            "k_proj_mot_gen" | "v_proj_mot_gen" => {
                // K/V output is num_kv_heads × head_dim. Caller must supply
                // dims that reflect this. For U1 8B-MoT: 8 × 128 = 1024.
                // We encode the K/V output dim via `hidden_size / 4` since
                // num_heads / num_kv_heads = 32/8 = 4 and head_dim is shared.
                (dims.hidden_size, dims.hidden_size / 4, true)
            }
            "o_proj_mot_gen" => (dims.hidden_size, dims.hidden_size, true),
            "mlp_mot_gen.gate_proj" | "mlp_mot_gen.up_proj" => {
                (dims.hidden_size, dims.intermediate_size, true)
            }
            "mlp_mot_gen.down_proj" => (dims.intermediate_size, dims.hidden_size, true),
            // fm_head.0: hidden(4096) → fm_head_hidden(4096)
            // fm_head.2: fm_head_hidden(4096) → fm_head_out(3072)
            "fm_modules.fm_head.0" => (dims.hidden_size, dims.fm_head_hidden, false),
            "fm_modules.fm_head.2" => (dims.fm_head_hidden, dims.fm_head_out, false),
            other => {
                return Err(Error::InvalidInput(format!(
                    "build_lora_adapters: unknown target {other:?}"
                )));
            }
        };
        let layers: Vec<Option<usize>> = if has_per_layer {
            (0..dims.num_layers).map(Some).collect()
        } else {
            vec![None]
        };
        for layer_idx in layers {
            let key = target_to_key(spec.target, layer_idx)?;
            let adapter = U1LoraAdapter::new(
                in_f, out_f, spec.r, spec.alpha,
                seed_base.wrapping_add(seed_counter),
                device.clone(),
            )?;
            seed_counter = seed_counter.wrapping_add(1);
            out.insert(key, adapter);
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Forward helper — base linear + optional LoRA delta
// ---------------------------------------------------------------------------

/// `y = base_linear(x, base_w) + scaling * x @ down^T @ up^T`. When
/// `adapter` is `None`, returns the bare base. The F32 master Parameters
/// are cast to BF16 with autograd recording so grads flow back through Cast.
///
/// For linear ops with bias, compute base separately and then call
/// `add_lora_delta` with the base result + the same `x` input.
pub fn linear_with_lora(
    x: &Tensor,
    base_w: &Tensor,
    adapter: Option<&U1LoraAdapter>,
) -> Result<Tensor> {
    let base = flame_core::ops::fused_inference::fused_linear3d_native(x, base_w, None)?;
    add_lora_delta(&base, x, adapter)
}

/// Add the LoRA delta to a precomputed `base` output. Returns `base` when
/// adapter is None. The delta is `scaling * (x @ down^T) @ up^T`.
pub fn add_lora_delta(
    base: &Tensor,
    x: &Tensor,
    adapter: Option<&U1LoraAdapter>,
) -> Result<Tensor> {
    let adapter = match adapter {
        Some(a) => a,
        None => return Ok(base.clone()),
    };
    let down_bf = adapter.down.tensor()?.to_dtype(DType::BF16)?;
    let h_low = flame_core::ops::fused_inference::fused_linear3d_native(x, &down_bf, None)?;
    let up_bf = adapter.up.tensor()?.to_dtype(DType::BF16)?;
    let delta = flame_core::ops::fused_inference::fused_linear3d_native(&h_low, &up_bf, None)?;
    let delta_scaled = delta.mul_scalar(adapter.scaling)?;
    base.add(&delta_scaled)
}

// ---------------------------------------------------------------------------
// Save / load in upstream PEFT format
// ---------------------------------------------------------------------------

/// Serialize all LoRA adapters to a single safetensors. Keys:
///   `<adapter_key>.lora_down.weight` — F32, shape `(r, in)`
///   `<adapter_key>.lora_up.weight`   — F32, shape `(out, r)`
///   `<adapter_key>.alpha`            — F32 scalar (shape `()`)
pub fn save_adapters(
    adapters: &HashMap<String, U1LoraAdapter>,
    path: &std::path::Path,
    device: &Arc<CudaDevice>,
) -> Result<()> {
    let mut tensors: HashMap<String, Tensor> = HashMap::with_capacity(adapters.len() * 3);
    for (k, a) in adapters.iter() {
        tensors.insert(format!("{k}.lora_down.weight"), a.down.tensor()?);
        tensors.insert(format!("{k}.lora_up.weight"), a.up.tensor()?);
        let alpha_t = Tensor::from_vec(
            vec![a.alpha],
            Shape::from_dims(&[]),
            device.clone(),
        )?;
        tensors.insert(format!("{k}.alpha"), alpha_t);
    }
    flame_core::serialization::save_file(&tensors, path)
        .map_err(|e| Error::Io(format!("save_adapters {:?}: {e}", path)))?;
    Ok(())
}

/// Load adapters from a previously-saved PEFT-format safetensors. Adapter
/// shapes are inferred from `<key>.lora_down.weight` (`[r, in]`) and
/// `<key>.lora_up.weight` (`[out, r]`).
pub fn load_adapters(
    path: &std::path::Path,
    device: Arc<CudaDevice>,
) -> Result<HashMap<String, U1LoraAdapter>> {
    let tensors = flame_core::serialization::load_file(path, &device)
        .map_err(|e| Error::Io(format!("load_adapters {:?}: {e}", path)))?;
    let mut out: HashMap<String, U1LoraAdapter> = HashMap::new();
    let mut keys: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for k in tensors.keys() {
        if let Some(stem) = k.strip_suffix(".lora_down.weight") {
            keys.insert(stem.to_string());
        }
    }
    for stem in keys {
        let down_t = tensors
            .get(&format!("{stem}.lora_down.weight"))
            .ok_or_else(|| {
                Error::InvalidInput(format!("load_adapters: missing {stem}.lora_down.weight"))
            })?
            .to_dtype(DType::F32)?
            .requires_grad_(true);
        let up_t = tensors
            .get(&format!("{stem}.lora_up.weight"))
            .ok_or_else(|| {
                Error::InvalidInput(format!("load_adapters: missing {stem}.lora_up.weight"))
            })?
            .to_dtype(DType::F32)?
            .requires_grad_(true);
        let alpha_t = tensors.get(&format!("{stem}.alpha")).ok_or_else(|| {
            Error::InvalidInput(format!("load_adapters: missing {stem}.alpha"))
        })?;
        let alpha: f32 = alpha_t.to_dtype(DType::F32)?.to_vec()?[0];
        let r = down_t.shape().dims()[0];
        let scaling = alpha / r as f32;
        out.insert(
            stem,
            U1LoraAdapter {
                r,
                alpha,
                scaling,
                down: Parameter::new(down_t),
                up: Parameter::new(up_t),
            },
        );
    }
    Ok(out)
}
