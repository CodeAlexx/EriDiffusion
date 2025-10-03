//! # Flux Double-Stream Runtime — Final Contracts (Phase-4)
//!
//! This file defines the **authoritative interfaces and key conventions** used by the
//! Flux double-stream path (image + text) for both training and inference.
//!
//! ## 1) Token shapes (Image/Text) — **tokens in, tokens out**
//! - The **trainer** is responsible for projecting image latents `[B,4,H/8,W/8]` into tokens
//!   `[B, T_img, D]` and preparing text tokens `[B, T_txt, D]`.
//! - The **block runtime** operates purely on tokens and returns tokens:
//!   ```ignore
//!   fn forward_image(&self, img_tokens: &Tensor, cond: &Tensor) -> Result<Tensor>;
//!   fn forward_text (&self, txt_tokens: &Tensor, cond: &Tensor) -> Result<Tensor>;
//!   ```
//! - There is **no implicit [4→D] projection** inside the runtime; keep the existing trainer
//!   projection path. (The strict packs do not include a standalone `[4,D]` projection tensor.)
//!
//! ## 2) Modulation layers — **linear → (scale, shift)** (no gate unless present in dump)
//! - The packs expose `img_mod.lin.{weight,bias}` and `text_mod.lin.{weight,bias}`.
//! - Treat these as the **AdaLN linear** producing **scale and shift** vectors from the conditioning
//!   input for the block normalization, i.e.:
//!   ```text
//!   [scale | shift] = mod_lin(cond)  // split evenly on the channel (D) axis
//!   x = scale * norm(x) + shift
//!   ```
//! - **Do not assume a “gate” output**. If a new dump version includes six slices
//!   (shift/scale/gate ×2), we will **feature-gate** that path behind a version flag and keep
//!   the default as (scale, shift) until the packs + DOC are updated. (See §4 Validation.)
//!   Existing legacy dumps still expose the six-slice layout; we auto-detect this width so the
//!   runtime can remain compatible while the metadata flag is rolled out.
//!
//! ## 3) Canonical keys (per block k)
//! Image branch (LoRA applies on the six sites below):
//! - Attention (fused QKV + O):
//!   - `double_blocks.{k:02}.image.qkv.weight`, `.bias` → slice into (q, k, v)
//!   - `double_blocks.{k:02}.image.o.weight`, `.bias`
//! - Norms:
//!   - `double_blocks.{k:02}.image.q_norm.weight`, `.bias`
//!   - `double_blocks.{k:02}.image.k_norm.weight`, `.bias`
//! - MLP:
//!   - `double_blocks.{k:02}.image.fc1.weight`, `.bias`
//!   - `double_blocks.{k:02}.image.fc2.weight`, `.bias`
//! - Modulation:
//!   - `double_blocks.{k:02}.image.img_mod.lin.weight`, `.bias`  // → (scale, shift)
//!
//! Text branch (no LoRA by default; symmetric keys under `.text.*`):
//! - `double_blocks.{k:02}.text.qkv.*`, `.o.*`, `.q_norm.*`, `.k_norm.*`, `.fc1.*`, `.fc2.*`,
//!   `text_mod.lin.{weight,bias}`.
//!
//! ## 4) Loader/runtime validation (fail-fast)
//! - **Shape rules** (error if violated):
//!   - `qkv.weight`: `[3*D, D]` and `qkv.bias`: `[3*D]` → splits (q,k,v) each `[D, D]` / `[D]`
//!   - `o.weight`: `[D, D]`, `o.bias`: `[D]`
//!   - `fc1.weight`: `[M, D]`, `fc1.bias`: `[M]`; `fc2.weight`: `[D, M]`, `fc2.bias`: `[D]`
//!   - `img_mod.lin.weight`: `[2*D, D_c]`, `img_mod.lin.bias`: `[2*D]` → split into `(scale, shift)`
//!     *If a dump presents `[3*D, D_c]` / `[3*D]`, treat as `(scale, shift, gate)` **only when**
//!     `DumpVersion>=X` is detected in `__metadata__`.*
//! - **DType/device**: storage **BF16**, compute **FP32**; no cross-device ops, no silent casts.
//! - **Exact-key usage**: the registry must report **missing** and **unused** keys; either case is
//!   a hard error. The training crate should bubble these as `flame_core::Error` with the offending key.
//!
//! ## 5) LoRA application (image branch only by default)
//! - Apply LoRA at construction time on **Q/K/V/O/FC1/FC2** image weights:
//!   ```text
//!   W ← W + (α/r) · B @ A    // A:[r, in], B:[out, r]; FP32 accumulate → cast to BF16 storage
//!   ```
//! - Stable site keys for checkpoints (A/B):
//!   - `lora.blocks.{k:02}.image.{q|k|v|o|fc1|fc2}.{A|B}`
//!
//! ## 6) Trainer & inference imports (Phase-4 alignment)
//! - Keep the trainer’s import surface **tokens in/out**; only swap the internal
//!   DiT-placeholder with `FluxBlockRuntime` built from strict provider/registry.
//! - Ensure all helpers are the CUDA/BF16 versions (no CPU fallback) as tracked in the Phase-4 doc.
//!   Re-enable the Flux modules in `training/lib.rs` after the new runtime compiles and the tests pass.
//!
//! ## 7) Implementation checklist
//! - [ ] Slice fused QKV and validate shapes/dtypes/devices.
//! - [ ] Run mod_lin(cond) → split (scale, shift); apply to norm(x).
//! - [ ] Attention: split heads, SDPA path (FlashAttention when feature gated), FP32 softmax, BF16 out.
//! - [ ] MLP: SiLU(fc1(·)) → fc2(·); residual merges in FP32, store BF16.
//! - [ ] LoRA apply on image sites; text path unchanged for now.
//! - [ ] Registry: exact-key check; report first N missing/unused with shapes to aid debugging.
//!
//! ## References
//! - Phase-4 “Flux / Pipeline Restoration” notes (device policy, strict loaders, gating, tests).  // see project doc

//! Flux registry helpers referenced by the trainer.
//! Provides lightweight adapter to compose stable LoRA keys and wire strict weights.

#![allow(dead_code)]
use std::collections::{BTreeMap, BTreeSet, HashMap};

use anyhow::{anyhow, Result};
use flame_core::Tensor;
use serde::{Deserialize, Serialize};

use super::{
    super::lora_keys::{compose_lora_tensor_key, enumerate_flux_lora_keys, LoraSpec},
    weights::FluxWeightProvider,
};
use crate::flux::runtime::{ExecutableBlock, FluxBlockRuntime, LayerRegistry as RuntimeRegistry};

const CANONICAL_KEYS: &[&str] = &[
    // Image branch
    "image.qkv.weight",
    "image.qkv.bias",
    "image.o.weight",
    "image.o.bias",
    "image.q_norm.scale",
    "image.k_norm.scale",
    "image.fc1.weight",
    "image.fc1.bias",
    "image.fc2.weight",
    "image.fc2.bias",
    "image.mod.lin.weight",
    "image.mod.lin.bias",
    // Text branch
    "text.qkv.weight",
    "text.qkv.bias",
    "text.o.weight",
    "text.o.bias",
    "text.q_norm.scale",
    "text.k_norm.scale",
    "text.fc1.weight",
    "text.fc1.bias",
    "text.fc2.weight",
    "text.fc2.bias",
    "text.mod.lin.weight",
    "text.mod.lin.bias",
];

fn canonical_from_physical(s: &str) -> Option<&'static str> {
    match s {
        // Image
        "img_attn.qkv.weight" => Some("image.qkv.weight"),
        "img_attn.qkv.bias" => Some("image.qkv.bias"),
        "img_attn.proj.weight" => Some("image.o.weight"),
        "img_attn.proj.bias" => Some("image.o.bias"),
        "img_attn.norm.query_norm.scale" => Some("image.q_norm.scale"),
        "img_attn.norm.key_norm.scale" => Some("image.k_norm.scale"),
        "img_mlp.0.weight" => Some("image.fc1.weight"),
        "img_mlp.0.bias" => Some("image.fc1.bias"),
        "img_mlp.2.weight" => Some("image.fc2.weight"),
        "img_mlp.2.bias" => Some("image.fc2.bias"),
        "img_mod.lin.weight" => Some("image.mod.lin.weight"),
        "img_mod.lin.bias" => Some("image.mod.lin.bias"),
        // Text
        "txt_attn.qkv.weight" => Some("text.qkv.weight"),
        "txt_attn.qkv.bias" => Some("text.qkv.bias"),
        "txt_attn.proj.weight" => Some("text.o.weight"),
        "txt_attn.proj.bias" => Some("text.o.bias"),
        "txt_attn.norm.query_norm.scale" => Some("text.q_norm.scale"),
        "txt_attn.norm.key_norm.scale" => Some("text.k_norm.scale"),
        "txt_mlp.0.weight" => Some("text.fc1.weight"),
        "txt_mlp.0.bias" => Some("text.fc1.bias"),
        "txt_mlp.2.weight" => Some("text.fc2.weight"),
        "txt_mlp.2.bias" => Some("text.fc2.bias"),
        "txt_mod.lin.weight" => Some("text.mod.lin.weight"),
        "txt_mod.lin.bias" => Some("text.mod.lin.bias"),
        _ => None,
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FluxRegistryPlan {
    pub num_blocks: usize,
    pub with_mid: bool,
    pub lora: Option<LoraSpec>,
}

#[derive(Clone, Debug)]
pub struct FluxRegistry {
    pub lora_keys: BTreeMap<String, (String, String)>, // stable_key -> (A_key, B_key)
}

impl FluxRegistry {
    pub fn new() -> Self {
        Self { lora_keys: BTreeMap::new() }
    }

    pub fn build(plan: &FluxRegistryPlan) -> Self {
        let map = enumerate_flux_lora_keys(plan.num_blocks, plan.with_mid);
        let mut out = BTreeMap::new();
        for (_, stable) in map.into_iter() {
            let a = compose_lora_tensor_key(&stable, "A");
            let b = compose_lora_tensor_key(&stable, "B");
            out.insert(stable, (a, b));
        }
        Self { lora_keys: out }
    }

    /// Create the set of expected keys (for strict validation) for a LoRA checkpoint.
    pub fn expected_ckpt_keys(&self) -> BTreeSet<String> {
        let mut s = BTreeSet::new();
        for (_stable, (a, b)) in &self.lora_keys {
            s.insert(a.clone());
            s.insert(b.clone());
        }
        s
    }
}

pub fn build_layer_registry(provider: &FluxWeightProvider) -> Result<RuntimeRegistry> {
    let mut blocks: Vec<Box<dyn ExecutableBlock>> = Vec::with_capacity(provider.num_blocks());
    for idx in 0..provider.num_blocks() {
        let named = provider.named_block_tensors(idx)?;
        let prefix = format!("double_blocks.{}.", idx);
        let mut params: HashMap<String, Tensor> = HashMap::new();
        let mut unused: Vec<(String, Vec<usize>)> = Vec::new();

        for (full_key, tensor) in named {
            let trimmed = full_key.strip_prefix(&prefix).unwrap_or(&full_key);
            if let Some(canon) = canonical_from_physical(trimmed) {
                params.insert(canon.to_string(), tensor.clone_result()?);
            } else {
                unused.push((trimmed.to_string(), tensor.shape().dims().to_vec()));
            }
        }

        let missing: Vec<&str> =
            CANONICAL_KEYS.iter().copied().filter(|key| !params.contains_key(*key)).collect();

        if !missing.is_empty() || !unused.is_empty() {
            let mut msg = format!("flux block {} canonical key audit failed", idx);
            if !missing.is_empty() {
                msg.push_str(&format!("; missing keys: {:?}", missing));
            }
            if !unused.is_empty() {
                let preview: Vec<String> =
                    unused.iter().take(4).map(|(k, shape)| format!("{}{:?}", k, shape)).collect();
                msg.push_str(&format!("; unused tensors: {:?}", preview));
                if unused.len() > preview.len() {
                    msg.push_str(&format!(" (+{} more)", unused.len() - preview.len()));
                }
            }
            return Err(anyhow!(msg));
        }

        let runtime = FluxBlockRuntime::from_params(params)?;
        blocks.push(Box::new(runtime));
    }
    Ok(RuntimeRegistry::new(blocks))
}
