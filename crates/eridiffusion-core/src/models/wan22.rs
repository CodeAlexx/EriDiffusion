//! Wan 2.2 video DiT — T2V LoRA training scaffold.
//!
//! ## Status
//!
//! This module is a **structural scaffold**. The CLI surface, dual-expert
//! dispatch, dataset, sampler, and modern feature wiring are complete.
//! The training-aware forward pass — re-implementing the Wan 2.2 block
//! math in pure broadcast tensor ops so gradients flow through the LoRA
//! deltas — is **deferred**: the inference reference at
//! `reference/inference-flame-master/src/models/wan22_dit.rs` (1255 LoC)
//! drives modulation through host round-trips, which breaks autograd, and
//! the archive's training port lives in
//! `flame-diffusion-archive/wan-trainer/src/forward_impl/*` (~1200 LoC of
//! re-implemented block / RoPE / head ops). Both must be ported.
//!
//! Until the forward lands, [`Wan22Model::forward`] returns a typed
//! `EriDiffusionError::Model("Wan22 forward not yet ported")`.
//!
//! ## Variants
//! - **TI2V-5B** (single expert): `dim=3072`, `ffn_dim=14336`, `heads=24`,
//!   `layers=30`, VAE in_channels = 48, `sample_shift=5.0`. No dual-expert.
//! - **T2V-A14B** (dual expert): `dim=5120`, `ffn_dim=13824`, `heads=40`,
//!   `head_dim=128`, `layers=40`, VAE in_channels = 16, `boundary=0.875`.
//!   Two checkpoints (high_noise + low_noise); per-step dispatch by t.
//! - **I2V-A14B**: same arch as T2V-A14B but with image conditioning
//!   plumbed (out of scope for the first port — focus T2V).
//!
//! ## LoRA targets per block (matches archive `model.rs::LoraTarget`)
//! ```text
//! self_attn.{q,k,v,o}      // 4 adapters
//! cross_attn.{q,k,v,o}     // 4 adapters
//! ```
//! 8 adapters × `num_layers` blocks per expert.
//!
//! ## Weight key prefixes (verified against the .safetensors files)
//! ```text
//! patch_embedding.{weight,bias}
//! text_embedding.{0,2}.{weight,bias}
//! time_embedding.{0,2}.{weight,bias}
//! time_projection.1.{weight,bias}
//! head.head.{weight,bias}
//! head.modulation
//!
//! blocks.{i}.modulation                       [1, 6, dim]
//! blocks.{i}.self_attn.{q,k,v,o}.{weight,bias}
//! blocks.{i}.self_attn.norm_{q,k}.weight
//! blocks.{i}.cross_attn.{q,k,v,o}.{weight,bias}
//! blocks.{i}.cross_attn.norm_{q,k}.weight
//! blocks.{i}.norm3.{weight,bias}              // cross_attn pre-norm
//! blocks.{i}.ffn.{0,2}.{weight,bias}
//! ```

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use cudarc::driver::CudaDevice;
use flame_core::{parameter::Parameter, DType, Tensor};

use crate::adapter::{AdapterModule, LycorisLinear};
use crate::config::TrainConfig;
use crate::lora::LoRALinear;
use crate::lycoris::{LycorisAlgo, LycorisBundleConfig};
use crate::Result;

// ---------------------------------------------------------------------------
// Variant config
// ---------------------------------------------------------------------------

/// Wan 2.2 architecture flavor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Wan22Variant {
    /// TI2V-5B (single expert; image+video text-to-video).
    Ti2v5b,
    /// T2V-A14B (dual expert; text-to-video).
    T2v14b,
    /// I2V-A14B (dual expert; image-to-video). Out of scope for the
    /// first port — listed for completeness.
    I2v14b,
}

impl Wan22Variant {
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "ti2v_5b" | "ti2v-5b" | "5b" => Ok(Self::Ti2v5b),
            "t2v_14b" | "t2v-14b" | "14b" | "t2v" => Ok(Self::T2v14b),
            "i2v_14b" | "i2v-14b" | "i2v" => Ok(Self::I2v14b),
            other => Err(crate::EriDiffusionError::Model(format!(
                "unknown wan22 variant '{other}' (expected ti2v_5b, t2v_14b, i2v_14b)"
            ))),
        }
    }

    pub fn is_dual_expert(self) -> bool {
        matches!(self, Self::T2v14b | Self::I2v14b)
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ti2v5b => "ti2v_5b",
            Self::T2v14b => "t2v_14b",
            Self::I2v14b => "i2v_14b",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Wan22Config {
    pub variant: Wan22Variant,
    pub num_layers: usize,
    pub dim: usize,
    pub ffn_dim: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub patch_size: [usize; 3],
    pub freq_dim: usize,
    pub text_dim: usize,
    pub text_len: usize,
    pub eps: f32,
    pub rope_theta: f64,
}

impl Wan22Config {
    pub fn ti2v_5b() -> Self {
        Self {
            variant: Wan22Variant::Ti2v5b,
            num_layers: 30,
            dim: 3072,
            ffn_dim: 14336,
            num_heads: 24,
            head_dim: 128,
            in_channels: 48,
            out_channels: 48,
            patch_size: [1, 2, 2],
            freq_dim: 256,
            text_dim: 4096,
            text_len: 512,
            eps: 1e-6,
            rope_theta: 10000.0,
        }
    }

    pub fn t2v_14b() -> Self {
        Self {
            variant: Wan22Variant::T2v14b,
            num_layers: 40,
            dim: 5120,
            ffn_dim: 13824,
            num_heads: 40,
            head_dim: 128,
            in_channels: 16,
            out_channels: 16,
            patch_size: [1, 2, 2],
            freq_dim: 256,
            text_dim: 4096,
            text_len: 512,
            eps: 1e-6,
            rope_theta: 10000.0,
        }
    }

    pub fn i2v_14b() -> Self {
        Self {
            variant: Wan22Variant::I2v14b,
            ..Self::t2v_14b()
        }
    }

    pub fn for_variant(v: Wan22Variant) -> Self {
        match v {
            Wan22Variant::Ti2v5b => Self::ti2v_5b(),
            Wan22Variant::T2v14b => Self::t2v_14b(),
            Wan22Variant::I2v14b => Self::i2v_14b(),
        }
    }
}

// ---------------------------------------------------------------------------
// Per-block LoRA targets (8 attention projections per block)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LoraTarget {
    SelfQ,
    SelfK,
    SelfV,
    SelfO,
    CrossQ,
    CrossK,
    CrossV,
    CrossO,
}

impl LoraTarget {
    pub fn key(self) -> &'static str {
        match self {
            Self::SelfQ => "self_attn.q",
            Self::SelfK => "self_attn.k",
            Self::SelfV => "self_attn.v",
            Self::SelfO => "self_attn.o",
            Self::CrossQ => "cross_attn.q",
            Self::CrossK => "cross_attn.k",
            Self::CrossV => "cross_attn.v",
            Self::CrossO => "cross_attn.o",
        }
    }

    pub fn all() -> &'static [LoraTarget] {
        &[
            Self::SelfQ,
            Self::SelfK,
            Self::SelfV,
            Self::SelfO,
            Self::CrossQ,
            Self::CrossK,
            Self::CrossV,
            Self::CrossO,
        ]
    }
}

// ---------------------------------------------------------------------------
// LoRA bundle — one instance per expert
// ---------------------------------------------------------------------------

/// Flat table of LoRA adapters keyed by `(block_idx, target)`. One bundle
/// per expert; the trainer holds two of these for the 14B dual-expert
/// path.
///
/// `Clone` so the bundle can be `Arc::new(b.clone())`-wrapped and captured
/// into the `AutogradContext::checkpoint_offload` closure (required for the
/// BlockOffloader path — same pattern as `ChromaLoraBundle`).
#[derive(Clone)]
pub struct Wan22LoraBundle {
    /// Legacy plain-LoRA adapters. Empty when a LyCORIS algo is active.
    pub adapters: HashMap<(usize, LoraTarget), LoRALinear>,
    /// LyCORIS adapters (LoCon/LoHa/LoKr/Full/OFT). Empty when
    /// `algo == LycorisAlgo::None`. Wrapped in `Arc` so the bundle's `Clone`
    /// (used by `wan22_fwd::forward.rs::bundle_arc.clone()` for the
    /// BlockOffloader checkpoint path) stays cheap (refcount bump per
    /// adapter rather than re-cloning leaves).
    pub lycoris_adapters: HashMap<(usize, LoraTarget), Arc<LycorisLinear>>,
    /// Currently active algo. `LycorisAlgo::None` means the legacy
    /// `LoRALinear` path is in use (see `adapters` above).
    pub algo: LycorisAlgo,
    pub rank: usize,
    pub alpha: f32,
    pub expert_label: &'static str,
}

impl Wan22LoraBundle {
    pub fn new(
        cfg: &Wan22Config,
        rank: usize,
        alpha: f32,
        device: Arc<CudaDevice>,
        seed: u64,
        expert_label: &'static str,
    ) -> Result<Self> {
        let dim = cfg.dim;
        let mut adapters = HashMap::new();
        for block_idx in 0..cfg.num_layers {
            for (t_idx, &target) in LoraTarget::all().iter().enumerate() {
                let adapter_seed = seed
                    .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                    .wrapping_add((block_idx * 37 + t_idx) as u64);
                let lora = LoRALinear::new(dim, dim, rank, alpha, device.clone(), adapter_seed)?;
                adapters.insert((block_idx, target), lora);
            }
        }
        Ok(Self {
            adapters,
            lycoris_adapters: HashMap::new(),
            algo: LycorisAlgo::None,
            rank,
            alpha,
            expert_label,
        })
    }

    /// LyCORIS-aware constructor. `config.algo == LycorisAlgo::None` falls
    /// back to plain `LoRALinear` (legacy byte-identical path). Other algos
    /// build `LycorisLinear` per target via the matching `lycoris_rs`
    /// `*_for_training` constructor and store them in `lycoris_adapters`.
    ///
    /// `Full` and `OFT` bundle-construction succeeds, but their
    /// `forward_delta` returns an error — wan22's call pattern is
    /// `base + delta_on_input`, which is incompatible with Full's
    /// "weight delta merged into base" or OFT's `R·(W·x+b)` semantics.
    /// Phase 2c will wire merge-into-base for those algos.
    pub fn new_with_config(
        cfg: &Wan22Config,
        lyc: &LycorisBundleConfig,
        device: Arc<CudaDevice>,
        seed: u64,
        expert_label: &'static str,
    ) -> Result<Self> {
        if lyc.algo == LycorisAlgo::None {
            return Self::new(cfg, lyc.rank, lyc.alpha, device, seed, expert_label);
        }
        let dim = cfg.dim;
        let mut lycoris_adapters: HashMap<(usize, LoraTarget), Arc<LycorisLinear>> =
            HashMap::new();
        for block_idx in 0..cfg.num_layers {
            for &target in LoraTarget::all() {
                let wrapper = build_wan22_lycoris_linear(lyc, dim, dim, device.clone())?;
                lycoris_adapters.insert((block_idx, target), Arc::new(wrapper));
            }
        }
        let _ = seed; // lycoris-rs uses its own internal RNG (kaiming/normal init).
        Ok(Self {
            adapters: HashMap::new(),
            lycoris_adapters,
            algo: lyc.algo,
            rank: lyc.rank,
            alpha: lyc.alpha,
            expert_label,
        })
    }

    /// SimpleTuner-style perturbed-normal init for LoKr.
    ///
    /// Phase 2b limitation: wan22's per-block weights are streamed via
    /// `BlockOffloader` and may not be resident at bundle-construction
    /// time, so this method (mirroring `qwenimage::apply_init_perturbed_normal`)
    /// logs a warning and returns Ok(()) without touching adapters when
    /// the resident base-weight map isn't available. Phase 2c will plumb
    /// the resident `block_weights` map into this method.
    pub fn apply_init_perturbed_normal(
        &self,
        _base_weights: &HashMap<String, Tensor>,
        scale: f32,
    ) -> Result<()> {
        if self.algo != LycorisAlgo::LoKr || scale <= 0.0 {
            return Ok(());
        }
        log::warn!(
            "[wan22:{}] init_lokr_norm={scale} requested but trainer-side base-weight \
             access is not yet wired for wan22 (block weights are streamed via \
             BlockOffloader and not always resident at bundle-construction time). \
             LoKr magnitude will use its lycoris-rs default init for this run. \
             Phase 2c will plumb the resident `block_weights` map into this method.",
            self.expert_label,
        );
        Ok(())
    }

    /// Look up the active adapter for `(block_idx, target)`. Prefers the
    /// LyCORIS map when populated; falls back to the legacy plain-LoRA map.
    /// Returns `None` when neither has an entry. Mirrors
    /// `QwenImageLoraBundle::adapter_for`.
    pub fn adapter_for(
        &self,
        block_idx: usize,
        target: LoraTarget,
    ) -> Option<&dyn AdapterModule> {
        if let Some(lyc) = self.lycoris_adapters.get(&(block_idx, target)) {
            return Some(lyc.as_ref());
        }
        if let Some(legacy) = self.adapters.get(&(block_idx, target)) {
            return Some(legacy);
        }
        None
    }

    pub fn parameters(&self) -> Vec<Parameter> {
        let mut out =
            Vec::with_capacity(self.adapters.len() * 2 + self.lycoris_adapters.len() * 2);
        for lora in self.adapters.values() {
            out.extend(lora.parameters());
        }
        for adapter in self.lycoris_adapters.values() {
            out.extend(adapter.to_parameters());
        }
        out
    }

    /// `(name, Parameter)` pairs in the same iteration order as `parameters()`.
    /// Used by `flame_core::diagnostics::assert_grad_flow` to report dead-grad
    /// params by their on-disk name. Mirrors `ChromaLoraBundle::named_parameters`
    /// — for the legacy `LoRALinear` path each adapter contributes
    /// `(prefix.lora_A.weight, prefix.lora_B.weight)`; for the LyCORIS path
    /// each adapter contributes `prefix.<leaf>` for every entry returned by
    /// [`AdapterModule::named_tensors`] (zipped with `to_parameters` order).
    pub fn named_parameters(&self) -> Vec<(String, Parameter)> {
        let mut out = Vec::new();
        for ((block_idx, target), lora) in &self.adapters {
            let prefix = self.key_prefix(*block_idx, *target);
            let params = lora.parameters();
            // LoRALinear::parameters() returns [lora_a, lora_b].
            if params.len() == 2 {
                out.push((format!("{prefix}.lora_A.weight"), params[0].clone()));
                out.push((format!("{prefix}.lora_B.weight"), params[1].clone()));
            } else {
                for p in params {
                    out.push((prefix.clone(), p));
                }
            }
        }
        for ((block_idx, target), adapter) in &self.lycoris_adapters {
            let prefix = self.key_prefix(*block_idx, *target);
            let params = adapter.to_parameters();
            let names = adapter.named_tensors();
            for (param, (leaf, _)) in params.into_iter().zip(names.into_iter()) {
                out.push((format!("{prefix}.{leaf}"), param));
            }
        }
        out
    }

    pub fn num_adapters(&self) -> usize {
        self.adapters.len() + self.lycoris_adapters.len()
    }

    /// Per-adapter key prefix for the modern PEFT-compliant save format
    /// (audit H5, 2026-05-09):
    ///   `blocks.{i}.{target.key()}.{lora_A,lora_B}.weight`
    /// where `target.key()` is e.g. `self_attn.q` / `cross_attn.o` —
    /// native Wan key paths, dot-separated, no underscore mangling.
    /// Matches the Diffusers/PEFT convention chroma uses; cross-loads
    /// with SimpleTuner / Comfy / diffusers consumers that read native-Wan
    /// keys.
    pub fn key_prefix(&self, block_idx: usize, target: LoraTarget) -> String {
        format!("blocks.{block_idx}.{}", target.key())
    }

    /// Legacy archive format key prefix:
    ///   `lora_wan_blocks_{i}_{target_key_with_dots_to_underscores}.{lora_A,lora_B}.weight`
    /// Kept for backwards-compat reads of LoRAs trained before the
    /// 2026-05-09 audit-H5 fix. New saves use [`Self::key_prefix`].
    pub fn key_prefix_legacy(&self, block_idx: usize, target: LoraTarget) -> String {
        format!(
            "lora_wan_blocks_{block_idx}_{}",
            target.key().replace('.', "_")
        )
    }

    /// Load saved LoRA tensors INTO this existing bundle in place.
    /// Tries the modern PEFT format first (`blocks.{i}.{target}.lora_*`)
    /// then falls back to the legacy `lora_wan_blocks_*` format. Used by
    /// `train_wan22 --resume-low-lora/--resume-high-lora` and by
    /// `sample_wan22 --low-lora/--high-lora`. Returns `(hits, total)`.
    pub fn rehydrate(
        &self,
        tensors: &HashMap<String, Tensor>,
    ) -> Result<(usize, usize)> {
        // LyCORIS-adapter rehydrate path: per-leaf `set_data` keyed by
        // `{prefix}.{leaf}` (modern PEFT convention), no legacy fallback.
        if !self.lycoris_adapters.is_empty() {
            let mut hits = 0usize;
            for ((block_idx, target), adapter) in &self.lycoris_adapters {
                let prefix = self.key_prefix(*block_idx, *target);
                let names = adapter.named_tensors();
                let params = adapter.to_parameters();
                let mut all_loaded = true;
                for ((leaf, _), param) in names.into_iter().zip(params.into_iter()) {
                    let key = format!("{prefix}.{leaf}");
                    if let Some(t) = tensors.get(&key) {
                        param.set_data(t.clone())?;
                    } else {
                        all_loaded = false;
                    }
                }
                if all_loaded {
                    hits += 1;
                }
            }
            return Ok((hits, self.lycoris_adapters.len()));
        }
        let mut hits = 0usize;
        let mut legacy_hits = 0usize;
        for ((block_idx, target), lora) in &self.adapters {
            // Try modern PEFT keys first.
            let prefix = self.key_prefix(*block_idx, *target);
            let a_key = format!("{prefix}.lora_A.weight");
            let b_key = format!("{prefix}.lora_B.weight");
            let mut loaded = false;
            if let (Some(a), Some(b)) = (tensors.get(&a_key), tensors.get(&b_key)) {
                lora.lora_a().set_data(a.clone())?;
                lora.lora_b().set_data(b.clone())?;
                hits += 1;
                loaded = true;
            }
            if !loaded {
                // Fall back to legacy `lora_wan_blocks_*` mangled format.
                let lp = self.key_prefix_legacy(*block_idx, *target);
                let la = format!("{lp}.lora_A.weight");
                let lb = format!("{lp}.lora_B.weight");
                if let (Some(a), Some(b)) = (tensors.get(&la), tensors.get(&lb)) {
                    lora.lora_a().set_data(a.clone())?;
                    lora.lora_b().set_data(b.clone())?;
                    hits += 1;
                    legacy_hits += 1;
                }
            }
        }
        if legacy_hits > 0 {
            log::warn!(
                "[wan22] {legacy_hits} adapters loaded from LEGACY `lora_wan_blocks_*` keys. \
                 Re-save (continue training to next save_every) to migrate to the modern \
                 `blocks.{{i}}.{{target}}.lora_*.weight` PEFT format."
            );
        }
        Ok((hits, self.adapters.len()))
    }

    /// Convenience: load + rehydrate from a safetensors path.
    pub fn rehydrate_from_path(
        &self,
        path: &Path,
        device: Arc<cudarc::driver::CudaDevice>,
    ) -> Result<(usize, usize)> {
        let tensors = flame_core::serialization::load_file(path, &device)?;
        self.rehydrate(&tensors)
    }

    /// Save trained LoRA tensors to a single safetensors file using the
    /// modern PEFT-compliant key format
    /// (`blocks.{i}.{target.key()}.{lora_A,lora_B}.weight`). Cross-loads
    /// with SimpleTuner / Comfy / diffusers consumers that read native
    /// Wan key paths. 2026-05-09 audit H5.
    pub fn save(&self, path: &Path) -> Result<()> {
        let mut out: HashMap<String, Tensor> = HashMap::new();
        for ((idx, target), lora) in &self.adapters {
            let prefix = self.key_prefix(*idx, *target);
            let a = lora.lora_a().tensor()?;
            let b = lora.lora_b().tensor()?;
            out.insert(format!("{prefix}.lora_A.weight"), a);
            out.insert(format!("{prefix}.lora_B.weight"), b);
        }
        for ((idx, target), adapter) in &self.lycoris_adapters {
            let prefix = self.key_prefix(*idx, *target);
            for (leaf, t) in adapter.named_tensors() {
                out.insert(format!("{prefix}.{leaf}"), t);
            }
        }
        flame_core::serialization::save_file(&out, path).map_err(Into::into)
    }
}

/// Build a single `LycorisLinear` for the configured algo. Mirrors
/// `chroma::build_lycoris_linear` byte-for-byte; only the docstring/log
/// site differs. For `LycorisAlgo::None` the caller (`new_with_config`)
/// short-circuits to `LoRALinear` instead — this helper bails on `None`
/// defensively. Wan22 attention projections are all square `[dim, dim]`
/// linears (no Conv, no MLP gate split), so we delegate to chroma's
/// `pub(crate)` helper.
fn build_wan22_lycoris_linear(
    config: &LycorisBundleConfig,
    in_features: usize,
    out_features: usize,
    device: Arc<CudaDevice>,
) -> Result<LycorisLinear> {
    Ok(crate::models::chroma::build_lycoris_linear(
        config,
        in_features,
        out_features,
        device,
    )?)
}

// ---------------------------------------------------------------------------
// Wan22Model — single-expert wrapper
// ---------------------------------------------------------------------------

/// `BlockFacilitator` impl for the Wan 2.2 transformer block layout.
/// Block keys all start with `blocks.{i}.` — that prefix classifies
/// every per-block weight; everything else is shared.
pub struct Wan22Facilitator {
    pub num_blocks: usize,
}

impl crate::training::block_offload::BlockFacilitator for Wan22Facilitator {
    fn block_count(&self) -> usize {
        self.num_blocks
    }
    fn classify_key(&self, name: &str) -> Option<usize> {
        let rest = name.strip_prefix("blocks.")?;
        rest.split('.').next()?.parse::<usize>().ok()
    }
}

/// Single-expert Wan 2.2 transformer wrapper. The trainer instantiates
/// ONE of these for TI2V-5B, or TWO of these (high+low) for T2V/I2V-14B.
pub struct Wan22Model {
    pub config: Wan22Config,
    pub device: Arc<CudaDevice>,
    /// Resident weights. Without offload: ALL keys (shared + block).
    /// With offload: shared (non-block) keys only — block weights live
    /// in `block_offloader`.
    pub weights: HashMap<String, Tensor>,
    pub lora: Wan22LoraBundle,
    pub expert_label: &'static str,
    /// Optional BlockOffloader. When Some, per-block weights stream from
    /// pinned CPU memory via `ensure_block(i)` inside the forward path,
    /// freeing ~num_layers × block_bytes of GPU memory at the cost of
    /// host-to-device copies (overlapped with compute on the prefetch
    /// stream). Required for fitting T2V/I2V-A14B on 24 GB; optional but
    /// useful for TI2V-5B.
    pub block_offloader:
        Option<std::sync::Arc<std::sync::Mutex<crate::training::block_offload::BlockOffloader>>>,
}

impl Wan22Model {
    /// Load a single Wan 2.2 expert from a single .safetensors file.
    /// Sharded directories are not yet supported in this scaffold (the
    /// official local checkpoints listed in the task prompt are all
    /// single-file).
    ///
    /// `weight_dtype` selects the *runtime* storage dtype for the
    /// frozen base weights. flame-core's safetensors loader handles
    /// the on-disk dtype (BF16 / FP16 / FP8E4M3 with optional scale)
    /// and converts to F32 during read; we then cast to the requested
    /// runtime dtype.  FP8-resident weights are NOT supported by
    /// flame-core today (no FP8 DType variant), so passing
    /// `weight_dtype=BF16` with the `*_fp8_scaled.safetensors` files is
    /// the realistic 14B path: on-disk savings only, runtime is BF16.
    /// LoRA params are always F32 regardless.
    /// Load a Wan22 checkpoint, transparently handling HuggingFace-sharded
    /// directories (e.g. TI2V-5B's `diffusion_pytorch_model-{1..3}-of-3`).
    /// Mirrors the shard-detection logic in `ChromaTrainingModel`.
    fn load_weights_with_shards(
        path: &Path,
        device: &Arc<CudaDevice>,
    ) -> Result<HashMap<String, Tensor>> {
        if path.is_dir() {
            return Self::load_shards_from_dir(path, device);
        }
        let parent = path.parent().unwrap_or(path);
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
        let index_path = parent.join(format!("{stem}.safetensors.index.json"));
        if index_path.exists() || stem.contains("-of-") || stem.contains("-00001-") {
            return Self::load_shards_from_dir(parent, device);
        }
        Ok(flame_core::serialization::load_file(path, device)?)
    }

    fn load_shards_from_dir(
        dir: &Path,
        device: &Arc<CudaDevice>,
    ) -> Result<HashMap<String, Tensor>> {
        let read_dir = std::fs::read_dir(dir).map_err(|e| {
            crate::EriDiffusionError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("{}: {}", dir.display(), e),
            ))
        })?;
        let mut shard_paths: Vec<std::path::PathBuf> = read_dir
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("safetensors"))
            .collect();
        shard_paths.sort();
        if shard_paths.is_empty() {
            return Err(crate::EriDiffusionError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("no safetensors shards found in {}", dir.display()),
            )));
        }
        log::info!(
            "[wan22] loading {} shards from {}",
            shard_paths.len(),
            dir.display()
        );
        let mut all = HashMap::new();
        for shard in &shard_paths {
            log::info!(
                "[wan22]   shard: {}",
                shard.file_name().unwrap().to_string_lossy()
            );
            let part = flame_core::serialization::load_file(shard, device)?;
            all.extend(part);
        }
        Ok(all)
    }

    pub fn load(
        ckpt_path: &Path,
        cfg: Wan22Config,
        rank: usize,
        alpha: f32,
        weight_dtype: DType,
        device: Arc<CudaDevice>,
        seed: u64,
        expert_label: &'static str,
    ) -> Result<Self> {
        log::info!(
            "[wan22:{expert_label}] loading variant={} from {}",
            cfg.variant.as_str(),
            ckpt_path.display()
        );
        let raw = Self::load_weights_with_shards(ckpt_path, &device)?;
        let mut weights = HashMap::with_capacity(raw.len());
        for (k, v) in raw {
            let cast = if v.dtype() == weight_dtype {
                v
            } else {
                v.to_dtype(weight_dtype)?
            };
            weights.insert(k, cast);
        }
        log::info!(
            "[wan22:{expert_label}] loaded {} weight tensors as {:?}",
            weights.len(),
            weight_dtype
        );

        let lora = Wan22LoraBundle::new(&cfg, rank, alpha, device.clone(), seed, expert_label)?;
        log::info!(
            "[wan22:{expert_label}] LoRA bundle: rank={rank} alpha={alpha} adapters={}",
            lora.num_adapters()
        );
        Ok(Self {
            config: cfg,
            device,
            weights,
            lora,
            expert_label,
            block_offloader: None,
        })
    }

    /// Load with BlockOffloader: block weights live in pinned CPU memory,
    /// streamed to a 2-slot GPU ring per forward step. Shared weights
    /// (`patch_embedding.*`, `text_embedding.*`, `time_embedding.*`,
    /// `time_projection.*`, `head.*`) stay resident. Mirrors
    /// `ChromaTrainingModel::load_swapped` — same checkpoint_offload
    /// closure pattern in the trainer.
    pub fn load_swapped(
        ckpt_path: &Path,
        cfg: Wan22Config,
        rank: usize,
        alpha: f32,
        weight_dtype: DType,
        device: Arc<CudaDevice>,
        seed: u64,
        expert_label: &'static str,
    ) -> Result<Self> {
        log::info!(
            "[wan22:{expert_label}] loading variant={} from {} via BlockOffloader",
            cfg.variant.as_str(),
            ckpt_path.display()
        );

        // BlockOffloader needs &[&str] shard paths.
        let shard_path = ckpt_path.to_string_lossy().into_owned();
        let path_refs: Vec<&str> = vec![shard_path.as_str()];

        let facilitator = Wan22Facilitator { num_blocks: cfg.num_layers };
        let offloader = crate::training::block_offload::BlockOffloader::load(
            &path_refs, &facilitator, device.clone(),
        ).map_err(|e| crate::EriDiffusionError::Model(format!("BlockOffloader load: {e}")))?;

        // Load shared (non-block) weights resident, casting to weight_dtype.
        let shared_raw = flame_core::serialization::load_file_filtered(
            ckpt_path, &device, |key| !key.starts_with("blocks."),
        )?;
        let mut weights = HashMap::with_capacity(shared_raw.len());
        for (k, v) in shared_raw {
            let cast = if v.dtype() == weight_dtype { v } else { v.to_dtype(weight_dtype)? };
            weights.insert(k, cast);
        }

        log::info!(
            "[wan22:{expert_label}] offloader: {} shared weights resident, {} blocks in {:.1} MB pinned",
            weights.len(), cfg.num_layers,
            offloader.pinned_bytes() as f64 / (1024.0 * 1024.0),
        );

        let lora = Wan22LoraBundle::new(&cfg, rank, alpha, device.clone(), seed, expert_label)?;
        log::info!(
            "[wan22:{expert_label}] LoRA bundle: rank={rank} alpha={alpha} adapters={}",
            lora.num_adapters()
        );

        Ok(Self {
            config: cfg,
            device,
            weights,
            lora,
            expert_label,
            block_offloader: Some(std::sync::Arc::new(std::sync::Mutex::new(offloader))),
        })
    }

    pub fn parameters(&self) -> Vec<Parameter> {
        self.lora.parameters()
    }

    pub fn refresh_lora_cache(&self) {
        for lora in self.lora.adapters.values() {
            lora.refresh_cache();
        }
        // LyCORIS adapters don't carry a transposed-BF16 cache — `forward_delta`
        // on `LycorisLinear` reads its leaves live each call. No-op here.
    }

    pub fn save_weights(&self, path: &Path) -> Result<()> {
        self.lora.save(path)
    }

    /// Training forward.
    ///
    /// Inputs:
    /// - `x`: noised latent `[C, F, H, W]` BF16 (single sample, B=1 implicit)
    /// - `timestep`: `[1]` F32 in `0..NUM_TRAIN_TIMESTEPS`
    /// - `context`: UMT5 text embedding `[1, text_len, text_dim]` BF16
    /// - `text_mask`: optional `[1, text_len]` F32 (1=real, 0=pad). When
    ///   set, threads through to cross-attention as a padding mask
    ///   (audit H1). When None, padded positions contribute to attention.
    ///
    /// Returns the predicted velocity `[C_out, F, H, W]` BF16.
    ///
    /// Implementation: dispatches to `super::wan22_fwd::forward_with_lora`
    /// (port of the archive's `forward_impl`). The training contract is
    /// `seq_len == n_patches` — no inference-style padding.
    pub fn forward(
        &mut self,
        x: &Tensor,
        timestep: &Tensor,
        context: &Tensor,
        text_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        // Read scalar timestep off the host (Wan's modulation tables are
        // scalar-time; timestep ∈ [0, 1000]). For B=1 training the [1]
        // shape collapses to a single value.
        let t_vals = timestep.to_dtype(DType::F32)?.to_vec()?;
        if t_vals.is_empty() {
            return Err(crate::EriDiffusionError::Model(
                "Wan22Model::forward: timestep tensor is empty".into(),
            ));
        }
        let t_scalar = t_vals[0];

        // seq_len = (F/p_t) * (H/p_h) * (W/p_w) for the training contract.
        let x_dims = x.shape().dims();
        if x_dims.len() != 4 {
            return Err(crate::EriDiffusionError::Model(format!(
                "Wan22Model::forward expects x as [C, F, H, W], got {:?}",
                x_dims
            )));
        }
        let (_c, f_in, h_in, w_in) = (x_dims[0], x_dims[1], x_dims[2], x_dims[3]);
        let (pt, ph, pw) = (
            self.config.patch_size[0],
            self.config.patch_size[1],
            self.config.patch_size[2],
        );
        let seq_len = (f_in / pt) * (h_in / ph) * (w_in / pw);

        super::wan22_fwd::forward_with_lora(
            &self.config,
            &self.weights,
            &self.lora,
            x,
            t_scalar,
            context,
            seq_len,
            text_mask,
            self.block_offloader.as_ref(),
        )
        .map_err(Into::into)
    }
}
