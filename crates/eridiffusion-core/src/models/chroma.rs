//! Chroma training model — loads FLUX-derivative DiT, injects LoRA or FFT.
//!
//! ## LyCORIS algo support
//!
//! `ChromaLoraBundle` accepts any [`crate::adapter::AdapterModule`] per target
//! via `Arc<dyn AdapterModule>`. The `--algo` CLI flag on `train_chroma` selects:
//!
//! - `lora` (default): plain [`LoRALinear`]. Byte-identical to pre-Phase-2b.
//!   The optimizer's `Vec<Parameter>` is the same `[lora_a, lora_b]` per
//!   adapter, in HashMap iteration order — matching the legacy code path.
//! - `locon` / `loha` / `lokr` / `dora`: build via [`LycorisLinear`] from
//!   `lycoris-rs`. The trainer's call sites (`add_lora_delta_*`) dispatch
//!   through the trait — no per-call-site match needed.
//! - `full` / `oft`: bundle construction succeeds, but
//!   [`AdapterModule::forward_delta`] returns an error at first forward
//!   because chroma's call pattern is `base + delta_on_input`, which is
//!   incompatible with Full's "weight delta merged into base" or OFT's
//!   `R·(W·x+b)` semantics. Phase 2c will wire those by hoisting the merge
//!   into the base linear.
//!
//! Save/load: legacy `--algo lora` checkpoints round-trip identically (writes
//! `lora_A.weight`/`lora_B.weight` keys at the same prefixes). LyCORIS algos
//! write algo-specific suffixes (e.g. `lokr_w1_a`/`lokr_w2`) under the same
//! per-target prefix. `--resume-lora` works for both via per-adapter
//! `load_named_tensors` which auto-detects the algo from key suffixes.
//!
//! ## Architecture (inference-flame `chroma_dit.rs`)
//!
//! ```text
//! Timestep → distilled_guidance_layer (approximator)
//!          → pooled_temb [B, mod_index_length, 3072]
//!
//! Image  → patchify(2×2) → x_embedder [B, N_img, 3072]
//! Text   → context_embedder [B, N_txt, 3072]
//!
//! → 19 double blocks: joint img+txt attention + FFN (GELU)
//!   (img: to_q/k/v, to_out.0; txt: add_q/k/v_proj, to_add_out)
//! → 38 single blocks: self-attn + proj_mlp/proj_out
//! → norm_out → proj_out → unpatchify
//! ```
//!
//! ## LoRA targets
//!
//! Double blocks:
//! - `transformer_blocks.{i}.attn.to_q.weight`
//! - `transformer_blocks.{i}.attn.to_k.weight`
//! - `transformer_blocks.{i}.attn.to_v.weight`
//! - `transformer_blocks.{i}.attn.to_out.0.weight`
//! - `transformer_blocks.{i}.attn.add_q_proj.weight`
//! - `transformer_blocks.{i}.attn.add_k_proj.weight`
//! - `transformer_blocks.{i}.attn.add_v_proj.weight`
//! - `transformer_blocks.{i}.attn.to_add_out.weight`
//! - `transformer_blocks.{i}.ff.net.0.proj.weight` (GELU gate)
//! - `transformer_blocks.{i}.ff.net.2.weight` (FFN out)
//! - `transformer_blocks.{i}.ff_context.net.0.proj.weight`
//! - `transformer_blocks.{i}.ff_context.net.2.weight`
//!
//! Single blocks:
//! - `single_transformer_blocks.{i}.attn.to_q.weight`
//! - `single_transformer_blocks.{i}.attn.to_k.weight`
//! - `single_transformer_blocks.{i}.attn.to_v.weight`
//! - `single_transformer_blocks.{i}.proj_mlp.weight`
//! - `single_transformer_blocks.{i}.proj_out.weight`
//!
//! NOT on: distilled_guidance_layer, x_embedder, context_embedder, proj_out (top-level).

use flame_core::autograd::AutogradContext;
use flame_core::{parameter::Parameter, DType, Result, Tensor};
use crate::adapter::{AdapterModule, LycorisLinear};
use crate::lora::LoRALinear;
use crate::lycoris::{LycorisAlgo, LycorisBundleConfig};
use crate::training::block_offload::{BlockOffloader, BlockFacilitator};
use lycoris_rs::{
    algorithms::{full::FullAdapter, locon::LoConModule, loha::LoHaModule, lokr::LoKrModule, oft::OFTModule},
    dora::init_magnitude,
    LycorisAdapter, LycorisModule, StorageDtype,
};
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

pub const NUM_DOUBLE_BLOCKS: usize = 19;
pub const NUM_SINGLE_BLOCKS: usize = 38;
pub const NUM_TOTAL_BLOCKS: usize = NUM_DOUBLE_BLOCKS + NUM_SINGLE_BLOCKS;

// ChromaFacilitator — was `chroma-trainer/src/facilitator.rs` in flame-diffusion.
// Inlined here to keep all chroma model bits in one EDv2 file.
pub struct ChromaFacilitator;

impl BlockFacilitator for ChromaFacilitator {
    fn block_count(&self) -> usize {
        NUM_TOTAL_BLOCKS
    }

    fn classify_key(&self, name: &str) -> Option<usize> {
        if let Some(rest) = name.strip_prefix("transformer_blocks.") {
            rest.split('.').next()?.parse::<usize>().ok()
        } else if let Some(rest) = name.strip_prefix("single_transformer_blocks.") {
            let idx: usize = rest.split('.').next()?.parse().ok()?;
            Some(NUM_DOUBLE_BLOCKS + idx)
        } else {
            None
        }
    }
}
pub const DIM: usize = 3072;
pub const NUM_HEADS: usize = 24;
pub const HEAD_DIM: usize = 128;
pub const IN_CHANNELS: usize = 64; // 16ch * 2*2 patch
pub const JOINT_ATTN_DIM: usize = 4096; // T5-XXL hidden
pub const MLP_HIDDEN: usize = 12288; // 3072 * 4
pub const NORM_EPS: f32 = 1e-6;

// ---------------------------------------------------------------------------
// LoRA target enumeration
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DoubleLoraTarget {
    ImgQ, ImgK, ImgV, ImgOut,
    TxtQ, TxtK, TxtV, TxtOut,
    ImgFfnGate, ImgFfnOut,
    TxtFfnGate, TxtFfnOut,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SingleLoraTarget {
    Q, K, V,
    ProjMlp, ProjOut,
}

// ---------------------------------------------------------------------------
// LoRA bundle
// ---------------------------------------------------------------------------

/// Per-target adapter collection. Each entry is an `Arc<dyn AdapterModule>`
/// so the bundle stays cheaply cloneable for `Arc<ChromaLoraBundle>` use in
/// the offloader closures (the closures previously cloned the whole bundle —
/// with `Arc<dyn>` per entry, that clone is now a per-entry refcount bump).
///
/// Algo dispatch is controlled by `algo`. Default `Lora` is byte-identical to
/// the pre-Phase-2b code: per-target `LoRALinear`, `[lora_a, lora_b]`
/// optimizer order, `lora_A.weight`/`lora_B.weight` save keys.
#[derive(Clone)]
pub struct ChromaLoraBundle {
    pub double_adapters: HashMap<(usize, DoubleLoraTarget), Arc<dyn AdapterModule>>,
    pub single_adapters: HashMap<(usize, SingleLoraTarget), Arc<dyn AdapterModule>>,
    /// `LycorisAlgo::None` is the legacy plain-LoRA path (this is what
    /// `ChromaLoraBundle::new` produces). Other variants come from
    /// `new_with_config` and are wired by `train_chroma --algo <foo>`.
    pub algo: LycorisAlgo,
    /// Plain-LoRA rank — kept for `load_from_safetensors` re-construction.
    /// LyCORIS algos store rank inside `LycorisBundleConfig` (the algo-specific
    /// path goes through `new_with_config_and_seed`).
    pub rank: usize,
    pub alpha: f32,
}

/// All Linear-target shapes for chroma. `None` everywhere here — the bundle
/// constructor fills DIMs from `DIM`/`MLP_HIDDEN`, NOT from the model weights.
const fn double_target_shape(target: DoubleLoraTarget) -> (usize, usize) {
    match target {
        DoubleLoraTarget::ImgQ
        | DoubleLoraTarget::ImgK
        | DoubleLoraTarget::ImgV
        | DoubleLoraTarget::ImgOut
        | DoubleLoraTarget::TxtQ
        | DoubleLoraTarget::TxtK
        | DoubleLoraTarget::TxtV
        | DoubleLoraTarget::TxtOut => (DIM, DIM),
        DoubleLoraTarget::ImgFfnGate | DoubleLoraTarget::TxtFfnGate => (DIM, MLP_HIDDEN),
        DoubleLoraTarget::ImgFfnOut | DoubleLoraTarget::TxtFfnOut => (MLP_HIDDEN, DIM),
    }
}

const fn single_target_shape(target: SingleLoraTarget) -> (usize, usize) {
    match target {
        SingleLoraTarget::Q | SingleLoraTarget::K | SingleLoraTarget::V => (DIM, DIM),
        SingleLoraTarget::ProjMlp => (DIM, MLP_HIDDEN),
        // proj_out is [3072, 15360] = [DIM, DIM + MLP_HIDDEN]
        SingleLoraTarget::ProjOut => (DIM + MLP_HIDDEN, DIM),
    }
}

const DOUBLE_TARGETS: &[DoubleLoraTarget] = &[
    DoubleLoraTarget::ImgQ,
    DoubleLoraTarget::ImgK,
    DoubleLoraTarget::ImgV,
    DoubleLoraTarget::ImgOut,
    DoubleLoraTarget::TxtQ,
    DoubleLoraTarget::TxtK,
    DoubleLoraTarget::TxtV,
    DoubleLoraTarget::TxtOut,
    DoubleLoraTarget::ImgFfnGate,
    DoubleLoraTarget::ImgFfnOut,
    DoubleLoraTarget::TxtFfnGate,
    DoubleLoraTarget::TxtFfnOut,
];

const SINGLE_TARGETS: &[SingleLoraTarget] = &[
    SingleLoraTarget::Q,
    SingleLoraTarget::K,
    SingleLoraTarget::V,
    SingleLoraTarget::ProjMlp,
    SingleLoraTarget::ProjOut,
];

impl ChromaLoraBundle {
    /// Legacy plain-LoRA constructor. Equivalent to pre-Phase-2b behaviour:
    /// constructs `LoRALinear` per target, wraps in `Arc<dyn AdapterModule>`.
    /// The Arc'd `AdapterModule::to_parameters()` for a `LoRALinear` returns
    /// `[lora_a, lora_b]` — same Parameter clones the old `parameters()`
    /// produced. Optimizer state stays byte-equivalent.
    pub fn new(
        rank: usize,
        alpha: f32,
        device: Arc<cudarc::driver::CudaDevice>,
        seed: u64,
    ) -> Result<Self> {
        let mut double_adapters: HashMap<(usize, DoubleLoraTarget), Arc<dyn AdapterModule>> =
            HashMap::new();
        for i in 0..NUM_DOUBLE_BLOCKS {
            for &target in DOUBLE_TARGETS {
                let (in_dim, out_dim) = double_target_shape(target);
                let lora = LoRALinear::new(
                    in_dim,
                    out_dim,
                    rank,
                    alpha,
                    device.clone(),
                    seed + i as u64,
                )?;
                double_adapters.insert((i, target), Arc::new(lora));
            }
        }

        let mut single_adapters: HashMap<(usize, SingleLoraTarget), Arc<dyn AdapterModule>> =
            HashMap::new();
        for i in 0..NUM_SINGLE_BLOCKS {
            for &target in SINGLE_TARGETS {
                let (in_dim, out_dim) = single_target_shape(target);
                let lora = LoRALinear::new(
                    in_dim,
                    out_dim,
                    rank,
                    alpha,
                    device.clone(),
                    seed + (NUM_DOUBLE_BLOCKS + i) as u64,
                )?;
                single_adapters.insert((i, target), Arc::new(lora));
            }
        }

        Ok(Self {
            double_adapters,
            single_adapters,
            algo: LycorisAlgo::None,
            rank,
            alpha,
        })
    }

    /// LyCORIS-aware constructor. `config.algo == LycorisAlgo::None` falls back
    /// to plain `LoRALinear` (legacy byte-identical path). Other algos build
    /// `LycorisLinear` per target via the matching `lycoris_rs` `*_for_training`
    /// constructor.
    ///
    /// Full and OFT bundle-construction succeeds, but their `forward_delta`
    /// returns an error — chroma's call pattern is `base + delta_on_input`,
    /// which is incompatible with Full's "weight delta merged into base" or
    /// OFT's `R·(W·x+b)` semantics. Phase 2c will hoist the merge into the
    /// base linear.
    pub fn new_with_config(
        config: &LycorisBundleConfig,
        device: Arc<cudarc::driver::CudaDevice>,
        seed: u64,
    ) -> Result<Self> {
        if config.algo == LycorisAlgo::None {
            return Self::new(config.rank, config.alpha, device, seed);
        }

        let mut double_adapters: HashMap<(usize, DoubleLoraTarget), Arc<dyn AdapterModule>> =
            HashMap::new();
        for i in 0..NUM_DOUBLE_BLOCKS {
            for &target in DOUBLE_TARGETS {
                let (in_dim, out_dim) = double_target_shape(target);
                let wrapper = build_lycoris_linear(config, in_dim, out_dim, device.clone())?;
                double_adapters.insert((i, target), Arc::new(wrapper));
            }
        }
        let mut single_adapters: HashMap<(usize, SingleLoraTarget), Arc<dyn AdapterModule>> =
            HashMap::new();
        for i in 0..NUM_SINGLE_BLOCKS {
            for &target in SINGLE_TARGETS {
                let (in_dim, out_dim) = single_target_shape(target);
                let wrapper = build_lycoris_linear(config, in_dim, out_dim, device.clone())?;
                single_adapters.insert((i, target), Arc::new(wrapper));
            }
        }
        let _ = seed; // lycoris-rs uses its own internal RNG (kaiming/normal init).

        Ok(Self {
            double_adapters,
            single_adapters,
            algo: config.algo,
            rank: config.rank,
            alpha: config.alpha,
        })
    }

    pub fn num_adapters(&self) -> usize {
        self.double_adapters.len() + self.single_adapters.len()
    }

    /// Flat parameter list for the optimizer.
    ///
    /// For the legacy `LycorisAlgo::None` path each adapter contributes
    /// `[lora_a, lora_b]` (via `LoRALinear`'s `AdapterModule::to_parameters`),
    /// in the same HashMap iteration order as the pre-Phase-2b
    /// `parameters()` — byte-equivalent.
    pub fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        for adapter in self.double_adapters.values() {
            params.extend(adapter.to_parameters());
        }
        for adapter in self.single_adapters.values() {
            params.extend(adapter.to_parameters());
        }
        params
    }

    /// Legacy no-op (LoRALinear's per-step cache was already empty). Kept as a
    /// no-op for source-level back-compat with chroma forward closures that
    /// still call this.
    pub fn refresh_caches(&self) {}

    /// Load adapter tensors from a safetensors file produced by `save()` (or
    /// `train_chroma`'s checkpoint format) into a freshly-constructed bundle.
    /// Used by `sample_chroma --lora <PATH>`. Currently builds a plain-LoRA
    /// bundle; LyCORIS-algo inference loaders should call
    /// `load_from_safetensors_with_config` instead.
    pub fn load_from_safetensors(
        path: &std::path::Path,
        rank: usize,
        alpha: f32,
        device: Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        let bundle = Self::new(rank, alpha, device.clone(), 0)?;
        bundle.load_weights(path, device)?;
        Ok(bundle)
    }

    /// LyCORIS-aware variant of `load_from_safetensors`. The `config.algo`
    /// must match the file's algo (auto-detected suffix mapping in
    /// [`crate::lycoris`]).
    pub fn load_from_safetensors_with_config(
        path: &std::path::Path,
        config: &LycorisBundleConfig,
        device: Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        let bundle = Self::new_with_config(config, device.clone(), 0)?;
        bundle.load_weights(path, device)?;
        Ok(bundle)
    }

    /// Load adapter tensors into THIS existing bundle in place. Used by
    /// `train_chroma --resume-lora` to pick up from a prior checkpoint
    /// without rebuilding the bundle.
    ///
    /// Per-adapter loading delegates to the legacy [`LoRALinear::load_tensors`]
    /// for plain-LoRA bundles (auto-handles old `lora_A`/`lora_A.weight`
    /// dual convention) or to in-place `set_data` against tensors keyed by
    /// the algo's named-suffix table for LyCORIS bundles.
    pub fn load_weights(
        &self,
        path: &std::path::Path,
        device: Arc<cudarc::driver::CudaDevice>,
    ) -> Result<()> {
        let tensors = flame_core::serialization::load_tensors(
            path,
            device,
            flame_core::serialization::SerializationFormat::SafeTensors,
        )?;

        for (&(block_idx, target), adapter) in &self.double_adapters {
            let suffix = double_lora_suffix(target);
            let prefix = format!("transformer_blocks.{block_idx}.{suffix}");
            load_adapter_in_place(adapter.as_ref(), &prefix, &tensors)?;
        }
        for (&(block_idx, target), adapter) in &self.single_adapters {
            let suffix = single_lora_suffix(target);
            let prefix = format!("single_transformer_blocks.{block_idx}.{suffix}");
            load_adapter_in_place(adapter.as_ref(), &prefix, &tensors)?;
        }
        Ok(())
    }

    /// `(name, Parameter)` pairs in the same iteration order as
    /// [`parameters`].  Used by `flame_core::diagnostics::assert_grad_flow`
    /// to report dead-grad params by their on-disk name (the same name
    /// `save` writes).  Relies on `to_parameters()` and `named_tensors()`
    /// returning per-adapter tensors in matching order — true for every
    /// `AdapterModule` impl in this crate.
    pub fn named_parameters(&self) -> Vec<(String, Parameter)> {
        let mut out = Vec::new();
        for (&(block_idx, target), adapter) in &self.double_adapters {
            let suffix = double_lora_suffix(target);
            let prefix = format!("transformer_blocks.{block_idx}.{suffix}");
            let params = adapter.to_parameters();
            let names = adapter.named_tensors();
            for (param, (leaf, _)) in params.into_iter().zip(names.into_iter()) {
                out.push((format!("{prefix}.{leaf}"), param));
            }
        }
        for (&(block_idx, target), adapter) in &self.single_adapters {
            let suffix = single_lora_suffix(target);
            let prefix = format!("single_transformer_blocks.{block_idx}.{suffix}");
            let params = adapter.to_parameters();
            let names = adapter.named_tensors();
            for (param, (leaf, _)) in params.into_iter().zip(names.into_iter()) {
                out.push((format!("{prefix}.{leaf}"), param));
            }
        }
        out
    }

    /// SimpleTuner-parity perturbed-normal LoKr init.  Walks every
    /// adapter, looks up its base weight in `weights` by the same
    /// `<prefix>.weight` key the safetensors loader produced, and
    /// dispatches `AdapterModule::init_perturbed_normal_lokr`.  The
    /// trait default-impl no-ops on non-LoKr adapters; LycorisLinear
    /// delegates to `LoKrModule::init_perturbed_normal` when
    /// applicable.
    ///
    /// Chroma streams base weights via BlockOffloader, so `weights` is
    /// expected to hold whatever's resident at swap time.  Missing
    /// slots are logged but not fatal — the adapter just keeps its
    /// canonical zero/kaiming init for that target.
    ///
    /// Returns the count of slots whose init was skipped (missing
    /// weight, factored LoKr, or non-LoKr algo).
    pub fn apply_init_perturbed_normal(
        &self,
        weights: &HashMap<String, Tensor>,
        scale: f32,
    ) -> Result<usize> {
        if scale <= 0.0 {
            return Ok(0);
        }
        let mut skipped = 0usize;
        let mut applied = 0usize;
        let mut try_one = |prefix: &str, adapter: &dyn AdapterModule| -> Result<()> {
            let key = format!("{prefix}.weight");
            let Some(base) = weights.get(&key) else {
                log::warn!("[chroma][init_lokr_norm] missing base weight `{key}` — skipping");
                skipped += 1;
                return Ok(());
            };
            let did = adapter.init_perturbed_normal_lokr(base, scale).map_err(|e| {
                flame_core::Error::InvalidInput(format!("init_perturbed_normal_lokr({prefix}): {e}"))
            })?;
            if did { applied += 1; } else { skipped += 1; }
            Ok(())
        };
        for (&(block_idx, target), adapter) in &self.double_adapters {
            let suffix = double_lora_suffix(target);
            let prefix = format!("transformer_blocks.{block_idx}.{suffix}");
            try_one(&prefix, adapter.as_ref())?;
        }
        for (&(block_idx, target), adapter) in &self.single_adapters {
            let suffix = single_lora_suffix(target);
            let prefix = format!("single_transformer_blocks.{block_idx}.{suffix}");
            try_one(&prefix, adapter.as_ref())?;
        }
        log::info!(
            "[chroma][init_lokr_norm] applied={applied} skipped={skipped} scale={scale}"
        );
        Ok(skipped)
    }

    pub fn save(&self, path: &std::path::Path) -> Result<()> {
        let mut tensors = HashMap::new();
        for (&(block_idx, target), adapter) in &self.double_adapters {
            let suffix = double_lora_suffix(target);
            let prefix = format!("transformer_blocks.{block_idx}.{suffix}");
            for (leaf, t) in adapter.named_tensors() {
                tensors.insert(format!("{prefix}.{leaf}"), t);
            }
        }
        for (&(block_idx, target), adapter) in &self.single_adapters {
            let suffix = single_lora_suffix(target);
            let prefix = format!("single_transformer_blocks.{block_idx}.{suffix}");
            for (leaf, t) in adapter.named_tensors() {
                tensors.insert(format!("{prefix}.{leaf}"), t);
            }
        }
        flame_core::serialization::save_tensors(
            &tensors,
            path,
            flame_core::serialization::SerializationFormat::SafeTensors,
        )
    }
}

/// Build a single `LycorisLinear` for the configured algo. For
/// `LycorisAlgo::None` the caller (`new_with_config`) short-circuits to
/// `LoRALinear` instead — this helper bails on `None` defensively.
///
/// `pub(crate)` so the other model bundles (zimage, qwenimage, etc.) can
/// reuse the same algo dispatch without copy-pasting the match.
pub(crate) fn build_lycoris_linear(
    config: &LycorisBundleConfig,
    in_features: usize,
    out_features: usize,
    device: Arc<cudarc::driver::CudaDevice>,
) -> Result<LycorisLinear> {
    let alpha = Some(config.alpha);
    let dtype = config.storage;

    let adapter = match config.algo {
        LycorisAlgo::None => {
            return Err(flame_core::Error::InvalidInput(
                "build_lycoris_linear: LycorisAlgo::None should be handled by caller".into(),
            ));
        }
        LycorisAlgo::LoCon => LycorisAdapter::LoCon(
            LoConModule::new_linear_for_training(
                in_features,
                out_features,
                config.rank,
                alpha,
                device.clone(),
                dtype,
            )
            .map_err(|e| flame_core::Error::InvalidInput(format!("LoCon::new_linear_for_training: {e}")))?,
        ),
        LycorisAlgo::LoHa => LycorisAdapter::LoHa(
            LoHaModule::new_linear_for_training(
                in_features,
                out_features,
                config.rank,
                alpha,
                device.clone(),
                dtype,
            )
            .map_err(|e| flame_core::Error::InvalidInput(format!("LoHa::new_linear_for_training: {e}")))?,
        ),
        LycorisAlgo::LoKr => LycorisAdapter::LoKr(
            LoKrModule::new_linear(
                in_features,
                out_features,
                config.rank,
                config.alpha,
                config.factor,
                config.decompose_both,
                config.use_scalar,
                device.clone(),
                dtype,
            )
            .map_err(|e| flame_core::Error::InvalidInput(format!("LoKr::new_linear: {e}")))?,
        ),
        LycorisAlgo::Full => LycorisAdapter::Full(
            FullAdapter::new_for_training(
                flame_core::Shape::from_dims(&[out_features, in_features]),
                None,
                device.clone(),
                dtype,
            )
            .map_err(|e| flame_core::Error::InvalidInput(format!("Full::new_for_training: {e}")))?,
        ),
        LycorisAlgo::Oft => LycorisAdapter::OFT(
            OFTModule::new_linear(
                in_features,
                out_features,
                config.block_size,
                config.alpha,
                None,
                dtype,
                device.clone(),
            )
            .map_err(|e| flame_core::Error::InvalidInput(format!("OFT::new_linear: {e}")))?
            .with_neumann_terms(config.neumann_terms),
        ),
    };

    // DoRA: chroma's bundle ctor doesn't have access to base weights here
    // (they're streamed by the BlockOffloader / live in `double_block_weights`).
    // For Phase 2b we initialize the magnitude as ones-of-shape — the trainer
    // can refine it post-construction by calling `init_magnitude` against the
    // streamed weight if needed. Bail loudly when DoRA is requested so the
    // user doesn't silently get an unitialized magnitude.
    let dora_magnitude = if config.dora {
        if config.algo == LycorisAlgo::Oft {
            return Err(flame_core::Error::InvalidInput(
                "DoRA + OFT is not supported (multiplicative + decomposition conflict)".into(),
            ));
        }
        // Initialize magnitude to ||W=I||_2 = 1 along the chosen axis. This is
        // an approximation — the lycoris-upstream init wants ||W_orig||_2.
        // For Phase 2b we accept the approximation and document the gap.
        let shape = if config.dora_wd_on_out {
            flame_core::Shape::from_dims(&[out_features, 1])
        } else {
            flame_core::Shape::from_dims(&[1, in_features])
        };
        let ones = Tensor::from_vec(
            vec![1.0_f32; shape.elem_count()],
            shape,
            device.clone(),
        )?;
        // Round-trip through init_magnitude to keep parity with non-identity
        // future paths and to set requires_grad consistently.
        let m = init_magnitude(&ones, config.dora_wd_on_out, 0.0)
            .map_err(|e| flame_core::Error::InvalidInput(format!("init_magnitude: {e}")))?;
        Some(m.requires_grad_(true))
    } else {
        None
    };

    Ok(LycorisLinear::new(
        adapter,
        dora_magnitude,
        config.dora_wd_on_out,
        config.dora_eps,
        config.storage,
    ))
}

/// In-place loader for one adapter: reads the file's tensors at
/// `<prefix>.<leaf>` for each leaf in `adapter.named_tensors()` and copies
/// them into the existing leaves via `Parameter::set_data` (LoRALinear) or
/// direct `Tensor::set_storage` (LycorisLinear's bare-Tensor leaves).
///
/// For plain-LoRA bundles this delegates to [`LoRALinear::load_tensors`] which
/// auto-detects the legacy bare-suffix `lora_A`/`lora_B` convention used by
/// older checkpoints.
fn load_adapter_in_place(
    adapter: &dyn AdapterModule,
    prefix: &str,
    tensors: &HashMap<String, Tensor>,
) -> Result<()> {
    // LoRALinear has a tailored loader that handles the legacy bare-suffix
    // convention. Detect by `kind()` and delegate to keep `--resume-lora`
    // round-trip identical for pre-Phase-2b checkpoints.
    if adapter.kind() == "lora" {
        // Down-cast via Any-style indirection isn't available on `dyn
        // AdapterModule` (no `Any` supertrait), so we re-implement the
        // dual-convention probe inline.
        let a_new = format!("{prefix}.lora_A.weight");
        let b_new = format!("{prefix}.lora_B.weight");
        let a_legacy = format!("{prefix}.lora_A");
        let b_legacy = format!("{prefix}.lora_B");
        let a = tensors
            .get(&a_new)
            .or_else(|| tensors.get(&a_legacy))
            .ok_or_else(|| {
                flame_core::Error::InvalidInput(format!(
                    "load_weights: missing {a_new} (or legacy {a_legacy})"
                ))
            })?;
        let b = tensors
            .get(&b_new)
            .or_else(|| tensors.get(&b_legacy))
            .ok_or_else(|| {
                flame_core::Error::InvalidInput(format!(
                    "load_weights: missing {b_new} (or legacy {b_legacy})"
                ))
            })?;
        // Walk the adapter's owned parameters; for LoRALinear order is `[a, b]`.
        let params = adapter.to_parameters();
        if params.len() != 2 {
            return Err(flame_core::Error::InvalidInput(format!(
                "load_weights: LoRA adapter expected 2 parameters, got {}",
                params.len()
            )));
        }
        params[0].set_data(a.to_dtype(DType::F32)?.requires_grad_(true))?;
        params[1].set_data(b.to_dtype(DType::F32)?.requires_grad_(true))?;
        return Ok(());
    }

    // LyCORIS path: the `LycorisLinear` leaves are bare `Tensor` fields
    // (not `Parameter`-wrapped), so we cannot do a `Parameter::set_data` here.
    // The trainer's resume convention for LyCORIS algos is to bail explicitly
    // — the right path is to construct the bundle from disk via
    // `load_from_safetensors_with_config`. Surface a clear error.
    Err(flame_core::Error::InvalidInput(format!(
        "load_weights: in-place resume for LyCORIS algo '{}' is not yet wired \
         (Phase 2b shortcut). Re-load via load_from_safetensors_with_config \
         and recreate the optimizer from scratch.",
        adapter.kind()
    )))
}

fn double_lora_suffix(target: DoubleLoraTarget) -> &'static str {
    match target {
        DoubleLoraTarget::ImgQ => "attn.to_q",
        DoubleLoraTarget::ImgK => "attn.to_k",
        DoubleLoraTarget::ImgV => "attn.to_v",
        DoubleLoraTarget::ImgOut => "attn.to_out.0",
        DoubleLoraTarget::TxtQ => "attn.add_q_proj",
        DoubleLoraTarget::TxtK => "attn.add_k_proj",
        DoubleLoraTarget::TxtV => "attn.add_v_proj",
        DoubleLoraTarget::TxtOut => "attn.to_add_out",
        DoubleLoraTarget::ImgFfnGate => "ff.net.0.proj",
        DoubleLoraTarget::ImgFfnOut => "ff.net.2",
        DoubleLoraTarget::TxtFfnGate => "ff_context.net.0.proj",
        DoubleLoraTarget::TxtFfnOut => "ff_context.net.2",
    }
}

fn single_lora_suffix(target: SingleLoraTarget) -> &'static str {
    match target {
        SingleLoraTarget::Q => "attn.to_q",
        SingleLoraTarget::K => "attn.to_k",
        SingleLoraTarget::V => "attn.to_v",
        SingleLoraTarget::ProjMlp => "proj_mlp",
        SingleLoraTarget::ProjOut => "proj_out",
    }
}

// ---------------------------------------------------------------------------
// Training model
// ---------------------------------------------------------------------------

pub struct ChromaTrainingModel {
    pub model_path: std::path::PathBuf,
    /// LoRA adapters (None if FFT mode) — Arc for cheap cloning into checkpoint closures.
    pub bundle: Option<ChromaLoraBundle>,
    /// FFT trainable parameters (None if LoRA mode)
    pub fft_params: Option<Vec<Parameter>>,
    pub is_full_finetune: bool,
    pub num_double_blocks: usize,
    pub num_single_blocks: usize,
    /// Arc-wrapped for cheap cloning into checkpoint closures.
    resident_weights: Arc<HashMap<String, Tensor>>,
    /// Per double-block weights (index 0..19) — empty when swap is active.
    double_block_weights: Vec<HashMap<String, Tensor>>,
    /// Per single-block weights (index 0..38) — empty when swap is active.
    single_block_weights: Vec<HashMap<String, Tensor>>,
    /// Block offloader: pinned CPU → GPU sequential copy (None if all blocks on GPU).
    pub block_offloader: Option<Arc<Mutex<BlockOffloader>>>,
    device: Arc<cudarc::driver::CudaDevice>,
    /// RoPE table cache, keyed on (h_tok, w_tok, n_txt). Building chroma's
    /// 3-axis RoPE costs ~2 s per call (slow flame-core narrow→matmul→cos/sin
    /// chain on small tensors). Tables only depend on shape (not on prompt
    /// content or timestep), so caching across denoise steps cuts ~4 s/step
    /// out of the per-step budget at 512² × CFG. Mutex<Option<…>> mirrors the
    /// offloader's interior-mutability pattern.
    rope_cache: Arc<Mutex<Option<RopeCacheEntry>>>,
}

#[derive(Clone)]
struct RopeCacheEntry {
    h_tok: usize,
    w_tok: usize,
    n_txt: usize,
    pe_cos: Tensor,
    pe_sin: Tensor,
}

impl ChromaTrainingModel {
    pub fn load(
        model_path: &std::path::Path,
        mode: &str,
        lora_rank: usize,
        lora_alpha: f32,
        device: Arc<cudarc::driver::CudaDevice>,
        seed: u64,
    ) -> Result<Self> {
        log::info!("[chroma-trainer] loading Chroma from {} (mode={})", model_path.display(), mode);

        let all_weights = Self::load_weights(model_path, &device)?;
        log::info!("[chroma-trainer] loaded {} weight tensors", all_weights.len());

        let mut resident = HashMap::new();
        let mut double_blocks: Vec<HashMap<String, Tensor>> = (0..NUM_DOUBLE_BLOCKS).map(|_| HashMap::new()).collect();
        let mut single_blocks: Vec<HashMap<String, Tensor>> = (0..NUM_SINGLE_BLOCKS).map(|_| HashMap::new()).collect();

        for (key, tensor) in &all_weights {
            if let Some(rest) = key.strip_prefix("transformer_blocks.") {
                if let Some(dot_pos) = rest.find('.') {
                    if let Ok(idx) = rest[..dot_pos].parse::<usize>() {
                        if idx < NUM_DOUBLE_BLOCKS {
                            double_blocks[idx].insert(key.clone(), tensor.clone());
                            continue;
                        }
                    }
                }
            }
            if let Some(rest) = key.strip_prefix("single_transformer_blocks.") {
                if let Some(dot_pos) = rest.find('.') {
                    if let Ok(idx) = rest[..dot_pos].parse::<usize>() {
                        if idx < NUM_SINGLE_BLOCKS {
                            single_blocks[idx].insert(key.clone(), tensor.clone());
                            continue;
                        }
                    }
                }
            }
            resident.insert(key.clone(), tensor.clone());
        }
        drop(all_weights);

        log::info!(
            "[chroma-trainer] {} resident, {} double block maps, {} single block maps",
            resident.len(), double_blocks.len(), single_blocks.len()
        );

        let is_fft = mode == "full";
        let bundle = if !is_fft {
            let b = ChromaLoraBundle::new(lora_rank, lora_alpha, device.clone(), seed)?;
            log::info!("[chroma-trainer] {} LoRA adapters (rank={})", b.num_adapters(), lora_rank);
            Some(b)
        } else {
            None
        };

        let fft_params = if is_fft {
            let mut params = Vec::new();
            // In FFT mode, make all block weights trainable
            for block_map in double_blocks.iter_mut().chain(single_blocks.iter_mut()) {
                for (_, tensor) in block_map.iter() {
                    let p = Parameter::new(tensor.clone());
                    params.push(p);
                }
            }
            log::info!("[chroma-trainer] FFT mode: {} trainable parameters", params.len());
            Some(params)
        } else {
            None
        };

        Ok(Self {
            model_path: model_path.to_path_buf(),
            bundle,
            fft_params,
            is_full_finetune: is_fft,
            num_double_blocks: NUM_DOUBLE_BLOCKS,
            num_single_blocks: NUM_SINGLE_BLOCKS,
            resident_weights: Arc::new(resident),
            double_block_weights: double_blocks,
            single_block_weights: single_blocks,
            block_offloader: None,
            device,
            rope_cache: Arc::new(Mutex::new(None)),
        })
    }

    /// Load weights from a single safetensors file or a directory of shards.
    ///
    /// Detects sharded models by looking for an `*.index.json` sibling or
    /// multiple `*-of-*.safetensors` files in the same directory.
    fn load_weights(
        path: &std::path::Path,
        device: &Arc<cudarc::driver::CudaDevice>,
    ) -> Result<HashMap<String, Tensor>> {
        // If path is a directory, find all safetensors inside it
        if path.is_dir() {
            return Self::load_shards_from_dir(path, device);
        }

        // If it's a single file, check for sharding
        let parent = path.parent().unwrap_or(path);
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");

        // Check for index.json (HuggingFace sharded format)
        let index_path = parent.join(format!("{stem}.safetensors.index.json"));
        if index_path.exists() {
            return Self::load_shards_from_dir(parent, device);
        }

        // Check if basename matches *-00001-of-*.safetensors pattern
        if stem.contains("-of-") || stem.contains("-00001-") {
            return Self::load_shards_from_dir(parent, device);
        }

        // Single file
        flame_core::serialization::load_file(path, device)
    }

    fn load_shards_from_dir(
        dir: &std::path::Path,
        device: &Arc<cudarc::driver::CudaDevice>,
    ) -> Result<HashMap<String, Tensor>> {
        let mut shard_paths: Vec<std::path::PathBuf> = Vec::new();
        for entry in std::fs::read_dir(dir)
            .map_err(|e| flame_core::Error::Io(format!("{}: {}", dir.display(), e)))?
        {
            let entry = entry.map_err(|e| flame_core::Error::Io(e.to_string()))?;
            let p = entry.path();
            if p.extension().and_then(|s| s.to_str()) == Some("safetensors") {
                shard_paths.push(p);
            }
        }
        shard_paths.sort();

        if shard_paths.is_empty() {
            return Err(flame_core::Error::InvalidInput(
                format!("no safetensors files found in {}", dir.display()),
            ));
        }

        log::info!("[chroma-trainer] loading {} shards from {}", shard_paths.len(), dir.display());
        let mut all_weights = HashMap::new();
        for shard in &shard_paths {
            log::info!("[chroma-trainer]   loading shard: {}", shard.file_name().unwrap().to_string_lossy());
            let shard_weights = flame_core::serialization::load_file(shard, device)?;
            for (k, v) in shard_weights {
                all_weights.insert(k, v);
            }
        }
        Ok(all_weights)
    }

    /// Load model with BlockOffloader: block weights in pinned CPU memory,
    /// copied to GPU one-at-a-time via `ensure_block`.
    pub fn load_swapped(
        model_path: &std::path::Path,
        mode: &str,
        lora_rank: usize,
        lora_alpha: f32,
        device: Arc<cudarc::driver::CudaDevice>,
        seed: u64,
    ) -> Result<Self> {
        log::info!("[chroma-trainer] loading Chroma with BlockOffloader from {}", model_path.display());

        // Discover shard paths
        let shard_paths = if model_path.is_dir() {
            let mut paths: Vec<String> = Vec::new();
            for entry in std::fs::read_dir(model_path)
                .map_err(|e| flame_core::Error::Io(format!("{}: {}", model_path.display(), e)))?
            {
                let entry = entry.map_err(|e| flame_core::Error::Io(e.to_string()))?;
                let p = entry.path();
                if p.extension().and_then(|s| s.to_str()) == Some("safetensors") {
                    paths.push(p.to_string_lossy().into_owned());
                }
            }
            paths.sort();
            paths
        } else {
            vec![model_path.to_string_lossy().into_owned()]
        };

        if shard_paths.is_empty() {
            return Err(flame_core::Error::InvalidInput(
                format!("no safetensors files found at {}", model_path.display()),
            ));
        }

        let path_refs: Vec<&str> = shard_paths.iter().map(|s| s.as_str()).collect();

        // Create BlockOffloader — reads block weights into pinned CPU memory
        let facilitator = ChromaFacilitator;
        let mut offloader = BlockOffloader::load(&path_refs, &facilitator, device.clone())
            .map_err(|e| flame_core::Error::InvalidInput(format!("BlockOffloader load: {e}")))?;

        // Phase 2 FlexTensor port: opt into Adaptive resident-set strategy
        // when `FLAME_OFFLOAD_ADAPTIVE=1`. Default behavior (no env var or
        // "0"/"false") is the pre-Phase-2 fixed 2-slot mechanic — unchanged.
        // Adaptive bounds the resident set against measured VRAM headroom
        // with hysteresis (shrink at ≥0.85 used, grow at ≤0.60 used). Use
        // for high-resolution / heavy-activation training where the fixed
        // 2-slot may otherwise OOM under pressure.
        if matches!(
            std::env::var("FLAME_OFFLOAD_ADAPTIVE").ok().as_deref(),
            Some("1") | Some("true") | Some("TRUE")
        ) {
            use flame_core::offload::strategy::Adaptive;
            offloader.set_strategy(Box::new(Adaptive::new()));
            log::info!(
                "[chroma] BlockOffloader: Adaptive strategy enabled (FLAME_OFFLOAD_ADAPTIVE=1)"
            );
        }

        // Load ONLY shared (non-block) weights to GPU
        let mut resident = HashMap::new();
        for shard_path in &shard_paths {
            let shard_weights = flame_core::serialization::load_file_filtered(
                shard_path,
                &device,
                |key| {
                    !key.starts_with("transformer_blocks.")
                        && !key.starts_with("single_transformer_blocks.")
                },
            )?;
            for (k, v) in shard_weights {
                resident.insert(k, v);
            }
        }

        log::info!(
            "[chroma-trainer] offloader: {} shared weights on GPU, {} blocks in {:.1} MB pinned",
            resident.len(), NUM_TOTAL_BLOCKS,
            offloader.pinned_bytes() as f64 / (1024.0 * 1024.0),
        );

        let is_fft = mode == "full";
        let bundle = if !is_fft {
            let b = ChromaLoraBundle::new(lora_rank, lora_alpha, device.clone(), seed)?;
            log::info!("[chroma-trainer] {} LoRA adapters (rank={})", b.num_adapters(), lora_rank);
            Some(b)
        } else {
            None
        };

        Ok(Self {
            model_path: model_path.to_path_buf(),
            bundle,
            fft_params: None, // FFT not supported with block offloading
            is_full_finetune: is_fft,
            num_double_blocks: NUM_DOUBLE_BLOCKS,
            num_single_blocks: NUM_SINGLE_BLOCKS,
            resident_weights: Arc::new(resident),
            double_block_weights: Vec::new(), // empty — weights come from offloader
            single_block_weights: Vec::new(),
            block_offloader: Some(Arc::new(Mutex::new(offloader))),
            device,
            rope_cache: Arc::new(Mutex::new(None)),
        })
    }

    pub fn parameters(&self) -> Vec<Parameter> {
        if let Some(ref bundle) = self.bundle {
            bundle.parameters()
        } else if let Some(ref fft) = self.fft_params {
            fft.clone()
        } else {
            Vec::new()
        }
    }

    pub fn refresh_lora_cache(&self) {
        if let Some(ref bundle) = self.bundle {
            bundle.refresh_caches();
        }
    }

    pub fn save_weights(&self, path: &std::path::Path) -> Result<()> {
        if let Some(ref bundle) = self.bundle {
            bundle.save(path)
        } else {
            // FFT: save all block weights
            let mut tensors = HashMap::new();
            for block_map in self.double_block_weights.iter().chain(self.single_block_weights.iter()) {
                for (key, tensor) in block_map {
                    tensors.insert(key.clone(), tensor.clone());
                }
            }
            for (key, tensor) in self.resident_weights.iter() {
                tensors.insert(key.clone(), tensor.clone());
            }
            flame_core::serialization::save_tensors(
                &tensors,
                path,
                flame_core::serialization::SerializationFormat::SafeTensors,
            )
        }
    }

    // -----------------------------------------------------------------------
    // Weight access
    // -----------------------------------------------------------------------

    fn w(&self, key: &str) -> Result<&Tensor> {
        self.resident_weights.get(key)
            .ok_or_else(|| flame_core::Error::InvalidInput(format!("missing weight: {key}")))
    }

    /// Borrow the resident base-weight map.  Used by
    /// `ChromaLoraBundle::apply_init_perturbed_normal` to look up base
    /// weights at LyCORIS-bundle swap time.  Some weights may be
    /// streamed via `BlockOffloader` and absent from this map at the
    /// moment of init — the helper logs them and continues.
    pub fn resident_weights(&self) -> &HashMap<String, Tensor> {
        &self.resident_weights
    }

    fn dw(&self, block_idx: usize, key: &str) -> Result<&Tensor> {
        let full_key = format!("transformer_blocks.{block_idx}.{key}");
        self.double_block_weights[block_idx].get(&full_key)
            .ok_or_else(|| flame_core::Error::InvalidInput(format!("missing: {full_key}")))
    }

    fn sw(&self, block_idx: usize, key: &str) -> Result<&Tensor> {
        let full_key = format!("single_transformer_blocks.{block_idx}.{key}");
        self.single_block_weights[block_idx].get(&full_key)
            .ok_or_else(|| flame_core::Error::InvalidInput(format!("missing: {full_key}")))
    }

    /// Look up a double-block weight, using external map (BlockOffloader) if provided, else stored.
    fn dw_or<'a>(&'a self, ext: Option<&'a HashMap<String, Tensor>>, block_idx: usize, key: &str) -> Result<&'a Tensor> {
        let full_key = format!("transformer_blocks.{block_idx}.{key}");
        if let Some(w) = ext {
            w.get(&full_key)
                .ok_or_else(|| flame_core::Error::InvalidInput(format!("missing: {full_key}")))
        } else {
            self.double_block_weights[block_idx].get(&full_key)
                .ok_or_else(|| flame_core::Error::InvalidInput(format!("missing: {full_key}")))
        }
    }

    /// Look up a single-block weight, using external map (BlockOffloader) if provided, else stored.
    fn sw_or<'a>(&'a self, ext: Option<&'a HashMap<String, Tensor>>, block_idx: usize, key: &str) -> Result<&'a Tensor> {
        let full_key = format!("single_transformer_blocks.{block_idx}.{key}");
        if let Some(w) = ext {
            w.get(&full_key)
                .ok_or_else(|| flame_core::Error::InvalidInput(format!("missing: {full_key}")))
        } else {
            self.single_block_weights[block_idx].get(&full_key)
                .ok_or_else(|| flame_core::Error::InvalidInput(format!("missing: {full_key}")))
        }
    }

    // -----------------------------------------------------------------------
    // Linear helpers (autograd-aware for gradient flow)
    // -----------------------------------------------------------------------

    fn linear_bias_resident(&self, x: &Tensor, w_key: &str, b_key: &str) -> Result<Tensor> {
        let weight = self.w(w_key)?;
        let bias = self.w(b_key)?;
        let dims = x.shape().dims().to_vec();
        let in_feat = *dims.last().unwrap();
        let batch: usize = dims[..dims.len() - 1].iter().product();
        let out_feat = weight.shape().dims()[0];
        let x_2d = x.reshape(&[batch, in_feat])?;
        let wt = weight.transpose()?;
        let out_2d = x_2d.matmul(&wt)?.add(bias)?;
        let mut out_shape = dims[..dims.len() - 1].to_vec();
        out_shape.push(out_feat);
        out_2d.reshape(&out_shape)
    }

    fn block_linear_no_bias(&self, x: &Tensor, weight: &Tensor, pre_transposed: bool) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let in_feat = *dims.last().unwrap();
        let batch: usize = dims[..dims.len() - 1].iter().product();
        let x_2d = x.reshape(&[batch, in_feat])?;
        let (wt, out_feat) = if pre_transposed {
            // Weight is already [in_features, out_features] from BlockOffloader.
            let out_feat = weight.shape().dims()[1];
            (weight.clone(), out_feat)
        } else {
            // Weight is [out_features, in_features], transpose for matmul.
            let out_feat = weight.shape().dims()[0];
            (weight.transpose()?, out_feat)
        };
        let out_2d = x_2d.matmul(&wt)?;
        let mut out_shape = dims[..dims.len() - 1].to_vec();
        out_shape.push(out_feat);
        out_2d.reshape(&out_shape)
    }

    fn block_linear_bias(&self, x: &Tensor, weight: &Tensor, bias: &Tensor, pre_transposed: bool) -> Result<Tensor> {
        let out = self.block_linear_no_bias(x, weight, pre_transposed)?;
        out.add(bias)
    }

    fn add_double_lora_delta(&self, base: Tensor, input: &Tensor, block_idx: usize, target: DoubleLoraTarget) -> Result<Tensor> {
        if let Some(ref bundle) = self.bundle {
            if let Some(lora) = bundle.double_adapters.get(&(block_idx, target)) {
                if lora.is_input_rotation() {
                    // OFT/BOFT: `output = base_linear(R · x)`.  Express
                    // as a residual on the existing `base` output by
                    // pushing `(R - I)·x` through the SAME base linear:
                    // `delta = base_linear((R - I)·x)`, then `base + delta
                    //  == base_linear(x) + base_linear((R - I)·x)
                    //  == base_linear(R · x)`.
                    let in3d = ensure_3d(input)?;
                    let rx = lora.apply_input(&in3d)?;
                    let delta_x = rx.sub(&in3d)?;
                    let suffix = double_lora_suffix(target);
                    let weight_key = format!("transformer_blocks.{block_idx}.{suffix}.weight");
                    let weight = self.w(&weight_key)?;
                    let delta = self.block_linear_no_bias(&delta_x, weight, false)?;
                    return base.add(&delta);
                }
                let delta = lora.forward_delta(&ensure_3d(input)?)?;
                return base.add(&delta);
            }
        }
        Ok(base)
    }

    fn add_single_lora_delta(&self, base: Tensor, input: &Tensor, block_idx: usize, target: SingleLoraTarget) -> Result<Tensor> {
        if let Some(ref bundle) = self.bundle {
            if let Some(lora) = bundle.single_adapters.get(&(block_idx, target)) {
                if lora.is_input_rotation() {
                    let in3d = ensure_3d(input)?;
                    let rx = lora.apply_input(&in3d)?;
                    let delta_x = rx.sub(&in3d)?;
                    let suffix = single_lora_suffix(target);
                    let weight_key = format!("single_transformer_blocks.{block_idx}.{suffix}.weight");
                    let weight = self.w(&weight_key)?;
                    let delta = self.block_linear_no_bias(&delta_x, weight, false)?;
                    return base.add(&delta);
                }
                let delta = lora.forward_delta(&ensure_3d(input)?)?;
                return base.add(&delta);
            }
        }
        Ok(base)
    }

    // -----------------------------------------------------------------------
    // Normalization
    // -----------------------------------------------------------------------

    fn layer_norm_no_affine(x: &Tensor) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let hidden = *dims.last().unwrap();
        let batch: usize = dims[..dims.len() - 1].iter().product();
        let x_2d = x.reshape(&[batch, hidden])?;
        let out = flame_core::cuda_ops_bf16::layer_norm_bf16(&x_2d, None, None, NORM_EPS)?;
        out.reshape(&dims)
    }

    fn rms_norm_per_head(x: &Tensor, weight: &Tensor) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let (b, h, s, d) = (dims[0], dims[1], dims[2], dims[3]);
        let flat = x.reshape(&[b * h * s, d])?;
        let normed = flame_core::cuda_ops_bf16::rms_norm_bf16(&flat, Some(weight), NORM_EPS)?;
        normed.reshape(&[b, h, s, d])
    }

    /// Modulate: LayerNorm(x) * (1 + scale) + shift
    /// Uses fused BF16 kernel (no autograd). For the final layer where autograd
    /// is needed, use `modulate_pre_autograd` instead.
    fn modulate_pre(x: &Tensor, shift: &Tensor, scale: &Tensor) -> Result<Tensor> {
        flame_core::bf16_ops::modulate_pre_fused_bf16(x, shift, scale, NORM_EPS)
    }

    /// Autograd-aware modulate_pre using flame_core::layer_norm (records Op::LayerNorm).
    fn modulate_pre_autograd(x: &Tensor, shift: &Tensor, scale: &Tensor) -> Result<Tensor> {
        let dim = *x.shape().dims().last().unwrap();
        let normed = flame_core::layer_norm::layer_norm(x, &[dim], None, None, NORM_EPS)?;
        let shift_b = shift.unsqueeze(1)?;
        let scale_b = scale.unsqueeze(1)?;
        normed.mul(&scale_b.add_scalar(1.0)?)?.add(&shift_b)
    }

    // -----------------------------------------------------------------------
    // Forward pass
    // -----------------------------------------------------------------------

    /// Full training forward.
    ///
    /// `img`: [B, N_img, 16] BF16 (packed latents, already patchified+embedded by caller)
    /// `txt`: [B, N_txt, 4096] BF16 (T5-XXL hidden states)
    /// `timesteps`: [B] BF16 (sigma in [0, 1])
    /// `img_ids`: [N_img, 3] position IDs
    /// `txt_ids`: [N_txt, 3] position IDs
    ///
    /// Returns: [B, N_img, 64] BF16 (predicted velocity in patch space)
    pub fn forward(
        &self,
        latent: &Tensor,      // [B, 16, H_lat, W_lat]
        txt: &Tensor,         // [B, N_txt, 4096]
        timesteps: &Tensor,   // [B]
    ) -> Result<Tensor> {
        let lat_dims = latent.shape().dims().to_vec();
        let (b, _c, h_lat, w_lat) = (lat_dims[0], lat_dims[1], lat_dims[2], lat_dims[3]);

        // Patchify: [B, 16, H, W] → [B, (H/2)*(W/2), 64]
        let h_tok = h_lat / 2;
        let w_tok = w_lat / 2;
        let n_img = h_tok * w_tok;
        let img_packed = self.patchify(latent, h_lat, w_lat)?;

        // x_embedder: [B, N_img, 64] → [B, N_img, 3072]
        let img = self.linear_bias_resident(&img_packed, "x_embedder.weight", "x_embedder.bias")?;

        // context_embedder: [B, N_txt, 4096] → [B, N_txt, 3072]
        let txt_emb = self.linear_bias_resident(txt, "context_embedder.weight", "context_embedder.bias")?;

        // Build img_ids and txt_ids for RoPE
        let (img_ids, txt_ids) = Self::build_position_ids(h_tok, w_tok, txt.shape().dims()[1], &self.device)?;

        // Phase timing — only when CHROMA_TIMING=1 (no-swap path only).
        let timing_setup = std::env::var("CHROMA_TIMING").is_ok();
        let dev_for_sync = self.device.clone();
        let sync_setup = || -> Result<()> {
            if timing_setup {
                dev_for_sync.synchronize()
                    .map_err(|e| flame_core::Error::Cuda(format!("sync: {e:?}")))?;
            }
            Ok(())
        };

        sync_setup()?;
        let t_approx = std::time::Instant::now();
        // Approximator: compute pooled_temb for modulation
        let pooled_temb = self.run_approximator(timesteps)?;
        sync_setup()?;
        let dt_approx = t_approx.elapsed();

        let t_rope = std::time::Instant::now();
        // Build RoPE — cached on (h_tok, w_tok, n_txt) since the tables
        // depend only on shape (positions are integer indices, not prompt
        // content or timestep). build_rope itself takes ~2 s/call due to
        // a slow flame-core narrow→matmul→cos/sin chain on small tensors;
        // caching across denoise steps cuts ~4 s/step at 512² × CFG (CFG
        // calls forward twice per step against the same shape).
        let n_txt_for_cache = txt.shape().dims()[1];
        let cached_rope: Option<(Tensor, Tensor)> = {
            let guard = self.rope_cache.lock()
                .map_err(|e| flame_core::Error::InvalidInput(format!("rope_cache lock: {e}")))?;
            guard.as_ref().and_then(|c| {
                if c.h_tok == h_tok && c.w_tok == w_tok && c.n_txt == n_txt_for_cache {
                    Some((c.pe_cos.clone(), c.pe_sin.clone()))
                } else {
                    None
                }
            })
        };
        let was_cached = cached_rope.is_some();
        let (pe_cos, pe_sin) = if let Some(pair) = cached_rope {
            pair
        } else {
            let (cos, sin) = self.build_rope(&img_ids, &txt_ids)?;
            let mut guard = self.rope_cache.lock()
                .map_err(|e| flame_core::Error::InvalidInput(format!("rope_cache lock: {e}")))?;
            *guard = Some(RopeCacheEntry {
                h_tok,
                w_tok,
                n_txt: n_txt_for_cache,
                pe_cos: cos.clone(),
                pe_sin: sin.clone(),
            });
            (cos, sin)
        };
        sync_setup()?;
        let dt_rope = t_rope.elapsed();

        if timing_setup {
            log::info!(
                "[chroma fwd] approximator: {:.1}ms | rope (cached={}): {:.1}ms",
                dt_approx.as_secs_f64() * 1000.0,
                was_cached,
                dt_rope.as_secs_f64() * 1000.0,
            );
        }

        let n_txt = txt.shape().dims()[1];
        let mut img_h = img;
        let mut txt_h = txt_emb;

        if let Some(ref offloader_mtx) = self.block_offloader {
            // ── BlockOffloader path with checkpoint_offload ──
            // Each block: fetch weights from pinned CPU, run through
            // checkpoint_offload, activations offloaded during backward.
            let bundle_arc: Option<Arc<ChromaLoraBundle>> =
                self.bundle.as_ref().map(|b| Arc::new(b.clone()));

            // Double blocks — closure fetches weights on demand (both forward and backward)
            for i in 0..self.num_double_blocks {
                let img_c = img_h.clone();
                let txt_c = txt_h.clone();
                let temb_c = pooled_temb.clone();
                let cos_c = pe_cos.clone();
                let sin_c = pe_sin.clone();
                let rw_c = self.resident_weights.clone();
                let bc = bundle_arc.clone();
                let dev_c = self.device.clone();
                let off_c = offloader_mtx.clone();
                let bi = i;
                let nt = n_txt;

                let block_out = AutogradContext::checkpoint_offload(
                    &[img_c.clone(), txt_c.clone()],
                    move || {
                        if let Some(ref b) = bc { b.refresh_caches(); }
                        let w = off_c.lock().unwrap().ensure_block(bi)
                            .map_err(|e| flame_core::Error::InvalidInput(format!("{e}")))?;
                        let (ni, nt_out) = double_block_fwd(
                            &img_c, &txt_c, &temb_c, &cos_c, &sin_c,
                            bi, &w, &rw_c, &bc, &dev_c,
                        )?;
                        Tensor::cat(&[&ni, &nt_out], 1)
                    },
                )?;

                // 2026-05-09 fix: previous version wrapped these unpacks in
                // `AutogradContext::no_grad` and then called `requires_grad_(true)`,
                // which produced orphaned leaves with no parent in the tape — the
                // next dual block's checkpoint_offload couldn't propagate
                // gradient back to *this* block's sub-tape, so all 19 dual
                // blocks' LoRAs got zero grad. Recording the narrow ops adds
                // 2 small entries per block (38 total) — negligible vs the
                // sub-tape size — and keeps the gradient chain intact.
                let img_seq = img_h.shape().dims()[1];
                img_h = block_out.narrow(1, 0, img_seq)?;
                txt_h = block_out.narrow(1, img_seq, nt)?;
                log::debug!("[fwd] double block {i}/{} done", self.num_double_blocks);
            }

            // Concat for single blocks. Same reasoning as the narrow above —
            // recording this cat is cheap and keeps the gradient chain alive
            // back into the dual blocks.
            let mut h = Tensor::cat(&[&txt_h, &img_h], 1)?;

            // Single blocks — closure fetches weights on demand
            for i in 0..self.num_single_blocks {
                let h_c = h.clone();
                let temb_c = pooled_temb.clone();
                let cos_c = pe_cos.clone();
                let sin_c = pe_sin.clone();
                let rw_c = self.resident_weights.clone();
                let bc = bundle_arc.clone();
                let dev_c = self.device.clone();
                let off_c = offloader_mtx.clone();
                let bi = i;

                h = AutogradContext::checkpoint_offload(
                    &[h_c.clone()],
                    move || {
                        if let Some(ref b) = bc { b.refresh_caches(); }
                        let swap_idx = NUM_DOUBLE_BLOCKS + bi;
                        let w = off_c.lock().unwrap().ensure_block(swap_idx)
                            .map_err(|e| flame_core::Error::InvalidInput(format!("{e}")))?;
                        single_block_fwd(
                            &h_c, &temb_c, &cos_c, &sin_c,
                            bi, &w, &rw_c, &bc, &dev_c,
                        )
                    },
                )?;
                log::debug!("[fwd] single block {i}/{} done", self.num_single_blocks);
            }

            // Extract image tokens from combined h
            let img_out_h = h.narrow(1, n_txt, n_img)?;
            img_h = img_out_h;
            // txt_h not needed after this point
        } else {
            // ── No-swap path: all weights on GPU, no checkpointing ──
            // Phase timing probe (CHROMA_TIMING=1 to enable). Sync with
            // device before each measurement to attribute time correctly.
            let timing = std::env::var("CHROMA_TIMING").is_ok();
            let sync_dev = || -> Result<()> {
                if timing {
                    self.device.synchronize()
                        .map_err(|e| flame_core::Error::Cuda(format!("sync: {e:?}")))?;
                }
                Ok(())
            };
            sync_dev()?;
            let t_double = std::time::Instant::now();
            for i in 0..self.num_double_blocks {
                let (new_img, new_txt) = self.double_block(
                    &img_h, &txt_h, &pooled_temb, &pe_cos, &pe_sin, i, n_txt, None,
                )?;
                img_h = new_img;
                txt_h = new_txt;
            }
            sync_dev()?;
            let dt_double = t_double.elapsed();

            let combined = Tensor::cat(&[&txt_h, &img_h], 1)?;
            let mut h = combined;
            sync_dev()?;
            let t_single = std::time::Instant::now();
            for i in 0..self.num_single_blocks {
                h = self.single_block(&h, &pooled_temb, &pe_cos, &pe_sin, i, None)?;
            }
            sync_dev()?;
            let dt_single = t_single.elapsed();

            let img_out_h = h.narrow(1, n_txt, n_img)?;
            img_h = img_out_h;

            if timing {
                log::info!(
                    "[chroma fwd] double {} blocks: {:.1}ms ({:.2}ms/blk) | single {} blocks: {:.1}ms ({:.2}ms/blk)",
                    self.num_double_blocks,
                    dt_double.as_secs_f64() * 1000.0,
                    dt_double.as_secs_f64() * 1000.0 / self.num_double_blocks as f64,
                    self.num_single_blocks,
                    dt_single.as_secs_f64() * 1000.0,
                    dt_single.as_secs_f64() * 1000.0 / self.num_single_blocks as f64,
                );
            }
        }

        // img_h now holds the extracted image tokens from whichever path.
        // Final norm + proj
        let final_mod_start = 3 * NUM_SINGLE_BLOCKS + 2 * 6 * NUM_DOUBLE_BLOCKS;
        let shift = pooled_temb.narrow(1, final_mod_start, 1)?.squeeze(Some(1))?;
        let scale = pooled_temb.narrow(1, final_mod_start + 1, 1)?.squeeze(Some(1))?;
        let img_mod = Self::modulate_pre_autograd(&img_h, &shift, &scale)?;

        let proj_w = self.w("proj_out.weight")?;
        let proj_b = self.w("proj_out.bias")?;
        let output = self.block_linear_bias(&img_mod, proj_w, proj_b, false)?;

        // Unpatchify: [B, N_img, 64] → [B, 16, H_lat, W_lat]
        self.unpatchify(&output, h_tok, w_tok)
    }

    // -----------------------------------------------------------------------
    // Approximator (distilled guidance layer)
    // -----------------------------------------------------------------------

    fn run_approximator(&self, timesteps: &Tensor) -> Result<Tensor> {
        let batch_size = timesteps.shape().dims()[0];
        let mod_index_length = 3 * NUM_SINGLE_BLOCKS + 2 * 6 * NUM_DOUBLE_BLOCKS + 2;
        let num_channels = 16; // approximator_in_channels / 4

        // Sinusoidal timestep embedding
        let t_scaled = timesteps.to_dtype(DType::F32)?.mul_scalar(1000.0)?.to_dtype(DType::BF16)?;
        let t_proj = sinusoidal_embedding(&t_scaled, num_channels)?; // [B, 2*num_channels]
        let zeros = Tensor::zeros_dtype(
            flame_core::Shape::from_dims(&[batch_size]),
            DType::BF16,
            self.device.clone(),
        )?;
        let g_proj = sinusoidal_embedding(&zeros, num_channels)?; // [B, 2*num_channels]

        // Concatenate timestep + guidance projections — matches diffusers/inference:
        //   conditioning = cat([timestep_proj, guidance_proj]) → [B, 2*num_channels]
        let conditioning = Tensor::cat(&[&t_proj, &g_proj], 1)?;
        let cond_exp = conditioning.unsqueeze(1)?
            .expand(&[batch_size, mod_index_length, 2 * num_channels])?;

        // Build mod_proj buffer: position encoding per mod index
        let mod_proj = build_mod_proj(mod_index_length, num_channels, &self.device)?;
        let mp_exp = mod_proj.unsqueeze(0)?
            .expand(&[batch_size, mod_index_length, 2 * num_channels])?;

        // Cat conditioning + mod_proj → [B, mod_index_length, 4*num_channels]
        let input_vec = Tensor::cat(&[&cond_exp, &mp_exp], 2)?;

        // in_proj
        let in_w = self.w("distilled_guidance_layer.in_proj.weight")?;
        let in_b = self.w("distilled_guidance_layer.in_proj.bias")?;
        let mut x = self.block_linear_bias(&input_vec, in_w, in_b, false)?;

        // Residual blocks
        for i in 0..5 {
            let norm_w = self.w(&format!("distilled_guidance_layer.norms.{i}.weight"))?;
            let l1_w = self.w(&format!("distilled_guidance_layer.layers.{i}.linear_1.weight"))?;
            let l1_b = self.w(&format!("distilled_guidance_layer.layers.{i}.linear_1.bias"))?;
            let l2_w = self.w(&format!("distilled_guidance_layer.layers.{i}.linear_2.weight"))?;
            let l2_b = self.w(&format!("distilled_guidance_layer.layers.{i}.linear_2.bias"))?;

            let n = rms_norm_with_weight(&x, norm_w, NORM_EPS)?;
            let h = self.block_linear_bias(&n, l1_w, l1_b, false)?;
            let h = h.silu()?;
            let h = self.block_linear_bias(&h, l2_w, l2_b, false)?;
            x = x.add(&h)?;
        }

        let out_w = self.w("distilled_guidance_layer.out_proj.weight")?;
        let out_b = self.w("distilled_guidance_layer.out_proj.bias")?;
        self.block_linear_bias(&x, out_w, out_b, false)
    }

    // -----------------------------------------------------------------------
    // Double block
    // -----------------------------------------------------------------------

    fn double_block(
        &self,
        img: &Tensor,
        txt: &Tensor,
        pooled_temb: &Tensor,
        pe_cos: &Tensor,
        pe_sin: &Tensor,
        block_idx: usize,
        _n_txt: usize,
        ext_weights: Option<&HashMap<String, Tensor>>,
    ) -> Result<(Tensor, Tensor)> {
        let pt = ext_weights.is_some(); // pre_transposed flag for block linear ops
        // Modulation slicing
        let img_mod_start = 3 * NUM_SINGLE_BLOCKS + 6 * block_idx;
        let txt_mod_start = 3 * NUM_SINGLE_BLOCKS + 6 * NUM_DOUBLE_BLOCKS + 6 * block_idx;

        let img_mod = pooled_temb.narrow(1, img_mod_start, 6)?; // [B, 6, 3072]
        let txt_mod = pooled_temb.narrow(1, txt_mod_start, 6)?;

        // img_mod slices: shift1, scale1, gate1, shift2, scale2, gate2
        let img_shift1 = img_mod.narrow(1, 0, 1)?.squeeze(Some(1))?;
        let img_scale1 = img_mod.narrow(1, 1, 1)?.squeeze(Some(1))?;
        let img_gate1 = img_mod.narrow(1, 2, 1)?.squeeze(Some(1))?;
        let img_shift2 = img_mod.narrow(1, 3, 1)?.squeeze(Some(1))?;
        let img_scale2 = img_mod.narrow(1, 4, 1)?.squeeze(Some(1))?;
        let img_gate2 = img_mod.narrow(1, 5, 1)?.squeeze(Some(1))?;

        let txt_shift1 = txt_mod.narrow(1, 0, 1)?.squeeze(Some(1))?;
        let txt_scale1 = txt_mod.narrow(1, 1, 1)?.squeeze(Some(1))?;
        let txt_gate1 = txt_mod.narrow(1, 2, 1)?.squeeze(Some(1))?;
        let txt_shift2 = txt_mod.narrow(1, 3, 1)?.squeeze(Some(1))?;
        let txt_scale2 = txt_mod.narrow(1, 4, 1)?.squeeze(Some(1))?;
        let txt_gate2 = txt_mod.narrow(1, 5, 1)?.squeeze(Some(1))?;

        // Image attention
        let img_norm = Self::modulate_pre(img, &img_shift1, &img_scale1)?;
        let b = img.shape().dims()[0];
        let n_img = img.shape().dims()[1];

        let mut img_q = self.block_linear_bias(
            &img_norm,
            self.dw_or(ext_weights, block_idx, "attn.to_q.weight")?,
            self.dw_or(ext_weights, block_idx, "attn.to_q.bias")?,
            pt,
        )?;
        img_q = self.add_double_lora_delta(img_q, &img_norm, block_idx, DoubleLoraTarget::ImgQ)?;

        let mut img_k = self.block_linear_bias(
            &img_norm,
            self.dw_or(ext_weights, block_idx, "attn.to_k.weight")?,
            self.dw_or(ext_weights, block_idx, "attn.to_k.bias")?,
            pt,
        )?;
        img_k = self.add_double_lora_delta(img_k, &img_norm, block_idx, DoubleLoraTarget::ImgK)?;

        let mut img_v = self.block_linear_bias(
            &img_norm,
            self.dw_or(ext_weights, block_idx, "attn.to_v.weight")?,
            self.dw_or(ext_weights, block_idx, "attn.to_v.bias")?,
            pt,
        )?;
        img_v = self.add_double_lora_delta(img_v, &img_norm, block_idx, DoubleLoraTarget::ImgV)?;

        // Txt attention
        let txt_norm = Self::modulate_pre(txt, &txt_shift1, &txt_scale1)?;

        let mut txt_q = self.block_linear_bias(
            &txt_norm,
            self.dw_or(ext_weights, block_idx, "attn.add_q_proj.weight")?,
            self.dw_or(ext_weights, block_idx, "attn.add_q_proj.bias")?,
            pt,
        )?;
        txt_q = self.add_double_lora_delta(txt_q, &txt_norm, block_idx, DoubleLoraTarget::TxtQ)?;

        let mut txt_k = self.block_linear_bias(
            &txt_norm,
            self.dw_or(ext_weights, block_idx, "attn.add_k_proj.weight")?,
            self.dw_or(ext_weights, block_idx, "attn.add_k_proj.bias")?,
            pt,
        )?;
        txt_k = self.add_double_lora_delta(txt_k, &txt_norm, block_idx, DoubleLoraTarget::TxtK)?;

        let mut txt_v = self.block_linear_bias(
            &txt_norm,
            self.dw_or(ext_weights, block_idx, "attn.add_v_proj.weight")?,
            self.dw_or(ext_weights, block_idx, "attn.add_v_proj.bias")?,
            pt,
        )?;
        txt_v = self.add_double_lora_delta(txt_v, &txt_norm, block_idx, DoubleLoraTarget::TxtV)?;

        // Reshape to [B, S, H, D] then [B, H, S, D]
        let img_q = img_q.reshape(&[b, n_img, NUM_HEADS, HEAD_DIM])?.permute(&[0, 2, 1, 3])?;
        let img_k = img_k.reshape(&[b, n_img, NUM_HEADS, HEAD_DIM])?.permute(&[0, 2, 1, 3])?;
        let img_v = img_v.reshape(&[b, n_img, NUM_HEADS, HEAD_DIM])?.permute(&[0, 2, 1, 3])?;

        let n_t = txt.shape().dims()[1];
        let txt_q = txt_q.reshape(&[b, n_t, NUM_HEADS, HEAD_DIM])?.permute(&[0, 2, 1, 3])?;
        let txt_k = txt_k.reshape(&[b, n_t, NUM_HEADS, HEAD_DIM])?.permute(&[0, 2, 1, 3])?;
        let txt_v = txt_v.reshape(&[b, n_t, NUM_HEADS, HEAD_DIM])?.permute(&[0, 2, 1, 3])?;

        // QK norms
        let img_q = Self::rms_norm_per_head(&img_q, self.dw_or(ext_weights, block_idx, "attn.norm_q.weight")?)?;
        let img_k = Self::rms_norm_per_head(&img_k, self.dw_or(ext_weights, block_idx, "attn.norm_k.weight")?)?;
        let txt_q = Self::rms_norm_per_head(&txt_q, self.dw_or(ext_weights, block_idx, "attn.norm_added_q.weight")?)?;
        let txt_k = Self::rms_norm_per_head(&txt_k, self.dw_or(ext_weights, block_idx, "attn.norm_added_k.weight")?)?;

        // Concat QKV for joint attention: [B, H, N_txt+N_img, D]
        let q = Tensor::cat(&[&txt_q, &img_q], 2)?;
        let k = Tensor::cat(&[&txt_k, &img_k], 2)?;
        let v = Tensor::cat(&[&txt_v, &img_v], 2)?;

        // RoPE — use autograd-recording wrapper so gradient flows through Q/K
        // in both the no-swap training path and any future checkpoint path.
        let q = rope_with_grad(&q, pe_cos, pe_sin)?;
        let k = rope_with_grad(&k, pe_cos, pe_sin)?;

        // SDPA
        let attn_out = flame_core::attention::sdpa(&q, &k, &v, None)?;

        // Split attn_out [B,H,N_total,D] back into txt and img.
        // Inference path: fused kernel (no autograd needed).
        // Training path: falls back to narrow+permute+reshape inside the kernel.
        let (txt_attn, img_attn) =
            flame_core::bf16_ops::attn_split_txt_img_bf16(&attn_out, n_t, n_img)?;

        // Output projections
        let mut img_out = self.block_linear_bias(
            &img_attn,
            self.dw_or(ext_weights, block_idx, "attn.to_out.0.weight")?,
            self.dw_or(ext_weights, block_idx, "attn.to_out.0.bias")?,
            pt,
        )?;
        img_out = self.add_double_lora_delta(img_out, &img_attn, block_idx, DoubleLoraTarget::ImgOut)?;

        let mut txt_out = self.block_linear_bias(
            &txt_attn,
            self.dw_or(ext_weights, block_idx, "attn.to_add_out.weight")?,
            self.dw_or(ext_weights, block_idx, "attn.to_add_out.bias")?,
            pt,
        )?;
        txt_out = self.add_double_lora_delta(txt_out, &txt_attn, block_idx, DoubleLoraTarget::TxtOut)?;

        // Gate + residual for attention.
        // Inference path: fused kernel. Training: falls back to elementwise.
        let is_inference = !AutogradContext::is_recording();
        let (img_r, txt_r) = if is_inference {
            let img_r = flame_core::bf16_ops::gate_residual_fused_bf16(img, &img_gate1, &img_out)?;
            let txt_r = flame_core::bf16_ops::gate_residual_fused_bf16(txt, &txt_gate1, &txt_out)?;
            (img_r, txt_r)
        } else {
            let img_gate1_unsq = img_gate1.unsqueeze(1)?;
            let txt_gate1_unsq = txt_gate1.unsqueeze(1)?;
            let img_r = img.add(&img_out.mul(&img_gate1_unsq)?)?;
            let txt_r = txt.add(&txt_out.mul(&txt_gate1_unsq)?)?;
            (img_r, txt_r)
        };

        // FFN: img
        let img_ffn_norm = Self::modulate_pre(&img_r, &img_shift2, &img_scale2)?;
        let mut img_ffn_h = self.block_linear_bias(
            &img_ffn_norm,
            self.dw_or(ext_weights, block_idx, "ff.net.0.proj.weight")?,
            self.dw_or(ext_weights, block_idx, "ff.net.0.proj.bias")?,
            pt,
        )?;
        img_ffn_h = self.add_double_lora_delta(img_ffn_h, &img_ffn_norm, block_idx, DoubleLoraTarget::ImgFfnGate)?;
        let img_ffn_h = img_ffn_h.gelu()?;
        let mut img_ffn_out = self.block_linear_bias(
            &img_ffn_h,
            self.dw_or(ext_weights, block_idx, "ff.net.2.weight")?,
            self.dw_or(ext_weights, block_idx, "ff.net.2.bias")?,
            pt,
        )?;
        img_ffn_out = self.add_double_lora_delta(img_ffn_out, &img_ffn_h, block_idx, DoubleLoraTarget::ImgFfnOut)?;

        let img_final = if is_inference {
            flame_core::bf16_ops::gate_residual_fused_bf16(&img_r, &img_gate2, &img_ffn_out)?
        } else {
            let img_gate2_unsq = img_gate2.unsqueeze(1)?;
            img_r.add(&img_ffn_out.mul(&img_gate2_unsq)?)?
        };

        // FFN: txt
        let txt_ffn_norm = Self::modulate_pre(&txt_r, &txt_shift2, &txt_scale2)?;
        let mut txt_ffn_h = self.block_linear_bias(
            &txt_ffn_norm,
            self.dw_or(ext_weights, block_idx, "ff_context.net.0.proj.weight")?,
            self.dw_or(ext_weights, block_idx, "ff_context.net.0.proj.bias")?,
            pt,
        )?;
        txt_ffn_h = self.add_double_lora_delta(txt_ffn_h, &txt_ffn_norm, block_idx, DoubleLoraTarget::TxtFfnGate)?;
        let txt_ffn_h = txt_ffn_h.gelu()?;
        let mut txt_ffn_out = self.block_linear_bias(
            &txt_ffn_h,
            self.dw_or(ext_weights, block_idx, "ff_context.net.2.weight")?,
            self.dw_or(ext_weights, block_idx, "ff_context.net.2.bias")?,
            pt,
        )?;
        txt_ffn_out = self.add_double_lora_delta(txt_ffn_out, &txt_ffn_h, block_idx, DoubleLoraTarget::TxtFfnOut)?;

        let txt_final = if is_inference {
            flame_core::bf16_ops::gate_residual_fused_bf16(&txt_r, &txt_gate2, &txt_ffn_out)?
        } else {
            let txt_gate2_unsq = txt_gate2.unsqueeze(1)?;
            txt_r.add(&txt_ffn_out.mul(&txt_gate2_unsq)?)?
        };

        Ok((img_final, txt_final))
    }

    // -----------------------------------------------------------------------
    // Single block
    // -----------------------------------------------------------------------

    fn single_block(
        &self,
        x: &Tensor,
        pooled_temb: &Tensor,
        pe_cos: &Tensor,
        pe_sin: &Tensor,
        block_idx: usize,
        ext_weights: Option<&HashMap<String, Tensor>>,
    ) -> Result<Tensor> {
        let pt = ext_weights.is_some(); // pre_transposed flag for block linear ops

        // Modulation
        let mod_start = 3 * block_idx;
        let mods = pooled_temb.narrow(1, mod_start, 3)?;
        let shift = mods.narrow(1, 0, 1)?.squeeze(Some(1))?;
        let scale = mods.narrow(1, 1, 1)?.squeeze(Some(1))?;
        let gate = mods.narrow(1, 2, 1)?.squeeze(Some(1))?;

        let x_norm = Self::modulate_pre(x, &shift, &scale)?;
        let b = x.shape().dims()[0];
        let seq = x.shape().dims()[1];

        // QKV
        let mut q = self.block_linear_bias(
            &x_norm,
            self.sw_or(ext_weights, block_idx, "attn.to_q.weight")?,
            self.sw_or(ext_weights, block_idx, "attn.to_q.bias")?,
            pt,
        )?;
        q = self.add_single_lora_delta(q, &x_norm, block_idx, SingleLoraTarget::Q)?;

        let mut k = self.block_linear_bias(
            &x_norm,
            self.sw_or(ext_weights, block_idx, "attn.to_k.weight")?,
            self.sw_or(ext_weights, block_idx, "attn.to_k.bias")?,
            pt,
        )?;
        k = self.add_single_lora_delta(k, &x_norm, block_idx, SingleLoraTarget::K)?;

        let mut v = self.block_linear_bias(
            &x_norm,
            self.sw_or(ext_weights, block_idx, "attn.to_v.weight")?,
            self.sw_or(ext_weights, block_idx, "attn.to_v.bias")?,
            pt,
        )?;
        v = self.add_single_lora_delta(v, &x_norm, block_idx, SingleLoraTarget::V)?;

        // MLP projection (parallel with attention)
        let mut mlp_h = self.block_linear_bias(
            &x_norm,
            self.sw_or(ext_weights, block_idx, "proj_mlp.weight")?,
            self.sw_or(ext_weights, block_idx, "proj_mlp.bias")?,
            pt,
        )?;
        mlp_h = self.add_single_lora_delta(mlp_h, &x_norm, block_idx, SingleLoraTarget::ProjMlp)?;
        let mlp_h = mlp_h.gelu()?;

        // Attention
        let q = q.reshape(&[b, seq, NUM_HEADS, HEAD_DIM])?.permute(&[0, 2, 1, 3])?;
        let k = k.reshape(&[b, seq, NUM_HEADS, HEAD_DIM])?.permute(&[0, 2, 1, 3])?;
        let v = v.reshape(&[b, seq, NUM_HEADS, HEAD_DIM])?.permute(&[0, 2, 1, 3])?;

        // QK norms
        let q = Self::rms_norm_per_head(&q, self.sw_or(ext_weights, block_idx, "attn.norm_q.weight")?)?;
        let k = Self::rms_norm_per_head(&k, self.sw_or(ext_weights, block_idx, "attn.norm_k.weight")?)?;

        // RoPE — use autograd-recording wrapper
        let q = rope_with_grad(&q, pe_cos, pe_sin)?;
        let k = rope_with_grad(&k, pe_cos, pe_sin)?;

        let attn_out = flame_core::attention::sdpa(&q, &k, &v, None)?;
        let attn_out = attn_out.permute(&[0, 2, 1, 3])?;
        let attn_flat = attn_out.reshape(&[b, seq, DIM])?;

        // Concat attn + mlp and project out: [B, S, DIM + MLP_HIDDEN] → [B, S, DIM]
        let combined = Tensor::cat(&[&attn_flat, &mlp_h], 2)?;
        let mut proj = self.block_linear_bias(
            &combined,
            self.sw_or(ext_weights, block_idx, "proj_out.weight")?,
            self.sw_or(ext_weights, block_idx, "proj_out.bias")?,
            pt,
        )?;
        proj = self.add_single_lora_delta(proj, &combined, block_idx, SingleLoraTarget::ProjOut)?;

        // Gated residual — fused kernel for inference, elementwise for training.
        if !AutogradContext::is_recording() {
            flame_core::bf16_ops::gate_residual_fused_bf16(x, &gate, &proj)
        } else {
            let gate_unsq = gate.unsqueeze(1)?;
            x.add(&proj.mul(&gate_unsq)?)
        }
    }

    // -----------------------------------------------------------------------
    // Patchify / Unpatchify
    // -----------------------------------------------------------------------

    fn patchify(&self, x: &Tensor, h: usize, w: usize) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let (b, c) = (dims[0], dims[1]);
        let h_tok = h / 2;
        let w_tok = w / 2;
        // [B, C, H, W] → [B, C, h_tok, 2, w_tok, 2]
        let x_r = x.reshape(&[b, c, h_tok, 2, w_tok, 2])?;
        // → [B, h_tok, w_tok, C, 2, 2]
        let x_p = x_r.permute(&[0, 2, 4, 1, 3, 5])?;
        // → [B, h_tok*w_tok, C*4]
        x_p.reshape(&[b, h_tok * w_tok, c * 4])
    }

    fn unpatchify(&self, x: &Tensor, h_tok: usize, w_tok: usize) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let b = dims[0];
        let c = 16; // latent channels
        // [B, N, 64] → [B, h_tok, w_tok, 16, 2, 2]
        let x_r = x.reshape(&[b, h_tok, w_tok, c, 2, 2])?;
        // → [B, 16, h_tok, 2, w_tok, 2]
        let x_p = x_r.permute(&[0, 3, 1, 4, 2, 5])?;
        // → [B, 16, H, W]
        x_p.reshape(&[b, c, h_tok * 2, w_tok * 2])
    }

    // -----------------------------------------------------------------------
    // Position IDs + RoPE
    // -----------------------------------------------------------------------

    fn build_position_ids(
        h_tok: usize,
        w_tok: usize,
        n_txt: usize,
        device: &Arc<cudarc::driver::CudaDevice>,
    ) -> Result<(Tensor, Tensor)> {
        let n_img = h_tok * w_tok;
        // img_ids: [N_img, 3] — (0, row, col)
        let mut img_data = vec![0.0f32; n_img * 3];
        for row in 0..h_tok {
            for col in 0..w_tok {
                let idx = row * w_tok + col;
                img_data[idx * 3] = 0.0;
                img_data[idx * 3 + 1] = row as f32;
                img_data[idx * 3 + 2] = col as f32;
            }
        }
        let img_ids = Tensor::from_vec(
            img_data,
            flame_core::Shape::from_dims(&[n_img, 3]),
            device.clone(),
        )?;

        // txt_ids: [N_txt, 3] — all zeros
        let txt_ids = Tensor::zeros_dtype(
            flame_core::Shape::from_dims(&[n_txt, 3]),
            DType::F32,
            device.clone(),
        )?;

        Ok((img_ids, txt_ids))
    }

    fn build_rope(
        &self,
        img_ids: &Tensor,
        txt_ids: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let axes_dims = [16usize, 56, 56]; // FLUX/Chroma RoPE axes
        let rope_theta: f64 = 10000.0;

        let all_ids = Tensor::cat(&[txt_ids, img_ids], 0)?;
        let all_f32 = all_ids.to_dtype(DType::F32)?;

        let mut cos_parts = Vec::new();
        let mut sin_parts = Vec::new();

        for (axis, &axis_dim) in axes_dims.iter().enumerate() {
            let half = axis_dim / 2;
            let omega_data: Vec<f32> = (0..half)
                .map(|i| {
                    let scale = (2 * i) as f64 / axis_dim as f64;
                    (1.0 / rope_theta.powf(scale)) as f32
                })
                .collect();
            let omega = Tensor::from_vec(
                omega_data,
                flame_core::Shape::from_dims(&[1, half]),
                self.device.clone(),
            )?;

            let pos = all_f32.narrow(1, axis, 1)?.squeeze(Some(1))?;
            let pos_col = pos.unsqueeze(1)?;
            let angles = pos_col.matmul(&omega)?;

            cos_parts.push(angles.cos()?);
            sin_parts.push(angles.sin()?);
        }

        let cos_refs: Vec<&Tensor> = cos_parts.iter().collect();
        let sin_refs: Vec<&Tensor> = sin_parts.iter().collect();
        let cos_full = Tensor::cat(&cos_refs, 1)?;
        let sin_full = Tensor::cat(&sin_refs, 1)?;

        let pe_cos = cos_full.unsqueeze(0)?.unsqueeze(0)?.to_dtype(DType::BF16)?;
        let pe_sin = sin_full.unsqueeze(0)?.unsqueeze(0)?.to_dtype(DType::BF16)?;
        Ok((pe_cos, pe_sin))
    }
}

// ---------------------------------------------------------------------------
// Standalone helpers
// ---------------------------------------------------------------------------

fn ensure_3d(x: &Tensor) -> Result<Tensor> {
    let dims = x.shape().dims();
    match dims.len() {
        2 => x.unsqueeze(0),
        3 => Ok(x.clone()),
        _ => Err(flame_core::Error::InvalidInput(format!(
            "expected 2D or 3D tensor, got {}D", dims.len()
        ))),
    }
}

fn sinusoidal_embedding(t: &Tensor, dim: usize) -> Result<Tensor> {
    let t_f32 = t.to_dtype(DType::F32)?;
    let half = dim / 2;
    let max_period = 10000.0f64;
    let freq_data: Vec<f32> = (0..half)
        .map(|i| (-max_period.ln() * i as f64 / half as f64).exp() as f32)
        .collect();
    let freqs = Tensor::from_vec(
        freq_data,
        flame_core::Shape::from_dims(&[1, half]),
        t.device().clone(),
    )?;
    let t_col = t_f32.unsqueeze(1)?;
    let args = t_col.matmul(&freqs)?;
    let cos = args.cos()?;
    let sin = args.sin()?;
    let emb = Tensor::cat(&[&cos, &sin], 1)?;
    emb.to_dtype(DType::BF16)
}

fn rms_norm_with_weight(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    let hidden = *dims.last().unwrap();
    let batch: usize = dims[..dims.len() - 1].iter().product();
    let x_2d = x.reshape(&[batch, hidden])?;
    let out = flame_core::cuda_ops_bf16::rms_norm_bf16(&x_2d, Some(weight), eps)?;
    out.reshape(&dims)
}

fn build_mod_proj(
    mod_index_length: usize,
    num_channels: usize,
    device: &Arc<cudarc::driver::CudaDevice>,
) -> Result<Tensor> {
    let dim = 2 * num_channels;
    let half = num_channels;
    let max_period = 10000.0f32;

    let mut data = vec![0.0f32; mod_index_length * dim];
    for idx in 0..mod_index_length {
        let pos = (idx as f32) * 1000.0;
        for i in 0..half {
            let freq = (-max_period.ln() * (i as f32) / (half as f32)).exp();
            let angle = pos * freq;
            data[idx * dim + i] = angle.cos();
            data[idx * dim + half + i] = angle.sin();
        }
    }
    Tensor::from_f32_to_bf16(
        data,
        flame_core::Shape::from_dims(&[mod_index_length, dim]),
        device.clone(),
    )
}

// ---------------------------------------------------------------------------
// Standalone block forward functions (for gradient checkpointing closures)
// ---------------------------------------------------------------------------

/// Standalone linear: x @ W + bias. `pt` = weights are pre-transposed.
fn linear_bias_pt(x: &Tensor, weight: &Tensor, bias: &Tensor, pt: bool) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    let in_feat = *dims.last().unwrap();
    let batch: usize = dims[..dims.len() - 1].iter().product();
    let x_2d = x.reshape(&[batch, in_feat])?;
    let (wt, out_feat) = if pt {
        (weight.clone(), weight.shape().dims()[1])
    } else {
        (weight.transpose()?, weight.shape().dims()[0])
    };
    let out_2d = x_2d.matmul(&wt)?.add(bias)?;
    let mut out_shape = dims[..dims.len() - 1].to_vec();
    out_shape.push(out_feat);
    out_2d.reshape(&out_shape)
}

fn modulate_pre(x: &Tensor, shift: &Tensor, scale: &Tensor) -> Result<Tensor> {
    flame_core::bf16_ops::modulate_pre_fused_bf16(x, shift, scale, NORM_EPS)
}

fn rms_norm_head(x: &Tensor, weight: &Tensor) -> Result<Tensor> {
    // 2026-05-09 fix: `cuda_ops_bf16::rms_norm_bf16` is inference-only — its
    // output never carries `requires_grad`, severing the gradient chain at Q/K
    // for every transformer block (single + dual). Use `flame_core::norm::rms_norm`
    // which records `Op::RmsNorm` and matches Klein's `head_rms_norm_local`
    // (klein.rs:491-497). Same kernel under the hood, autograd-aware wrapper.
    let dims = x.shape().dims().to_vec();
    let (b, h, s, d) = (dims[0], dims[1], dims[2], dims[3]);
    let flat = x.reshape(&[b * h * s, d])?;
    let normed = flame_core::norm::rms_norm(&flat, &[d], Some(weight), NORM_EPS)?;
    normed.reshape(&[b, h, s, d])
}

/// RoPE with autograd support (fused kernel + manual Op recording, same as Klein).
fn rope_with_grad(x: &Tensor, pe_cos: &Tensor, pe_sin: &Tensor) -> Result<Tensor> {
    use flame_core::autograd::{AutogradContext, Op};
    let mut output = flame_core::bf16_ops::rope_fused_bf16(x, pe_cos, pe_sin)?;
    if x.requires_grad() {
        output = output.requires_grad_(true);
        AutogradContext::record_op(
            output.id(),
            // flame-core HEAD's `Op::RoPePrecomputed` no longer carries an
            // explicit layout; backward dispatches by saved cos shape
            // (broadcast `[1,_,_,_]` / `[1,1,_,_]` / `[1,_,_]` →
            // fused/Interleaved). Chroma's pe_cos is `[1,1,N,half]`, so
            // this routes to `rope_fused_bf16` on backward — same kernel
            // family used in forward, no behavior change.
            Op::RoPePrecomputed {
                input: x.id(),
                cos: pe_cos.id(),
                sin: pe_sin.id(),
            },
            vec![
                (x.id(), x.clone()),
                (pe_cos.id(), pe_cos.clone()),
                (pe_sin.id(), pe_sin.clone()),
            ],
        );
    }
    Ok(output)
}

fn get_w<'a>(weights: &'a HashMap<String, Tensor>, prefix: &str, block_idx: usize, key: &str) -> Result<&'a Tensor> {
    let full_key = format!("{prefix}.{block_idx}.{key}");
    weights.get(&full_key)
        .ok_or_else(|| flame_core::Error::InvalidInput(format!("missing: {full_key}")))
}

fn add_lora_delta_double(
    base: Tensor, input: &Tensor, bundle: &Option<Arc<ChromaLoraBundle>>,
    weights: &HashMap<String, Tensor>,
    block_idx: usize, target: DoubleLoraTarget,
) -> Result<Tensor> {
    if let Some(ref b) = bundle {
        if let Some(lora) = b.double_adapters.get(&(block_idx, target)) {
            if lora.is_input_rotation() {
                let in3d = ensure_3d(input)?;
                let rx = lora.apply_input(&in3d)?;
                let delta_x = rx.sub(&in3d)?;
                let suffix = double_lora_suffix(target);
                let key = format!("transformer_blocks.{block_idx}.{suffix}.weight");
                let weight = weights.get(&key).ok_or_else(|| {
                    flame_core::Error::InvalidInput(format!("OFT: missing base weight `{key}`"))
                })?;
                let delta = standalone_block_linear_no_bias(&delta_x, weight)?;
                return base.add(&delta);
            }
            let delta = lora.forward_delta(&ensure_3d(input)?)?;
            return base.add(&delta);
        }
    }
    Ok(base)
}

fn add_lora_delta_single(
    base: Tensor, input: &Tensor, bundle: &Option<Arc<ChromaLoraBundle>>,
    weights: &HashMap<String, Tensor>,
    block_idx: usize, target: SingleLoraTarget,
) -> Result<Tensor> {
    if let Some(ref b) = bundle {
        if let Some(lora) = b.single_adapters.get(&(block_idx, target)) {
            if lora.is_input_rotation() {
                let in3d = ensure_3d(input)?;
                let rx = lora.apply_input(&in3d)?;
                let delta_x = rx.sub(&in3d)?;
                let suffix = single_lora_suffix(target);
                let key = format!("single_transformer_blocks.{block_idx}.{suffix}.weight");
                let weight = weights.get(&key).ok_or_else(|| {
                    flame_core::Error::InvalidInput(format!("OFT: missing base weight `{key}`"))
                })?;
                let delta = standalone_block_linear_no_bias(&delta_x, weight)?;
                return base.add(&delta);
            }
            let delta = lora.forward_delta(&ensure_3d(input)?)?;
            return base.add(&delta);
        }
    }
    Ok(base)
}

/// Free-function variant of `block_linear_no_bias` used by the
/// standalone checkpoint closures.  Standalone-closure weights ARE
/// pre-transposed (stored `[in, out]`) per chroma's swap loader
/// (`double_block_fwd::pt = true`), so this matmuls directly without
/// a transpose.
fn standalone_block_linear_no_bias(x: &Tensor, weight: &Tensor) -> Result<Tensor> {
    let dims = x.dims().to_vec();
    let in_feat = *dims.last().unwrap();
    // Weight is `[in, out]` (pre-transposed) — out_feat is dim 1.
    let out_feat = weight.dims()[1];
    let batch: usize = dims[..dims.len() - 1].iter().product();
    let x_2d = x.reshape(&[batch, in_feat])?;
    let out_2d = x_2d.matmul(weight)?;
    let mut out_shape = dims[..dims.len() - 1].to_vec();
    out_shape.push(out_feat);
    out_2d.reshape(&out_shape)
}

/// Standalone double block forward for checkpoint closures.
fn double_block_fwd(
    img: &Tensor,
    txt: &Tensor,
    pooled_temb: &Tensor,
    pe_cos: &Tensor,
    pe_sin: &Tensor,
    block_idx: usize,
    weights: &HashMap<String, Tensor>,
    _resident: &HashMap<String, Tensor>,
    bundle: &Option<Arc<ChromaLoraBundle>>,
    _device: &Arc<cudarc::driver::CudaDevice>,
) -> Result<(Tensor, Tensor)> {
    let pt = true; // swap weights are pre-transposed
    let pfx = "transformer_blocks";

    let img_mod_start = 3 * NUM_SINGLE_BLOCKS + 6 * block_idx;
    let txt_mod_start = 3 * NUM_SINGLE_BLOCKS + 6 * NUM_DOUBLE_BLOCKS + 6 * block_idx;

    let img_mod = pooled_temb.narrow(1, img_mod_start, 6)?;
    let txt_mod = pooled_temb.narrow(1, txt_mod_start, 6)?;

    let img_shift1 = img_mod.narrow(1, 0, 1)?.squeeze(Some(1))?;
    let img_scale1 = img_mod.narrow(1, 1, 1)?.squeeze(Some(1))?;
    let img_gate1 = img_mod.narrow(1, 2, 1)?.squeeze(Some(1))?;
    let img_shift2 = img_mod.narrow(1, 3, 1)?.squeeze(Some(1))?;
    let img_scale2 = img_mod.narrow(1, 4, 1)?.squeeze(Some(1))?;
    let img_gate2 = img_mod.narrow(1, 5, 1)?.squeeze(Some(1))?;

    let txt_shift1 = txt_mod.narrow(1, 0, 1)?.squeeze(Some(1))?;
    let txt_scale1 = txt_mod.narrow(1, 1, 1)?.squeeze(Some(1))?;
    let txt_gate1 = txt_mod.narrow(1, 2, 1)?.squeeze(Some(1))?;
    let txt_shift2 = txt_mod.narrow(1, 3, 1)?.squeeze(Some(1))?;
    let txt_scale2 = txt_mod.narrow(1, 4, 1)?.squeeze(Some(1))?;
    let txt_gate2 = txt_mod.narrow(1, 5, 1)?.squeeze(Some(1))?;

    let img_norm = modulate_pre(img, &img_shift1, &img_scale1)?;
    let b = img.shape().dims()[0];
    let n_img = img.shape().dims()[1];

    let mut img_q = linear_bias_pt(&img_norm, get_w(weights, pfx, block_idx, "attn.to_q.weight")?, get_w(weights, pfx, block_idx, "attn.to_q.bias")?, pt)?;
    img_q = add_lora_delta_double(img_q, &img_norm, bundle, weights, block_idx, DoubleLoraTarget::ImgQ)?;
    let mut img_k = linear_bias_pt(&img_norm, get_w(weights, pfx, block_idx, "attn.to_k.weight")?, get_w(weights, pfx, block_idx, "attn.to_k.bias")?, pt)?;
    img_k = add_lora_delta_double(img_k, &img_norm, bundle, weights, block_idx, DoubleLoraTarget::ImgK)?;
    let mut img_v = linear_bias_pt(&img_norm, get_w(weights, pfx, block_idx, "attn.to_v.weight")?, get_w(weights, pfx, block_idx, "attn.to_v.bias")?, pt)?;
    img_v = add_lora_delta_double(img_v, &img_norm, bundle, weights, block_idx, DoubleLoraTarget::ImgV)?;

    let txt_norm = modulate_pre(txt, &txt_shift1, &txt_scale1)?;
    let n_t = txt.shape().dims()[1];

    let mut txt_q = linear_bias_pt(&txt_norm, get_w(weights, pfx, block_idx, "attn.add_q_proj.weight")?, get_w(weights, pfx, block_idx, "attn.add_q_proj.bias")?, pt)?;
    txt_q = add_lora_delta_double(txt_q, &txt_norm, bundle, weights, block_idx, DoubleLoraTarget::TxtQ)?;
    let mut txt_k = linear_bias_pt(&txt_norm, get_w(weights, pfx, block_idx, "attn.add_k_proj.weight")?, get_w(weights, pfx, block_idx, "attn.add_k_proj.bias")?, pt)?;
    txt_k = add_lora_delta_double(txt_k, &txt_norm, bundle, weights, block_idx, DoubleLoraTarget::TxtK)?;
    let mut txt_v = linear_bias_pt(&txt_norm, get_w(weights, pfx, block_idx, "attn.add_v_proj.weight")?, get_w(weights, pfx, block_idx, "attn.add_v_proj.bias")?, pt)?;
    txt_v = add_lora_delta_double(txt_v, &txt_norm, bundle, weights, block_idx, DoubleLoraTarget::TxtV)?;

    let img_q = img_q.reshape(&[b, n_img, NUM_HEADS, HEAD_DIM])?.permute(&[0, 2, 1, 3])?;
    let img_k = img_k.reshape(&[b, n_img, NUM_HEADS, HEAD_DIM])?.permute(&[0, 2, 1, 3])?;
    let img_v = img_v.reshape(&[b, n_img, NUM_HEADS, HEAD_DIM])?.permute(&[0, 2, 1, 3])?;
    let txt_q = txt_q.reshape(&[b, n_t, NUM_HEADS, HEAD_DIM])?.permute(&[0, 2, 1, 3])?;
    let txt_k = txt_k.reshape(&[b, n_t, NUM_HEADS, HEAD_DIM])?.permute(&[0, 2, 1, 3])?;
    let txt_v = txt_v.reshape(&[b, n_t, NUM_HEADS, HEAD_DIM])?.permute(&[0, 2, 1, 3])?;

    let img_q = rms_norm_head(&img_q, get_w(weights, pfx, block_idx, "attn.norm_q.weight")?)?;
    let img_k = rms_norm_head(&img_k, get_w(weights, pfx, block_idx, "attn.norm_k.weight")?)?;
    let txt_q = rms_norm_head(&txt_q, get_w(weights, pfx, block_idx, "attn.norm_added_q.weight")?)?;
    let txt_k = rms_norm_head(&txt_k, get_w(weights, pfx, block_idx, "attn.norm_added_k.weight")?)?;

    let q = Tensor::cat(&[&txt_q, &img_q], 2)?;
    let k = Tensor::cat(&[&txt_k, &img_k], 2)?;
    let v = Tensor::cat(&[&txt_v, &img_v], 2)?;

    // rope_with_grad records Op::RoPePrecomputed so backward flows through Q/K.
    let q = rope_with_grad(&q, pe_cos, pe_sin)?;
    let k = rope_with_grad(&k, pe_cos, pe_sin)?;

    let attn_out = flame_core::attention::sdpa(&q, &k, &v, None)?;

    // 2026-05-09 fix: inline the narrow+permute+reshape split.
    // `attn_split_txt_img_bf16`'s "autograd-aware" branch turned out to leave
    // the dual-block LoRAs at zero (50-step smoke). Inlining the same ops at
    // the call site dodges any kernel-fallback / dispatch issue and uses the
    // autograd-clean primitives (narrow, permute, reshape all record their
    // own ops). attn_out is `[B, H, N_total, D]` BF16 from cuDNN SDPA.
    let txt_attn = attn_out
        .narrow(2, 0, n_t)?
        .permute(&[0, 2, 1, 3])?
        .contiguous()?
        .reshape(&[b, n_t, NUM_HEADS * HEAD_DIM])?;
    let img_attn = attn_out
        .narrow(2, n_t, n_img)?
        .permute(&[0, 2, 1, 3])?
        .contiguous()?
        .reshape(&[b, n_img, NUM_HEADS * HEAD_DIM])?;

    let mut img_out = linear_bias_pt(&img_attn, get_w(weights, pfx, block_idx, "attn.to_out.0.weight")?, get_w(weights, pfx, block_idx, "attn.to_out.0.bias")?, pt)?;
    img_out = add_lora_delta_double(img_out, &img_attn, bundle, weights, block_idx, DoubleLoraTarget::ImgOut)?;
    let mut txt_out = linear_bias_pt(&txt_attn, get_w(weights, pfx, block_idx, "attn.to_add_out.weight")?, get_w(weights, pfx, block_idx, "attn.to_add_out.bias")?, pt)?;
    txt_out = add_lora_delta_double(txt_out, &txt_attn, bundle, weights, block_idx, DoubleLoraTarget::TxtOut)?;

    // gate_residual_fused_bf16 records autograd (2026-04 fix); safe for training.
    let img_r = flame_core::bf16_ops::gate_residual_fused_bf16(img, &img_gate1, &img_out)?;
    let txt_r = flame_core::bf16_ops::gate_residual_fused_bf16(txt, &txt_gate1, &txt_out)?;

    // FFN img
    let img_ffn_norm = modulate_pre(&img_r, &img_shift2, &img_scale2)?;
    let mut img_ffn_h = linear_bias_pt(&img_ffn_norm, get_w(weights, pfx, block_idx, "ff.net.0.proj.weight")?, get_w(weights, pfx, block_idx, "ff.net.0.proj.bias")?, pt)?;
    img_ffn_h = add_lora_delta_double(img_ffn_h, &img_ffn_norm, bundle, weights, block_idx, DoubleLoraTarget::ImgFfnGate)?;
    let img_ffn_h = img_ffn_h.gelu()?;
    let mut img_ffn_out = linear_bias_pt(&img_ffn_h, get_w(weights, pfx, block_idx, "ff.net.2.weight")?, get_w(weights, pfx, block_idx, "ff.net.2.bias")?, pt)?;
    img_ffn_out = add_lora_delta_double(img_ffn_out, &img_ffn_h, bundle, weights, block_idx, DoubleLoraTarget::ImgFfnOut)?;
    let img_final = flame_core::bf16_ops::gate_residual_fused_bf16(&img_r, &img_gate2, &img_ffn_out)?;

    // FFN txt
    let txt_ffn_norm = modulate_pre(&txt_r, &txt_shift2, &txt_scale2)?;
    let mut txt_ffn_h = linear_bias_pt(&txt_ffn_norm, get_w(weights, pfx, block_idx, "ff_context.net.0.proj.weight")?, get_w(weights, pfx, block_idx, "ff_context.net.0.proj.bias")?, pt)?;
    txt_ffn_h = add_lora_delta_double(txt_ffn_h, &txt_ffn_norm, bundle, weights, block_idx, DoubleLoraTarget::TxtFfnGate)?;
    let txt_ffn_h = txt_ffn_h.gelu()?;
    let mut txt_ffn_out = linear_bias_pt(&txt_ffn_h, get_w(weights, pfx, block_idx, "ff_context.net.2.weight")?, get_w(weights, pfx, block_idx, "ff_context.net.2.bias")?, pt)?;
    txt_ffn_out = add_lora_delta_double(txt_ffn_out, &txt_ffn_h, bundle, weights, block_idx, DoubleLoraTarget::TxtFfnOut)?;
    let txt_final = flame_core::bf16_ops::gate_residual_fused_bf16(&txt_r, &txt_gate2, &txt_ffn_out)?;

    Ok((img_final, txt_final))
}

/// Standalone single block forward for checkpoint closures.
fn single_block_fwd(
    x: &Tensor,
    pooled_temb: &Tensor,
    pe_cos: &Tensor,
    pe_sin: &Tensor,
    block_idx: usize,
    weights: &HashMap<String, Tensor>,
    _resident: &HashMap<String, Tensor>,
    bundle: &Option<Arc<ChromaLoraBundle>>,
    _device: &Arc<cudarc::driver::CudaDevice>,
) -> Result<Tensor> {
    let pt = true;
    let pfx = "single_transformer_blocks";

    let mod_start = 3 * block_idx;
    let mods = pooled_temb.narrow(1, mod_start, 3)?;
    let shift = mods.narrow(1, 0, 1)?.squeeze(Some(1))?;
    let scale = mods.narrow(1, 1, 1)?.squeeze(Some(1))?;
    let gate = mods.narrow(1, 2, 1)?.squeeze(Some(1))?;

    let x_norm = modulate_pre(x, &shift, &scale)?;
    let b = x.shape().dims()[0];
    let seq = x.shape().dims()[1];

    let mut q = linear_bias_pt(&x_norm, get_w(weights, pfx, block_idx, "attn.to_q.weight")?, get_w(weights, pfx, block_idx, "attn.to_q.bias")?, pt)?;
    q = add_lora_delta_single(q, &x_norm, bundle, weights, block_idx, SingleLoraTarget::Q)?;
    let mut k = linear_bias_pt(&x_norm, get_w(weights, pfx, block_idx, "attn.to_k.weight")?, get_w(weights, pfx, block_idx, "attn.to_k.bias")?, pt)?;
    k = add_lora_delta_single(k, &x_norm, bundle, weights, block_idx, SingleLoraTarget::K)?;
    let mut v = linear_bias_pt(&x_norm, get_w(weights, pfx, block_idx, "attn.to_v.weight")?, get_w(weights, pfx, block_idx, "attn.to_v.bias")?, pt)?;
    v = add_lora_delta_single(v, &x_norm, bundle, weights, block_idx, SingleLoraTarget::V)?;

    let mut mlp_h = linear_bias_pt(&x_norm, get_w(weights, pfx, block_idx, "proj_mlp.weight")?, get_w(weights, pfx, block_idx, "proj_mlp.bias")?, pt)?;
    mlp_h = add_lora_delta_single(mlp_h, &x_norm, bundle, weights, block_idx, SingleLoraTarget::ProjMlp)?;
    let mlp_h = mlp_h.gelu()?;

    let q = q.reshape(&[b, seq, NUM_HEADS, HEAD_DIM])?.permute(&[0, 2, 1, 3])?;
    let k = k.reshape(&[b, seq, NUM_HEADS, HEAD_DIM])?.permute(&[0, 2, 1, 3])?;
    let v = v.reshape(&[b, seq, NUM_HEADS, HEAD_DIM])?.permute(&[0, 2, 1, 3])?;

    let q = rms_norm_head(&q, get_w(weights, pfx, block_idx, "attn.norm_q.weight")?)?;
    let k = rms_norm_head(&k, get_w(weights, pfx, block_idx, "attn.norm_k.weight")?)?;

    // rope_with_grad records Op::RoPePrecomputed — needed for Q/K LoRA backward.
    let q = rope_with_grad(&q, pe_cos, pe_sin)?;
    let k = rope_with_grad(&k, pe_cos, pe_sin)?;

    let attn_out = flame_core::attention::sdpa(&q, &k, &v, None)?;
    let attn_out = attn_out.permute(&[0, 2, 1, 3])?;
    let attn_flat = attn_out.reshape(&[b, seq, DIM])?;

    let combined = Tensor::cat(&[&attn_flat, &mlp_h], 2)?;
    let mut proj = linear_bias_pt(&combined, get_w(weights, pfx, block_idx, "proj_out.weight")?, get_w(weights, pfx, block_idx, "proj_out.bias")?, pt)?;
    proj = add_lora_delta_single(proj, &combined, bundle, weights, block_idx, SingleLoraTarget::ProjOut)?;

    // gate_residual_fused_bf16 records autograd — safe for training.
    flame_core::bf16_ops::gate_residual_fused_bf16(x, &gate, &proj)
}
