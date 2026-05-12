//! Z-Image training model — loads S3-DiT via FlameSwap, injects LoRA.
//!
//! ## Architecture (musubi-tuner `zimage_model.py`, flame-core `zimage_inference.rs`)
//!
//! ```text
//! Timestep → t_embedder → adaln_input [B, 256]
//! Image    → patchify(2×2) → x_embedder [B, N_img, 3840]
//! Text     → cap_embedder [B, N_txt, 3840]
//!
//! x → noise_refiner[0..2] (self-attn + FFN, no adaln for ctx_refiner)
//! cap → context_refiner[0..2] (self-attn + FFN, no modulation)
//!
//! unified = cat(x, cap)
//! unified → layers[0..30] (joint self-attn + SwiGLU FFN, adaln modulation)
//! unified[:N_img] → final_layer → unpatchify → output
//! ```
//!
//! ## LoRA targets (musubi-tuner `lora_zimage.py:17`)
//!
//! Target module: `ZImageTransformerBlock`
//! Exclude patterns: `.*(_modulation|_refiner).*`
//!
//! In practice this means LoRA on:
//! - `layers.{i}.attention.to_q.weight`
//! - `layers.{i}.attention.to_k.weight`
//! - `layers.{i}.attention.to_v.weight`
//! - `layers.{i}.attention.out.weight`
//! - `layers.{i}.feed_forward.w1.weight`
//! - `layers.{i}.feed_forward.w2.weight`
//! - `layers.{i}.feed_forward.w3.weight`
//!
//! NOT on noise_refiner, context_refiner, or adaLN_modulation.

use flame_core::autograd::AutogradContext;
use flame_core::{parameter::Parameter, DType, Result, Shape, Tensor, TensorId};
use crate::adapter::{AdapterModule, LycorisLinear};
use crate::lora::LoRALinear;
use crate::lycoris::{LycorisAlgo, LycorisBundleConfig};
use lycoris_rs::{
    algorithms::{full::FullAdapter, locon::LoConModule, loha::LoHaModule, lokr::LoKrModule, oft::OFTModule},
    dora::init_magnitude,
    LycorisAdapter,
};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

static BLOCK_GRAD_IDS_LOCK: OnceLock<Mutex<Vec<(String, TensorId)>>> = OnceLock::new();
pub fn block_grad_ids() -> &'static Mutex<Vec<(String, TensorId)>> {
    BLOCK_GRAD_IDS_LOCK.get_or_init(|| Mutex::new(Vec::new()))
}

pub const NUM_LAYERS: usize = 30;
pub const DIM: usize = 3840;
pub const NUM_HEADS: usize = 30;
pub const HEAD_DIM: usize = 128; // 3840 / 30
pub const MLP_HIDDEN: usize = 10240; // dim / 3 * 8
pub const IN_CHANNELS: usize = 16;
pub const PATCH_SIZE: usize = 2;
pub const CAP_FEAT_DIM: usize = 2560;
pub const ADALN_EMBED_DIM: usize = 256;
pub const NORM_EPS: f32 = 1e-5;
pub const T_SCALE: f32 = 1000.0;
pub const ROPE_THETA: f64 = 256.0;
pub const ROPE_AXES_DIMS: [usize; 3] = [32, 48, 48];
pub const ROPE_AXES_LENS: [usize; 3] = [1536, 512, 512];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LoraTarget {
    AttnQ,
    AttnK,
    AttnV,
    AttnOut,
    FfnW1,
    FfnW2,
    FfnW3,
}

/// Per-layer target shapes — single source of truth shared by `new` /
/// `new_with_config`. Same order = same HashMap iteration determinism for
/// the legacy plain-LoRA path.
const ZIMAGE_TARGETS: &[(LoraTarget, usize, usize)] = &[
    (LoraTarget::AttnQ, DIM, DIM),
    (LoraTarget::AttnK, DIM, DIM),
    (LoraTarget::AttnV, DIM, DIM),
    (LoraTarget::AttnOut, DIM, DIM),
    (LoraTarget::FfnW1, DIM, MLP_HIDDEN),
    (LoraTarget::FfnW2, MLP_HIDDEN, DIM),
    (LoraTarget::FfnW3, DIM, MLP_HIDDEN),
];

/// LoRA bundle for Z-Image. The legacy `adapters` field carries plain
/// `LoRALinear` per target — this is the byte-identical pre-Phase-2b path
/// used when `--algo lora` (or no `--algo` flag at all). When a non-`lora`
/// LyCORIS algo is selected, `lycoris_adapters` is populated instead and
/// the legacy `adapters` field is left empty; the per-block forward
/// dispatches via `adapter_for(...)` which checks `lycoris_adapters` first.
#[derive(Clone)]
pub struct ZImageLoraBundle {
    /// Legacy plain-LoRA adapters. Empty when a LyCORIS algo is active.
    pub adapters: HashMap<(usize, LoraTarget), LoRALinear>,
    /// LyCORIS adapters (LoCon/LoHa/LoKr/Full/OFT). Empty when `algo == None`.
    /// Wrapped in `Arc` so the per-adapter clone into the bundle's `Clone`
    /// impl stays cheap (refcount bump).
    pub lycoris_adapters: HashMap<(usize, LoraTarget), Arc<LycorisLinear>>,
    /// Currently active algo. `LycorisAlgo::None` = legacy `LoRALinear`.
    pub algo: LycorisAlgo,
    /// Plain-LoRA rank — kept for save/load reconstruction.
    pub rank: usize,
    pub alpha: f32,
}

impl ZImageLoraBundle {
    pub fn new(
        num_layers: usize,
        rank: usize,
        alpha: f32,
        device: Arc<cudarc::driver::CudaDevice>,
        seed: u64,
    ) -> Result<Self> {
        let mut adapters = HashMap::new();
        for i in 0..num_layers {
            for &(target, in_dim, out_dim) in ZIMAGE_TARGETS {
                let lora = LoRALinear::new(in_dim, out_dim, rank, alpha, device.clone(), seed + i as u64).map_err(|e| flame_core::FlameError::InvalidInput(format!("lora new: {e}")))?;
                adapters.insert((i, target), lora);
            }
        }
        Ok(Self {
            adapters,
            lycoris_adapters: HashMap::new(),
            algo: LycorisAlgo::None,
            rank,
            alpha,
        })
    }

    /// LyCORIS-aware constructor. `config.algo == LycorisAlgo::None` falls
    /// back to plain `LoRALinear` (legacy byte-identical path). Other algos
    /// build `LycorisLinear` per target via the matching `lycoris_rs`
    /// `*_for_training` constructor and store them in `lycoris_adapters`.
    ///
    /// `Full` and `OFT` bundle-construction succeeds, but their
    /// `forward_delta` returns an error — Z-Image's call pattern is
    /// `base + delta_on_input`, which is incompatible with Full's
    /// "weight delta merged into base" or OFT's `R·(W·x+b)` semantics.
    /// Phase 2c will wire merge-into-base for those algos.
    pub fn new_with_config(
        config: &LycorisBundleConfig,
        device: Arc<cudarc::driver::CudaDevice>,
        seed: u64,
    ) -> Result<Self> {
        if config.algo == LycorisAlgo::None {
            return Self::new(NUM_LAYERS, config.rank, config.alpha, device, seed);
        }
        let mut lycoris_adapters: HashMap<(usize, LoraTarget), Arc<LycorisLinear>> =
            HashMap::new();
        for i in 0..NUM_LAYERS {
            for &(target, in_dim, out_dim) in ZIMAGE_TARGETS {
                let wrapper = build_lycoris_linear(config, in_dim, out_dim, device.clone())?;
                lycoris_adapters.insert((i, target), Arc::new(wrapper));
            }
        }
        let _ = seed; // lycoris-rs uses its own internal RNG.
        Ok(Self {
            adapters: HashMap::new(),
            lycoris_adapters,
            algo: config.algo,
            rank: config.rank,
            alpha: config.alpha,
        })
    }

    /// SimpleTuner-style perturbed-normal init for LoKr (Phase 2c — wired).
    ///
    /// Breaks the LoKr dead-leaf at step 0: default LoKr init zeros `w2_b`
    /// so only `w2_b` receives non-zero gradient, leaving `w1` and `w2_a`
    /// frozen for many steps (catastrophic with ScheduleFree optimizers).
    /// Perturbed-normal init replaces both factors with small noise so
    /// every leaf sees gradient from step 1.
    ///
    /// `block_weights[block_idx]` is the per-block weight map populated by
    /// `ZImageModel::load`. Base key format: `layers.{block_idx}.{suffix}.weight`
    /// where `suffix` matches `ai_toolkit_suffix(target)`.
    ///
    /// Returns the number of slots whose init was skipped (missing weight,
    /// adapter declined). Logged but non-fatal.
    pub fn apply_init_perturbed_normal(
        &self,
        block_weights: &[HashMap<String, Tensor>],
        scale: f32,
    ) -> Result<usize> {
        if self.algo != LycorisAlgo::LoKr || scale <= 0.0 {
            return Ok(0);
        }
        // On-disk Z-Image base weight keys (NOT the ai-toolkit save keys):
        //   attention.qkv.weight    — fused [3*dim, dim], chunk(3, 0) into Q/K/V
        //   attention.out.weight    — output projection [dim, dim]
        //   feed_forward.w{1,2,3}.weight
        // ai-toolkit save format (`to_q`/`to_out.0`) is for the LoRA file, not the model.
        let mut applied = 0usize;
        let mut skipped = 0usize;
        let mut qkv_slice_cache: HashMap<usize, [Tensor; 3]> = HashMap::new();

        for (&(block_idx, target), adapter) in &self.lycoris_adapters {
            let Some(block_map) = block_weights.get(block_idx) else {
                log::warn!(
                    "[zimage][init_lokr_norm] block_idx={block_idx} out of range \
                     (block_weights.len()={}) — skipping",
                    block_weights.len()
                );
                skipped += 1;
                continue;
            };

            // Resolve the actual base tensor for this LoRA target.
            let base_owned: Option<Tensor> = match target {
                LoraTarget::AttnQ | LoraTarget::AttnK | LoraTarget::AttnV => {
                    // Fuse fetch once per block, slice into 3 chunks. Cached
                    // so we don't reslice for K and V after Q.
                    if !qkv_slice_cache.contains_key(&block_idx) {
                        let qkv_key = format!("layers.{block_idx}.attention.qkv.weight");
                        let Some(fused) = block_map.get(&qkv_key) else {
                            log::warn!(
                                "[zimage][init_lokr_norm] missing fused base weight \
                                 `{qkv_key}` — skipping Q/K/V for block {block_idx}"
                            );
                            skipped += 1;
                            continue;
                        };
                        // Fused shape is [3*dim, dim]; chunk along dim 0.
                        let chunks = fused.chunk(3, 0).map_err(|e| {
                            flame_core::FlameError::InvalidOperation(format!(
                                "qkv chunk({qkv_key}): {e}"
                            ))
                        })?;
                        if chunks.len() != 3 {
                            log::warn!(
                                "[zimage][init_lokr_norm] expected 3 qkv chunks from \
                                 `{qkv_key}`, got {}", chunks.len()
                            );
                            skipped += 1;
                            continue;
                        }
                        let arr: [Tensor; 3] = [
                            chunks[0].clone(),
                            chunks[1].clone(),
                            chunks[2].clone(),
                        ];
                        qkv_slice_cache.insert(block_idx, arr);
                    }
                    let slice_idx = match target {
                        LoraTarget::AttnQ => 0,
                        LoraTarget::AttnK => 1,
                        LoraTarget::AttnV => 2,
                        _ => unreachable!(),
                    };
                    qkv_slice_cache.get(&block_idx).map(|arr| arr[slice_idx].clone())
                }
                LoraTarget::AttnOut => {
                    let key = format!("layers.{block_idx}.attention.out.weight");
                    block_map.get(&key).cloned()
                }
                LoraTarget::FfnW1 | LoraTarget::FfnW2 | LoraTarget::FfnW3 => {
                    let leaf = match target {
                        LoraTarget::FfnW1 => "feed_forward.w1",
                        LoraTarget::FfnW2 => "feed_forward.w2",
                        LoraTarget::FfnW3 => "feed_forward.w3",
                        _ => unreachable!(),
                    };
                    let key = format!("layers.{block_idx}.{leaf}.weight");
                    block_map.get(&key).cloned()
                }
            };

            let Some(base) = base_owned else {
                log::warn!(
                    "[zimage][init_lokr_norm] base lookup failed for \
                     (block={block_idx}, target={:?}) — skipping", target
                );
                skipped += 1;
                continue;
            };

            let did = adapter.as_ref().init_perturbed_normal_lokr(&base, scale)
                .map_err(|e| flame_core::FlameError::InvalidOperation(format!(
                    "init_perturbed_normal_lokr(block={block_idx}, target={:?}): {e}",
                    target
                )))?;
            if did { applied += 1; } else { skipped += 1; }
        }
        log::info!(
            "[zimage][init_lokr_norm] applied={applied} skipped={skipped} scale={scale}"
        );
        Ok(skipped)
    }

    /// Adapter accessor used by the per-block forward closures. Checks
    /// `lycoris_adapters` first (Phase 2b active path) then falls back to
    /// the legacy `adapters` map. Returns `None` when no adapter exists for
    /// `(block_idx, target)` (shouldn't happen on a fully-populated bundle).
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

    pub fn num_adapters(&self) -> usize {
        self.adapters.len() + self.lycoris_adapters.len()
    }

    pub fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        for lora in self.adapters.values() {
            params.extend(lora.parameters());
        }
        for adapter in self.lycoris_adapters.values() {
            params.extend(adapter.to_parameters());
        }
        params
    }

    /// Canonical (name, Parameter) pairs in deterministic order. Names match
    /// what `save()` writes to disk so a saved checkpoint round-trips
    /// optimizer state by name across runs (TensorIds are unstable, names
    /// are stable). Order: (block_idx, target_idx) ascending, then A then B.
    pub fn named_parameters(&self) -> Vec<(String, Parameter)> {
        // Legacy plain-LoRA path.
        let mut keys: Vec<&(usize, LoraTarget)> = self.adapters.keys().collect();
        keys.sort_by_key(|(b, t)| (*b, *t as usize));
        let mut out: Vec<(String, Parameter)> =
            Vec::with_capacity((self.adapters.len() + self.lycoris_adapters.len()) * 2);
        for &(block_idx, target) in keys {
            let lora = &self.adapters[&(block_idx, target)];
            let suffix = Self::ai_toolkit_suffix(target);
            let prefix = format!("diffusion_model.layers.{block_idx}.{suffix}");
            out.push((format!("{prefix}.lora_A.weight"), lora.lora_a().clone()));
            out.push((format!("{prefix}.lora_B.weight"), lora.lora_b().clone()));
        }

        // LyCORIS path. Each adapter's `to_parameters()` order matches its
        // `named_tensors()` so the per-leaf zip is stable. Suffix uses
        // `lycoris-upstream` conventions (`lora_down.weight` /
        // `lora_up.weight` for LoCon, `hada_*` for LoHa, `lokr_*` for LoKr,
        // `dora_scale` last when DoRA is on — see
        // `LycorisLinear::named_tensors`). Without this branch a `--algo
        // lokr` (or any non-`lora` algo) save silently writes 0 keys.
        let mut lyc_keys: Vec<&(usize, LoraTarget)> =
            self.lycoris_adapters.keys().collect();
        lyc_keys.sort_by_key(|(b, t)| (*b, *t as usize));
        for &(block_idx, target) in lyc_keys {
            let adapter = &self.lycoris_adapters[&(block_idx, target)];
            let suffix = Self::ai_toolkit_suffix(target);
            let prefix = format!("diffusion_model.layers.{block_idx}.{suffix}");
            let params = adapter.to_parameters();
            let named = adapter.named_tensors();
            if params.len() != named.len() {
                log::warn!(
                    "ZImageLoraBundle::named_parameters: adapter at \
                     ({block_idx}, {:?}) has {} params but {} named entries; \
                     skipping (lycoris-rs/AdapterModule contract bug)",
                    target,
                    params.len(),
                    named.len(),
                );
                continue;
            }
            for ((leaf, _), p) in named.into_iter().zip(params.into_iter()) {
                out.push((format!("{prefix}.{leaf}"), p));
            }
        }
        out
    }

    pub fn refresh_caches(&self) {
        for lora in self.adapters.values() {
            lora.refresh_cache();
        }
    }

    /// Per-target ai-toolkit module suffix. Matches the key naming in
    /// `/home/alex/ai-toolkit/output/zimage/zimage.safetensors`:
    /// `attention.to_{q,k,v}`, `attention.to_out.0` (ModuleList index),
    /// `feed_forward.w{1,2,3}`.
    fn ai_toolkit_suffix(target: LoraTarget) -> &'static str {
        match target {
            LoraTarget::AttnQ => "attention.to_q",
            LoraTarget::AttnK => "attention.to_k",
            LoraTarget::AttnV => "attention.to_v",
            LoraTarget::AttnOut => "attention.to_out.0",
            LoraTarget::FfnW1 => "feed_forward.w1",
            LoraTarget::FfnW2 => "feed_forward.w2",
            LoraTarget::FfnW3 => "feed_forward.w3",
        }
    }

    /// Save in ai-toolkit's PEFT-style format:
    ///   `diffusion_model.layers.{i}.{module}.lora_{A,B}.weight`
    /// Verified against `ai-toolkit/output/zimage/zimage.safetensors` — same
    /// prefix, same dotted path, same suffix, no `.alpha`. Per-module alpha
    /// is implicitly `alpha=rank` → `scale=1.0` (matches `LoraStack::load`'s
    /// fallback in `inference-flame/src/lora.rs:255`).
    pub fn save(&self, path: &std::path::Path) -> Result<()> {
        let mut tensors = HashMap::new();
        // Legacy plain-LoRA path (used when --algo lora). Empty when a
        // LyCORIS algo is active.
        for (&(block_idx, target), lora) in &self.adapters {
            let suffix = Self::ai_toolkit_suffix(target);
            let prefix = format!("diffusion_model.layers.{block_idx}.{suffix}");
            // ai-toolkit uses `.lora_A.weight` / `.lora_B.weight`.
            let lora_a = lora.lora_a().tensor()?;
            let lora_b = lora.lora_b().tensor()?;
            tensors.insert(format!("{prefix}.lora_A.weight"), lora_a);
            tensors.insert(format!("{prefix}.lora_B.weight"), lora_b);
        }
        // LyCORIS path — must save when --algo lokr/loha/locon/... is active,
        // otherwise the ckpt is just an empty `{}` safetensors file (10 bytes).
        // Each adapter contributes its own per-leaf tensors via
        // `named_tensors()`: LoKr → (lokr_w1, lokr_w2 or lokr_w2_a/b),
        // LoCon → (lora_down/up), LoHa → (hada_*), etc.
        for (&(block_idx, target), adapter) in &self.lycoris_adapters {
            let suffix = Self::ai_toolkit_suffix(target);
            let prefix = format!("diffusion_model.layers.{block_idx}.{suffix}");
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

    /// Load ai-toolkit-format Z-Image LoRA. Accepts either the new format
    /// (`diffusion_model.<...>.lora_A.weight`) or the legacy trainer format
    /// (`layers.{i}.<...>.lora_A`) for back-compat with previously-saved
    /// checkpoints. `attention.out` (legacy) is also accepted as an alias
    /// for `attention.to_out.0` (ai-toolkit) on load.
    pub fn load(&self, path: &std::path::Path, device: &Arc<cudarc::driver::CudaDevice>) -> Result<()> {
        let tensors = flame_core::serialization::load_file(path, device)?;
        for (&(block_idx, target), lora) in &self.adapters {
            let suffix = Self::ai_toolkit_suffix(target);
            // Try ai-toolkit format first.
            let new_prefix = format!("diffusion_model.layers.{block_idx}.{suffix}");
            let a_key = format!("{new_prefix}.lora_A.weight");
            let b_key = format!("{new_prefix}.lora_B.weight");
            if tensors.contains_key(&a_key) && tensors.contains_key(&b_key) {
                lora.load_tensors(&new_prefix, &tensors_with_weight_alias(&tensors, &new_prefix)).map_err(|e| flame_core::FlameError::InvalidInput(format!("lora load: {e}")))?;
                continue;
            }
            // Legacy trainer format: `layers.{i}.<legacy_suffix>.lora_A` / `.lora_B`.
            let legacy_suffix = match target {
                LoraTarget::AttnOut => "attention.out",
                _ => suffix,
            };
            let legacy_prefix = format!("layers.{block_idx}.{legacy_suffix}");
            lora.load_tensors(&legacy_prefix, &tensors).map_err(|e| flame_core::FlameError::InvalidInput(format!("lora load: {e}")))?;
        }
        for lora in self.adapters.values() {
            lora.refresh_cache();
        }
        Ok(())
    }
}

/// Helper for `load`: synthesizes the legacy `<prefix>.lora_A`/`.lora_B`
/// keys from the ai-toolkit `<prefix>.lora_A.weight`/`.lora_B.weight`
/// keys so `LoRALinear::load_tensors` (which expects bare suffixes) can
/// consume them. Returns a map containing only the two synthesized
/// entries — `load_tensors` only reads those exact keys for the given
/// prefix.
fn tensors_with_weight_alias(
    src: &HashMap<String, Tensor>,
    prefix: &str,
) -> HashMap<String, Tensor> {
    let mut out = HashMap::new();
    let a_key_aitk = format!("{prefix}.lora_A.weight");
    let b_key_aitk = format!("{prefix}.lora_B.weight");
    if let Some(t) = src.get(&a_key_aitk) {
        out.insert(format!("{prefix}.lora_A"), t.clone());
    }
    if let Some(t) = src.get(&b_key_aitk) {
        out.insert(format!("{prefix}.lora_B"), t.clone());
    }
    out
}

/// Build a single `LycorisLinear` for the configured algo. For
/// `LycorisAlgo::None` the caller (`ZImageLoraBundle::new_with_config`)
/// short-circuits to `LoRALinear` instead — this helper bails on `None`
/// defensively. Mirrors `chroma::build_lycoris_linear`.
fn build_lycoris_linear(
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
                Shape::from_dims(&[out_features, in_features]),
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

    // DoRA: Z-Image's bundle ctor doesn't have access to base weights here
    // (block weights are loaded into `ZImageModel::block_weights` after
    // bundle construction). Initialize magnitude to ||W=I||_2 = 1 along the
    // chosen axis as an approximation — lycoris-upstream wants ||W_orig||_2.
    // The trainer's first few hundred steps adjust it; Phase 2c will plumb
    // the resident weight map for a proper init.
    let dora_magnitude = if config.dora {
        if config.algo == LycorisAlgo::Oft {
            return Err(flame_core::Error::InvalidInput(
                "DoRA + OFT is not supported (multiplicative + decomposition conflict)".into(),
            ));
        }
        let shape = if config.dora_wd_on_out {
            Shape::from_dims(&[out_features, 1])
        } else {
            Shape::from_dims(&[1, in_features])
        };
        let ones = Tensor::from_vec(
            vec![1.0_f32; shape.elem_count()],
            shape,
            device.clone(),
        )?;
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

pub struct ZImageModel {
    pub model_path: std::path::PathBuf,
    pub bundle: ZImageLoraBundle,
    /// Optional LyCORIS bundle (Phase 2b wiring). When `Some`, takes priority
    /// over the legacy `bundle` per-target lookup in the per-block forward.
    /// When `None` (default), the legacy `LoRALinear` path is byte-identical
    /// to the pre-Phase-2b behaviour.
    pub lycoris: Option<Arc<crate::lycoris::LycorisBundle>>,
    pub num_blocks: usize,
    resident_weights: HashMap<String, Tensor>,
    block_weights: Vec<HashMap<String, Tensor>>,
    device: Arc<cudarc::driver::CudaDevice>,
    /// Full fine-tune mode: ALL model weights are trainable Parameters.
    is_full_finetune: bool,
    /// FFT Parameters (F32, requires_grad=true) for the optimizer.
    fft_params: Option<HashMap<String, Parameter>>,
}

impl ZImageModel {
    pub fn load(
        model_path: &std::path::Path,
        lora_rank: usize,
        lora_alpha: f32,
        device: Arc<cudarc::driver::CudaDevice>,
        seed: u64,
    ) -> Result<Self> {
        Self::load_with_mode(model_path, lora_rank, lora_alpha, device, seed, false)
    }

    pub fn load_with_mode(
        model_path: &std::path::Path,
        lora_rank: usize,
        lora_alpha: f32,
        device: Arc<cudarc::driver::CudaDevice>,
        seed: u64,
        full_finetune: bool,
    ) -> Result<Self> {
        log::info!("[zimage-trainer] loading Z-Image from {}", model_path.display());

        // Load ALL weights at once (Z-Image is ~12 GB BF16, fits in 24 GB VRAM)
        let all_weights = flame_core::serialization::load_file(model_path, &device)?;
        log::info!("[zimage-trainer] loaded {} weight tensors", all_weights.len());

        // Separate resident (shared) vs block weights
        let mut resident = HashMap::new();
        let mut per_block: Vec<HashMap<String, Tensor>> = (0..NUM_LAYERS).map(|_| HashMap::new()).collect();

        for (key, tensor) in &all_weights {
            let is_block = key.starts_with("layers.");
            if is_block {
                // Parse block index: "layers.{i}...."
                if let Some(rest) = key.strip_prefix("layers.") {
                    if let Some(dot_pos) = rest.find('.') {
                        if let Ok(idx) = rest[..dot_pos].parse::<usize>() {
                            if idx < NUM_LAYERS {
                                per_block[idx].insert(key.clone(), tensor.clone());
                            }
                        }
                    }
                }
            } else {
                resident.insert(key.clone(), tensor.clone());
            }
        }
        drop(all_weights);

        log::info!("[zimage-trainer] {} resident keys, {} block maps", resident.len(), per_block.len());

        let bundle = ZImageLoraBundle::new(NUM_LAYERS, lora_rank, lora_alpha, device.clone(), seed)?;
        if !full_finetune {
            log::info!("[zimage-trainer] {} LoRA adapters (rank={})", bundle.num_adapters(), lora_rank);
        }

        // FFT mode: build F32 master Parameters. Forward casts each F32
        // Parameter to BF16 on the fly via `w()` / `bw()` / `fft_block_cast`
        // with autograd recording, so backward flows through Cast → F32
        // Parameter and the optimizer sees real gradients. Previous
        // pattern stored BF16 copies with `requires_grad_(true)` at
        // fresh TensorIds disconnected from the F32 Parameters — that
        // silently trained with zero updates.
        let fft_params = if full_finetune {
            let mut params = HashMap::new();
            for (key, tensor) in &resident {
                let t = tensor.to_dtype(flame_core::DType::F32)?.requires_grad_(true);
                params.insert(key.clone(), Parameter::new(t));
            }
            for (_block_idx, block_map) in per_block.iter().enumerate() {
                for (key, tensor) in block_map {
                    let t = tensor.to_dtype(flame_core::DType::F32)?.requires_grad_(true);
                    params.insert(key.clone(), Parameter::new(t));
                }
            }
            log::info!("[zimage-trainer] FFT mode: {} trainable parameters", params.len());
            Some(params)
        } else {
            None
        };

        Ok(Self {
            model_path: model_path.to_path_buf(),
            bundle,
            lycoris: None,
            num_blocks: NUM_LAYERS,
            resident_weights: resident,
            block_weights: per_block,
            device,
            is_full_finetune: full_finetune,
            fft_params,
        })
    }

    pub fn parameters(&self) -> Vec<Parameter> {
        if self.is_full_finetune {
            self.fft_parameters()
        } else {
            self.bundle.parameters()
        }
    }

    /// Drop all per-block weights from GPU. Frees ~12GB for things that need
    /// big contiguous allocations (e.g. 1024² VAE decode conv workspace).
    /// Must be paired with `reload_blocks` before the next forward pass.
    /// LoRA-mode only — FFT mode keeps weights in `fft_params` and would
    /// need a separate offload path.
    pub fn unload_blocks(&mut self) {
        if self.is_full_finetune {
            return;
        }
        for block in self.block_weights.iter_mut() {
            block.clear();
        }
        flame_core::cuda_alloc_pool::clear_pool_cache();
        flame_core::trim_cuda_mempool(0);
    }

    /// Reload per-block weights from disk via mmap. Pair with `unload_blocks`.
    pub fn reload_blocks(&mut self) -> Result<()> {
        if self.is_full_finetune {
            return Ok(());
        }
        let block_data = flame_core::serialization::load_file_filtered(
            &self.model_path,
            &self.device,
            |key| key.starts_with("layers."),
        )?;
        for block in self.block_weights.iter_mut() {
            block.clear();
        }
        for (key, tensor) in block_data {
            if let Some(rest) = key.strip_prefix("layers.") {
                if let Some(dot_pos) = rest.find('.') {
                    if let Ok(idx) = rest[..dot_pos].parse::<usize>() {
                        if idx < self.block_weights.len() {
                            self.block_weights[idx].insert(key, tensor);
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Return all model weight Parameters (FFT mode only).
    pub fn fft_parameters(&self) -> Vec<Parameter> {
        match &self.fft_params {
            Some(map) => map.values().cloned().collect(),
            None => Vec::new(),
        }
    }

    pub fn is_full_finetune(&self) -> bool {
        self.is_full_finetune
    }

    /// Per-block resident weight maps. Indexed by `block_idx`. Used by
    /// `ZImageLoraBundle::apply_init_perturbed_normal` to look up base
    /// weights for SimpleTuner-style LoKr init. LoRA mode only — in FFT
    /// mode the block weights live in `fft_params` as F32 Parameters.
    pub fn block_weights(&self) -> &[HashMap<String, Tensor>] {
        &self.block_weights
    }

    /// Save weights: FFT saves ALL model weights as BF16 safetensors,
    /// LoRA saves only the adapter deltas.
    pub fn save_weights(&self, path: &std::path::Path) -> Result<()> {
        if self.is_full_finetune {
            self.save_fft_weights(path)
        } else {
            self.bundle.save(path)
        }
    }

    /// Save all model weights as BF16 safetensors (FFT mode).
    fn save_fft_weights(&self, path: &std::path::Path) -> Result<()> {
        let fft = self.fft_params.as_ref().ok_or_else(|| {
            flame_core::Error::InvalidInput("save_fft_weights called but fft_params is None".into())
        })?;
        let mut output = HashMap::new();
        for (key, param) in fft {
            let tensor = param.tensor()?.to_dtype(DType::BF16)?;
            output.insert(key.clone(), tensor);
        }
        flame_core::serialization::save_tensors(
            &output,
            path,
            flame_core::serialization::SerializationFormat::SafeTensors,
        )
    }

    pub fn refresh_lora_cache(&self) {
        self.bundle.refresh_caches();
    }

    /// Resident (non-block) weight lookup. In FFT mode, casts the F32
    /// Parameter to BF16 on the fly with autograd recording so backward
    /// flows through Cast → Parameter. In LoRA mode, returns a cheap
    /// clone of the frozen BF16 tensor.
    fn w(&self, key: &str) -> Result<Tensor> {
        if let Some(ref params) = self.fft_params {
            if let Some(p) = params.get(key) {
                return p.tensor()?.to_dtype(DType::BF16);
            }
        }
        self.resident_weights.get(key)
            .cloned()
            .ok_or_else(|| flame_core::Error::InvalidInput(format!("missing weight: {key}")))
    }

    /// Per-block weight lookup. Same FFT-aware dispatch as `w()`.
    fn bw(&self, block_idx: usize, key: &str) -> Result<Tensor> {
        let full_key = format!("layers.{block_idx}.{key}");
        if let Some(ref params) = self.fft_params {
            if let Some(p) = params.get(&full_key) {
                return p.tensor()?.to_dtype(DType::BF16);
            }
        }
        self.block_weights[block_idx].get(&full_key)
            .cloned()
            .ok_or_else(|| flame_core::Error::InvalidInput(format!("missing block weight: {full_key}")))
    }

    /// Build the BF16-cast weight map for block `idx`, used to feed
    /// `block_forward_standalone` inside the checkpoint closure. In
    /// FFT mode casts each matching F32 Parameter to BF16 with autograd
    /// recording; in LoRA mode returns a plain clone of the frozen map.
    fn fft_block_cast(&self, idx: usize) -> Result<HashMap<String, Tensor>> {
        if let Some(ref params) = self.fft_params {
            let prefix = format!("layers.{idx}.");
            let mut out: HashMap<String, Tensor> = HashMap::new();
            for (key, p) in params {
                if key.starts_with(&prefix) {
                    let bf16 = p.tensor()?.to_dtype(DType::BF16)?;
                    out.insert(key.clone(), bf16);
                }
            }
            Ok(out)
        } else {
            Ok(self.block_weights[idx].clone())
        }
    }

    /// Full training forward.
    ///
    /// `x`: [B, 16, H, W] BF16 (noisy latent, already scaled)
    /// `t`: [B] BF16 (timestep in [0, 1], already inverted: t = 1 - sigma)
    /// `cap_feats`: [B, seq, 2560] BF16 (Qwen3 embeddings)
    /// `cap_mask`: optional [B, seq] (True for valid tokens)
    ///
    /// Returns: [B, 16, H, W] BF16 (predicted velocity, sign-inverted by caller if needed)
    pub fn forward(
        &self,
        x: &Tensor,
        t: &Tensor,
        cap_feats: &Tensor,
        cap_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let (b, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);

        // Timestep embedding: t * T_SCALE → sinusoidal → MLP
        let t_scaled = t.mul_scalar(T_SCALE)?;
        let adaln_input = self.timestep_embed(&t_scaled)?;

        // Patchify: [B, 16, H, W] → [B, (H/2)*(W/2), 16*2*2=64]
        let h_tok = h / PATCH_SIZE;
        let w_tok = w / PATCH_SIZE;
        let n_img = h_tok * w_tok;
        let patch_dim = c * PATCH_SIZE * PATCH_SIZE;
        let x_patched = self.patchify(x, h, w)?;
        // x_embedder: Linear(64, 3840)
        let mut x_emb = self.linear_bias(&x_patched, "x_embedder.weight", "x_embedder.bias")?;
        if let Some(ref d) = std::env::var("FLAME_ZIMAGE_DUMP_DIR").ok().map(std::path::PathBuf::from) {
            zimage_dump(&d, "x_patched", &x_patched)?;
        }

        let adaln_input = adaln_input.to_dtype(DType::BF16)?;

        // FLAME_ZIMAGE_DUMP_DIR: dump per-block forward outputs for
        // musubi parity checks. Matches the key names used by
        // `output/zimage_parity_dump_musubi.py`.
        let dump_dir: Option<std::path::PathBuf> = std::env::var("FLAME_ZIMAGE_DUMP_DIR")
            .ok().map(std::path::PathBuf::from);
        if let Some(ref d) = dump_dir {
            std::fs::create_dir_all(d).ok();
            zimage_dump(d, "t_emb", &adaln_input)?;
            zimage_dump(d, "x_embed", &x_emb)?;
        }

        // Build RoPE tables for image + caption
        let cap_seq = cap_feats.shape().dims()[1];
        let (img_cos, img_sin) = self.build_rope_image(h_tok, w_tok, cap_seq)?;
        let (cap_cos, cap_sin) = self.build_rope_caption(cap_seq)?;

        // Debug: dump RoPE values
        {
        }
        // Debug: dump after patchify+embed
        {
        }

        // Noise refiner (2 blocks, no LoRA, with modulation)
        for ri in 0..2 {
            x_emb = self.refiner_block(&x_emb, &img_cos, &img_sin, Some(&adaln_input), &format!("noise_refiner.{ri}"))?;
            if let Some(ref d) = dump_dir { zimage_dump(d, &format!("noise_refiner_{ri}"), &x_emb)?; }
        }

        // Caption embedding: RMSNorm(.0) → Linear(.1).
        // BUG FIX (was LayerNorm): musubi `zimage_model.py:424-425` and the
        // official Tongyi-MAI/Z-Image both use RMSNorm here. Using LayerNorm
        // mean-centers the cap features, shifting them by their per-token
        // mean — wrong. Caused garbage at step 0 of in-process sampling
        // (no LoRA contribution = the bug isn't masked by adapter delta).
        // F32-internal primitive RMSNorm. Same fix as everywhere else —
        // fused kernel backward has BF16-accumulation precision bug.
        let cap_norm = primitive_rms_norm(cap_feats, &self.w("cap_embedder.0.weight")?, NORM_EPS)?;
        let mut cap_emb = self.linear_bias(&cap_norm, "cap_embedder.1.weight", "cap_embedder.1.bias")?;
        if let Some(ref d) = dump_dir { zimage_dump(d, "cap_embed", &cap_emb)?; }

        log::debug!("cap_emb after embedder: {:?}", cap_emb.shape().dims());

        // Replace pad-position embeddings with the learned `cap_pad_token`,
        // matching musubi-tuner `zimage_model.py:677-679` and the inference
        // NextDiT `pad_to_multiple` path. Without this, training feeds the
        // model real Qwen3 outputs for the PAD_TOKEN_ID at every position
        // past the prompt length — uninformative noise that the LoRA must
        // attend to alongside the actual content. That dilutes the
        // identity-learning gradient and was producing zero subject
        // convergence even after 1500 steps with healthy LoRA grad flow.
        //
        // Formula (musubi):
        //   cap_pad_mask = ~cap_mask                                  # 1.0 at pad
        //   cap_feats[pad] = 0; cap_feats += cap_pad_token * cap_pad_mask
        // ⇒ cap_feats = cap_feats * cap_mask + cap_pad_token * (1 - cap_mask)
        if let Some(m) = cap_mask {
            let cap_pad_token = self.w("cap_pad_token")?; // [1, DIM]
            let pad_3d = cap_pad_token.reshape(&[1, 1, DIM])?;
            // m is [B, seq] BF16 — broadcast-mul against [B, seq, DIM].
            let m_3d = m.unsqueeze(2)?; // [B, seq, 1]
            let inv_3d = m_3d.mul_scalar(-1.0)?.add_scalar(1.0)?; // [B, seq, 1] = (1 - mask)
            let kept = cap_emb.mul(&m_3d)?;          // [B, seq, DIM]
            let pad_contrib = pad_3d.mul(&inv_3d)?;  // [B, seq, DIM] (broadcast)
            cap_emb = kept.add(&pad_contrib)?;
            log::debug!("applied cap_pad_token at {} pad positions", m.shape().dims()[1]);
        }

        // Context refiner (2 blocks, no LoRA, no modulation)
        for ri in 0..2 {
            cap_emb = self.refiner_block(&cap_emb, &cap_cos, &cap_sin, None, &format!("context_refiner.{ri}"))?;
            if let Some(ref d) = dump_dir { zimage_dump(d, &format!("context_refiner_{ri}"), &cap_emb)?; }
            log::debug!("cap_emb after context_refiner.{}: {:?}", ri, cap_emb.shape().dims());
        }

        // Debug: dump pre-concat stats
        {
        }

        // Concatenate: [B, N_img + N_cap, dim] — image first, caption
        // second. Matches musubi's `torch.cat([x, cap_feats], dim=1)` at
        // `musubi_tuner/zimage/zimage_model.py:696`. The old order
        // (`[cap, img]`) disagreed with the module doc comment and with
        // musubi, so the final-layer narrow pulled the wrong token range
        // during training.
        let unified = Tensor::cat(&[&x_emb, &cap_emb], 1)?;
        let unified_cos = Tensor::cat(&[&img_cos, &cap_cos], 1)?;
        let unified_sin = Tensor::cat(&[&img_sin, &cap_sin], 1)?;
        if let Some(ref d) = dump_dir {
            zimage_dump(d, "unified", &unified)?;
            zimage_dump(d, "unified_cos", &unified_cos)?;
            zimage_dump(d, "unified_sin", &unified_sin)?;
        }

        // Main transformer layers (30 blocks, with LoRA + adaln)
        let n_blocks = std::env::var("ZIMAGE_MAX_BLOCKS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .map(|n| n.min(NUM_LAYERS))
            .unwrap_or(NUM_LAYERS);
        if n_blocks != NUM_LAYERS {
            log::warn!("[zimage-trainer] ZIMAGE_MAX_BLOCKS={n_blocks} (debug — forward stops early)");
        }
        let use_checkpoint = std::env::var("FLAME_CHECKPOINT")
            .ok().map(|v| v != "0").unwrap_or(true);

        // When autograd is not recording (inference / sample_image call) we
        // bypass the checkpoint machinery entirely and use the fused-kernel
        // inference path.  Training keeps the existing checkpoint path so
        // activation-offload, gradient bookkeeping, and LoRA training are
        // byte-identical to the pre-port behavior.
        let is_inference = !flame_core::autograd::AutogradContext::is_recording();

        let mut h_state = unified;
        // FLAME_RETAIN_BLOCK_GRADS=1 records block-boundary tensor IDs to
        // a global so callers can call retain_intermediate_grads on them.
        let retain_block_grads = std::env::var("FLAME_RETAIN_BLOCK_GRADS")
            .ok().as_deref() == Some("1");
        if retain_block_grads {
            block_grad_ids().lock().unwrap().clear();
            block_grad_ids().lock().unwrap().push(("unified".to_string(), h_state.id()));
        }
        for i in 0..n_blocks {
            if is_inference {
                // Inference fast path — fused kernels, no checkpoint overhead.
                // `block_forward_iflame` mirrors inference-flame's
                // `transformer_block`: fused_rms_norm, rope_fused_bf16,
                // swiglu_fused_bf16, gate_residual_fused_bf16, linear_3d.
                let block_w = self.fft_block_cast(i)?;
                self.bundle.refresh_caches();
                h_state = block_forward_iflame(
                    &h_state, &unified_cos, &unified_sin, &adaln_input, i,
                    &self.bundle, None, &block_w,
                )?;
            } else if use_checkpoint {
                let h_c = h_state.clone();
                let cos_c = unified_cos.clone();
                let sin_c = unified_sin.clone();
                let adaln_c = adaln_input.clone();
                let bundle_c = self.bundle.clone();
                // In FFT mode, `fft_block_cast` returns a BF16 HashMap
                // produced by autograd-recording F32→BF16 casts of the
                // block's F32 Parameters, so backward flows through
                // Cast → Parameter. In LoRA mode it's a plain clone.
                let block_w_c = self.fft_block_cast(i)?;
                // 2026-05-12: migrated from `checkpoint` to `checkpoint_offload`
                // (Stage 1, plan keen-crafting-jellyfish). Signatures are
                // identical; `checkpoint_offload` falls back to `checkpoint`
                // when no ActivationOffloadPool is installed (autograd.rs:
                // 1876-1879), so this is byte-equivalent without the pool.
                // With the pool installed by `train_zimage.rs`, saved
                // activations live on CPU instead of being recomputed,
                // eliminating ~1.5 s/step of recompute overhead per block.
                h_state = AutogradContext::checkpoint_offload(
                    &[h_c.clone()],
                    move || {
                        bundle_c.refresh_caches();
                        block_forward_standalone(
                            &h_c, &cos_c, &sin_c, &adaln_c, i,
                            &bundle_c, None, &block_w_c,
                        )
                    },
                )?;
                if retain_block_grads {
                    block_grad_ids().lock().unwrap().push((format!("block_{i}_out"), h_state.id()));
                }
            } else {
                // Use the same standalone forward as the checkpoint path so
                // sample-time output exactly matches what the LoRA was trained
                // against. The previous `transformer_block_with_lora` path
                // diverged in subtle ways from `block_forward_standalone`
                // (different rms_norm helpers, different chunk-then-reshape
                // order) and produced un-converged-looking samples even when
                // the saved LoRA rendered correctly through `zimage_lora_infer`.
                let block_w = self.fft_block_cast(i)?;
                self.bundle.refresh_caches();
                h_state = block_forward_standalone(
                    &h_state, &unified_cos, &unified_sin, &adaln_input, i,
                    &self.bundle, None, &block_w,
                )?;
            }
            if let Some(ref d) = dump_dir {
                zimage_dump(d, &format!("layer_{i:02}"), &h_state)?;
            }
        }

        // Take only image tokens. With `[img, cap]` concat order, image
        // tokens occupy positions 0..n_img.
        let _ = cap_seq;
        let img_out = h_state.narrow(1, 0, n_img)?;
        if retain_block_grads {
            block_grad_ids().lock().unwrap().push(("img_out".to_string(), img_out.id()));
        }

        // Final layer: adaln modulation + layer norm + linear
        let final_out = self.final_layer(&img_out, &adaln_input)?;
        if retain_block_grads {
            block_grad_ids().lock().unwrap().push(("final_out".to_string(), final_out.id()));
        }
        if let Some(ref d) = dump_dir { zimage_dump(d, "final_out", &final_out)?; }

        // Unpatchify: [B, N_img, patch_dim] → [B, C, H, W]
        let pred = self.unpatchify(&final_out, h_tok, w_tok)?;
        if retain_block_grads {
            block_grad_ids().lock().unwrap().push(("pred".to_string(), pred.id()));
        }
        if let Some(ref d) = dump_dir { zimage_dump(d, "pred", &pred)?; }
        Ok(pred)
    }

    fn timestep_embed(&self, t: &Tensor) -> Result<Tensor> {
        // musubi-tuner zimage_model.py: TimestepEmbedder
        // Sinusoidal embedding (256 channels) → MLP(256 → 1024 → 256)
        let freq_embed = sinusoidal_embedding(t, ADALN_EMBED_DIM)?;
        let h = self.linear_bias(&freq_embed, "t_embedder.mlp.0.weight", "t_embedder.mlp.0.bias")?;
        let h = h.silu()?;
        self.linear_bias(&h, "t_embedder.mlp.2.weight", "t_embedder.mlp.2.bias")
    }

    fn linear_no_bias(&self, x: &Tensor, weight_key: &str) -> Result<Tensor> {
        // `fused_linear3d_native` now records `Op::Linear` when called
        // inside `AutogradContext::is_recording()` (flame-core fix
        // 2026-04-28). Safe to use on the training path; perf benefit
        // of the fused cuBLASLt kernel is preserved.
        let weight = self.w(weight_key)?;
        flame_core::ops::fused_inference::fused_linear3d_native(
            &ensure_3d(x)?, &weight, None,
        ).map(|t| squeeze_if_needed(t, x.shape().dims().len()))
    }

    fn linear_bias(&self, x: &Tensor, weight_key: &str, bias_key: &str) -> Result<Tensor> {
        let weight = self.w(weight_key)?;
        let bias = self.w(bias_key)?;
        flame_core::ops::fused_inference::fused_linear3d_native(
            &ensure_3d(x)?, &weight, Some(&bias),
        ).map(|t| squeeze_if_needed(t, x.shape().dims().len()))
    }

    /// Block linear using autograd-aware matmul so LoRA gradients flow.
    /// weight is [out, in] (PyTorch layout) — we transpose and matmul.
    fn block_linear_no_bias(&self, x: &Tensor, block_idx: usize, suffix: &str) -> Result<Tensor> {
        let weight = self.bw(block_idx, suffix)?;
        let dims = x.shape().dims().to_vec();
        let in_feat = *dims.last().unwrap();
        let batch: usize = dims[..dims.len()-1].iter().product();
        let out_feat = weight.shape().dims()[0];
        let x_2d = x.reshape(&[batch, in_feat])?;
        let wt = weight.transpose()?;
        let out_2d = x_2d.matmul(&wt)?;
        let mut out_shape = dims[..dims.len()-1].to_vec();
        out_shape.push(out_feat);
        out_2d.reshape(&out_shape)
    }

    fn block_linear_bias(&self, x: &Tensor, block_idx: usize, suffix_w: &str, suffix_b: &str) -> Result<Tensor> {
        let out = self.block_linear_no_bias(x, block_idx, suffix_w)?;
        let bias = self.bw(block_idx, suffix_b)?;
        out.add(&bias)
    }

    fn add_lora_delta(&self, base: Tensor, input: &Tensor, block_idx: usize, target: LoraTarget) -> Result<Tensor> {
        // Dispatch via `adapter_for` so `--algo locon`/etc populate
        // `lycoris_adapters` and the legacy `adapters` map can be empty —
        // fixes the silent-skip pitfall that would otherwise train the
        // LoRA to noise.
        if let Some(lora) = self.bundle.adapter_for(block_idx, target) {
            let delta = lora.forward_delta(&ensure_3d(input)?).map_err(|e| flame_core::FlameError::InvalidInput(format!("lora delta: {e}")))?;
            base.add(&delta)
        } else {
            Ok(base)
        }
    }

    fn refiner_block(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        adaln_input: Option<&Tensor>,
        prefix: &str,
    ) -> Result<Tensor> {
        // musubi-tuner zimage_model.py:260-287
        let has_mod = adaln_input.is_some()
            && self.resident_weights.contains_key(&format!("{prefix}.adaLN_modulation.0.weight"));

        if has_mod {
            let adaln = adaln_input.unwrap();
            // adaLN_modulation is ModuleList([Linear]) — NO SiLU here
            let mod_out = self.linear_bias(adaln,
                &format!("{prefix}.adaLN_modulation.0.weight"),
                &format!("{prefix}.adaLN_modulation.0.bias"),
            )?;
            let mod_2d = if mod_out.shape().dims().len() == 3 && mod_out.shape().dims()[0] == 1 {
                mod_out.squeeze(Some(0))?
            } else {
                mod_out
            };
            let chunks = mod_2d.unsqueeze(1)?.chunk(4, 2)?;
            let scale_msa = chunks[0].add_scalar(1.0)?;
            let gate_msa = chunks[1].tanh()?;
            let scale_mlp = chunks[2].add_scalar(1.0)?;
            let gate_mlp = chunks[3].tanh()?;

            // Attention
            let x_norm = self.resident_rms_norm(x, &format!("{prefix}.attention_norm1.weight"))?;
            if prefix == "noise_refiner.0" {
            }
            if prefix == "noise_refiner.0" {
                // Verify the weight keys are present; tensors themselves
                // aren't needed here.
                let wkey = format!("{prefix}.adaLN_modulation.0.weight");
                let bkey = format!("{prefix}.adaLN_modulation.0.bias");
                let _ = self.w(&wkey)?;
                let _ = self.w(&bkey)?;
            }
            let x_mod = x_norm.mul(&scale_msa)?;
            if prefix == "noise_refiner.0" {
            }
            let attn_out = self.refiner_attention(&x_mod, cos, sin, prefix)?;
            if prefix == "noise_refiner.0" {
            }
            let attn_post = self.resident_rms_norm(&attn_out, &format!("{prefix}.attention_norm2.weight"))?;
            let x = x.add(&gate_msa.mul(&attn_post)?)?;
            if prefix == "noise_refiner.0" {
            }

            // FFN
            let ffn_norm = self.resident_rms_norm(&x, &format!("{prefix}.ffn_norm1.weight"))?;
            let ffn_mod = ffn_norm.mul(&scale_mlp)?;
            let ffn_out = self.refiner_ffn(&ffn_mod, prefix)?;
            let ffn_post = self.resident_rms_norm(&ffn_out, &format!("{prefix}.ffn_norm2.weight"))?;
            x.add(&gate_mlp.mul(&ffn_post)?)
        } else {
            // No modulation (context refiner)
            log::debug!("[refiner {prefix}] x: {:?}", x.shape().dims());
            let x_norm = self.resident_rms_norm(x, &format!("{prefix}.attention_norm1.weight"))?;
            log::debug!("[refiner {prefix}] x_norm: {:?}", x_norm.shape().dims());
            let attn_out = self.refiner_attention(&x_norm, cos, sin, prefix)?;
            log::debug!("[refiner {prefix}] attn_out: {:?}", attn_out.shape().dims());
            let attn_post = self.resident_rms_norm(&attn_out, &format!("{prefix}.attention_norm2.weight"))?;
            log::debug!("[refiner {prefix}] attn_post: {:?}", attn_post.shape().dims());
            let x = x.add(&attn_post)?;
            log::debug!("[refiner {prefix}] after SA: {:?}", x.shape().dims());

            let ffn_norm = self.resident_rms_norm(&x, &format!("{prefix}.ffn_norm1.weight"))?;
            log::debug!("[refiner {prefix}] ffn_norm: {:?}", ffn_norm.shape().dims());
            let ffn_out = self.refiner_ffn(&ffn_norm, prefix)?;
            log::debug!("[refiner {prefix}] ffn_out: {:?}", ffn_out.shape().dims());
            let ffn_post = self.resident_rms_norm(&ffn_out, &format!("{prefix}.ffn_norm2.weight"))?;
            log::debug!("[refiner {prefix}] ffn_post: {:?}", ffn_post.shape().dims());
            x.add(&ffn_post)
        }
    }

    fn refiner_attention(&self, x: &Tensor, cos: &Tensor, sin: &Tensor, prefix: &str) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let (b, seq) = (dims[0], dims[1]);

        // Fused QKV then split (checkpoint stores fused qkv.weight)
        let qkv = self.linear_no_bias(x, &format!("{prefix}.attention.qkv.weight"))?;
        let qkv_chunks = qkv.chunk(3, 2)?;
        let (q, k, v) = (qkv_chunks[0].clone(), qkv_chunks[1].clone(), qkv_chunks[2].clone());

        let q = self.resident_rms_norm_per_head(&q, &format!("{prefix}.attention.q_norm.weight"))?;
        let k = self.resident_rms_norm_per_head(&k, &format!("{prefix}.attention.k_norm.weight"))?;

        // RoPE + SDPA
        let q = q.reshape(&[b, seq, NUM_HEADS, HEAD_DIM])?.permute(&[0, 2, 1, 3])?;
        let k = k.reshape(&[b, seq, NUM_HEADS, HEAD_DIM])?.permute(&[0, 2, 1, 3])?;
        let v = v.reshape(&[b, seq, NUM_HEADS, HEAD_DIM])?.permute(&[0, 2, 1, 3])?;

        let q = apply_rope_complex(&q, cos, sin)?;
        let k = apply_rope_complex(&k, cos, sin)?;

        let out = flame_core::attention::sdpa(&q, &k, &v, None)?;
        let out = out.permute(&[0, 2, 1, 3])?.reshape(&[b, seq, DIM])?;
        self.linear_no_bias(&out, &format!("{prefix}.attention.out.weight"))
    }

    fn refiner_ffn(&self, x: &Tensor, prefix: &str) -> Result<Tensor> {
        let w1 = self.linear_no_bias(x, &format!("{prefix}.feed_forward.w1.weight"))?;
        let w3 = self.linear_no_bias(x, &format!("{prefix}.feed_forward.w3.weight"))?;
        let h = w1.swiglu(&w3)?;
        self.linear_no_bias(&h, &format!("{prefix}.feed_forward.w2.weight"))
    }

    fn transformer_block_with_lora(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        adaln_input: &Tensor,
        block_idx: usize,
    ) -> Result<Tensor> {
        // musubi-tuner zimage_model.py:266-281
        // adaLN_modulation is ModuleList([Linear]) — NO SiLU
        let mod_out = self.block_linear_bias(adaln_input, block_idx,
            "adaLN_modulation.0.weight", "adaLN_modulation.0.bias")?;
        // Squeeze back to 2D if ensure_3d added a dim, then unsqueeze for broadcast
        let mod_2d = if mod_out.shape().dims().len() == 3 && mod_out.shape().dims()[0] == 1 {
            mod_out.squeeze(Some(0))?
        } else {
            mod_out
        };
        let chunks = mod_2d.unsqueeze(1)?.chunk(4, 2)?;
        let scale_msa = chunks[0].add_scalar(1.0)?;
        let gate_msa = chunks[1].tanh()?;
        let scale_mlp = chunks[2].add_scalar(1.0)?;
        let gate_mlp = chunks[3].tanh()?;

        // Self-attention with LoRA
        let x_norm = self.block_rms_norm(x, block_idx, "attention_norm1.weight")?;
        let x_mod = x_norm.mul(&scale_msa)?;

        let dims = x.shape().dims().to_vec();
        let (b, seq) = (dims[0], dims[1]);

        // Fused QKV projection, then split + LoRA delta per component
        let qkv = self.block_linear_no_bias(&x_mod, block_idx, "attention.qkv.weight")?;
        let qkv_chunks = qkv.chunk(3, 2)?;
        let q = self.add_lora_delta(qkv_chunks[0].clone(), &x_mod, block_idx, LoraTarget::AttnQ)?;
        let k = self.add_lora_delta(qkv_chunks[1].clone(), &x_mod, block_idx, LoraTarget::AttnK)?;
        let v = self.add_lora_delta(qkv_chunks[2].clone(), &x_mod, block_idx, LoraTarget::AttnV)?;

        // QK norm — per-head: reshape to [B*seq*num_heads, head_dim], norm, reshape back.
        // Must match inference: rms_norm over head_dim=128, NOT full dim=3840.
        //
        // BUG FIX: previously called `flame_core::norm::rms_norm` which on this
        // Q/K-after-LoRA-delta path produced *exactly zero* grad_input for
        // some reason (Q_B/K_B stayed at the LoRA zero-init across all
        // training steps while V_B trained normally — V skips qk_norm).
        // Using primitive ops (mul/sum/add_scalar/rsqrt/mul) here makes the
        // autograd chain explicit and gradient flows to lora_a/lora_b.
        let q = q.reshape(&[b * seq * NUM_HEADS, HEAD_DIM])?;
        let k = k.reshape(&[b * seq * NUM_HEADS, HEAD_DIM])?;
        let q_w = self.bw(block_idx, "attention.q_norm.weight")?;
        let k_w = self.bw(block_idx, "attention.k_norm.weight")?;
        let q = primitive_rms_norm(&q, &q_w, NORM_EPS)?
            .reshape(&[b, seq, NUM_HEADS, HEAD_DIM])?;
        let k = primitive_rms_norm(&k, &k_w, NORM_EPS)?
            .reshape(&[b, seq, NUM_HEADS, HEAD_DIM])?;
        let v = v.reshape(&[b, seq, NUM_HEADS, HEAD_DIM])?;

        // Permute to [B, heads, seq, head_dim] for attention
        let q = q.permute(&[0, 2, 1, 3])?;
        let k = k.permute(&[0, 2, 1, 3])?;
        let v = v.permute(&[0, 2, 1, 3])?;

        let q = apply_rope_complex(&q, cos, sin)?;
        let k = apply_rope_complex(&k, cos, sin)?;

        let out = flame_core::attention::sdpa(&q, &k, &v, None)?;
        let out = out.permute(&[0, 2, 1, 3])?.reshape(&[b, seq, DIM])?;

        // Output projection with LoRA
        let attn_out = self.add_lora_delta(
            self.block_linear_no_bias(&out, block_idx, "attention.out.weight")?,
            &out, block_idx, LoraTarget::AttnOut,
        )?;
        let attn_post = self.block_rms_norm(&attn_out, block_idx, "attention_norm2.weight")?;
        let x = x.add(&gate_msa.mul(&attn_post)?)?;

        // FFN with LoRA (SwiGLU: w2(silu(w1(x)) * w3(x)))
        let ffn_norm = self.block_rms_norm(&x, block_idx, "ffn_norm1.weight")?;
        let ffn_mod = ffn_norm.mul(&scale_mlp)?;

        let w1 = self.add_lora_delta(
            self.block_linear_no_bias(&ffn_mod, block_idx, "feed_forward.w1.weight")?,
            &ffn_mod, block_idx, LoraTarget::FfnW1,
        )?;
        let w3 = self.add_lora_delta(
            self.block_linear_no_bias(&ffn_mod, block_idx, "feed_forward.w3.weight")?,
            &ffn_mod, block_idx, LoraTarget::FfnW3,
        )?;
        let h = w1.swiglu(&w3)?;
        let ffn_out = self.add_lora_delta(
            self.block_linear_no_bias(&h, block_idx, "feed_forward.w2.weight")?,
            &h, block_idx, LoraTarget::FfnW2,
        )?;

        let ffn_post = self.block_rms_norm(&ffn_out, block_idx, "ffn_norm2.weight")?;
        x.add(&gate_mlp.mul(&ffn_post)?)
    }

    fn final_layer(&self, x: &Tensor, adaln_input: &Tensor) -> Result<Tensor> {
        // musubi-tuner zimage_model.py:315-318
        // Checkpoint: adaLN_modulation.0 = SiLU (no weights), .1 = Linear
        let mod_silu = adaln_input.silu()?;
        let scale = self.linear_bias(&mod_silu,
            "final_layer.adaLN_modulation.1.weight",
            "final_layer.adaLN_modulation.1.bias")?;
        let scale = scale.add_scalar(1.0)?.unsqueeze(1)?;

        let _hidden = x.shape().dims()[2];
        // F32-internal LayerNorm — fused-kernel backward at hidden=3840
        // has the same BF16-accumulation precision bug documented on
        // `primitive_rms_norm`. Surfaced by single-block grad parity:
        // pre-fix layer 29's LoRA grad had cos_sim 0.5 against EriDiffusion,
        // tracing to the post-final_layer backward chain.
        let x_norm = primitive_layer_norm(x, 1e-6)?;
        let x_scaled = x_norm.mul(&scale)?;

        // Use autograd-aware matmul so gradients flow through from LoRA blocks.
        // `.contiguous()` after transpose: flame's BF16 matmul backward reads
        // saved tensors as if contiguous (CONVENTIONS doc) — strided view here
        // would scramble grad_input direction.
        let w = self.w("final_layer.linear.weight")?;
        let b = self.w("final_layer.linear.bias")?;
        let dims = x_scaled.shape().dims().to_vec();
        let batch: usize = dims[..dims.len()-1].iter().product();
        let in_feat = *dims.last().unwrap();
        let out_feat = w.shape().dims()[0];
        let x_2d = x_scaled.reshape(&[batch, in_feat])?;
        let wt = w.transpose()?.contiguous()?;
        let out_2d = x_2d.matmul(&wt)?.add(&b)?;
        let mut out_shape = dims[..dims.len()-1].to_vec();
        out_shape.push(out_feat);
        out_2d.reshape(&out_shape)
    }

    fn patchify(&self, x: &Tensor, h: usize, w: usize) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let (b, c) = (dims[0], dims[1]);
        let p = PATCH_SIZE;
        let h_tok = h / p;
        let w_tok = w / p;
        // [B, C, H, W] → [B, C, h_tok, p, w_tok, p] → [B, h_tok, w_tok, p, p, C]
        // Must match inference NextDiT ordering: permute [0,2,4,3,5,1]
        let x = x.reshape(&[b, c, h_tok, p, w_tok, p])?;
        let x = x.permute(&[0, 2, 4, 3, 5, 1])?;
        x.reshape(&[b, h_tok * w_tok, p * p * c])
    }

    fn unpatchify(&self, x: &Tensor, h_tok: usize, w_tok: usize) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let b = dims[0];
        let p = PATCH_SIZE;
        let c = IN_CHANNELS;
        // [B, N, p*p*C] → [B, h_tok, w_tok, p, p, C] → [B, C, h_tok*p, w_tok*p]
        // Inverse of patchify: permute [0,5,1,3,2,4]
        let x = x.reshape(&[b, h_tok, w_tok, p, p, c])?;
        let x = x.permute(&[0, 5, 1, 3, 2, 4])?;
        x.reshape(&[b, c, h_tok * p, w_tok * p])
    }

    fn build_rope_image(&self, h_tok: usize, w_tok: usize, cap_seq: usize) -> Result<(Tensor, Tensor)> {
        // musubi-tuner zimage_model.py:662
        // Position IDs: (cap_seq + 1 + 0, h_idx, w_idx) for image tokens (F=1)
        build_3d_rope(1, h_tok, w_tok, cap_seq + 1, &self.device)
    }

    fn build_rope_caption(&self, cap_seq: usize) -> Result<(Tensor, Tensor)> {
        // musubi-tuner zimage_model.py:613
        // Position IDs: (i + 1, 0, 0) for i in range(cap_seq)
        build_1d_rope(cap_seq, &self.device)
    }

    fn resident_rms_norm(&self, x: &Tensor, key: &str) -> Result<Tensor> {
        // Use the F32-internal primitive ops chain. flame_core::norm::rms_norm's
        // fused-kernel backward at HEAD_DIM=128 has the same direction-randomizing
        // BF16 accumulation bug that primitive_rms_norm's BF16 chain had —
        // surfaced by the block_parity harness. Until flame-core's fused
        // rms_norm_backward gets an F32-accumulation path, route through
        // primitive ops (F32-internal) here.
        let w = self.w(key)?;
        primitive_rms_norm(x, &w, NORM_EPS)
    }

    fn resident_rms_norm_per_head(&self, x: &Tensor, key: &str) -> Result<Tensor> {
        // Per-head RMSNorm: norm over head_dim=128, NOT full dim=3840.
        // Input x is [B, seq, dim]. Reshape to [B*seq*num_heads, head_dim] for norm.
        // primitive_rms_norm normalizes over the last dim → flat layout works.
        // F32-internal chain keeps the backward direction faithful (the fused
        // kernel's BF16 backward at HEAD_DIM=128 randomizes direction).
        let w = self.w(key)?;
        let dims = x.shape().dims().to_vec();
        let b = dims[0];
        let seq = dims[1];
        let flat = x.reshape(&[b * seq * NUM_HEADS, HEAD_DIM])?;
        let normed = primitive_rms_norm(&flat, &w, NORM_EPS)?;
        normed.reshape(&dims)
    }

    fn block_rms_norm(&self, x: &Tensor, block_idx: usize, suffix: &str) -> Result<Tensor> {
        let w = self.bw(block_idx, suffix)?;
        let hidden = w.shape().dims()[0];
        flame_core::norm::rms_norm(x, &[hidden], Some(&w), NORM_EPS)
    }

    fn block_rms_norm_per_head(&self, x: &Tensor, block_idx: usize, suffix: &str) -> Result<Tensor> {
        let w = self.bw(block_idx, suffix)?;
        let dims = x.shape().dims().to_vec();
        let (batch, hidden) = (dims[..dims.len()-1].iter().product::<usize>(), *dims.last().unwrap());
        let flat = x.reshape(&[batch, hidden])?;
        let normed = flame_core::norm::rms_norm(&flat.unsqueeze(0)?, &[hidden], Some(&w), NORM_EPS)?;
        normed.reshape(&dims)
    }
}

fn ensure_3d(x: &Tensor) -> Result<Tensor> {
    let dims = x.shape().dims();
    if dims.len() == 3 { return Ok(x.clone()); }
    if dims.len() == 2 { return x.unsqueeze(0); }
    if dims.len() == 1 { return x.unsqueeze(0)?.unsqueeze(0); }
    if dims.len() > 3 {
        log::error!("ensure_3d got {}D tensor: {:?}", dims.len(), dims);
        return Err(flame_core::Error::InvalidInput(format!(
            "ensure_3d: expected <=3D, got {:?}", dims
        )));
    }
    Ok(x.clone())
}

fn squeeze_if_needed(t: Tensor, orig_rank: usize) -> Tensor {
    if orig_rank == 2 { t.squeeze(Some(0)).unwrap_or(t) } else { t }
}

/// Build 3D RoPE cos/sin tables for image tokens.
/// Returns (cos, sin) each [1, N, head_dim/2] BF16 where N = f_tok * h_tok * w_tok.
fn build_3d_rope(
    f_tok: usize,
    h_tok: usize,
    w_tok: usize,
    f_offset: usize,
    device: &Arc<cudarc::driver::CudaDevice>,
) -> Result<(Tensor, Tensor)> {
    // musubi-tuner zimage_model.py:331-355
    // axes_dims = [32, 48, 48], total head_dim/2 = 64 (complex pairs)
    // freqs_cis[axis] = polar(ones, outer(timestep, 1/theta^(2i/d)))
    let seq = f_tok * h_tok * w_tok;
    let half_dim = ROPE_AXES_DIMS.iter().sum::<usize>() / 2; // 64 = (32+48+48)/2

    // Wait — Z-Image uses complex RoPE, not split-half. The axes_dims are
    // [32, 48, 48] and sum to 128 = head_dim. Each axis contributes
    // half its dims as complex pairs: 16, 24, 24 = 64 complex pairs total.
    let mut cos_host = vec![0.0f32; seq * (ROPE_AXES_DIMS[0] + ROPE_AXES_DIMS[1] + ROPE_AXES_DIMS[2]) / 2];
    let mut sin_host = vec![0.0f32; seq * (ROPE_AXES_DIMS[0] + ROPE_AXES_DIMS[1] + ROPE_AXES_DIMS[2]) / 2];
    let total_half = (ROPE_AXES_DIMS[0] + ROPE_AXES_DIMS[1] + ROPE_AXES_DIMS[2]) / 2;

    let freqs_for_axis = |dim: usize| -> Vec<f64> {
        let half = dim / 2;
        (0..half).map(|i| 1.0 / ROPE_THETA.powf(2.0 * i as f64 / dim as f64)).collect()
    };
    let fq = [freqs_for_axis(ROPE_AXES_DIMS[0]), freqs_for_axis(ROPE_AXES_DIMS[1]), freqs_for_axis(ROPE_AXES_DIMS[2])];
    let half_per_axis = [ROPE_AXES_DIMS[0] / 2, ROPE_AXES_DIMS[1] / 2, ROPE_AXES_DIMS[2] / 2];

    for fi in 0..f_tok {
        for hi in 0..h_tok {
            for wi in 0..w_tok {
                let si = fi * h_tok * w_tok + hi * w_tok + wi;
                let row = si * total_half;
                let positions = [(fi + f_offset) as f64, hi as f64, wi as f64];
                let mut offset = 0;
                for axis in 0..3 {
                    for i in 0..half_per_axis[axis] {
                        let angle = positions[axis] * fq[axis][i];
                        cos_host[row + offset + i] = angle.cos() as f32;
                        sin_host[row + offset + i] = angle.sin() as f32;
                    }
                    offset += half_per_axis[axis];
                }
            }
        }
    }

    let cos = Tensor::from_vec(cos_host, flame_core::Shape::from_dims(&[1, seq, total_half]), device.clone())?
        .to_dtype(DType::BF16)?;
    let sin = Tensor::from_vec(sin_host, flame_core::Shape::from_dims(&[1, seq, total_half]), device.clone())?
        .to_dtype(DType::BF16)?;
    Ok((cos, sin))
}

fn build_1d_rope(cap_seq: usize, device: &Arc<cudarc::driver::CudaDevice>) -> Result<(Tensor, Tensor)> {
    // Caption positions: (i+1, 0, 0)
    build_3d_rope(cap_seq, 1, 1, 1, device)
}

/// Sinusoidal timestep embedding.
///
/// musubi-tuner `zimage_model.py:65-73`:
/// ```python
/// half = dim // 2
/// freqs = exp(-log(max_period) * arange(0, half) / half)
/// args = t[:, None] * freqs[None]
/// RMSNorm wrapper. Now a thin delegate to `flame_core::norm::rms_norm`.
///
/// History: this function used to build the norm by hand from primitive
/// autograd ops (mul + sum_dim_keepdim + add_scalar + rsqrt + mul + cast)
/// because the older `flame_core::norm::rms_norm` backward suffered a
/// BF16-accumulation drift that compounded across the 30-block stack —
/// per-call grad direction error of 1-22%, gradient magnitude blow-up of
/// ~1.25× per layer. That bug was fixed in flame-core commit `bcc37a7`
/// (forward + backward kernels now F32-accumulate internally over BF16
/// storage). Parity pinned by
/// `flame-core/tests/rms_norm_vs_primitive_zimage.rs`:
///
///   forward [1,4096,2560]      max_rel = 7.8e-3 (BF16 floor)
///   forward [1,24,4096,128]    max_rel = 7.5e-3 (BF16 floor)
///   backward grad_x            cos = 1.000000, mag_ratio = 1.000000
///   backward grad_w            cos = 1.000000, L1 = 1.0399e7 (both)
///
/// i.e. bit-exact backward against the primitive F32 chain. Name kept
/// for source-diff minimization; signature unchanged.
fn primitive_rms_norm(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    let last_dim = weight
        .shape()
        .dims()
        .last()
        .copied()
        .unwrap_or_else(|| x.shape().dims()[x.shape().dims().len() - 1]);
    flame_core::norm::rms_norm(x, &[last_dim], Some(weight), eps)
}

/// Primitive LayerNorm (no scale/bias) with F32-internal chain.
///
/// Kept as a primitive chain pending tighter parity work. The fused
/// `flame_core::layer_norm::layer_norm` agrees in forward (max_rel 7.8e-3,
/// BF16 floor) and on backward magnitude (mag_ratio 1.003) but currently
/// drifts in backward direction at cos ≈ 0.997 — see
/// `flame-core/tests/rms_norm_vs_primitive_zimage.rs::
/// layer_norm_no_affine_backward_parity_zimage_block_shape`. That ~0.3%
/// per-call drift may compound across the 30-block stack; revisit after
/// fused LN backward switches to a two-pass `E[(x-mean)^2]` accumulator
/// matching this primitive chain.
fn primitive_layer_norm(x: &Tensor, eps: f32) -> Result<Tensor> {
    let out_dtype = x.dtype();
    let x_f32 = if out_dtype == DType::F32 { x.clone() } else { x.to_dtype(DType::F32)? };
    let dims = x_f32.shape().dims().to_vec();
    let last = dims.len() - 1;
    let n = dims[last] as f32;
    // mean over last dim
    let mean = x_f32.sum_dim_keepdim(last)?.mul_scalar(1.0 / n)?;
    let centered = x_f32.sub(&mean)?;
    // var = mean(centered²)
    let sq = centered.mul(&centered)?;
    let var = sq.sum_dim_keepdim(last)?.mul_scalar(1.0 / n)?;
    let inv_std = var.add_scalar(eps)?.rsqrt()?;
    let normed = centered.mul(&inv_std)?;
    if out_dtype == DType::F32 {
        Ok(normed)
    } else {
        normed.to_dtype(out_dtype)
    }
}

/// embedding = cat([cos(args), sin(args)], dim=-1)
/// ```
fn sinusoidal_embedding(t: &Tensor, dim: usize) -> Result<Tensor> {
    let half = dim / 2;
    let max_period: f32 = 10000.0;

    // Build frequency vector on host
    let mut freqs = Vec::with_capacity(half);
    for i in 0..half {
        freqs.push((-max_period.ln() * i as f32 / half as f32).exp());
    }

    // t: [B] → [B, 1]
    let t_f32 = t.to_dtype(DType::F32)?;
    let t_2d = t_f32.unsqueeze(1)?;

    // freqs: [1, half]
    let freqs_t = Tensor::from_vec(
        freqs,
        flame_core::Shape::from_dims(&[1, half]),
        t.device().clone(),
    )?;

    // args = t_2d * freqs → [B, half]
    let args = t_2d.mul(&freqs_t)?;
    let cos_part = args.cos()?;
    let sin_part = args.sin()?;

    // [B, dim]
    Tensor::cat(&[&cos_part, &sin_part], 1)?.to_dtype(DType::BF16)
}

/// Resolve `(block_idx, target)` into the dotted-path adapter name used by
/// the LyCORIS wiring (Phase 2b). Mirrors `ZImageLoraBundle::ai_toolkit_suffix`
/// + the saved-key prefix scheme so both code paths agree on adapter names.
pub(crate) fn lycoris_target_name(block_idx: usize, target: LoraTarget) -> String {
    let suffix = match target {
        LoraTarget::AttnQ => "attention.to_q",
        LoraTarget::AttnK => "attention.to_k",
        LoraTarget::AttnV => "attention.to_v",
        LoraTarget::AttnOut => "attention.to_out.0",
        LoraTarget::FfnW1 => "feed_forward.w1",
        LoraTarget::FfnW2 => "feed_forward.w2",
        LoraTarget::FfnW3 => "feed_forward.w3",
    };
    format!("layers.{block_idx}.{suffix}")
}

/// Standalone block forward for use inside `checkpoint_offload` closures.
/// Duplicates `ZImageModel::transformer_block_with_lora` but takes
/// explicit weight maps instead of `&self`.
fn block_forward_standalone(
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    adaln_input: &Tensor,
    block_idx: usize,
    bundle: &ZImageLoraBundle,
    lycoris: Option<&Arc<crate::lycoris::LycorisBundle>>,
    block_weights: &HashMap<String, Tensor>,
) -> Result<Tensor> {
    let bw = |suffix: &str| -> Result<&Tensor> {
        let key = format!("layers.{block_idx}.{suffix}");
        block_weights.get(&key)
            .ok_or_else(|| flame_core::Error::InvalidInput(format!("missing: {key}")))
    };

    // Was: `weight.transpose()?.contiguous()?` + `matmul` → materialized
    // every weight transpose via `permute_generic_bf16_kernel` (rank-2 [1,0]
    // is not a fast-path perm). 6 linears × 30 blocks × ~2 (checkpoint
    // replay) ≈ 360 permute_generic launches/step, ~70 ms wall.
    //
    // `fused_linear3d_native` takes weight in PyTorch `[Cout, Cin]` layout
    // and lets cuBLASLt apply TRANSA=T inline — zero transposes, zero
    // materialize. Autograd records `Op::Linear` whose backward is also
    // transpose-free (autograd.rs:3019).
    let linear_no_bias = |x: &Tensor, suffix: &str| -> Result<Tensor> {
        let weight = bw(suffix)?;
        let orig_rank = x.shape().dims().len();
        let out_3d = flame_core::ops::fused_inference::fused_linear3d_native(
            &ensure_3d(x)?,
            weight,
            None,
        )?;
        Ok(squeeze_if_needed(out_3d, orig_rank))
    };

    let linear_bias = |x: &Tensor, w_suffix: &str, b_suffix: &str| -> Result<Tensor> {
        let weight = bw(w_suffix)?;
        let bias = bw(b_suffix)?;
        let orig_rank = x.shape().dims().len();
        let out_3d = flame_core::ops::fused_inference::fused_linear3d_native(
            &ensure_3d(x)?,
            weight,
            Some(bias),
        )?;
        Ok(squeeze_if_needed(out_3d, orig_rank))
    };

    let rms_norm = |x: &Tensor, suffix: &str| -> Result<Tensor> {
        // F32-internal primitive ops — matches the same fix applied to
        // `resident_rms_norm` to bypass the fused-kernel BF16-backward bug.
        primitive_rms_norm(x, bw(suffix)?, NORM_EPS)
    };

    let rms_norm_per_head = |x: &Tensor, suffix: &str| -> Result<Tensor> {
        // See note on `primitive_rms_norm` — using primitive ops is required
        // here so backward propagates to LoRA Q/K (the autograd path through
        // `flame_core::norm::rms_norm` produces zero grad_input on this
        // Q/K-after-LoRA-delta pattern, killing Q_B/K_B training).
        let w = bw(suffix)?;
        let dims = x.shape().dims().to_vec();
        let batch: usize = dims[..dims.len() - 1].iter().product();
        let flat = x.reshape(&[batch * NUM_HEADS, HEAD_DIM])?;
        let normed = primitive_rms_norm(&flat, w, NORM_EPS)?;
        normed.reshape(&dims)
    };

    let add_lora = |base: Tensor, input: &Tensor, target: LoraTarget| -> Result<Tensor> {
        // Phase 2b: LyCORIS bundle takes priority when wired. When absent
        // (default) the legacy LoRALinear branch runs unchanged → byte-identical.
        if let Some(lyc) = lycoris {
            let name = lycoris_target_name(block_idx, target);
            let in3d = ensure_3d(input)?;
            if let Some(delta) = lyc.forward_delta(&name, &in3d).map_err(|e| {
                flame_core::FlameError::InvalidInput(format!("lycoris forward_delta({name}): {e}"))
            })? {
                return base.add(&delta);
            }
            return Ok(base);
        }
        // Dispatch via `adapter_for` so `--algo locon`/etc on the model's own
        // bundle is honored (the legacy `adapters` map is empty when LyCORIS
        // is the active path).  Fixes the silent-skip pitfall.
        if let Some(lora) = bundle.adapter_for(block_idx, target) {
            let delta = lora.forward_delta(&ensure_3d(input)?).map_err(|e| flame_core::FlameError::InvalidInput(format!("lora delta: {e}")))?;
            base.add(&delta)
        } else {
            Ok(base)
        }
    };

    // AdaLN modulation
    let mod_out = linear_bias(adaln_input, "adaLN_modulation.0.weight", "adaLN_modulation.0.bias")?;
    let mod_2d = if mod_out.shape().dims().len() == 3 && mod_out.shape().dims()[0] == 1 {
        mod_out.squeeze(Some(0))?
    } else { mod_out };
    let chunks = mod_2d.unsqueeze(1)?.chunk(4, 2)?;
    let scale_msa = chunks[0].add_scalar(1.0)?;
    let gate_msa = chunks[1].tanh()?;
    let scale_mlp = chunks[2].add_scalar(1.0)?;
    let gate_mlp = chunks[3].tanh()?;

    // Self-attention with LoRA
    let x_norm = rms_norm(x, "attention_norm1.weight")?;
    let x_mod = x_norm.mul(&scale_msa)?;
    let dims = x.shape().dims().to_vec();
    let (b, seq) = (dims[0], dims[1]);

    let qkv = linear_no_bias(&x_mod, "attention.qkv.weight")?;
    let qkv_chunks = qkv.chunk(3, 2)?;
    let q = add_lora(qkv_chunks[0].clone(), &x_mod, LoraTarget::AttnQ)?;
    let k = add_lora(qkv_chunks[1].clone(), &x_mod, LoraTarget::AttnK)?;
    let v = add_lora(qkv_chunks[2].clone(), &x_mod, LoraTarget::AttnV)?;

    let q = rms_norm_per_head(&q, "attention.q_norm.weight")?;
    let k = rms_norm_per_head(&k, "attention.k_norm.weight")?;

    let q = q.reshape(&[b, seq, NUM_HEADS, HEAD_DIM])?.permute(&[0, 2, 1, 3])?;
    let k = k.reshape(&[b, seq, NUM_HEADS, HEAD_DIM])?.permute(&[0, 2, 1, 3])?;
    let v = v.reshape(&[b, seq, NUM_HEADS, HEAD_DIM])?.permute(&[0, 2, 1, 3])?;

    let q = apply_rope_complex(&q, cos, sin)?;
    let k = apply_rope_complex(&k, cos, sin)?;

    let out = flame_core::attention::sdpa(&q, &k, &v, None)?;
    let out = out.permute(&[0, 2, 1, 3])?.reshape(&[b, seq, DIM])?;

    let attn_out = add_lora(linear_no_bias(&out, "attention.out.weight")?, &out, LoraTarget::AttnOut)?;
    let attn_post = rms_norm(&attn_out, "attention_norm2.weight")?;
    let x = x.add(&gate_msa.mul(&attn_post)?)?;

    // FFN with LoRA (SwiGLU)
    let ffn_norm = rms_norm(&x, "ffn_norm1.weight")?;
    let ffn_mod = ffn_norm.mul(&scale_mlp)?;
    let w1 = add_lora(linear_no_bias(&ffn_mod, "feed_forward.w1.weight")?, &ffn_mod, LoraTarget::FfnW1)?;
    let w3 = add_lora(linear_no_bias(&ffn_mod, "feed_forward.w3.weight")?, &ffn_mod, LoraTarget::FfnW3)?;
    let h = w1.swiglu(&w3)?;
    let ffn_out = add_lora(linear_no_bias(&h, "feed_forward.w2.weight")?, &h, LoraTarget::FfnW2)?;
    let ffn_post = rms_norm(&ffn_out, "ffn_norm2.weight")?;
    x.add(&gate_mlp.mul(&ffn_post)?)
}

/// Inference-mode block forward — fused kernels, no autograd requirements.
///
/// Mirrors inference-flame `zimage_nextdit.rs::transformer_block` but reads
/// weights from the block-weight HashMap (same source as
/// `block_forward_standalone`) and applies LoRA deltas at inference time.
///
/// Called from the forward loop when `!AutogradContext::is_recording()`.
/// The training path continues to use `block_forward_standalone` unchanged
/// so that gradient flow, checkpoint semantics, and LoRA training are
/// unaffected.
///
/// # Fused ops used
/// * `fused_rms_norm`          — `flame_core::ops::fused_inference`
/// * `rope_fused_bf16`         — interleaved complex RoPE (Z-Image style)
/// * `swiglu_fused_bf16`       — fused silu(w1)*w3
/// * `gate_residual_fused_bf16`— fused x + tanh(gate)*residual
///
/// # linear_3d helper
/// We must NOT call `fused_linear3d_native` here (EDv2 bug #6). Instead we
/// use the `linear_3d` helper that does explicit reshape→matmul→reshape.
fn block_forward_iflame(
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    adaln_input: &Tensor,
    block_idx: usize,
    bundle: &ZImageLoraBundle,
    lycoris: Option<&Arc<crate::lycoris::LycorisBundle>>,
    block_weights: &HashMap<String, Tensor>,
) -> Result<Tensor> {
    use flame_core::ops::fused_inference::fused_rms_norm;
    use flame_core::bf16_ops::{gate_residual_fused_bf16, swiglu_fused_bf16};

    // Weight accessor: `layers.{block_idx}.{suffix}`
    let bw = |suffix: &str| -> Result<&Tensor> {
        let key = format!("layers.{block_idx}.{suffix}");
        block_weights.get(&key)
            .ok_or_else(|| flame_core::Error::InvalidInput(format!("missing: {key}")))
    };

    // linear_3d: explicit reshape→matmul(W^T)→reshape.
    // Weights arrive as [out, in] (PyTorch layout). We transpose once and
    // matmul so the fused cuBLASLt path is NOT triggered (bug #6).
    let linear_3d = |x: &Tensor, w: &Tensor| -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let m: usize = dims[..dims.len() - 1].iter().product();
        let c = *dims.last().unwrap();
        let x_2d = x.reshape(&[m, c])?;
        let out = x_2d.matmul(&w.transpose()?)?;
        let out_dim = *out.shape().dims().last().unwrap();
        let mut new_dims = dims.clone();
        *new_dims.last_mut().unwrap() = out_dim;
        out.reshape(&new_dims)
    };

    // linear_3d with bias add.
    let linear_3d_bias = |x: &Tensor, suffix_w: &str, suffix_b: &str| -> Result<Tensor> {
        let out = linear_3d(x, bw(suffix_w)?)?;
        out.add(bw(suffix_b)?)
    };

    // LoRA delta: adds LoRA output on top of base projection when an adapter
    // exists for this block and target.
    let add_lora = |base: Tensor, input: &Tensor, target: LoraTarget| -> Result<Tensor> {
        // Phase 2b: LyCORIS bundle takes priority when wired. When absent
        // (default) the legacy LoRALinear branch runs unchanged → byte-identical.
        if let Some(lyc) = lycoris {
            let name = lycoris_target_name(block_idx, target);
            let in3d = ensure_3d(input)?;
            if let Some(delta) = lyc.forward_delta(&name, &in3d).map_err(|e| {
                flame_core::FlameError::InvalidInput(format!("lycoris forward_delta({name}): {e}"))
            })? {
                return base.add(&delta);
            }
            return Ok(base);
        }
        // Dispatch via `adapter_for` to honour `--algo locon`/etc on the
        // model's own bundle (legacy `adapters` map is empty under LyCORIS).
        if let Some(lora) = bundle.adapter_for(block_idx, target) {
            let delta = lora.forward_delta(&ensure_3d(input)?)
                .map_err(|e| flame_core::FlameError::InvalidInput(
                    format!("lora delta: {e}")))?;
            base.add(&delta)
        } else {
            Ok(base)
        }
    };

    // --- AdaLN modulation ---------------------------------------------------
    // adaLN_modulation is ModuleList([Linear]) — NO SiLU, matching
    // musubi-tuner `zimage_model.py:266`.
    let mod_out = linear_3d_bias(adaln_input, "adaLN_modulation.0.weight", "adaLN_modulation.0.bias")?;
    // mod_out shape: [B, 4*DIM] (2-D) or [1, B, 4*DIM] (3-D from ensure_3d).
    // Squeeze batch-dim if ensure_3d added one so chunk splits on the right dim.
    let mod_2d = if mod_out.shape().dims().len() == 3 && mod_out.shape().dims()[0] == 1 {
        mod_out.squeeze(Some(0))?
    } else {
        mod_out
    };
    // Broadcast shape: [B, 1, 4*DIM] → chunk(4, 2) → each [B, 1, DIM].
    let chunks = mod_2d.unsqueeze(1)?.chunk(4, 2)?;
    let scale_msa = chunks[0].add_scalar(1.0)?;
    let gate_msa  = chunks[1].tanh()?;
    let scale_mlp = chunks[2].add_scalar(1.0)?;
    let gate_mlp  = chunks[3].tanh()?;

    // --- Attention branch ----------------------------------------------------
    let dims = x.shape().dims().to_vec();
    let (b, seq) = (dims[0], dims[1]);

    // Pre-attn norm + scale modulation
    let x_norm = fused_rms_norm(x, bw("attention_norm1.weight")?, NORM_EPS)?;
    let x_mod  = x_norm.mul(&scale_msa)?;

    // Fused QKV projection, then split
    let qkv = linear_3d(&x_mod, bw("attention.qkv.weight")?)?;
    let qkv_chunks = qkv.chunk(3, 2)?;
    let q = add_lora(qkv_chunks[0].clone(), &x_mod, LoraTarget::AttnQ)?;
    let k = add_lora(qkv_chunks[1].clone(), &x_mod, LoraTarget::AttnK)?;
    let v = add_lora(qkv_chunks[2].clone(), &x_mod, LoraTarget::AttnV)?;

    // Per-head QK RMSNorm (over head_dim=128, not full dim=3840).
    // Flatten to [B*seq*heads, head_dim], norm, reshape back.
    let q_flat = q.reshape(&[b * seq * NUM_HEADS, HEAD_DIM])?;
    let k_flat = k.reshape(&[b * seq * NUM_HEADS, HEAD_DIM])?;
    let q = fused_rms_norm(&q_flat, bw("attention.q_norm.weight")?, NORM_EPS)?
        .reshape(&[b, seq, NUM_HEADS, HEAD_DIM])?;
    let k = fused_rms_norm(&k_flat, bw("attention.k_norm.weight")?, NORM_EPS)?
        .reshape(&[b, seq, NUM_HEADS, HEAD_DIM])?;
    let v = v.reshape(&[b, seq, NUM_HEADS, HEAD_DIM])?;

    // Permute to [B, heads, seq, head_dim] for SDPA
    let q = q.permute(&[0, 2, 1, 3])?;
    let k = k.permute(&[0, 2, 1, 3])?;
    let v = v.permute(&[0, 2, 1, 3])?;

    // RoPE: Z-Image uses interleaved complex RoPE (rope_fused_bf16).
    // cos/sin arrive as [1, seq, head_dim/2]; need [1, 1, seq, half] for the kernel.
    let q = apply_rope_complex(&q, cos, sin)?;
    let k = apply_rope_complex(&k, cos, sin)?;

    // Scaled dot-product attention
    let out = flame_core::attention::sdpa(&q, &k, &v, None)?;
    let out = out.permute(&[0, 2, 1, 3])?.reshape(&[b, seq, DIM])?;

    // Output projection + LoRA
    let attn_out = add_lora(
        linear_3d(&out, bw("attention.out.weight")?)?,
        &out, LoraTarget::AttnOut,
    )?;

    // Post-attn norm + gated residual: x + tanh(gate_msa) * norm(attn_out)
    let attn_post = fused_rms_norm(&attn_out, bw("attention_norm2.weight")?, NORM_EPS)?;
    let x = gate_residual_fused_bf16(x, &gate_msa, &attn_post)?;

    // --- FFN branch ----------------------------------------------------------
    // Pre-FFN norm + scale modulation
    let ffn_norm = fused_rms_norm(&x, bw("ffn_norm1.weight")?, NORM_EPS)?;
    let ffn_mod  = ffn_norm.mul(&scale_mlp)?;

    // SwiGLU: w2(silu(w1(x)) * w3(x)) with LoRA on each projection
    let w1 = add_lora(linear_3d(&ffn_mod, bw("feed_forward.w1.weight")?)?, &ffn_mod, LoraTarget::FfnW1)?;
    let w3 = add_lora(linear_3d(&ffn_mod, bw("feed_forward.w3.weight")?)?, &ffn_mod, LoraTarget::FfnW3)?;
    let h  = swiglu_fused_bf16(&w1, &w3)?;
    let ffn_out = add_lora(linear_3d(&h, bw("feed_forward.w2.weight")?)?, &h, LoraTarget::FfnW2)?;

    // Post-FFN norm + gated residual: x + tanh(gate_mlp) * norm(ffn_out)
    let ffn_post = fused_rms_norm(&ffn_out, bw("ffn_norm2.weight")?, NORM_EPS)?;
    gate_residual_fused_bf16(&x, &gate_mlp, &ffn_post)
}

/// Apply complex RoPE (Z-Image style): interleaved (2i, 2i+1) pairs.
/// x: [B, H, N, D], cos/sin: [1, N, D/2]
fn apply_rope_complex(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    // Reshape cos/sin from [1, N, half] to [1, 1, N, half] for rope_fused_bf16
    let cos_4d = cos.reshape(&[1, 1, cos.shape().dims()[1], cos.shape().dims()[2]])?;
    let sin_4d = sin.reshape(&[1, 1, sin.shape().dims()[1], sin.shape().dims()[2]])?;
    // Z-Image uses interleaved (complex) RoPE, same as Wan/Klein/Flux
    flame_core::bf16_ops::rope_fused_bf16(x, &cos_4d, &sin_4d)
}

/// Dump one tensor for parity comparison.
/// Writes `<dir>/<name>.safetensors` with a single entry `"t"` as F32.
fn zimage_dump(dir: &std::path::Path, name: &str, t: &Tensor) -> Result<()> {
    let path = dir.join(format!("{name}.safetensors"));
    let f32 = t.to_dtype(DType::F32)?;
    let mut m: HashMap<String, Tensor> = HashMap::new();
    m.insert("t".to_string(), f32);
    flame_core::serialization::save_tensors(
        &m, &path, flame_core::serialization::SerializationFormat::SafeTensors,
    )
}

// ─── ED-v2 `TrainableModel` bridge ────────────────────────────────────────
// `context` slot 0 = caption embeddings ([B, seq, 2560]).
// `context` slot 1 = optional caption mask ([B, seq]). pooled is unused.
impl crate::models::TrainableModel for ZImageModel {
    fn forward(
        &mut self,
        noisy: &Tensor,
        timestep: &Tensor,
        context: &[Tensor],
        _pooled: Option<&Tensor>,
    ) -> crate::Result<Tensor> {
        let cap_feats = context.first().ok_or_else(|| {
            crate::EriDiffusionError::Model("ZImageModel needs caption embeddings in context[0]".into())
        })?;
        let cap_mask = context.get(1);
        ZImageModel::forward(self, noisy, timestep, cap_feats, cap_mask)
            .map_err(crate::EriDiffusionError::from)
    }

    fn parameters(&self) -> Vec<Parameter> { ZImageModel::parameters(self) }

    fn post_optimizer_step(&mut self) { self.refresh_lora_cache(); }

    fn save_weights(&self, path: &str) -> crate::Result<()> {
        ZImageModel::save_weights(self, std::path::Path::new(path))
            .map_err(crate::EriDiffusionError::from)
    }

    fn load_weights(&mut self, path: &str) -> crate::Result<()> {
        let device = self.bundle.adapters.values().next()
            .map(|l| l.lora_a.tensor().map(|t| t.device().clone()))
            .transpose()?
            .ok_or_else(|| crate::EriDiffusionError::Model("no LoRA adapters to load into".into()))?;
        self.bundle.load(std::path::Path::new(path), &device)
            .map_err(crate::EriDiffusionError::from)
    }
}

