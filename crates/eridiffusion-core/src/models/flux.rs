//! Flux 1 DiT model — correct implementation ported from flame-diffusion flux1-trainer.
//! Architecture constants match BFL/flux-1-dev.

use std::collections::HashMap;
use std::sync::Arc;
use cudarc::driver::CudaDevice;
use flame_core::{parameter::Parameter, DType, Shape, Tensor};
use flame_core::autograd::AutogradContext;
use crate::adapter::AdapterModule;
use crate::config::TrainConfig;
use crate::lora::LoRALinear;
use crate::lycoris::{AdapterStore, LycorisAlgo, LycorisBundleConfig};
use crate::models::TrainableModel;
use crate::Result;

pub const NUM_DOUBLE: usize = 19;
pub const NUM_SINGLE: usize = 38;
pub const DIM: usize = 3072;
pub const NUM_HEADS: usize = 24;
pub const HEAD_DIM: usize = 128;
pub const IN_CHANNELS: usize = 64;
pub const T5_DIM: usize = 4096;
pub const MLP_HIDDEN: usize = 12288;
pub const TIMESTEP_DIM: usize = 256;
pub const VECTOR_DIM: usize = 768;
pub const ROPE_AXES: [usize; 3] = [16, 56, 56];
pub const ROPE_THETA: f64 = 10000.0;
pub const NORM_EPS: f32 = 1e-6;

// ── LoRA targets ────────────────────────────────────────────────────
//
// Audit fix FLUX_VERIFY §H4 / §H5 / SKEPTIC §H4 / §H5: the OT-Python wrapper
// hooks **every** `nn.Linear` in the diffusers `FluxTransformer2DModel` whose
// dotted path matches the `attn-mlp` filter. BFL's fused QKV (`img_attn.qkv`,
// `txt_attn.qkv`) appears in diffusers as 3 separate linears (`to_q`, `to_k`,
// `to_v`) and BFL's fused `linear1` in single blocks splits into
// `attn.{to_q,to_k,to_v}` + `proj_mlp`. So the canonical OT adapter granularity
// per double block is 12 LoRAs (4 attn + 4 attn + 4 MLP) and per single block
// is 5 LoRAs (3 QKV + proj_mlp + proj_out).
//
// Pre-fix the ED-v2 impl had only 4 doubles/block + 2 singles/block (152 total
// vs canonical 418), and the single `Out` adapter wrapped a phantom `DIM→DIM`
// matrix when `linear2` is actually `5*DIM→DIM` (silent shape add over `attn`
// only — MLP-up half got no LoRA correction). This module now matches the
// canonical layout exactly.
//
// DEFERRED (Wave-2 BUG-3): OT's permissive `attn-mlp` regex *also* matches a
// few extra `nn.Linear`s that this module does not adapt — the per-block
// modulation linears (`img_mod.lin`, `txt_mod.lin`, single-block `modulation.lin`),
// the time/vector/text embedders (`time_in`, `vector_in`, `txt_in`,
// `guidance_in`), `img_in`, and `final_layer.linear`. Adding them widens
// the bundle by ~150 adapters but is rarely user-visible at standard LoRA
// rank. Decide convention-vs-completeness in a future session.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DoubleLoraTarget {
    // Attention path (img/txt symmetric).
    ImgQ, ImgK, ImgV, ImgProj,
    TxtQ, TxtK, TxtV, TxtProj,
    // MLP path (Linear → GELU → Linear).
    ImgMlp0, ImgMlp2,
    TxtMlp0, TxtMlp2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SingleLoraTarget {
    // Q/K/V split from BFL fused linear1[:3*DIM].
    Q, K, V,
    // MLP-up half from BFL fused linear1[3*DIM:7*DIM] (DIM → 4*DIM).
    ProjMlp,
    // BFL `linear2` (5*DIM → DIM) — wraps the full fused [attn ‖ mlp_out].
    ProjOut,
}

// ── Model struct ────────────────────────────────────────────────────

pub struct FluxModel {
    pub config: TrainConfig,
    pub device: Arc<CudaDevice>,

    pub shared_weights: HashMap<String, Tensor>,
    pub double_block_weights: Vec<HashMap<String, Tensor>>,
    pub single_block_weights: Vec<HashMap<String, Tensor>>,

    // LoRA — legacy plain-LoRA path. Byte-identical to pre-LyCORIS commits
    // when this is `Some` and `lycoris_bundle` is `None`.
    pub bundle: Option<FluxLoraBundle>,
    /// LyCORIS adapter store (`--algo locon|loha|lokr|full|oft`, optional DoRA).
    /// Mutually exclusive with `bundle`.
    pub lycoris_bundle: Option<FluxLycorisBundle>,
    pub fft_params: Option<HashMap<String, Parameter>>,
    pub is_full_finetune: bool,

    pub has_guidance: bool,

    /// When Some, double/single block weights live in pinned host RAM
    /// (or mmap-backed streaming staging buffers) and are H2D-streamed into
    /// reusable GPU slots per block, per step. LoRA-only.
    ///
    /// Unified block index space: `0..NUM_DOUBLE` → double_blocks.{i},
    /// `NUM_DOUBLE..NUM_DOUBLE + NUM_SINGLE` → single_blocks.{i}.
    pub offloader: Option<std::sync::Arc<std::sync::Mutex<crate::training::block_offload::BlockOffloader>>>,

    /// Default guidance value passed to the model at training time.
    /// 1.0 for Schnell, 3.5 for Dev (matches sd-scripts and EriDiffusion defaults).
    pub guidance_value: f32,
}

#[derive(Clone)]
pub struct FluxLoraBundle {
    pub double_adapters: HashMap<(usize, DoubleLoraTarget), LoRALinear>,
    pub single_adapters: HashMap<(usize, SingleLoraTarget), LoRALinear>,
}

impl FluxLoraBundle {
    /// Build the canonical 418-adapter LoRA bundle.
    /// `seed` is **fixed** (and ignored beyond the single seed=42 invariant) —
    /// each adapter is initialised from the same seed; the autograd graph and
    /// per-adapter shape differentiate them. Audit fix FLUX_VERIFY §H7 /
    /// SKEPTIC §H10 (`feedback_default_seed_42.md`).
    pub fn new(rank: usize, alpha: f32, device: Arc<CudaDevice>, seed: u64) -> Result<Self> {
        let mut da = HashMap::new();
        let mut sa = HashMap::new();
        // Double blocks: 12 adapters/block × 19 blocks = 228.
        for i in 0..NUM_DOUBLE {
            // Attention: Q/K/V split from BFL fused img_attn.qkv (3*DIM → 3 separate DIM→DIM).
            da.insert((i, DoubleLoraTarget::ImgQ),    LoRALinear::new(DIM, DIM, rank, alpha, device.clone(), seed)?);
            da.insert((i, DoubleLoraTarget::ImgK),    LoRALinear::new(DIM, DIM, rank, alpha, device.clone(), seed)?);
            da.insert((i, DoubleLoraTarget::ImgV),    LoRALinear::new(DIM, DIM, rank, alpha, device.clone(), seed)?);
            da.insert((i, DoubleLoraTarget::ImgProj), LoRALinear::new(DIM, DIM, rank, alpha, device.clone(), seed)?);
            da.insert((i, DoubleLoraTarget::TxtQ),    LoRALinear::new(DIM, DIM, rank, alpha, device.clone(), seed)?);
            da.insert((i, DoubleLoraTarget::TxtK),    LoRALinear::new(DIM, DIM, rank, alpha, device.clone(), seed)?);
            da.insert((i, DoubleLoraTarget::TxtV),    LoRALinear::new(DIM, DIM, rank, alpha, device.clone(), seed)?);
            da.insert((i, DoubleLoraTarget::TxtProj), LoRALinear::new(DIM, DIM, rank, alpha, device.clone(), seed)?);
            // MLP up + down (DIM → MLP_HIDDEN → DIM).
            da.insert((i, DoubleLoraTarget::ImgMlp0), LoRALinear::new(DIM, MLP_HIDDEN, rank, alpha, device.clone(), seed)?);
            da.insert((i, DoubleLoraTarget::ImgMlp2), LoRALinear::new(MLP_HIDDEN, DIM, rank, alpha, device.clone(), seed)?);
            da.insert((i, DoubleLoraTarget::TxtMlp0), LoRALinear::new(DIM, MLP_HIDDEN, rank, alpha, device.clone(), seed)?);
            da.insert((i, DoubleLoraTarget::TxtMlp2), LoRALinear::new(MLP_HIDDEN, DIM, rank, alpha, device.clone(), seed)?);
        }
        // Single blocks: 5 adapters/block × 38 blocks = 190.
        // BFL `linear1` is fused [Q | K | V | proj_mlp] = 7*DIM output (3*DIM + 4*DIM).
        // BFL `linear2` is `5*DIM → DIM` (input is cat([attn, mlp_out]) where mlp_out is 4*DIM).
        for i in 0..NUM_SINGLE {
            sa.insert((i, SingleLoraTarget::Q),       LoRALinear::new(DIM, DIM, rank, alpha, device.clone(), seed)?);
            sa.insert((i, SingleLoraTarget::K),       LoRALinear::new(DIM, DIM, rank, alpha, device.clone(), seed)?);
            sa.insert((i, SingleLoraTarget::V),       LoRALinear::new(DIM, DIM, rank, alpha, device.clone(), seed)?);
            sa.insert((i, SingleLoraTarget::ProjMlp), LoRALinear::new(DIM, 4*DIM, rank, alpha, device.clone(), seed)?);
            sa.insert((i, SingleLoraTarget::ProjOut), LoRALinear::new(5*DIM, DIM, rank, alpha, device.clone(), seed)?);
        }
        Ok(Self { double_adapters: da, single_adapters: sa })
    }

    pub fn parameters(&self) -> Vec<Parameter> {
        let mut p = Vec::new();
        for l in self.double_adapters.values() { p.extend(l.parameters()); }
        for l in self.single_adapters.values() { p.extend(l.parameters()); }
        p
    }

    /// Canonical (name, Parameter) pairs for full-checkpoint save/resume.
    /// Names match exactly what `<FluxModel as TrainableModel>::save_weights`
    /// writes: `double_blocks.{i}.{double_target_suffix(target)}.lora_{A,B}.weight`
    /// then `single_blocks.{i}.{single_target_suffix(target)}.lora_{A,B}.weight`.
    /// Order is deterministic via sorted (block_idx, target_idx) keys. The
    /// `.alpha` scalars that `save_weights` emits are NOT Parameters and are
    /// intentionally skipped (alpha is restored from CkptHeader on load).
    pub fn named_parameters(&self) -> Vec<(String, Parameter)> {
        let mut out = Vec::with_capacity(
            (self.double_adapters.len() + self.single_adapters.len()) * 2,
        );
        let mut dkeys: Vec<(usize, DoubleLoraTarget)> = self.double_adapters.keys().copied().collect();
        dkeys.sort_by_key(|(i, t)| (*i, *t as usize));
        for (i, target) in dkeys {
            let lora = &self.double_adapters[&(i, target)];
            let prefix = format!("double_blocks.{}.{}", i, double_target_suffix(target));
            out.push((format!("{prefix}.lora_A.weight"), lora.lora_a().clone()));
            out.push((format!("{prefix}.lora_B.weight"), lora.lora_b().clone()));
        }
        let mut skeys: Vec<(usize, SingleLoraTarget)> = self.single_adapters.keys().copied().collect();
        skeys.sort_by_key(|(i, t)| (*i, *t as usize));
        for (i, target) in skeys {
            let lora = &self.single_adapters[&(i, target)];
            let prefix = format!("single_blocks.{}.{}", i, single_target_suffix(target));
            out.push((format!("{prefix}.lora_A.weight"), lora.lora_a().clone()));
            out.push((format!("{prefix}.lora_B.weight"), lora.lora_b().clone()));
        }
        out
    }
}

// ── LyCORIS bundle (Phase 2b) ──────────────────────────────────────
//
// Wraps an `AdapterStore` keyed by per-target dotted names matching the
// legacy `FluxLoraBundle` save format. Per-target adapter granularity is
// identical to the legacy bundle (12 doubles/block × 19 + 5 singles/block × 38
// = 418 adapters); only the leaf-tensor algo differs.
//
// **Mutual exclusion**: `FluxModel.bundle` (legacy LoRA) and
// `FluxModel.lycoris_bundle` are mutually exclusive — the trainer picks one
// based on `--algo` at construction time. The legacy path is byte-identical
// for `--algo lora|none` (different `LoRALinear::new` seed strategy than
// `AdapterStore::build_and_push_linear` would use, so we keep the original
// `seed=42` constructor for parity).
//
// **Known infrastructure limitation** (NOT Phase 2b plumbing): the
// lycoris-rs adapter modules cache leaf `Tensor` fields and the `flame_core`
// `shared_storage` feature COWs the param storage at AdamW step time,
// silently desyncing the cached field from the optimized weight. Real
// LyCORIS training runs need the lycoris-rs forward path reworked
// (round-trip through `param.tensor()` per call, matching
// `LoRALinear::forward_delta`). Phase 2b is build-clean, review-ready —
// not GPU-tested.
pub struct FluxLycorisBundle {
    pub config: LycorisBundleConfig,
    pub store: AdapterStore,
    double_index: HashMap<(usize, DoubleLoraTarget), usize>,
    single_index: HashMap<(usize, SingleLoraTarget), usize>,
    parameters: Vec<Parameter>,
}

impl FluxLycorisBundle {
    /// Build the canonical 418-adapter LyCORIS bundle for FLUX.
    pub fn new(config: LycorisBundleConfig, device: Arc<CudaDevice>) -> Result<Self> {
        if config.algo == LycorisAlgo::None {
            return Err(crate::EriDiffusionError::Lora(
                "FluxLycorisBundle::new: algo=None — caller should use FluxLoraBundle".into(),
            ));
        }
        let mut store = AdapterStore::new(config.clone(), device.clone());
        let mut double_index: HashMap<(usize, DoubleLoraTarget), usize> = HashMap::new();
        let mut single_index: HashMap<(usize, SingleLoraTarget), usize> = HashMap::new();

        const DOUBLE_TARGETS: [DoubleLoraTarget; 12] = [
            DoubleLoraTarget::ImgQ, DoubleLoraTarget::ImgK,
            DoubleLoraTarget::ImgV, DoubleLoraTarget::ImgProj,
            DoubleLoraTarget::TxtQ, DoubleLoraTarget::TxtK,
            DoubleLoraTarget::TxtV, DoubleLoraTarget::TxtProj,
            DoubleLoraTarget::ImgMlp0, DoubleLoraTarget::ImgMlp2,
            DoubleLoraTarget::TxtMlp0, DoubleLoraTarget::TxtMlp2,
        ];
        const SINGLE_TARGETS: [SingleLoraTarget; 5] = [
            SingleLoraTarget::Q, SingleLoraTarget::K, SingleLoraTarget::V,
            SingleLoraTarget::ProjMlp, SingleLoraTarget::ProjOut,
        ];

        for i in 0..NUM_DOUBLE {
            for &target in &DOUBLE_TARGETS {
                let (in_f, out_f) = double_target_io(target);
                let name = format!("double_blocks.{}.{}", i, double_target_suffix(target));
                store.build_and_push_linear(&name, in_f, out_f, /*w_orig=*/ None)
                    .map_err(|e| crate::EriDiffusionError::Lora(format!(
                        "FluxLycorisBundle: build_and_push_linear({name}): {e}"
                    )))?;
                double_index.insert((i, target), store.adapters.len() - 1);
            }
        }
        for i in 0..NUM_SINGLE {
            for &target in &SINGLE_TARGETS {
                let (in_f, out_f) = single_target_io(target);
                let name = format!("single_blocks.{}.{}", i, single_target_suffix(target));
                store.build_and_push_linear(&name, in_f, out_f, /*w_orig=*/ None)
                    .map_err(|e| crate::EriDiffusionError::Lora(format!(
                        "FluxLycorisBundle: build_and_push_linear({name}): {e}"
                    )))?;
                single_index.insert((i, target), store.adapters.len() - 1);
            }
        }
        let parameters = store.to_parameters();
        log::info!(
            "[Flux] LyCORIS bundle: algo={} dora={} {} adapters {} parameters",
            config.algo.as_str(), config.dora,
            store.adapters.len(), parameters.len(),
        );
        Ok(Self { config, store, double_index, single_index, parameters })
    }

    pub fn parameters(&self) -> Vec<Parameter> {
        self.parameters.clone()
    }

    pub fn lookup_double(&self, idx: usize, target: DoubleLoraTarget) -> Option<&dyn AdapterModule> {
        self.double_index
            .get(&(idx, target))
            .map(|i| self.store.adapters[*i].as_ref())
    }

    pub fn lookup_single(&self, idx: usize, target: SingleLoraTarget) -> Option<&dyn AdapterModule> {
        self.single_index
            .get(&(idx, target))
            .map(|i| self.store.adapters[*i].as_ref())
    }

    /// `(name, Parameter)` pairs for full-checkpoint save/resume. Names match
    /// what `save_weights` writes: `<adapter_name>.<algo_suffix>`.
    pub fn named_parameters(&self) -> Vec<(String, Parameter)> {
        let mut out = Vec::with_capacity(self.parameters.len());
        for (i, adapter) in self.store.adapters.iter().enumerate() {
            let base = &self.store.names[i];
            let pairs = adapter.to_parameters();
            let names = adapter.named_tensors();
            for (param, (suffix, _)) in pairs.into_iter().zip(names.into_iter()) {
                out.push((format!("{base}.{suffix}"), param));
            }
        }
        out
    }
}

fn double_target_io(t: DoubleLoraTarget) -> (usize, usize) {
    match t {
        DoubleLoraTarget::ImgQ | DoubleLoraTarget::ImgK | DoubleLoraTarget::ImgV
        | DoubleLoraTarget::ImgProj
        | DoubleLoraTarget::TxtQ | DoubleLoraTarget::TxtK | DoubleLoraTarget::TxtV
        | DoubleLoraTarget::TxtProj => (DIM, DIM),
        DoubleLoraTarget::ImgMlp0 | DoubleLoraTarget::TxtMlp0 => (DIM, MLP_HIDDEN),
        DoubleLoraTarget::ImgMlp2 | DoubleLoraTarget::TxtMlp2 => (MLP_HIDDEN, DIM),
    }
}

fn single_target_io(t: SingleLoraTarget) -> (usize, usize) {
    match t {
        SingleLoraTarget::Q | SingleLoraTarget::K | SingleLoraTarget::V => (DIM, DIM),
        SingleLoraTarget::ProjMlp => (DIM, 4 * DIM),
        SingleLoraTarget::ProjOut => (5 * DIM, DIM),
    }
}

impl FluxModel {
    pub fn load(
        model_path: &std::path::Path, config: &TrainConfig, device: Arc<CudaDevice>,
    ) -> Result<Self> {
        Self::load_inner(model_path, config, device, /*skip_blocks=*/ false, /*lyc_cfg=*/ None)
    }

    /// Phase 2b: same as `load` but with optional `LycorisBundleConfig`.
    pub fn load_with_lycoris(
        model_path: &std::path::Path,
        config: &TrainConfig,
        device: Arc<CudaDevice>,
        lyc_cfg: Option<LycorisBundleConfig>,
    ) -> Result<Self> {
        Self::load_inner(model_path, config, device, /*skip_blocks=*/ false, lyc_cfg)
    }

    /// Like `load`, but skips per-block weights (`double_blocks.*` /
    /// `single_blocks.*`) at GPU-load time. Use this when the caller will
    /// immediately call `enable_offload` — avoids the transient ~24 GB GPU
    /// spike from the full-load + drop pattern. Per-block weights are
    /// streamed in via `stage_*_block` during forward instead.
    pub fn load_offload(
        model_path: &std::path::Path, config: &TrainConfig, device: Arc<CudaDevice>,
    ) -> Result<Self> {
        Self::load_inner(model_path, config, device, /*skip_blocks=*/ true, /*lyc_cfg=*/ None)
    }

    /// `load_offload` + LyCORIS — combinator for offload + lycoris-algo runs.
    pub fn load_offload_with_lycoris(
        model_path: &std::path::Path,
        config: &TrainConfig,
        device: Arc<CudaDevice>,
        lyc_cfg: Option<LycorisBundleConfig>,
    ) -> Result<Self> {
        Self::load_inner(model_path, config, device, /*skip_blocks=*/ true, lyc_cfg)
    }

    fn load_inner(
        model_path: &std::path::Path, config: &TrainConfig, device: Arc<CudaDevice>,
        skip_blocks: bool,
        lyc_cfg: Option<LycorisBundleConfig>,
    ) -> Result<Self> {
        log::info!(
            "[Flux] loading from {} (skip_blocks={})",
            model_path.display(), skip_blocks,
        );
        let all = if skip_blocks {
            flame_core::serialization::load_file_filtered(model_path, &device, |k| {
                !k.starts_with("double_blocks.") && !k.starts_with("single_blocks.")
            })?
        } else {
            flame_core::serialization::load_file(model_path, &device)?
        };
        log::info!("[Flux] {} weight tensors", all.len());

        let has_guidance = all.contains_key("guidance_in.in_layer.weight");
        log::info!("[Flux] guidance_in: {} ({})", has_guidance, if has_guidance {"Dev"} else {"Schnell"});

        let mut shared = HashMap::new();
        let mut db: Vec<_> = (0..NUM_DOUBLE).map(|_| HashMap::new()).collect();
        let mut sb: Vec<_> = (0..NUM_SINGLE).map(|_| HashMap::new()).collect();

        for (key, t) in &all {
            if let Some(rest) = key.strip_prefix("double_blocks.") {
                if let Some(idx) = parse_block_idx(rest, NUM_DOUBLE) { db[idx].insert(key.clone(), t.clone()); continue; }
            }
            if let Some(rest) = key.strip_prefix("single_blocks.") {
                if let Some(idx) = parse_block_idx(rest, NUM_SINGLE) { sb[idx].insert(key.clone(), t.clone()); continue; }
            }
            shared.insert(key.clone(), t.clone());
        }
        drop(all);

        log::info!("[Flux] {} shared, {} double blocks, {} single blocks", shared.len(), db.len(), sb.len());

        let is_fft = !config.is_lora();
        // Resolve effective LyCORIS config: an explicit `algo=None` (or no
        // config at all) takes the legacy `FluxLoraBundle` path. The two
        // bundle paths are mutually exclusive at the trainer level.
        let lyc_active: Option<LycorisBundleConfig> = lyc_cfg
            .as_ref()
            .filter(|c| c.algo != LycorisAlgo::None)
            .cloned();
        let (bundle, lycoris_bundle, fft_params) = if is_fft {
            let mut params = HashMap::new();
            for (k, t) in &shared { params.insert(k.clone(), Parameter::new(t.to_dtype(DType::F32)?.requires_grad_(true))); }
            for block in &db { for (k, t) in block { params.insert(k.clone(), Parameter::new(t.to_dtype(DType::F32)?.requires_grad_(true))); } }
            for block in &sb { for (k, t) in block { params.insert(k.clone(), Parameter::new(t.to_dtype(DType::F32)?.requires_grad_(true))); } }
            (None, None, Some(params))
        } else if let Some(lyc) = lyc_active {
            // LyCORIS path. Override rank/alpha with TrainConfig values so the
            // CLI `--rank` / `--lora-alpha` flags propagate identically.
            let lyc = LycorisBundleConfig {
                rank: config.lora_rank as usize,
                alpha: config.lora_alpha as f32,
                ..lyc
            };
            let lb = FluxLycorisBundle::new(lyc, device.clone())?;
            (None, Some(lb), None)
        } else {
            // SEED=42 (single fixed seed — see FluxLoraBundle::new for rationale).
            let b = FluxLoraBundle::new(config.lora_rank as usize, config.lora_alpha as f32, device.clone(), 42u64)?;
            (Some(b), None, None)
        };

        // Audit fix FLUX_VERIFY §H7 / SKEPTIC §H7: OT canonical training-time
        // guidance is `config.transformer.guidance_scale` (TrainConfig default
        // 1.0 — see `/home/alex/upstream Python/modules/util/config/TrainConfig.py:289`). 3.5
        // is the *inference* default; the sampler binary overrides this field
        // explicitly when generating images. Hardcoding 3.5 here at training
        // time shifted the guidance MLP's input distribution away from where
        // BFL distillation expects it.
        let guidance_value = 1.0;
        Ok(Self { config: config.clone(), device, shared_weights: shared, double_block_weights: db, single_block_weights: sb, bundle, lycoris_bundle, fft_params, is_full_finetune: is_fft, has_guidance, offloader: None, guidance_value })
    }

    /// Drop double/single block weights from VRAM and build a `BlockOffloader`
    /// that streams them from pinned host RAM into reusable GPU slots per block,
    /// per step. Unified index space: `0..NUM_DOUBLE` → double_blocks.{i},
    /// `NUM_DOUBLE..NUM_DOUBLE+NUM_SINGLE` → single_blocks.{i}.
    /// LoRA or base inference — both supported.
    pub fn enable_offload(&mut self, shards: Vec<std::path::PathBuf>) -> Result<()> {
        // Drop per-block GPU tensors so the offloader controls staging.
        let mut dropped = 0usize;
        for block in &mut self.double_block_weights {
            dropped += block.len();
            block.clear();
        }
        for block in &mut self.single_block_weights {
            dropped += block.len();
            block.clear();
        }
        log::info!("[Flux] offload: dropped {} per-block weight tensors", dropped);
        flame_core::cuda_alloc_pool::clear_pool_cache();
        flame_core::trim_cuda_mempool(0);

        struct FluxFacilitator;
        impl crate::training::block_offload::BlockFacilitator for FluxFacilitator {
            fn block_count(&self) -> usize { NUM_DOUBLE + NUM_SINGLE }
            fn classify_key(&self, key: &str) -> Option<usize> {
                if let Some(rest) = key.strip_prefix("double_blocks.") {
                    let idx: usize = rest.split('.').next()?.parse().ok()?;
                    if idx < NUM_DOUBLE { return Some(idx); }
                }
                if let Some(rest) = key.strip_prefix("single_blocks.") {
                    let idx: usize = rest.split('.').next()?.parse().ok()?;
                    if idx < NUM_SINGLE { return Some(NUM_DOUBLE + idx); }
                }
                None
            }
        }

        let shard_strs: Vec<String> = shards.iter()
            .map(|p| p.to_string_lossy().into_owned())
            .collect();
        let path_refs: Vec<&str> = shard_strs.iter().map(|s| s.as_str()).collect();

        let use_streaming = std::env::var("FLUX_BLOCK_STREAMING")
            .ok()
            .map(|v| !matches!(v.as_str(), "0" | "" | "false" | "False"))
            .unwrap_or(true); // default: streaming (low pinned-RAM footprint)

        let offloader = if use_streaming {
            log::info!("[Flux] BlockOffloader: streaming mode");
            crate::training::block_offload::BlockOffloader::load_streaming(
                &path_refs, &FluxFacilitator, self.device.clone(),
            )
        } else {
            log::info!("[Flux] BlockOffloader: pinned-RAM mode");
            crate::training::block_offload::BlockOffloader::load(
                &path_refs, &FluxFacilitator, self.device.clone(),
            )
        }
        // native_layout=true: leave 2D .weight tensors in on-disk [Cout, Cin] layout.
        // Flux model code calls `.transpose()` itself before matmul — pre-transposing
        // here would invert the shape and cause a matmul dimension mismatch.
        .map(|o| o.with_native_layout(true))
        .map_err(|e| crate::EriDiffusionError::Model(format!("BlockOffloader: {e}")))?;

        self.offloader = Some(std::sync::Arc::new(std::sync::Mutex::new(offloader)));
        log::info!("[Flux] BlockOffloader ready ({} unified blocks)", NUM_DOUBLE + NUM_SINGLE);
        Ok(())
    }

    // ── Primitives ──────────────────────────────────────────────────

    fn dw(&self, idx: usize, suffix: &str) -> Result<&Tensor> {
        self.double_block_weights[idx].get(&format!("double_blocks.{}.{}", idx, suffix))
            .ok_or_else(|| crate::EriDiffusionError::Model(format!("missing DW: {}.{}", idx, suffix)))
    }
    fn singw(&self, idx: usize, suffix: &str) -> Result<&Tensor> {
        self.single_block_weights[idx].get(&format!("single_blocks.{}.{}", idx, suffix))
            .ok_or_else(|| crate::EriDiffusionError::Model(format!("missing SW: {}.{}", idx, suffix)))
    }
    fn sw(&self, key: &str) -> Result<&Tensor> {
        self.shared_weights.get(key)
            .ok_or_else(|| crate::EriDiffusionError::Model(format!("missing shared: {}", key)))
    }

    /// Linear with bias: x @ weight^T + bias. Autograd-recording, handles 3D input.
    fn linear(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let in_feat = *dims.last().unwrap();
        let batch: usize = dims[..dims.len()-1].iter().product();
        let out_feat = weight.shape().dims()[0];
        let x_2d = x.reshape(&[batch, in_feat])?;
        let wt = weight.transpose()?;
        let out_2d = x_2d.matmul(&wt)?.add(bias)?;
        let mut out_shape = dims[..dims.len()-1].to_vec();
        out_shape.push(out_feat);
        Ok(out_2d.reshape(&out_shape)?)
    }

    /// MLP embedder: Linear → SiLU → Linear
    fn mlp_embedder(&self, x: &Tensor, w1k: &str, b1k: &str, w2k: &str, b2k: &str) -> Result<Tensor> {
        let h = Self::linear(x, self.sw(w1k)?, self.sw(b1k)?)?;
        let h = h.silu()?;
        Self::linear(&h, self.sw(w2k)?, self.sw(b2k)?)
    }

    /// Sinusoidal timestep embedding. BFL model.py: `timestep_embedding`.
    ///
    /// Caller passes `t ∈ [0, 1)` (sigma directly); this function multiplies
    /// by `time_factor=1000` exactly once before forming the sinusoid arguments.
    /// Audit fix FLUX_VERIFY §H1 / SKEPTIC §H1: previously the trainer also
    /// passed `t ∈ [0, 1000)` so the multiply produced `t * 1_000_000`,
    /// wrapping the sinusoid into the wrong frequency band entirely.
    /// Mirrors the Klein fix at `models/klein.rs::timestep_embedding`.
    fn timestep_embedding(t: &Tensor, dim: usize, device: &Arc<CudaDevice>) -> Result<Tensor> {
        let t_f32 = t.to_dtype(DType::F32)?;
        let t_vec = t_f32.to_vec()?;
        let b = t_vec.len();
        let half = dim / 2;
        let mut data = vec![0f32; b * dim];
        for (bi, &tv) in t_vec.iter().enumerate() {
            let scaled = tv * 1000.0; // BFL `time_factor`
            for j in 0..half {
                let freq = (-(10000.0f64.ln()) * (j as f64) / (half as f64)).exp() as f32;
                let angle = scaled * freq;
                data[bi * dim + j] = angle.cos();
                data[bi * dim + half + j] = angle.sin();
            }
        }
        Tensor::from_slice(&data, Shape::from_dims(&[b, dim]), device.clone())?
            .to_dtype(DType::BF16).map_err(Into::into)
    }

    /// Per-head RMSNorm: reshape to [B*H, HEAD_DIM], norm, reshape back.
    fn rms_norm_per_head(x: &Tensor, scale: &Tensor) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let batch: usize = dims[..dims.len()-1].iter().product();
        let full_dim = *dims.last().unwrap();
        let x_heads = x.reshape(&[batch * NUM_HEADS, HEAD_DIM])?;
        let normed = flame_core::norm::rms_norm(&x_heads, &[HEAD_DIM], Some(scale), NORM_EPS)?;
        normed.reshape(&dims).map_err(Into::into)
    }

    /// 3-axis RoPE: ids [N, 3], works with bfs4d::rope_fused_bf16.
    fn build_rope(ids: &Tensor) -> Result<(Tensor, Tensor)> {
        let ids_f32 = ids.to_dtype(DType::F32)?.to_vec()?;
        let n = ids.shape().dims()[0];
        let half_dim = HEAD_DIM / 2;
        let mut cos_data = vec![0f32; n * half_dim];
        let mut sin_data = vec![0f32; n * half_dim];
        let mut offset = 0;
        for (axis, &axis_dim) in ROPE_AXES.iter().enumerate() {
            let half_ax = axis_dim / 2;
            for (i, row) in ids_f32.chunks(3).enumerate() {
                let pos = row[axis];
                for j in 0..half_ax {
                    let freq = (ROPE_THETA.powf(-(2.0 * j as f64) / axis_dim as f64)) as f32;
                    let angle = pos * freq;
                    cos_data[i * half_dim + offset + j] = angle.cos();
                    sin_data[i * half_dim + offset + j] = angle.sin();
                }
            }
            offset += half_ax;
        }
        // Keep cos/sin in F32. Per inference-flame's `build_rope_2d`
        // (flux1_dit.rs:458-461): the ~4e-3 BF16 floor on cos/sin
        // accumulates across `blocks × steps × (Q+K) ≈ 2280` RoPE
        // applications per inference and shows up as dense per-pixel
        // speckle noise on the output. Use the `_f32pe` variant of the
        // fused RoPE kernel which accepts F32 cos/sin against BF16 q/k.
        let cos = Tensor::from_slice(&cos_data, Shape::from_dims(&[1, 1, n, half_dim]), flame_core::global_cuda_device())?;
        let sin = Tensor::from_slice(&sin_data, Shape::from_dims(&[1, 1, n, half_dim]), flame_core::global_cuda_device())?;
        Ok((cos, sin))
    }

    fn apply_rope(q: &Tensor, k: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<(Tensor, Tensor)> {
        Ok((flame_core::bf16_ops::rope_fused_bf16_f32pe(q, cos, sin)?, flame_core::bf16_ops::rope_fused_bf16_f32pe(k, cos, sin)?))
    }

    // ── Double block forward ────────────────────────────────────────

    /// Apply 3 split-QKV LoRAs to a fused QKV output. Each adapter sees the
    /// same input (`x_in`) and produces a `[B, N, DIM]` delta; deltas are
    /// concatenated along the last axis to match the fused base output of
    /// shape `[B, N, 3*DIM]`. Audit fix FLUX_VERIFY §H4 / SKEPTIC §H4 / §H5
    /// (Klein parity — see `models/klein.rs::linear_with_split_qkv_lora`).
    fn add_split_qkv_lora(
        base: &Tensor,
        x_in: &Tensor,
        lora_q: Option<&dyn AdapterModule>,
        lora_k: Option<&dyn AdapterModule>,
        lora_v: Option<&dyn AdapterModule>,
    ) -> Result<Tensor> {
        if lora_q.is_none() && lora_k.is_none() && lora_v.is_none() {
            return Ok(base.clone());
        }
        // Each delta is [B, N, DIM]; missing adapter → zeros of matching shape.
        let zeros_dim = || -> Result<Tensor> {
            let mut shape = base.shape().dims().to_vec();
            *shape.last_mut().unwrap() = DIM;
            Ok(Tensor::zeros_dtype(Shape::from_dims(&shape), base.dtype(), base.device().clone())?)
        };
        // `AdapterModule::forward_delta` returns `flame_core::Result`; map back
        // to the EDv2 `Result` so callsites stay typed against `crate::Result`.
        let call = |a: &dyn AdapterModule, x: &Tensor| -> Result<Tensor> {
            a.forward_delta(x)
                .map_err(|e| crate::EriDiffusionError::Lora(format!("AdapterModule::forward_delta: {e}")))
        };
        let dq = match lora_q { Some(a) => call(a, x_in)?, None => zeros_dim()? };
        let dk = match lora_k { Some(a) => call(a, x_in)?, None => zeros_dim()? };
        let dv = match lora_v { Some(a) => call(a, x_in)?, None => zeros_dim()? };
        let delta = Tensor::cat(&[&dq, &dk, &dv], 2)?.contiguous()?;
        base.add(&delta).map_err(Into::into)
    }

    /// Per-call adapter lookup. Returns `Some(&dyn AdapterModule)` from
    /// whichever bundle is active. Mutually exclusive at construction time.
    fn lookup_double_adapter(&self, idx: usize, target: DoubleLoraTarget) -> Option<&dyn AdapterModule> {
        if let Some(ref b) = self.bundle {
            if let Some(l) = b.double_adapters.get(&(idx, target)) {
                return Some(l as &dyn AdapterModule);
            }
        }
        if let Some(ref lb) = self.lycoris_bundle {
            return lb.lookup_double(idx, target);
        }
        None
    }

    fn lookup_single_adapter(&self, idx: usize, target: SingleLoraTarget) -> Option<&dyn AdapterModule> {
        if let Some(ref b) = self.bundle {
            if let Some(l) = b.single_adapters.get(&(idx, target)) {
                return Some(l as &dyn AdapterModule);
            }
        }
        if let Some(ref lb) = self.lycoris_bundle {
            return lb.lookup_single(idx, target);
        }
        None
    }

    fn double_block_forward(&self, img: &Tensor, txt: &Tensor, vec: &Tensor, cos: &Tensor, sin: &Tensor, idx: usize) -> Result<(Tensor, Tensor)> {
        let dims = img.shape().dims().to_vec();
        let (b, n_img) = (dims[0], dims[1]);
        let n_txt = txt.shape().dims()[1];

        // Modulation
        let img_mod = Self::linear(&vec.silu()?, self.dw(idx, "img_mod.lin.weight")?, self.dw(idx, "img_mod.lin.bias")?)?;
        let img_mods = img_mod.unsqueeze(1)?.chunk(6, 2)?;
        let (img_s1, img_scale1, img_g1): (_,&Tensor,&Tensor) = (&img_mods[0], &img_mods[1], &img_mods[2]);
        let (img_s2, img_scale2, img_g2) = (&img_mods[3], &img_mods[4], &img_mods[5]);

        let txt_mod = Self::linear(&vec.silu()?, self.dw(idx, "txt_mod.lin.weight")?, self.dw(idx, "txt_mod.lin.bias")?)?;
        let txt_mods = txt_mod.unsqueeze(1)?.chunk(6, 2)?;
        let (txt_s1, txt_scale1, txt_g1) = (&txt_mods[0], &txt_mods[1], &txt_mods[2]);
        let (txt_s2, txt_scale2, txt_g2) = (&txt_mods[3], &txt_mods[4], &txt_mods[5]);

        // --- Img attention --- (split Q/K/V LoRA; H4/H5)
        let img_norm = flame_core::layer_norm::layer_norm(img, &[DIM], None, None, NORM_EPS)?;
        let img_mod_in = img_norm.mul(&img_scale1.add_scalar(1.0)?)?.add(img_s1)?;
        let img_qkv = Self::linear(&img_mod_in, self.dw(idx, "img_attn.qkv.weight")?, self.dw(idx, "img_attn.qkv.bias")?)?;
        let img_qkv = Self::add_split_qkv_lora(
            &img_qkv, &img_mod_in,
            self.lookup_double_adapter(idx, DoubleLoraTarget::ImgQ),
            self.lookup_double_adapter(idx, DoubleLoraTarget::ImgK),
            self.lookup_double_adapter(idx, DoubleLoraTarget::ImgV),
        )?;
        let c = img_qkv.chunk(3, 2)?;
        let (img_q, img_k, img_v) = (c[0].clone(), c[1].clone(), c[2].clone());

        // --- Txt attention --- (split Q/K/V LoRA)
        let txt_norm = flame_core::layer_norm::layer_norm(txt, &[DIM], None, None, NORM_EPS)?;
        let txt_mod_in = txt_norm.mul(&txt_scale1.add_scalar(1.0)?)?.add(txt_s1)?;
        let txt_qkv = Self::linear(&txt_mod_in, self.dw(idx, "txt_attn.qkv.weight")?, self.dw(idx, "txt_attn.qkv.bias")?)?;
        let txt_qkv = Self::add_split_qkv_lora(
            &txt_qkv, &txt_mod_in,
            self.lookup_double_adapter(idx, DoubleLoraTarget::TxtQ),
            self.lookup_double_adapter(idx, DoubleLoraTarget::TxtK),
            self.lookup_double_adapter(idx, DoubleLoraTarget::TxtV),
        )?;
        let c = txt_qkv.chunk(3, 2)?;
        let (txt_q, txt_k, txt_v) = (c[0].clone(), c[1].clone(), c[2].clone());

        // QK norm
        let img_q = Self::rms_norm_per_head(&img_q, self.dw(idx, "img_attn.norm.query_norm.scale")?)?;
        let img_k = Self::rms_norm_per_head(&img_k, self.dw(idx, "img_attn.norm.key_norm.scale")?)?;
        let txt_q = Self::rms_norm_per_head(&txt_q, self.dw(idx, "txt_attn.norm.query_norm.scale")?)?;
        let txt_k = Self::rms_norm_per_head(&txt_k, self.dw(idx, "txt_attn.norm.key_norm.scale")?)?;

        // Reshape → [B, H, N, D]
        let (img_q, img_k, img_v) = (reshape_qkv(&img_q, b, n_img)?, reshape_qkv(&img_k, b, n_img)?, reshape_qkv(&img_v, b, n_img)?);
        let (txt_q, txt_k, txt_v) = (reshape_qkv(&txt_q, b, n_txt)?, reshape_qkv(&txt_k, b, n_txt)?, reshape_qkv(&txt_v, b, n_txt)?);

        // Joint attention. `.contiguous()` after each cat — H9 / GOTCHAS §2.4:
        // Tensor::cat may return non-contiguous views; downstream BF16 SDPA /
        // rope_fused_bf16 read as if contig and silently garble (cos≈0.99,
        // max_abs ~1.8). See `feedback_flame_core_cat_contig.md`.
        let q = Tensor::cat(&[&txt_q, &img_q], 2)?.contiguous()?;
        let k = Tensor::cat(&[&txt_k, &img_k], 2)?.contiguous()?;
        let v = Tensor::cat(&[&txt_v, &img_v], 2)?.contiguous()?;
        let (q, k) = Self::apply_rope(&q, &k, cos, sin)?;

        let attn = flame_core::attention::sdpa(&q, &k, &v, None)?;
        // FLUX speckle bug bisect (2026-05-07): the prior BlockOffloader
        // port replaced the fused `attn_split_txt_img_bf16` with manual
        // permute+reshape+narrow. The fused kernel handles the
        // [B,H,N,D] → ([B,n_txt,DIM], [B,n_img,DIM]) transform correctly;
        // the manual replacement (with .contiguous()) produced byte-
        // identical speckle output. Restoring the fused kernel.
        let (txt_attn, img_attn) = flame_core::bf16_ops::attn_split_txt_img_bf16(&attn, n_txt, n_img)?;

        // Adapter-delta helper: dispatches `forward_delta` through the
        // AdapterModule trait. `LoRALinear` impls AdapterModule directly so
        // the legacy path still hits its tape-recording matmul fast path.
        let apply_delta = |a: &dyn AdapterModule, x: &Tensor| -> Result<Tensor> {
            a.forward_delta(x)
                .map_err(|e| crate::EriDiffusionError::Lora(format!("AdapterModule::forward_delta: {e}")))
        };

        // Output proj + gate + residual
        let img_proj = Self::linear(&img_attn, self.dw(idx, "img_attn.proj.weight")?, self.dw(idx, "img_attn.proj.bias")?)?;
        let img_proj = if let Some(lora) = self.lookup_double_adapter(idx, DoubleLoraTarget::ImgProj) {
            img_proj.add(&apply_delta(lora, &img_attn)?)?
        } else { img_proj };
        // FLUX speckle bisect (2026-05-07): replace manual mul+add with the
        // fused gate_residual_fused_bf16 kernel (the OLD inference path the
        // BlockOffloader port removed). Eliminates any potential broadcast
        // bug between [B,1,DIM] gate and [B,N,DIM] proj.
        let img = flame_core::bf16_ops::gate_residual_fused_bf16(&img, &img_g1.squeeze(Some(1))?, &img_proj)?;

        let txt_proj = Self::linear(&txt_attn, self.dw(idx, "txt_attn.proj.weight")?, self.dw(idx, "txt_attn.proj.bias")?)?;
        let txt_proj = if let Some(lora) = self.lookup_double_adapter(idx, DoubleLoraTarget::TxtProj) {
            txt_proj.add(&apply_delta(lora, &txt_attn)?)?
        } else { txt_proj };
        let txt = flame_core::bf16_ops::gate_residual_fused_bf16(&txt, &txt_g1.squeeze(Some(1))?, &txt_proj)?;

        // --- GELU MLP --- (img + txt MLP up/down LoRAs added per H5)
        let img_norm2 = flame_core::layer_norm::layer_norm(&img, &[DIM], None, None, NORM_EPS)?;
        let img_mlp_in = img_norm2.mul(&img_scale2.add_scalar(1.0)?)?.add(img_s2)?;
        let img_mlp_h_base = Self::linear(&img_mlp_in, self.dw(idx, "img_mlp.0.weight")?, self.dw(idx, "img_mlp.0.bias")?)?;
        let img_mlp_h = if let Some(lora) = self.lookup_double_adapter(idx, DoubleLoraTarget::ImgMlp0) {
            img_mlp_h_base.add(&apply_delta(lora, &img_mlp_in)?)?
        } else { img_mlp_h_base };
        let img_mlp_h = img_mlp_h.gelu()?;
        let img_mlp_out_base = Self::linear(&img_mlp_h, self.dw(idx, "img_mlp.2.weight")?, self.dw(idx, "img_mlp.2.bias")?)?;
        let img_mlp_out = if let Some(lora) = self.lookup_double_adapter(idx, DoubleLoraTarget::ImgMlp2) {
            img_mlp_out_base.add(&apply_delta(lora, &img_mlp_h)?)?
        } else { img_mlp_out_base };
        let img = flame_core::bf16_ops::gate_residual_fused_bf16(&img, &img_g2.squeeze(Some(1))?, &img_mlp_out)?;

        let txt_norm2 = flame_core::layer_norm::layer_norm(&txt, &[DIM], None, None, NORM_EPS)?;
        let txt_mlp_in = txt_norm2.mul(&txt_scale2.add_scalar(1.0)?)?.add(txt_s2)?;
        let txt_mlp_h_base = Self::linear(&txt_mlp_in, self.dw(idx, "txt_mlp.0.weight")?, self.dw(idx, "txt_mlp.0.bias")?)?;
        let txt_mlp_h = if let Some(lora) = self.lookup_double_adapter(idx, DoubleLoraTarget::TxtMlp0) {
            txt_mlp_h_base.add(&apply_delta(lora, &txt_mlp_in)?)?
        } else { txt_mlp_h_base };
        let txt_mlp_h = txt_mlp_h.gelu()?;
        let txt_mlp_out_base = Self::linear(&txt_mlp_h, self.dw(idx, "txt_mlp.2.weight")?, self.dw(idx, "txt_mlp.2.bias")?)?;
        let txt_mlp_out = if let Some(lora) = self.lookup_double_adapter(idx, DoubleLoraTarget::TxtMlp2) {
            txt_mlp_out_base.add(&apply_delta(lora, &txt_mlp_h)?)?
        } else { txt_mlp_out_base };
        let txt = flame_core::bf16_ops::gate_residual_fused_bf16(&txt, &txt_g2.squeeze(Some(1))?, &txt_mlp_out)?;

        Ok((img, txt))
    }

    // ── Single block forward ────────────────────────────────────────

    fn single_block_forward(&self, x: &Tensor, vec: &Tensor, cos: &Tensor, sin: &Tensor, _txt_len: usize, idx: usize) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let (b, n) = (dims[0], dims[1]);

        // Adapter-delta helper (see double_block_forward for rationale).
        let apply_delta = |a: &dyn AdapterModule, xx: &Tensor| -> Result<Tensor> {
            a.forward_delta(xx)
                .map_err(|e| crate::EriDiffusionError::Lora(format!("AdapterModule::forward_delta: {e}")))
        };

        // Modulation: Linear(vec.silu()) → 3*DIM
        let m = Self::linear(&vec.silu()?, self.singw(idx, "modulation.lin.weight")?, self.singw(idx, "modulation.lin.bias")?)?;
        let mc = m.unsqueeze(1)?.chunk(3, 2)?;
        let (shift, scale, gate) = (&mc[0], &mc[1], &mc[2]);

        let x_norm = flame_core::layer_norm::layer_norm(x, &[DIM], None, None, NORM_EPS)?;
        let x_mod = x_norm.mul(&scale.add_scalar(1.0)?)?.add(shift)?;

        // linear1: [B, N, 7*DIM] → QKV (3*DIM) + MLP (4*DIM)
        let l1 = Self::linear(&x_mod, self.singw(idx, "linear1.weight")?, self.singw(idx, "linear1.bias")?)?;
        let qkv = l1.narrow(2, 0, 3*DIM)?;
        let mlp_in_base = l1.narrow(2, 3*DIM, 4*DIM)?;

        // Q/K/V split LoRAs (H4/H5: 3 separate adapters on the 3 slices of the
        // fused QKV output — Klein parity).
        let qkv = Self::add_split_qkv_lora(
            &qkv, &x_mod,
            self.lookup_single_adapter(idx, SingleLoraTarget::Q),
            self.lookup_single_adapter(idx, SingleLoraTarget::K),
            self.lookup_single_adapter(idx, SingleLoraTarget::V),
        )?;

        // ProjMlp LoRA on the MLP-up half of the fused linear1 (H5: previously
        // the entire MLP-up branch had no LoRA correction).
        let mlp_in = if let Some(lora) = self.lookup_single_adapter(idx, SingleLoraTarget::ProjMlp) {
            mlp_in_base.add(&apply_delta(lora, &x_mod)?)?
        } else { mlp_in_base };

        let c = qkv.chunk(3, 2)?;
        let (q, k, v) = (c[0].clone(), c[1].clone(), c[2].clone());

        let q = Self::rms_norm_per_head(&q, self.singw(idx, "norm.query_norm.scale")?)?;
        let k = Self::rms_norm_per_head(&k, self.singw(idx, "norm.key_norm.scale")?)?;

        let (q, k, v) = (reshape_qkv(&q, b, n)?, reshape_qkv(&k, b, n)?, reshape_qkv(&v, b, n)?);
        let (q, k) = Self::apply_rope(&q, &k, cos, sin)?;
        let attn = flame_core::attention::sdpa(&q, &k, &v, None)?;
        let attn = attn.permute(&[0, 2, 1, 3])?.reshape(&[b, n, DIM])?;

        // GELU MLP
        let mlp_out = mlp_in.gelu()?;
        // H9: explicit `.contiguous()` after cat — `linear2` is `5*DIM → DIM`
        // and reads the fused buffer as a flat `[B, N, 5*DIM]` 3D matmul input.
        let fused = Tensor::cat(&[&attn, &mlp_out], 2)?.contiguous()?;
        let l2 = Self::linear(&fused, self.singw(idx, "linear2.weight")?, self.singw(idx, "linear2.bias")?)?;
        // H4: `ProjOut` LoRA wraps the **full fused 5*DIM input** mapping to
        // DIM, matching BFL's actual `linear2` shape. Pre-fix this was a
        // phantom `DIM→DIM` adapter receiving only `attn` (1/5 of the input)
        // — silently shape-checked, MLP-up half got no correction.
        let l2 = if let Some(lora) = self.lookup_single_adapter(idx, SingleLoraTarget::ProjOut) {
            l2.add(&apply_delta(lora, &fused)?)?
        } else { l2 };

        flame_core::bf16_ops::gate_residual_fused_bf16(x, &gate.squeeze(Some(1))?, &l2)
            .map_err(Into::into)
    }

    // ── Full forward ─────────────────────────────────────────────────

    pub fn forward(
        &mut self,
        img: &Tensor,
        txt: &Tensor,
        timesteps: &Tensor,
        img_ids: &Tensor,
        txt_ids: &Tensor,
        guidance: Option<&Tensor>,
        vector: &Tensor,
    ) -> Result<Tensor> {
        let n_img = img.shape().dims()[1];
        let n_txt = txt.shape().dims()[1];

        // Input projections
        let img = Self::linear(img, self.sw("img_in.weight")?, self.sw("img_in.bias")?)?;
        let txt = Self::linear(txt, self.sw("txt_in.weight")?, self.sw("txt_in.bias")?)?;

        // Time + guidance + vector embeddings
        let t_emb = Self::timestep_embedding(timesteps, TIMESTEP_DIM, &self.device)?;
        let mut vec = self.mlp_embedder(&t_emb, "time_in.in_layer.weight", "time_in.in_layer.bias", "time_in.out_layer.weight", "time_in.out_layer.bias")?;

        if let Some(g) = guidance {
            if self.has_guidance {
                let g_emb = Self::timestep_embedding(g, TIMESTEP_DIM, &self.device)?;
                let gv = self.mlp_embedder(&g_emb, "guidance_in.in_layer.weight", "guidance_in.in_layer.bias", "guidance_in.out_layer.weight", "guidance_in.out_layer.bias")?;
                vec = vec.add(&gv)?;
            }
        }

        let vv = self.mlp_embedder(vector, "vector_in.in_layer.weight", "vector_in.in_layer.bias", "vector_in.out_layer.weight", "vector_in.out_layer.bias")?;
        vec = vec.add(&vv)?;

        // RoPE — `build_rope` reads `all_ids` via `to_vec()` so non-contiguous
        // input is harmless here (CPU-side gather). Kept idiomatic for clarity.
        let all_ids = Tensor::cat(&[txt_ids, img_ids], 0)?;
        // Keep cos/sin F32. The fused `rope_fused_bf16_f32pe` kernel accepts
        // F32 PE against BF16 q/k. Pre-fix this dtype-cast to BF16 was the
        // load-bearing FLUX speckle bug — the BF16 floor on cos/sin
        // accumulated across 57×20×2 RoPE applications and produced dense
        // per-pixel speckle on the output (per inference-flame's comment in
        // `flux1_dit.rs::build_rope_2d`).
        let (cos, sin) = Self::build_rope(&all_ids)?;

        // Double blocks
        // Note: Flux double blocks return (img, txt) — gradient checkpointing
        // via flame_core::AutogradContext::checkpoint() returns a single Tensor,
        // so we'd need to fuse along a fresh axis to checkpoint. Left as a
        // future optimization — VRAM headroom is gained via the BlockOffloader
        // (per-block weight streaming) instead.
        let (mut img, mut txt) = (img, txt);
        for i in 0..NUM_DOUBLE {
            // BlockOffloader: stream block i from pinned host RAM into GPU slot.
            if let Some(ref off) = self.offloader {
                let arc = off.lock()
                    .map_err(|e| crate::EriDiffusionError::Model(format!("offloader lock: {e}")))?
                    .ensure_block(i)
                    .map_err(|e| crate::EriDiffusionError::Model(format!("offloader ensure_block({i}): {e}")))?;
                self.double_block_weights[i] = (*arc).clone();
            }
            let (ni, nt) = self.double_block_forward(&img, &txt, &vec, &cos, &sin, i)?;
            if let Some(ref off) = self.offloader {
                self.double_block_weights[i].clear();
                off.lock()
                    .map_err(|e| crate::EriDiffusionError::Model(format!("offloader lock: {e}")))?
                    .evict_block();
            }
            img = ni; txt = nt;
        }

        // Merge + single blocks. H9: `.contiguous()` after cat — the merged
        // joint sequence feeds into single_block_forward whose first op is a
        // BF16 layer_norm + linear matmul.
        let mut merged = Tensor::cat(&[&txt, &img], 1)?.contiguous()?;
        for i in 0..NUM_SINGLE {
            let unified_idx = NUM_DOUBLE + i;
            if let Some(ref off) = self.offloader {
                let arc = off.lock()
                    .map_err(|e| crate::EriDiffusionError::Model(format!("offloader lock: {e}")))?
                    .ensure_block(unified_idx)
                    .map_err(|e| crate::EriDiffusionError::Model(format!("offloader ensure_block({unified_idx}): {e}")))?;
                self.single_block_weights[i] = (*arc).clone();
            }
            merged = self.single_block_forward(&merged, &vec, &cos, &sin, n_txt, i)?;
            if let Some(ref off) = self.offloader {
                self.single_block_weights[i].clear();
                off.lock()
                    .map_err(|e| crate::EriDiffusionError::Model(format!("offloader lock: {e}")))?
                    .evict_block();
            }
        }

        // Extract img + final layer.
        //
        // FLUX speckle bug FIX (2026-05-07): the final layer in BFL FLUX is
        //     out = linear(modulate(layer_norm(x), shift, scale))
        // where (shift, scale) come from `Linear(silu(vec))` against the
        // `final_layer.adaLN_modulation.1.{weight,bias}` weights. Pre-fix
        // EDv2 only did `layer_norm → linear`, skipping the modulate. The
        // missing modulation produced dense per-pixel speckle output (the
        // post-norm activations were never centred/scaled by the
        // timestep-conditioned (shift, scale) so the final linear emitted
        // a noisy magnitude-band that VAE-decoded into per-pixel noise).
        // Mirrors `inference-flame::flux1_dit::final_layer_forward`
        // (flux1_dit.rs:1010-1033).
        let img_out = merged.narrow(1, n_txt, n_img)?;
        let vec_act = vec.silu()?;
        let mods = Self::linear(
            &vec_act.unsqueeze(1)?,
            self.sw("final_layer.adaLN_modulation.1.weight")?,
            self.sw("final_layer.adaLN_modulation.1.bias")?,
        )?.squeeze(Some(1))?;
        let final_shift = mods.narrow(1, 0, DIM)?;
        let final_scale = mods.narrow(1, DIM, DIM)?;

        let i_norm = flame_core::layer_norm::layer_norm(&img_out, &[DIM], None, None, NORM_EPS)?;
        // modulate: x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        let i_mod = i_norm
            .mul(&final_scale.add_scalar(1.0)?.unsqueeze(1)?)?
            .add(&final_shift.unsqueeze(1)?)?;
        let i_linear = Self::linear(&i_mod, self.sw("final_layer.linear.weight")?, self.sw("final_layer.linear.bias")?)?;

        Ok(i_linear)
    }

    // ── Helpers ──────────────────────────────────────────────────────

    fn dw_optional(&self, idx: usize, suffix: &str) -> Result<Option<&Tensor>> {
        match self.double_block_weights[idx].get(&format!("double_blocks.{}.{}", idx, suffix)) {
            Some(t) => Ok(Some(t)),
            None => Ok(None),
        }
    }
    fn sw_optional(&self, key: &str) -> Result<Option<&Tensor>> {
        Ok(self.shared_weights.get(key))
    }
}

// ── Standalone helpers ──────────────────────────────────────────────

fn parse_block_idx(rest: &str, max: usize) -> Option<usize> {
    rest.find('.').and_then(|d| rest[..d].parse::<usize>().ok()).filter(|&i| i < max)
}

fn reshape_qkv(x: &Tensor, b: usize, n: usize) -> Result<Tensor> {
    x.reshape(&[b, n, NUM_HEADS, HEAD_DIM])?.permute(&[0, 2, 1, 3]).map_err(Into::into)
}

// ── TrainableModel trait ────────────────────────────────────────────

impl TrainableModel for FluxModel {
    /// Training forward — accepts pre-cached data from CachedDataset.
    /// `noisy`:    packed latents [B, N_img, 64] (already pack_latents'd at cache time)
    /// `context[0]`: T5-XXL embeddings [B, N_txt, 4096]
    /// `context[1]`: img_ids [N_img, 3]  — pre-computed at cache time
    /// `context[2]`: txt_ids [N_txt, 3]  — pre-computed at cache time
    /// `pooled`:   CLIP-L pooled [B, 768]
    ///
    /// Position IDs MUST be supplied — generating zeros here would silently break
    /// RoPE for image tokens (Flux uses row/col coords on axis 1/2).
    fn forward(&mut self, noisy: &Tensor, timestep: &Tensor, context: &[Tensor], pooled: Option<&Tensor>) -> Result<Tensor> {
        let t5 = context.first()
            .ok_or_else(|| crate::EriDiffusionError::Model("Flux requires T5 embeddings (context[0])".into()))?;
        let img_ids = context.get(1)
            .ok_or_else(|| crate::EriDiffusionError::Model("Flux requires img_ids (context[1])".into()))?;
        let txt_ids = context.get(2)
            .ok_or_else(|| crate::EriDiffusionError::Model("Flux requires txt_ids (context[2])".into()))?;

        let b = noisy.shape().dims()[0];
        let guidance_val = self.guidance_value;
        let guidance = Tensor::from_vec(
            vec![guidance_val; b],
            Shape::from_dims(&[b]),
            self.device.clone(),
        )?;

        let default_pool = Tensor::zeros(
            Shape::from_dims(&[b, VECTOR_DIM]),
            self.device.clone(),
        )?;
        let vector = pooled.unwrap_or(&default_pool);

        let guidance_opt = if self.has_guidance { Some(&guidance) } else { None };
        FluxModel::forward(self, noisy, t5, timestep, img_ids, txt_ids, guidance_opt, vector)
    }

    fn parameters(&self) -> Vec<Parameter> {
        if let Some(ref fft) = self.fft_params { return fft.values().cloned().collect(); }
        if let Some(ref b) = self.bundle { return b.parameters(); }
        if let Some(ref lb) = self.lycoris_bundle { return lb.parameters(); }
        Vec::new()
    }

    fn post_optimizer_step(&mut self) {
        if let Some(ref b) = self.bundle {
            for l in b.double_adapters.values() { l.refresh_cache(); }
            for l in b.single_adapters.values() { l.refresh_cache(); }
        }
        // LyCORIS adapters do not currently expose a `refresh_cache()` analog
        // (lycoris-rs leaves are bare Tensors, not `LoRALinear`-style cached
        // matrices); no-op here. See the `FluxLycorisBundle` doc comment for
        // the shared_storage / COW concern that motivates the future fix.
        if let Some(ref fft) = self.fft_params {
            // Sync FFT weights back to BF16 for forward pass
            for (key, param) in fft {
                if let Ok(t) = param.tensor() {
                    let t = t.to_dtype(DType::BF16).unwrap_or(t);
                    let key = key.clone();
                    if let Some(rest) = key.strip_prefix("double_blocks.") {
                        if let Some(idx) = parse_block_idx(rest, NUM_DOUBLE) {
                            self.double_block_weights[idx].insert(key, t);
                            continue;
                        }
                    }
                    if let Some(rest) = key.strip_prefix("single_blocks.") {
                        if let Some(idx) = parse_block_idx(rest, NUM_SINGLE) {
                            self.single_block_weights[idx].insert(key, t);
                            continue;
                        }
                    }
                    self.shared_weights.insert(key, t);
                }
            }
        }
    }

    /// Save LoRA adapters in the diffusers/PEFT-style key naming PLUS a
    /// per-module `.alpha` scalar tensor. Audit fix FLUX_VERIFY §H5 / §H6 /
    /// SKEPTIC §H6: ecosystem loaders (kohya, ComfyUI, A1111, diffusers
    /// `load_lora_weights`) read `.alpha` to compute `scale = alpha / rank`.
    /// Without it they fall back to `scale = 1.0`, so a checkpoint trained
    /// with `alpha=1, rank=16` (in-trainer scale = 1/16) is silently amplified
    /// 16× on load → blown-out output.
    ///
    /// Per-block diffusers-style key paths used (matches `convert_flux_lora`
    /// expectations after Q/K/V split):
    ///   double_blocks.{i}.img_attn.{to_q,to_k,to_v,proj}
    ///   double_blocks.{i}.txt_attn.{to_q,to_k,to_v,proj}
    ///   double_blocks.{i}.{img,txt}_mlp.{0,2}
    ///   single_blocks.{i}.attn.{to_q,to_k,to_v}
    ///   single_blocks.{i}.{proj_mlp,proj_out}
    ///
    /// `LoRALinear::save_tensors` writes `<prefix>.lora_A.weight` and
    /// `<prefix>.lora_B.weight` (PEFT diffusers convention — same matrices as
    /// kohya `lora_down`/`lora_up`, different suffix). The `.alpha` companion
    /// is always emitted (matches SDXL's recently-landed pattern).
    fn save_weights(&self, path: &str) -> Result<()> {
        let mut tensors = HashMap::new();
        let emit_alpha = |prefix: &str, alpha: f32, out: &mut HashMap<String, Tensor>| -> Result<()> {
            let alpha_t = Tensor::from_vec(
                vec![alpha],
                Shape::from_dims(&[]),
                self.device.clone(),
            )
            .and_then(|t| t.to_dtype(DType::BF16))
            .map_err(|e| crate::EriDiffusionError::Lora(format!(
                "alpha tensor for {prefix}: {e}")))?;
            out.insert(format!("{prefix}.alpha"), alpha_t);
            Ok(())
        };
        if let Some(ref bundle) = self.bundle {
            // Legacy LoRA save — byte-identical to pre-LyCORIS commits.
            for (&(idx, target), lora) in &bundle.double_adapters {
                let prefix = format!("double_blocks.{}.{}", idx, double_target_suffix(target));
                lora.save_tensors(&prefix, &mut tensors)?;
                emit_alpha(&prefix, lora.alpha, &mut tensors)?;
            }
            for (&(idx, target), lora) in &bundle.single_adapters {
                let prefix = format!("single_blocks.{}.{}", idx, single_target_suffix(target));
                lora.save_tensors(&prefix, &mut tensors)?;
                emit_alpha(&prefix, lora.alpha, &mut tensors)?;
            }
        } else if let Some(ref lb) = self.lycoris_bundle {
            // LyCORIS save: per-adapter `<adapter_name>.<algo_suffix>` + alpha.
            // `AdapterStore.names[i]` already includes the full block-prefixed
            // path (e.g. `double_blocks.0.img_attn.to_q`); the algo suffix
            // (e.g. `lora_A.weight`, `hada_w1_a`) comes from
            // `AdapterModule::named_tensors`.
            for (i, adapter) in lb.store.adapters.iter().enumerate() {
                let prefix = &lb.store.names[i];
                for (suffix, t) in adapter.named_tensors() {
                    tensors.insert(format!("{prefix}.{suffix}"), t);
                }
                emit_alpha(prefix, lb.config.alpha, &mut tensors)?;
            }
        } else {
            return Err(crate::EriDiffusionError::Model(
                "Flux save_weights requires LoRA or LyCORIS mode".into(),
            ));
        }
        let p = std::path::Path::new(path);
        flame_core::serialization::save_tensors(&tensors, p, flame_core::serialization::SerializationFormat::SafeTensors)?;
        Ok(())
    }

    fn load_weights(&mut self, path: &str) -> Result<()> {
        let source = flame_core::serialization::load_file(
            std::path::Path::new(path), &self.device,
        ).map_err(|e| crate::EriDiffusionError::Safetensors(format!("load_file: {e}")))?;
        if let Some(ref bundle) = self.bundle {
            for (&(idx, target), lora) in &bundle.double_adapters {
                let prefix = format!("double_blocks.{}.{}", idx, double_target_suffix(target));
                lora.load_tensors(&prefix, &source)?;
            }
            for (&(idx, target), lora) in &bundle.single_adapters {
                let prefix = format!("single_blocks.{}.{}", idx, single_target_suffix(target));
                lora.load_tensors(&prefix, &source)?;
            }
            Ok(())
        } else if let Some(ref lb) = self.lycoris_bundle {
            // LyCORIS in-place load: validation-only stub for Phase 2b. The
            // underlying `AdapterStore::load_safetensors` validates that the
            // file's algo + DoRA match the bundle's config and bails on
            // mismatch. Per-algo tensor population requires lycoris-rs setter
            // APIs that don't exist yet (see `lycoris.rs::load_safetensors`
            // doc comment) — same contract as the legacy
            // `LycorisBundle::load_safetensors` stub.
            let mut tmp = AdapterStore::new(lb.config.clone(), self.device.clone());
            tmp.load_safetensors("", std::path::Path::new(path))
                .map_err(|e| crate::EriDiffusionError::Safetensors(format!(
                    "FluxLycorisBundle::load (validation-only stub): {e}"
                )))?;
            log::warn!(
                "[Flux] load_weights: LyCORIS in-place load is a validation-only stub \
                 — file algo/DoRA matched bundle config; per-algo tensor population \
                 requires lycoris-rs setter API (Phase 2b follow-up)"
            );
            Ok(())
        } else {
            Err(crate::EriDiffusionError::Model(
                "Flux load_weights requires LoRA or LyCORIS mode".into(),
            ))
        }
    }
}

/// Diffusers-style suffix for each per-double-block LoRA target. Q/K/V are
/// split out from BFL's fused `img_attn.qkv` / `txt_attn.qkv` to match the
/// adapter granularity that `convert_flux_lora.py` expects.
fn double_target_suffix(t: DoubleLoraTarget) -> &'static str {
    match t {
        DoubleLoraTarget::ImgQ => "img_attn.to_q",
        DoubleLoraTarget::ImgK => "img_attn.to_k",
        DoubleLoraTarget::ImgV => "img_attn.to_v",
        DoubleLoraTarget::ImgProj => "img_attn.proj",
        DoubleLoraTarget::TxtQ => "txt_attn.to_q",
        DoubleLoraTarget::TxtK => "txt_attn.to_k",
        DoubleLoraTarget::TxtV => "txt_attn.to_v",
        DoubleLoraTarget::TxtProj => "txt_attn.proj",
        DoubleLoraTarget::ImgMlp0 => "img_mlp.0",
        DoubleLoraTarget::ImgMlp2 => "img_mlp.2",
        DoubleLoraTarget::TxtMlp0 => "txt_mlp.0",
        DoubleLoraTarget::TxtMlp2 => "txt_mlp.2",
    }
}

/// Diffusers-style suffix for each per-single-block LoRA target. Q/K/V split
/// from BFL `linear1[:3*DIM]`; `proj_mlp` is BFL `linear1[3*DIM:]`; `proj_out`
/// is BFL `linear2`.
fn single_target_suffix(t: SingleLoraTarget) -> &'static str {
    match t {
        SingleLoraTarget::Q => "attn.to_q",
        SingleLoraTarget::K => "attn.to_k",
        SingleLoraTarget::V => "attn.to_v",
        SingleLoraTarget::ProjMlp => "proj_mlp",
        SingleLoraTarget::ProjOut => "proj_out",
    }
}
