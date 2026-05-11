//! SD 3.5 Medium / Large MMDiT — dual-stream transformer with LoRA training.
//!
//! ## Reference
//!
//! - **upstream Python (canonical)**: `modules/model/StableDiffusion3Model.py`,
//!   `modules/modelSetup/BaseStableDiffusion3Setup.py`. Loss is plain MSE on
//!   `target = noise - clean`, `predicted = transformer(noisy, t, ctx, pooled)`
//!   — NO velocity preconditioning at training time. Verified against
//!   `BaseStableDiffusion3Setup.py:319-333`.
//! - **Working pure-Rust SD3 forward**: `EriDiffusion/inference-flame/src/bin/sd3_medium_infer.rs`
//!   (verified end-to-end). The forward pass here is ported from
//!   `flame-diffusion/sd3-trainer/src/model.rs` which itself ports from
//!   `inference-flame/sd3_mmdit.rs` and is the closest existing flame-core
//!   wiring of MMDiT. flame-diffusion's *training loop* was never validated
//!   (memory: feedback_zimage_trainer_not_converged) so we keep ONLY the
//!   forward-pass math; the training loop here is ED-v2 idiomatic and matches
//!   train_klein.rs.
//!
//! ## Architecture (auto-detected from checkpoint)
//!
//! - SD3.5-medium: depth=24, hidden=1536, heads=24, dual-attention blocks 0..12
//! - SD3.5-large : depth=38, hidden=2432, heads=38, dual-attention disabled
//! - Patch size: 2  ;  in_channels: 16  ;  out_channels: 16
//! - Joint blocks: dual stream (image x_block + text context_block)
//! - Last block has `is_last`: context stream emits no MLP (only attention)
//! - QK normalization: per-head LN with weight (and optional bias)
//! - Time embed: sinusoidal(256) → MLP × 2 → hidden
//! - Pooled-text (`y`): MLP × 2 (`y_embedder.mlp.{0,2}`) → hidden, added to t_emb
//! - Conditioning `c` is fed into per-block `adaLN_modulation.1` to produce
//!   shift / scale / gate triples (6 per stream, 9 if dual-attention)
//!
//! ## LoRA targets
//!
//! Matching kohya `SD3_TARGET_REPLACE_MODULE = ["SingleDiTBlock"]`
//! (`networks/lora_sd3.py:231`) and the working flame-diffusion sd3-trainer:
//!   - per joint block: `x_block.{attn.qkv, attn.proj, mlp.fc1, mlp.fc2}` (4)
//!   - context_block:   same 4 (skipped on the LAST block where `is_last=True`)
//!   - dual-attention blocks (SD3.5-medium 0..12): `x_block.attn2.{qkv, proj}` (2)
//!
//! NOTE: upstream Python with the SD3 LoRA preset uses an EMPTY `layer_filter`
//! which causes `LoRAModuleWrapper` to wrap *every* `nn.Linear` in the
//! transformer — that's broader than the kohya set above (it also catches
//! `adaLN_modulation`, embedder MLPs, `final_layer.linear`). The kohya set
//! is what the public SD3 LoRA ecosystem ships and what the verified
//! flame-diffusion trainer used; matching it keeps trained adapters
//! interchangeable with `inference-flame/sd3_lora_infer`. **TODO(user audit):
//! widen to OT-Python-style "all linears" if you want bit-for-bit parity
//! with upstream Python's training output.**
//!
//! Wave-2 audit (HIGH-2, 2026-05-08): the gap is concrete. OT preset
//! `#sd 3 LoRA.json` has no `layer_filter` field → empty filter →
//! `LoRAModuleWrapper` walks `named_modules()` and wraps every
//! `nn.Linear | nn.Conv2d`. Compared to the kohya set above, the missing
//! adapters are:
//!   - per joint block: `x_block.adaLN_modulation.1`,
//!     `context_block.adaLN_modulation.1` (last-block has
//!     `swap_chunks=True` per `convert_sd3_lora.py:24`)
//!   - global: `pos_embed.proj` (Conv2d!), `context_embedder`,
//!     `t_embedder.mlp.{0,2}`, `y_embedder.mlp.{0,2}`,
//!     `final_layer.linear`, `final_layer.adaLN_modulation.1`
//!     (also `swap_chunks=True`)
//! The widening adds ~20% trainable params at rank=16 and changes the
//! gradient flow on the modulation/time-embed paths. NOT applied in this
//! Wave-2 fix — the Conv2d LoRA target requires lora_conv2d support in
//! flame-core which is out of scope here. Tracked as deferred follow-up.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use cudarc::driver::CudaDevice;
use flame_core::{
    attention::sdpa,
    layer_norm::layer_norm,
    parameter::Parameter,
    serialization::load_file_filtered,
    DType, Shape, Tensor,
};

use crate::adapter::{AdapterModule, LycorisLinear};
use crate::config::TrainConfig;
use crate::lora::LoRALinear;
use crate::lycoris::{LycorisAlgo, LycorisBundleConfig};
use crate::models::chroma::build_lycoris_linear;
use crate::models::TrainableModel;
use crate::{EriDiffusionError, Result};

// ---------------------------------------------------------------------------
// Auto-detected config
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct SD35Config {
    pub depth: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub mlp_ratio: f32,
    pub patch_size: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub context_dim: usize,
    pub pooled_dim: usize,
    pub has_dual_attention: Vec<bool>,
}

// ---------------------------------------------------------------------------
// Per-joint-block LoRA adapters
// ---------------------------------------------------------------------------

/// Per-block LoRA target. Matches the suffix strings used by
/// [`JointBlockAdapters::iter_with_keys`] one-to-one — used as the `target`
/// argument to [`SD35Model::adapter_for`] for LyCORIS dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Sd35LoraTarget {
    XAttnQkv,
    XAttnProj,
    XMlpFc1,
    XMlpFc2,
    CtxAttnQkv,
    CtxAttnProj,
    CtxMlpFc1,
    CtxMlpFc2,
    XAttn2Qkv,
    XAttn2Proj,
}

impl Sd35LoraTarget {
    /// Save-key suffix matching `JointBlockAdapters::iter_with_keys` strings.
    pub fn suffix(self) -> &'static str {
        match self {
            Sd35LoraTarget::XAttnQkv    => "x_block.attn.qkv",
            Sd35LoraTarget::XAttnProj   => "x_block.attn.proj",
            Sd35LoraTarget::XMlpFc1     => "x_block.mlp.fc1",
            Sd35LoraTarget::XMlpFc2     => "x_block.mlp.fc2",
            Sd35LoraTarget::CtxAttnQkv  => "context_block.attn.qkv",
            Sd35LoraTarget::CtxAttnProj => "context_block.attn.proj",
            Sd35LoraTarget::CtxMlpFc1   => "context_block.mlp.fc1",
            Sd35LoraTarget::CtxMlpFc2   => "context_block.mlp.fc2",
            Sd35LoraTarget::XAttn2Qkv   => "x_block.attn2.qkv",
            Sd35LoraTarget::XAttn2Proj  => "x_block.attn2.proj",
        }
    }
}

struct JointBlockAdapters {
    x_attn_qkv: LoRALinear,
    x_attn_proj: LoRALinear,
    x_mlp_fc1: LoRALinear,
    x_mlp_fc2: LoRALinear,
    ctx_attn_qkv: Option<LoRALinear>,
    ctx_attn_proj: Option<LoRALinear>,
    ctx_mlp_fc1: Option<LoRALinear>,
    ctx_mlp_fc2: Option<LoRALinear>,
    x_attn2_qkv: Option<LoRALinear>,
    x_attn2_proj: Option<LoRALinear>,
}

impl JointBlockAdapters {
    fn iter_with_keys(&self) -> Vec<(&'static str, &LoRALinear)> {
        let mut v: Vec<(&'static str, &LoRALinear)> = vec![
            ("x_block.attn.qkv", &self.x_attn_qkv),
            ("x_block.attn.proj", &self.x_attn_proj),
            ("x_block.mlp.fc1", &self.x_mlp_fc1),
            ("x_block.mlp.fc2", &self.x_mlp_fc2),
        ];
        if let Some(ref l) = self.ctx_attn_qkv {  v.push(("context_block.attn.qkv", l)); }
        if let Some(ref l) = self.ctx_attn_proj { v.push(("context_block.attn.proj", l)); }
        if let Some(ref l) = self.ctx_mlp_fc1 {   v.push(("context_block.mlp.fc1", l)); }
        if let Some(ref l) = self.ctx_mlp_fc2 {   v.push(("context_block.mlp.fc2", l)); }
        if let Some(ref l) = self.x_attn2_qkv {   v.push(("x_block.attn2.qkv", l)); }
        if let Some(ref l) = self.x_attn2_proj {  v.push(("x_block.attn2.proj", l)); }
        v
    }

    fn all(&self) -> Vec<&LoRALinear> {
        self.iter_with_keys().into_iter().map(|(_, l)| l).collect()
    }

    /// Look up the legacy plain-LoRA adapter by target. Returns `None` for
    /// targets that don't exist on this block (e.g. `CtxAttnQkv` on `is_last`,
    /// or `XAttn2*` on a non-dual block).
    fn get(&self, target: Sd35LoraTarget) -> Option<&LoRALinear> {
        match target {
            Sd35LoraTarget::XAttnQkv    => Some(&self.x_attn_qkv),
            Sd35LoraTarget::XAttnProj   => Some(&self.x_attn_proj),
            Sd35LoraTarget::XMlpFc1     => Some(&self.x_mlp_fc1),
            Sd35LoraTarget::XMlpFc2     => Some(&self.x_mlp_fc2),
            Sd35LoraTarget::CtxAttnQkv  => self.ctx_attn_qkv.as_ref(),
            Sd35LoraTarget::CtxAttnProj => self.ctx_attn_proj.as_ref(),
            Sd35LoraTarget::CtxMlpFc1   => self.ctx_mlp_fc1.as_ref(),
            Sd35LoraTarget::CtxMlpFc2   => self.ctx_mlp_fc2.as_ref(),
            Sd35LoraTarget::XAttn2Qkv   => self.x_attn2_qkv.as_ref(),
            Sd35LoraTarget::XAttn2Proj  => self.x_attn2_proj.as_ref(),
        }
    }
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

pub struct SD35Model {
    pub config: TrainConfig,
    pub device: Arc<CudaDevice>,
    /// Frozen base weights. 2D `*.weight` matrices are pre-transposed `[in, out]`
    /// for direct `matmul` (matching the flame-diffusion sd3-trainer convention
    /// — the inference-flame SD3 forward also pre-transposes).
    pub weights: HashMap<String, Tensor>,
    pub mmdit_config: SD35Config,

    /// LoRA adapters, parallel to `0..mmdit_config.depth`. `None` means
    /// non-LoRA load (full-FT not supported in this minimal port — would need
    /// promoting every weight to F32 trainable, ~40 GB on SD3.5-large).
    pub block_adapters: Option<Vec<JointBlockAdapters>>,

    /// LyCORIS adapters (LoCon/LoHa/LoKr/Full/OFT). Empty when `algo == None`.
    /// When [`SD35Model::install_lycoris_bundle`] swaps in a LyCORIS algo,
    /// `block_adapters` is dropped and this map is populated instead. The
    /// per-block-target geometry mirrors `JointBlockAdapters` (i.e. ctx_*
    /// keys are skipped on `is_last`, x_attn2_* keys are gated by
    /// `has_dual_attention[i]`).
    pub lycoris_adapters: HashMap<(usize, Sd35LoraTarget), Arc<LycorisLinear>>,

    /// Currently active algo. `LycorisAlgo::None` means the legacy plain
    /// `LoRALinear` path is in use (`block_adapters`).
    pub algo: LycorisAlgo,

    /// Flattened parameter list (returned by `parameters()`).
    pub parameters: Vec<Parameter>,
    pub is_lora: bool,
}

impl SD35Model {
    pub fn load(
        paths: &[std::path::PathBuf],
        config: &TrainConfig,
        device: Arc<CudaDevice>,
    ) -> Result<Self> {
        // Combined SD3.5 checkpoints prefix DiT keys with `model.diffusion_model.`;
        // LoRA-only safetensors and bare DiT exports do not.
        let prefix = "model.diffusion_model.";
        let predicate = |key: &str| -> bool {
            key.starts_with(prefix)
                || key.starts_with("joint_blocks.")
                || key == "pos_embed"
                || key.starts_with("x_embedder.")
                || key.starts_with("t_embedder.")
                || key.starts_with("y_embedder.")
                || key.starts_with("context_embedder.")
                || key.starts_with("final_layer.")
        };

        let mut weights: HashMap<String, Tensor> = HashMap::new();
        for p in paths {
            let raw = load_file_filtered(
                p.to_str().ok_or_else(|| EriDiffusionError::Model(
                    format!("non-UTF8 model path {:?}", p)))?,
                &device,
                predicate,
            ).map_err(|e| EriDiffusionError::Safetensors(format!(
                "load_file_filtered {}: {e}", p.display())))?;

            for (key, val) in raw {
                let k = key.strip_prefix(prefix).unwrap_or(&key).to_string();
                let val = if val.dtype() != DType::BF16 {
                    val.to_dtype(DType::BF16)?
                } else {
                    val
                };
                let val = val.requires_grad_(false);
                // Pre-transpose 2D weight matrices `[out, in] → [in, out]`
                // for `x @ wt` matmul. Conv2d weights are 4D and must NOT be
                // transposed.
                let val = if k.ends_with(".weight") && val.dims().len() == 2 {
                    val.permute(&[1, 0])?.requires_grad_(false)
                } else {
                    val
                };
                weights.insert(k, val);
            }
        }

        let mmdit_config = Self::detect_config(&weights)?;
        log::info!(
            "SD3.5 MMDiT loaded: depth={}, hidden={}, heads={}, head_dim={}, dual_attn_blocks={}",
            mmdit_config.depth, mmdit_config.hidden_size, mmdit_config.num_heads,
            mmdit_config.head_dim,
            mmdit_config.has_dual_attention.iter().filter(|&&b| b).count(),
        );

        // Build LoRA adapters (or skip for non-LoRA load).
        let is_lora = config.is_lora();
        let mut parameters: Vec<Parameter> = Vec::new();
        let block_adapters = if is_lora {
            let rank = config.lora_rank as usize;
            let alpha = config.lora_alpha as f32;
            let h = mmdit_config.hidden_size;
            let mlp_h = (h as f32 * mmdit_config.mlp_ratio) as usize;

            let mut blocks = Vec::with_capacity(mmdit_config.depth);
            for i in 0..mmdit_config.depth {
                let is_last = i == mmdit_config.depth - 1;
                let has_dual = mmdit_config.has_dual_attention[i];
                let seed = 42u64 + (i as u64) * 20;

                blocks.push(JointBlockAdapters {
                    x_attn_qkv:  LoRALinear::new(h,    3 * h, rank, alpha, device.clone(), seed)?,
                    x_attn_proj: LoRALinear::new(h,    h,     rank, alpha, device.clone(), seed + 1)?,
                    x_mlp_fc1:   LoRALinear::new(h,    mlp_h, rank, alpha, device.clone(), seed + 2)?,
                    x_mlp_fc2:   LoRALinear::new(mlp_h, h,    rank, alpha, device.clone(), seed + 3)?,
                    ctx_attn_qkv:  if !is_last { Some(LoRALinear::new(h,    3 * h, rank, alpha, device.clone(), seed + 4)?) } else { None },
                    ctx_attn_proj: if !is_last { Some(LoRALinear::new(h,    h,     rank, alpha, device.clone(), seed + 5)?) } else { None },
                    ctx_mlp_fc1:   if !is_last { Some(LoRALinear::new(h,    mlp_h, rank, alpha, device.clone(), seed + 6)?) } else { None },
                    ctx_mlp_fc2:   if !is_last { Some(LoRALinear::new(mlp_h, h,    rank, alpha, device.clone(), seed + 7)?) } else { None },
                    x_attn2_qkv:   if has_dual { Some(LoRALinear::new(h,    3 * h, rank, alpha, device.clone(), seed + 8)?) } else { None },
                    x_attn2_proj:  if has_dual { Some(LoRALinear::new(h,    h,     rank, alpha, device.clone(), seed + 9)?) } else { None },
                });
            }
            for b in &blocks {
                for l in b.all() {
                    parameters.extend(l.parameters());
                }
            }
            log::info!(
                "SD3.5 LoRA: {} adapters across {} blocks, rank={}, alpha={}",
                blocks.iter().flat_map(|b| b.all()).count(),
                blocks.len(), rank, alpha,
            );
            Some(blocks)
        } else {
            log::warn!("SD3.5 non-LoRA mode is not supported in this port — empty parameter list");
            None
        };

        Ok(Self {
            config: config.clone(),
            device,
            weights,
            mmdit_config,
            block_adapters,
            lycoris_adapters: HashMap::new(),
            algo: LycorisAlgo::None,
            parameters,
            is_lora,
        })
    }

    /// Phase 2b: swap the legacy plain-LoRA `block_adapters` for a fresh
    /// LyCORIS-algo bundle (LoCon/LoHa/LoKr/Full/OFT). Drops `block_adapters`,
    /// populates `lycoris_adapters` keyed by `(block_idx, Sd35LoraTarget)`,
    /// and rebuilds `parameters`. Caller must invoke this BEFORE reading
    /// [`Self::parameters`] (the trainer's optimizer captures Parameter IDs
    /// at construction time).
    ///
    /// Per-block target gating mirrors the legacy ctor: `Ctx*` skipped on
    /// `is_last`, `XAttn2*` gated by `has_dual_attention[i]`.
    ///
    /// `Full` and `OFT` bundle-construction succeeds, but their
    /// `forward_delta` errors inside the `base + delta_on_input` call
    /// pattern. Phase 2c will wire merge-into-base.
    pub fn install_lycoris_bundle(
        &mut self,
        config: &LycorisBundleConfig,
        device: Arc<CudaDevice>,
        seed: u64,
    ) -> Result<()> {
        if config.algo == LycorisAlgo::None {
            return Ok(());
        }
        if !self.is_lora {
            return Err(EriDiffusionError::Model(
                "install_lycoris_bundle requires LoRA mode (is_lora=true)".into(),
            ));
        }
        let _ = seed; // lycoris-rs uses its own internal RNG.

        let h = self.mmdit_config.hidden_size;
        let mlp_h = (h as f32 * self.mmdit_config.mlp_ratio) as usize;
        let depth = self.mmdit_config.depth;

        let mut adapters: HashMap<(usize, Sd35LoraTarget), Arc<LycorisLinear>> =
            HashMap::new();
        for i in 0..depth {
            let is_last = i == depth - 1;
            let has_dual = self.mmdit_config.has_dual_attention[i];

            // Per-target (in_dim, out_dim) — mirrors the legacy
            // `JointBlockAdapters` ctor exactly.
            let mut targets: Vec<(Sd35LoraTarget, usize, usize)> = vec![
                (Sd35LoraTarget::XAttnQkv,  h,     3 * h),
                (Sd35LoraTarget::XAttnProj, h,     h),
                (Sd35LoraTarget::XMlpFc1,   h,     mlp_h),
                (Sd35LoraTarget::XMlpFc2,   mlp_h, h),
            ];
            if !is_last {
                targets.extend([
                    (Sd35LoraTarget::CtxAttnQkv,  h,     3 * h),
                    (Sd35LoraTarget::CtxAttnProj, h,     h),
                    (Sd35LoraTarget::CtxMlpFc1,   h,     mlp_h),
                    (Sd35LoraTarget::CtxMlpFc2,   mlp_h, h),
                ]);
            }
            if has_dual {
                targets.extend([
                    (Sd35LoraTarget::XAttn2Qkv,  h, 3 * h),
                    (Sd35LoraTarget::XAttn2Proj, h, h),
                ]);
            }

            for (target, in_dim, out_dim) in targets {
                let wrapper = build_lycoris_linear(config, in_dim, out_dim, device.clone())
                    .map_err(|e| EriDiffusionError::Lora(format!(
                        "build_lycoris_linear({:?}): {e}", target,
                    )))?;
                adapters.insert((i, target), Arc::new(wrapper));
            }
        }

        log::info!(
            "[SD3.5] LyCORIS algo='{}' installed: {} adapters across {} blocks",
            config.algo.as_str(),
            adapters.len(),
            depth,
        );

        // Rebuild parameter list from the LyCORIS adapters. Drop
        // `block_adapters` so the legacy path doesn't double-count.
        let mut parameters: Vec<Parameter> = Vec::new();
        for adapter in adapters.values() {
            parameters.extend(adapter.to_parameters());
        }

        self.block_adapters = None;
        self.lycoris_adapters = adapters;
        self.algo = config.algo;
        self.parameters = parameters;
        Ok(())
    }

    /// Look up the active adapter for `(block_idx, target)`. Prefers the
    /// LyCORIS map when populated; falls back to the legacy plain-LoRA
    /// `block_adapters`. Returns `None` when neither has an entry (e.g.
    /// non-LoRA load, or a target that doesn't exist on the given block).
    pub fn adapter_for(
        &self,
        block_idx: usize,
        target: Sd35LoraTarget,
    ) -> Option<&dyn AdapterModule> {
        if let Some(lyc) = self.lycoris_adapters.get(&(block_idx, target)) {
            return Some(lyc.as_ref());
        }
        if let Some(blocks) = self.block_adapters.as_ref() {
            if let Some(legacy) = blocks[block_idx].get(target) {
                return Some(legacy);
            }
        }
        None
    }

    /// Auto-detect MMDiT shape from the loaded weights.
    /// Mirrors `flame-diffusion/sd3-trainer/src/model.rs::detect_config`.
    fn detect_config(weights: &HashMap<String, Tensor>) -> Result<SD35Config> {
        let proj_w = weights.get("x_embedder.proj.weight")
            .ok_or_else(|| EriDiffusionError::Model("missing x_embedder.proj.weight".into()))?;
        let proj_dims = proj_w.dims();
        // Conv weight is 4D `[out, in, kH, kW]` (NOT pre-transposed).
        let (hidden_size, in_channels, patch_size) = if proj_dims.len() == 4 {
            (proj_dims[0], proj_dims[1], proj_dims[2])
        } else {
            // 2D after pre-transpose: `[in, out]`
            (proj_dims[1], proj_dims[0], 2)
        };
        let head_dim = 64;
        if hidden_size % head_dim != 0 {
            return Err(EriDiffusionError::Model(format!(
                "hidden_size {hidden_size} not divisible by head_dim {head_dim}")));
        }
        let num_heads = hidden_size / head_dim;

        let final_w = weights.get("final_layer.linear.weight")
            .ok_or_else(|| EriDiffusionError::Model("missing final_layer.linear.weight".into()))?;
        let total_out = final_w.dims()[1]; // pre-transposed [hidden, out]
        let out_channels = total_out / (patch_size * patch_size);

        let ctx_w = weights.get("context_embedder.weight")
            .ok_or_else(|| EriDiffusionError::Model("missing context_embedder.weight".into()))?;
        // pre-transposed [context_dim, hidden]
        let context_dim = ctx_w.dims()[0];

        let y_w = weights.get("y_embedder.mlp.0.weight")
            .ok_or_else(|| EriDiffusionError::Model("missing y_embedder.mlp.0.weight".into()))?;
        let pooled_dim = y_w.dims()[0];

        // Probe per-block adaLN to detect depth + dual attention.
        let mut depth = 0usize;
        let mut has_dual = Vec::new();
        loop {
            let key = format!("joint_blocks.{depth}.x_block.adaLN_modulation.1.weight");
            match weights.get(&key) {
                Some(t) => {
                    // pre-transposed [hidden, n*hidden], n=6 (single) or 9 (dual).
                    let ada_out = t.dims()[1];
                    has_dual.push(ada_out / hidden_size == 9);
                    depth += 1;
                }
                None => break,
            }
        }
        if depth == 0 {
            return Err(EriDiffusionError::Model("no joint_blocks found".into()));
        }

        Ok(SD35Config {
            depth,
            hidden_size,
            num_heads,
            head_dim,
            mlp_ratio: 4.0,
            patch_size,
            in_channels,
            out_channels,
            context_dim,
            pooled_dim,
            has_dual_attention: has_dual,
        })
    }

    // ------------------------------------------------------------------
    // Internal weight access + autograd-aware ops
    // ------------------------------------------------------------------

    fn w(&self, key: &str) -> Result<&Tensor> {
        self.weights.get(key)
            .ok_or_else(|| EriDiffusionError::Model(format!("missing weight: {key}")))
    }

    /// Autograd-aware `out = x @ wt + bias`. `wt` is pre-transposed `[in, out]`.
    fn linear(&self, weight_key: &str, bias_key: &str, x: &Tensor) -> Result<Tensor> {
        let wt = self.w(weight_key)?;
        let dims = x.dims().to_vec();
        let batch: usize = dims[..dims.len() - 1].iter().product();
        let in_feat = *dims.last().unwrap();
        let out_feat = wt.dims()[1];
        let x_2d = x.reshape(&[batch, in_feat])?;
        let mut out = x_2d.matmul(wt)?;
        let bias = self.w(bias_key)?;
        out = out.add(&bias.unsqueeze(0)?.expand(&[batch, out_feat])?)?;
        let mut out_shape = dims[..dims.len() - 1].to_vec();
        out_shape.push(out_feat);
        out.reshape(&out_shape).map_err(Into::into)
    }

    /// `linear` + adapter delta when an adapter is provided. Accepts any
    /// [`AdapterModule`] — plain `LoRALinear` (legacy) or `LycorisLinear`
    /// (LoCon / LoHa / LoKr / DoRA, etc.). Dispatch happens through the
    /// `forward_delta` trait method so a `--algo locon` swap works
    /// transparently in every call site.
    fn linear_lora(
        &self,
        weight_key: &str,
        bias_key: &str,
        x: &Tensor,
        adapter: Option<&dyn AdapterModule>,
    ) -> Result<Tensor> {
        let base = self.linear(weight_key, bias_key, x)?;
        match adapter {
            None => Ok(base),
            Some(a) => {
                let delta = a.forward_delta(x)?;
                // forward_delta returns a tensor whose final dim is out_features;
                // reshape to base's exact shape (it always matches since
                // LoRA in/out match the base linear).
                let base_dims = base.dims().to_vec();
                let delta = delta.reshape(&base_dims)?;
                base.add(&delta).map_err(Into::into)
            }
        }
    }

    fn layer_norm_no_affine(&self, x: &Tensor) -> Result<Tensor> {
        layer_norm(x, &[self.mmdit_config.hidden_size], None, None, 1e-6)
            .map_err(Into::into)
    }

    /// `x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)`.
    fn modulate(&self, x: &Tensor, shift: &Tensor, scale: &Tensor) -> Result<Tensor> {
        let scale_unsq = scale.unsqueeze(1)?;
        let shift_unsq = shift.unsqueeze(1)?;
        let ones = Tensor::from_vec_dtype(
            vec![1.0f32], Shape::from_dims(&[1, 1, 1]),
            self.device.clone(), DType::BF16,
        )?;
        let factor = ones.add(&scale_unsq)?;
        x.mul(&factor)?.add(&shift_unsq).map_err(Into::into)
    }

    /// SD3 sinusoidal time embedding (256 freq dim) → MLP → hidden_size.
    fn timestep_embed(&self, t: &Tensor) -> Result<Tensor> {
        let t_data = t.to_dtype(DType::F32)?.to_vec()?;
        let batch = t_data.len();
        let freq_dim = 256usize;
        let half = freq_dim / 2;
        let max_period: f32 = 10000.0;

        let mut emb_data = vec![0.0f32; batch * freq_dim];
        for b in 0..batch {
            let tv = t_data[b];
            for i in 0..half {
                let freq = (-f32::ln(max_period) * (i as f32) / (half as f32)).exp();
                let ang = tv * freq;
                emb_data[b * freq_dim + i]        = ang.cos();
                emb_data[b * freq_dim + half + i] = ang.sin();
            }
        }
        let emb = Tensor::from_vec_dtype(
            emb_data, Shape::from_dims(&[batch, freq_dim]),
            self.device.clone(), DType::BF16,
        )?;
        let h = self.linear("t_embedder.mlp.0.weight", "t_embedder.mlp.0.bias", &emb)?;
        let h = h.silu()?;
        self.linear("t_embedder.mlp.2.weight", "t_embedder.mlp.2.bias", &h)
    }

    /// 2x2 patch embed → `[B, N, hidden]`. Conv2d weight stays 4D (NOT
    /// pre-transposed) — handled in `load`.
    fn patch_embed(&self, x: &Tensor) -> Result<Tensor> {
        let w = self.w("x_embedder.proj.weight")?;
        let b = self.w("x_embedder.proj.bias")?;
        let p = self.mmdit_config.patch_size;

        let mut conv = flame_core::conv::Conv2d::new_with_bias(
            self.mmdit_config.in_channels,
            self.mmdit_config.hidden_size,
            p, p, 0,
            self.device.clone(),
            true,
        )?;
        conv.copy_weight_from(w)?;
        conv.copy_bias_from(b)?;
        let out = conv.forward(x)?; // [B, hidden, H/p, W/p]

        let d = out.dims().to_vec();
        let out = out.reshape(&[d[0], d[1], d[2] * d[3]])?;
        out.permute(&[0, 2, 1]).map_err(Into::into) // [B, N, hidden]
    }

    fn cropped_pos_embed(&self, h: usize, w: usize) -> Result<Tensor> {
        let p = self.mmdit_config.patch_size;
        let ph = (h + 1) / p;
        let pw = (w + 1) / p;
        let pos = self.w("pos_embed")?; // [1, max_s*max_s, hidden]
        let max_s = (pos.dims()[1] as f64).sqrt().round() as usize;
        let hidden = self.mmdit_config.hidden_size;
        let spatial = pos.reshape(&[1, max_s, max_s, hidden])?;
        let top = (max_s - ph) / 2;
        let left = (max_s - pw) / 2;
        let crop = spatial.narrow(1, top, ph)?.narrow(2, left, pw)?;
        crop.reshape(&[1, ph * pw, hidden]).map_err(Into::into)
    }

    fn unpatchify(&self, x: &Tensor, h: usize, w: usize) -> Result<Tensor> {
        let c = self.mmdit_config.out_channels;
        let p = self.mmdit_config.patch_size;
        let ph = (h + 1) / p;
        let pw = (w + 1) / p;
        let b = x.dims()[0];
        let x = x.reshape(&[b, ph, pw, p, p, c])?;
        let x = x.permute(&[0, 5, 1, 3, 2, 4])?;
        x.reshape(&[b, c, ph * p, pw * p]).map_err(Into::into)
    }

    fn final_layer(&self, x: &Tensor, c: &Tensor) -> Result<Tensor> {
        let x_norm = self.layer_norm_no_affine(x)?;
        let c_silu = c.silu()?;
        let mods = self.linear(
            "final_layer.adaLN_modulation.1.weight",
            "final_layer.adaLN_modulation.1.bias",
            &c_silu,
        )?;
        let chunks = mods.chunk(2, mods.dims().len() - 1)?;
        let x_mod = self.modulate(&x_norm, &chunks[0], &chunks[1])?;
        self.linear("final_layer.linear.weight", "final_layer.linear.bias", &x_mod)
    }

    fn pre_attn_qkv(
        &self,
        x: &Tensor,
        prefix: &str,
        attn_name: &str,
        adapter: Option<&dyn AdapterModule>,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let b = x.dims()[0];
        let n = x.dims()[1];
        let nh = self.mmdit_config.num_heads;
        let hd = self.mmdit_config.head_dim;

        let qkv = self.linear_lora(
            &format!("{prefix}.{attn_name}.qkv.weight"),
            &format!("{prefix}.{attn_name}.qkv.bias"),
            x, adapter,
        )?;

        let qkv = qkv.reshape(&[b, n, 3, nh, hd])?
            .permute(&[2, 0, 3, 1, 4])?
            .contiguous()?;
        // .contiguous() after narrow: zero-copy views feeding matmul/sdpa
        // kernels read scrambled storage (memory: feedback_flame_conv3d_contiguous
        // applies to sdpa/matmul too — kernels assume contig stride).
        let q = qkv.narrow(0, 0, 1)?.squeeze(Some(0))?.contiguous()?;
        let k = qkv.narrow(0, 1, 1)?.squeeze(Some(0))?.contiguous()?;
        let v = qkv.narrow(0, 2, 1)?.squeeze(Some(0))?.contiguous()?;

        // Per-head QK normalization (LN with weight, optional bias).
        let q = self.qk_norm_4d(&q, &format!("{prefix}.{attn_name}.ln_q"))?;
        let k = self.qk_norm_4d(&k, &format!("{prefix}.{attn_name}.ln_k"))?;
        Ok((q, k, v))
    }

    fn qk_norm_4d(&self, x: &Tensor, prefix: &str) -> Result<Tensor> {
        // SD 3.5 uses RMSNorm (NOT LayerNorm) for q/k normalization. The
        // inference-flame impl explicitly notes this — using layer_norm
        // produces a wrong forward even before LoRA. The bias key (if present
        // in the checkpoint) is ignored by rms_norm convention.
        let d = x.dims().to_vec();
        let (b, h, n, dim) = (d[0], d[1], d[2], d[3]);
        let flat = x.reshape(&[b * h * n, dim])?;
        let weight = self.w(&format!("{prefix}.weight"))?;
        let normed = flame_core::norm::rms_norm(&flat, &[dim], Some(weight), 1e-6)?;
        normed.reshape(&[b, h, n, dim]).map_err(Into::into)
    }

    fn gelu_mlp(
        &self,
        x: &Tensor,
        block_prefix: &str,
        is_ctx: bool,
        block_idx: usize,
    ) -> Result<Tensor> {
        let fc1_w = format!("{block_prefix}.mlp.fc1.weight");
        let fc1_b = format!("{block_prefix}.mlp.fc1.bias");
        let fc2_w = format!("{block_prefix}.mlp.fc2.weight");
        let fc2_b = format!("{block_prefix}.mlp.fc2.bias");

        // Phase 2b: dispatch via the unified `adapter_for` accessor so the
        // LyCORIS-swapped path picks up `lycoris_adapters` entries while the
        // default `--algo lora` path keeps reading `block_adapters`.
        let (fc1_target, fc2_target) = if is_ctx {
            (Sd35LoraTarget::CtxMlpFc1, Sd35LoraTarget::CtxMlpFc2)
        } else {
            (Sd35LoraTarget::XMlpFc1, Sd35LoraTarget::XMlpFc2)
        };
        let fc1_adapter = self.adapter_for(block_idx, fc1_target);
        let fc2_adapter = self.adapter_for(block_idx, fc2_target);

        let h = self.linear_lora(&fc1_w, &fc1_b, x, fc1_adapter)?;
        let h = h.gelu()?;
        self.linear_lora(&fc2_w, &fc2_b, &h, fc2_adapter)
    }

    /// One MMDiT joint block. Returns `(new_context, new_x)`. On the last
    /// block the context stream emits no further state (`new_context = None`).
    fn joint_block(
        &self,
        context: &Tensor,
        x: &Tensor,
        c: &Tensor,
        block_idx: usize,
        is_last: bool,
    ) -> Result<(Option<Tensor>, Tensor)> {
        let prefix = format!("joint_blocks.{block_idx}");
        let x_prefix = format!("{prefix}.x_block");
        let ctx_prefix = format!("{prefix}.context_block");
        let hidden = self.mmdit_config.hidden_size;

        // ---- Context stream pre-attention ----
        let ctx_mods = {
            let c_silu = c.silu()?;
            self.linear(
                &format!("{ctx_prefix}.adaLN_modulation.1.weight"),
                &format!("{ctx_prefix}.adaLN_modulation.1.bias"),
                &c_silu,
            )?
        };

        let (ctx_q, ctx_k, ctx_v, ctx_intermediates) = if is_last {
            // Last block: context only feeds attention; only 2 mods needed.
            let chunks = ctx_mods.chunk(2, ctx_mods.dims().len() - 1)?;
            let ctx_norm = self.layer_norm_no_affine(context)?;
            let ctx_mod = self.modulate(&ctx_norm, &chunks[0], &chunks[1])?;
            let (q, k, v) = self.pre_attn_qkv(
                &ctx_mod, &ctx_prefix, "attn",
                self.adapter_for(block_idx, Sd35LoraTarget::CtxAttnQkv),
            )?;
            (q, k, v, None)
        } else {
            let chunks = ctx_mods.chunk(6, ctx_mods.dims().len() - 1)?;
            let ctx_norm = self.layer_norm_no_affine(context)?;
            let ctx_mod = self.modulate(&ctx_norm, &chunks[0], &chunks[1])?;
            let (q, k, v) = self.pre_attn_qkv(
                &ctx_mod, &ctx_prefix, "attn",
                self.adapter_for(block_idx, Sd35LoraTarget::CtxAttnQkv),
            )?;
            (
                q, k, v,
                Some((
                    context.clone(),
                    chunks[2].clone(), chunks[3].clone(),
                    chunks[4].clone(), chunks[5].clone(),
                )),
            )
        };

        // ---- X stream pre-attention ----
        let x_mods = {
            let c_silu = c.silu()?;
            self.linear(
                &format!("{x_prefix}.adaLN_modulation.1.weight"),
                &format!("{x_prefix}.adaLN_modulation.1.bias"),
                &c_silu,
            )?
        };
        let ada_out = x_mods.dims()[x_mods.dims().len() - 1];
        let block_has_dual = ada_out / hidden == 9;
        let num_mods = if block_has_dual { 9 } else { 6 };
        let x_chunks = x_mods.chunk(num_mods, x_mods.dims().len() - 1)?;

        let x_norm = self.layer_norm_no_affine(x)?;
        let x_mod = self.modulate(&x_norm, &x_chunks[0], &x_chunks[1])?;
        let (x_q, x_k, x_v) = self.pre_attn_qkv(
            &x_mod, &x_prefix, "attn",
            self.adapter_for(block_idx, Sd35LoraTarget::XAttnQkv),
        )?;

        // ---- Joint attention: concat over token dim, single SDPA call ----
        let q = Tensor::cat(&[&ctx_q, &x_q], 2)?;
        let k = Tensor::cat(&[&ctx_k, &x_k], 2)?;
        let v = Tensor::cat(&[&ctx_v, &x_v], 2)?;
        let attn_out = sdpa(&q, &k, &v, None)?;

        let n_ctx = ctx_q.dims()[2];
        let n_x = x_q.dims()[2];
        let batch = attn_out.dims()[0];

        // .contiguous() after narrow: same kernel-stride trap as the QKV split.
        let ctx_attn = attn_out.narrow(2, 0, n_ctx)?.contiguous()?
            .permute(&[0, 2, 1, 3])?.contiguous()?.reshape(&[batch, n_ctx, hidden])?;
        let x_attn = attn_out.narrow(2, n_ctx, n_x)?.contiguous()?
            .permute(&[0, 2, 1, 3])?.contiguous()?.reshape(&[batch, n_x, hidden])?;

        // ---- Context post-attention (skipped on last block) ----
        let context_out = if let Some((ctx_res, gate_msa, shift_mlp, scale_mlp, gate_mlp)) = ctx_intermediates {
            let ctx_proj = self.linear_lora(
                &format!("{ctx_prefix}.attn.proj.weight"),
                &format!("{ctx_prefix}.attn.proj.bias"),
                &ctx_attn,
                self.adapter_for(block_idx, Sd35LoraTarget::CtxAttnProj),
            )?;
            let gated = gate_msa.unsqueeze(1)?.mul(&ctx_proj)?;
            let ctx_out = ctx_res.add(&gated)?;
            let ctx_norm2 = self.layer_norm_no_affine(&ctx_out)?;
            let ctx_mlp_in = self.modulate(&ctx_norm2, &shift_mlp, &scale_mlp)?;
            let ctx_mlp = self.gelu_mlp(&ctx_mlp_in, &ctx_prefix, true, block_idx)?;
            let ctx_gated = gate_mlp.unsqueeze(1)?.mul(&ctx_mlp)?;
            Some(ctx_out.add(&ctx_gated)?)
        } else {
            None
        };

        // ---- X post-attention ----
        let x_proj = self.linear_lora(
            &format!("{x_prefix}.attn.proj.weight"),
            &format!("{x_prefix}.attn.proj.bias"),
            &x_attn,
            self.adapter_for(block_idx, Sd35LoraTarget::XAttnProj),
        )?;
        let x_gated = x_chunks[2].unsqueeze(1)?.mul(&x_proj)?;
        let mut x_out = x.add(&x_gated)?;

        // ---- Optional dual attention (SD3.5 Medium blocks 0..12) ----
        if block_has_dual {
            let x_mod2 = self.modulate(&x_norm, &x_chunks[6], &x_chunks[7])?;
            let (q2, k2, v2) = self.pre_attn_qkv(
                &x_mod2, &x_prefix, "attn2",
                self.adapter_for(block_idx, Sd35LoraTarget::XAttn2Qkv),
            )?;
            let attn2_out = sdpa(&q2, &k2, &v2, None)?;
            let attn2_flat = attn2_out.permute(&[0, 2, 1, 3])?.reshape(&[batch, n_x, hidden])?;
            let attn2_proj = self.linear_lora(
                &format!("{x_prefix}.attn2.proj.weight"),
                &format!("{x_prefix}.attn2.proj.bias"),
                &attn2_flat,
                self.adapter_for(block_idx, Sd35LoraTarget::XAttn2Proj),
            )?;
            let attn2_gated = x_chunks[8].unsqueeze(1)?.mul(&attn2_proj)?;
            x_out = x_out.add(&attn2_gated)?;
        }

        // ---- X MLP ----
        let x_norm2 = self.layer_norm_no_affine(&x_out)?;
        let x_mlp_in = self.modulate(&x_norm2, &x_chunks[3], &x_chunks[4])?;
        let x_mlp = self.gelu_mlp(&x_mlp_in, &x_prefix, false, block_idx)?;
        let x_mlp_gated = x_chunks[5].unsqueeze(1)?.mul(&x_mlp)?;
        let x_out = x_out.add(&x_mlp_gated)?;

        Ok((context_out, x_out))
    }

    /// Full SD3.5 MMDiT forward.
    ///
    /// - `x`        : `[B, in_channels, H, W]` BF16 noisy latent
    /// - `timestep` : `[B]` floats in `[0, 1000]` (BF16 or F32 — converted)
    /// - `context`  : `[B, seq, 4096]` BF16 combined text hidden (CLIP-L+G+T5)
    /// - `pooled`   : `[B, 2048]` BF16 pooled (CLIP-L+G)
    pub fn forward_inner(
        &self,
        x: &Tensor,
        timestep: &Tensor,
        context: &Tensor,
        pooled: &Tensor,
    ) -> Result<Tensor> {
        let x_dims = x.dims().to_vec();
        let (h, w) = (x_dims[2], x_dims[3]);

        let x_tokens = self.patch_embed(x)?;
        let pos_embed = self.cropped_pos_embed(h, w)?;
        let mut x_tokens = x_tokens.add(&pos_embed)?;

        let t_emb = self.timestep_embed(timestep)?;
        let y_emb = self.linear("y_embedder.mlp.0.weight", "y_embedder.mlp.0.bias", pooled)?;
        let y_emb = y_emb.silu()?;
        let y_emb = self.linear("y_embedder.mlp.2.weight", "y_embedder.mlp.2.bias", &y_emb)?;
        let c = t_emb.add(&y_emb)?;

        let mut ctx_tokens = self.linear(
            "context_embedder.weight", "context_embedder.bias", context,
        )?;

        let depth = self.mmdit_config.depth;
        for i in 0..depth {
            let is_last = i == depth - 1;
            let (new_ctx, new_x) = self.joint_block(&ctx_tokens, &x_tokens, &c, i, is_last)?;
            x_tokens = new_x;
            if let Some(ctx) = new_ctx {
                ctx_tokens = ctx;
            }
        }

        let x_out = self.final_layer(&x_tokens, &c)?;
        self.unpatchify(&x_out, h, w)
    }
}

// ---------------------------------------------------------------------------
// TrainableModel impl
// ---------------------------------------------------------------------------

impl TrainableModel for SD35Model {
    fn forward(
        &mut self,
        noisy: &Tensor,
        timestep: &Tensor,
        context: &[Tensor],
        pooled: Option<&Tensor>,
    ) -> Result<Tensor> {
        let ctx = context.first().ok_or_else(||
            EriDiffusionError::Model("SD3.5 expects combined text hidden as context[0]".into()))?;
        let pool = pooled.ok_or_else(||
            EriDiffusionError::Model("SD3.5 expects pooled CLIP-L+G [B, 2048]".into()))?;
        self.forward_inner(noisy, timestep, ctx, pool)
    }

    fn parameters(&self) -> Vec<Parameter> {
        self.parameters.clone()
    }

    fn post_optimizer_step(&mut self) {
        // Legacy path: refresh per-LoRA transposed BF16 cache.
        if let Some(ref blocks) = self.block_adapters {
            for b in blocks {
                for l in b.all() {
                    l.refresh_cache();
                }
            }
        }
        // LyCORIS path: `LycorisLinear::forward_delta` re-reads its leaves
        // live each call — no per-step cache to refresh.
    }

    /// Save LoRA adapters in PEFT/diffusers convention with a per-module
    /// `.alpha` scalar (matches `train_sdxl` save format and is what
    /// `inference-flame/sd3_lora_infer` expects via `LoraStack::load`'s
    /// AiToolkit-compatible path: bare `<prefix>.lora_{A,B}` keys whose
    /// `<prefix>.weight` matches a base key).
    ///
    /// HIGH-3 audit (2026-05-08): OT/kohya/ComfyUI ecosystem expects
    /// `lora_down.weight / lora_up.weight` keys with `lora_transformer_*`
    /// prefixes and SPLIT QKV (per `convert_sd3_lora_key_sets` in
    /// OneTrainer). Producing that format requires splitting fused QKV LoRAs
    /// (`lora_B` row-split into 3) AND remapping prefixes — a non-trivial
    /// change with risk of breaking the in-tree `inference-flame/sd3_lora_infer`
    /// consumer (its `detect_format()` would fall back to KleinTrainer for
    /// bare `lora_down/up` without `lora_unet_` prefix, breaking load). For
    /// now: PEFT-format save is preserved. Follow-up: add a
    /// `--export-format kohya|peft` CLI flag + full OMI conversion.
    fn save_weights(&self, path: &str) -> Result<()> {
        if !self.is_lora {
            return Err(EriDiffusionError::Model(
                "SD3.5 non-LoRA save not implemented".into()));
        }
        let mut out: HashMap<String, Tensor> = HashMap::new();
        if let Some(blocks) = self.block_adapters.as_ref() {
            // Legacy plain-LoRA path — preserved byte-for-byte.
            for (i, block) in blocks.iter().enumerate() {
                for (suffix, lora) in block.iter_with_keys() {
                    let prefix = format!("joint_blocks.{i}.{suffix}");
                    lora.save_tensors(&prefix, &mut out)
                        .map_err(|e| EriDiffusionError::Lora(format!(
                            "save {prefix}: {e}")))?;
                    // Per-module .alpha — what every downstream loader expects.
                    let alpha_t = Tensor::from_vec(
                        vec![lora.alpha],
                        Shape::from_dims(&[]),
                        self.device.clone(),
                    ).and_then(|t| t.to_dtype(DType::BF16))
                     .map_err(|e| EriDiffusionError::Lora(format!(
                         "alpha tensor for {prefix}: {e}")))?;
                    out.insert(format!("{prefix}.alpha"), alpha_t);
                }
            }
        } else if !self.lycoris_adapters.is_empty() {
            // LyCORIS path — emit each adapter's `named_tensors()` under the
            // same `joint_blocks.{i}.{suffix}.` prefix scheme. No `.alpha`
            // scalar (LyCORIS adapters carry alpha internally).
            for (&(block_idx, target), adapter) in &self.lycoris_adapters {
                let prefix = format!("joint_blocks.{block_idx}.{}", target.suffix());
                for (leaf, t) in adapter.named_tensors() {
                    out.insert(format!("{prefix}.{leaf}"), t);
                }
            }
        } else {
            return Err(EriDiffusionError::Model(
                "SD3.5 save_weights: no adapters present (block_adapters=None, \
                 lycoris_adapters empty)".into()));
        }
        flame_core::serialization::save_file(&out, Path::new(path))
            .map_err(|e| EriDiffusionError::Safetensors(format!("save_file: {e}")))?;
        Ok(())
    }

    fn load_weights(&mut self, path: &str) -> Result<()> {
        if !self.is_lora {
            return Err(EriDiffusionError::Model(
                "SD3.5 non-LoRA load not implemented".into()));
        }
        let source = flame_core::serialization::load_file(Path::new(path), &self.device)
            .map_err(|e| EriDiffusionError::Safetensors(format!("load_file: {e}")))?;
        if let Some(blocks) = self.block_adapters.as_ref() {
            for (i, block) in blocks.iter().enumerate() {
                for (suffix, lora) in block.iter_with_keys() {
                    let prefix = format!("joint_blocks.{i}.{suffix}");
                    lora.load_tensors(&prefix, &source)
                        .map_err(|e| EriDiffusionError::Lora(format!(
                            "load {prefix}: {e}")))?;
                }
            }
        } else if !self.lycoris_adapters.is_empty() {
            return Err(EriDiffusionError::Model(
                "SD3.5 LyCORIS load_weights not yet wired (Phase 2c). Use \
                 plain `--algo lora` resume for now.".into()));
        }
        Ok(())
    }
}

impl SD35Model {
    /// Canonical (name, Parameter) pairs for full-checkpoint save/resume.
    /// Mirrors `<SD35Model as TrainableModel>::save_weights`'s key layout:
    /// `joint_blocks.{i}.{suffix}.lora_{A,B}.weight`, where `suffix` is the
    /// per-target string from `JointBlockAdapters::iter_with_keys`. The
    /// `.alpha` scalars that `save_weights` also emits are NOT Parameters and
    /// are intentionally skipped (alpha is restored from CkptHeader on load).
    /// Returns an empty Vec if the model isn't in LoRA mode.
    pub fn named_parameters(&self) -> Vec<(String, Parameter)> {
        let mut out = Vec::new();
        if let Some(blocks) = self.block_adapters.as_ref() {
            for (i, block) in blocks.iter().enumerate() {
                for (suffix, lora) in block.iter_with_keys() {
                    let prefix = format!("joint_blocks.{i}.{suffix}");
                    out.push((format!("{prefix}.lora_A.weight"), lora.lora_a().clone()));
                    out.push((format!("{prefix}.lora_B.weight"), lora.lora_b().clone()));
                }
            }
        } else {
            // LyCORIS path — zip `to_parameters()` with `named_tensors()` to
            // recover the on-disk-name → Parameter mapping. Same convention
            // as `chroma::ChromaLoraBundle::named_parameters`.
            for (&(block_idx, target), adapter) in &self.lycoris_adapters {
                let prefix = format!("joint_blocks.{block_idx}.{}", target.suffix());
                let params = adapter.to_parameters();
                let names = adapter.named_tensors();
                for (param, (leaf, _)) in params.into_iter().zip(names.into_iter()) {
                    out.push((format!("{prefix}.{leaf}"), param));
                }
            }
        }
        out
    }

    /// Phase 2c — SimpleTuner-style perturbed-normal LoKr init.
    ///
    /// Walks `lycoris_adapters`, looks up the corresponding base weight in
    /// `self.weights` under `joint_blocks.{block_idx}.{target.suffix()}.weight`,
    /// and calls `AdapterModule::init_perturbed_normal_lokr(base, scale)` on
    /// each. The adapter method internally dispatches between the full-W2
    /// `init_perturbed_normal` (perturbs W2 in its statistical envelope) and
    /// the factored fallback `init_perturbed_normal_factored` (perturbs W2_B
    /// to break the `factorize_w2 + zero-init → dead-leaf` failure mode that
    /// stalls factored LoKr training under ScheduleFree warmup damping).
    ///
    /// No-op when `algo != LoKr` or `scale <= 0.0`. Returns the number of
    /// adapters skipped (missing base weight or skipped by `is_last`/dual-
    /// attention gating); applied count is logged at info level.
    pub fn apply_init_perturbed_normal(&self, scale: f32) -> Result<usize> {
        use crate::adapter::AdapterModule;
        if self.algo != LycorisAlgo::LoKr || scale <= 0.0 {
            return Ok(0);
        }
        let mut applied = 0usize;
        let mut skipped = 0usize;
        for (&(block_idx, target), adapter) in &self.lycoris_adapters {
            let key = format!("joint_blocks.{block_idx}.{}.weight", target.suffix());
            let Some(base) = self.weights.get(&key) else {
                log::warn!("[sd35][init_lokr_norm] missing base weight `{key}` — skipping");
                skipped += 1;
                continue;
            };
            let did = adapter
                .as_ref()
                .init_perturbed_normal_lokr(base, scale)
                .map_err(|e| EriDiffusionError::Model(format!(
                    "init_perturbed_normal_lokr({key}): {e}"
                )))?;
            if did { applied += 1; } else { skipped += 1; }
        }
        log::info!(
            "[sd35][init_lokr_norm] applied={applied} skipped={skipped} scale={scale}"
        );
        Ok(skipped)
    }
}
