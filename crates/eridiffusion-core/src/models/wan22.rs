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

use crate::config::TrainConfig;
use crate::lora::LoRALinear;
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
pub struct Wan22LoraBundle {
    pub adapters: HashMap<(usize, LoraTarget), LoRALinear>,
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
            rank,
            alpha,
            expert_label,
        })
    }

    pub fn parameters(&self) -> Vec<Parameter> {
        let mut out = Vec::with_capacity(self.adapters.len() * 2);
        for lora in self.adapters.values() {
            out.extend(lora.parameters());
        }
        out
    }

    pub fn num_adapters(&self) -> usize {
        self.adapters.len()
    }

    /// Save trained LoRA tensors to a single safetensors file. Naming
    /// matches archive `model.rs::WanLoraBundle::save`:
    /// `lora_wan_blocks_{i}_{target_key_with_dots_to_underscores}.{lora_A,lora_B,alpha}`.
    pub fn save(&self, path: &Path) -> Result<()> {
        let mut out: HashMap<String, Tensor> = HashMap::new();
        for ((idx, target), lora) in &self.adapters {
            let prefix = format!(
                "lora_wan_blocks_{idx}_{}",
                target.key().replace('.', "_")
            );
            let a = lora.lora_a().tensor()?;
            let b = lora.lora_b().tensor()?;
            out.insert(format!("{prefix}.lora_A.weight"), a);
            out.insert(format!("{prefix}.lora_B.weight"), b);
        }
        flame_core::serialization::save_file(&out, path).map_err(Into::into)
    }
}

// ---------------------------------------------------------------------------
// Wan22Model — single-expert wrapper
// ---------------------------------------------------------------------------

/// Single-expert Wan 2.2 transformer wrapper. The trainer instantiates
/// ONE of these for TI2V-5B, or TWO of these (high+low) for T2V/I2V-14B.
pub struct Wan22Model {
    pub config: Wan22Config,
    pub device: Arc<CudaDevice>,
    /// Frozen base weights. For 14B the trainer SHOULD load these in
    /// FP8 (`weight_dtype=fp8_scaled`) per
    /// `feedback_wan22_quant_exception.md` so 28B params fit on 24 GB.
    /// LoRA params remain F32.
    pub weights: HashMap<String, Tensor>,
    pub lora: Wan22LoraBundle,
    pub expert_label: &'static str,
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
        let raw = flame_core::serialization::load_file(ckpt_path, &device)?;
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
        })
    }

    pub fn parameters(&self) -> Vec<Parameter> {
        self.lora.parameters()
    }

    pub fn refresh_lora_cache(&self) {
        for lora in self.lora.adapters.values() {
            lora.refresh_cache();
        }
    }

    pub fn save_weights(&self, path: &Path) -> Result<()> {
        self.lora.save(path)
    }

    /// Training forward.
    ///
    /// Inputs:
    /// - `x`: noised latent `[B, C, F, H, W]` BF16
    /// - `timestep`: `[B]` F32 in `0..NUM_TRAIN_TIMESTEPS`
    /// - `context`: UMT5 text embedding `[B, text_len, text_dim]` BF16
    ///
    /// Returns the predicted velocity `[B, C, F, H, W]`.
    ///
    /// **Status:** not yet implemented. The block-level forward needs
    /// porting from `flame-diffusion-archive/wan-trainer/src/forward_impl/`
    /// (block.rs, head.rs, rope.rs, forward.rs) onto EDv2's tensor ops.
    /// Until then, every training step will hit this error and the
    /// dual-expert dispatch can be exercised with `--max-steps 0` /
    /// dry-run paths only.
    pub fn forward(
        &mut self,
        _x: &Tensor,
        _timestep: &Tensor,
        _context: &Tensor,
    ) -> Result<Tensor> {
        Err(crate::EriDiffusionError::Model(format!(
            "Wan22Model::forward[{}] not yet ported — see crates/eridiffusion-core/src/models/wan22.rs \
             module docs and flame-diffusion-archive/wan-trainer/src/forward_impl/ for the work \
             items (block.rs, head.rs, rope.rs, forward.rs).",
            self.expert_label
        )))
    }
}
