//! Klein (Flux 2) DiT — pure flame_core port for EriDiffusion-v2.
//!
//! Architecture (from inference-flame/src/models/klein.rs, verified pure-Rust):
//! - Klein 4B: inner=3072, heads=24, head_dim=128, double=5, single=20, mlp_hidden=9216, joint_dim=7680
//! - Klein 9B: inner=4096, heads=32, head_dim=128, double=8, single=24, mlp_hidden=12288, joint_dim=12288
//! - NO biases anywhere (every linear is `x @ W^T`, no add).
//! - SwiGLU MLP with 3x ratio (gate+up fused, then silu(gate)*up, then down).
//! - Shared modulation: 3 lin layers at model-level produce per-block shift/scale/gate.
//! - Single block fuses [QKV(3*inner) | gate+up(2*mlp)] into `linear1`, [attn_out(inner) | down(mlp)] into `linear2`.
//! - Joint attention concatenates txt-then-img across the sequence axis.
//! - 4-axis RoPE on `axes_dims=[32,32,32,32]` with `theta=2000.0`; img_ids=[N,4]=[0,row,col,0],
//!   txt_ids=[N,4]=[0,0,0,L_idx] (L axis varies, matches upstream Python `prepare_text_ids`).
//!
//! LoRA targets follow OT preset `transformer_block` (BaseFlux2Setup.LAYER_PRESETS["blocks"])
//! and upstream Python's per-Linear adapter granularity (12 attn linears per double block):
//!   - per double block (12 adapters): img_attn.{to_q,to_k,to_v,proj},
//!     txt_attn.{to_q,to_k,to_v,proj}, img_mlp.{0,2}, txt_mlp.{0,2}.
//!     The Q/K/V splits each train an `inner -> inner` adapter; their deltas
//!     are concatenated and added to the BFL fused-QKV base output. Matches
//!     diffusers `Flux2Attention.{to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj}`.
//!   - per single block (2 adapters): linear1, linear2 — diffusers
//!     `attn.to_qkv_mlp_proj` is fused at the same granularity, so 1:1 mapping.
//!
//! Variant auto-detection: `img_in.weight` first dim → inner_dim → 3072 (4B) or 4096 (9B).

use std::collections::HashMap;
use std::sync::Arc;
use cudarc::driver::CudaDevice;
use flame_core::{parameter::Parameter, DType, Shape, Tensor};
use crate::config::TrainConfig;
use crate::lora::LoRALinear;
use crate::models::TrainableModel;
use crate::Result;

#[derive(Debug, Clone)]
pub struct KleinConfig {
    pub inner_dim: usize,
    pub in_channels: usize,
    pub joint_attention_dim: usize,
    pub num_double: usize,
    pub num_single: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub mlp_hidden: usize,
    pub timestep_dim: usize,
    pub axes_dims: [usize; 4],
    pub theta: f32,
}

impl KleinConfig {
    pub fn klein_4b() -> Self {
        Self {
            inner_dim: 3072, in_channels: 128, joint_attention_dim: 7680,
            num_double: 5, num_single: 20, num_heads: 24, head_dim: 128,
            mlp_hidden: 3072 * 3, timestep_dim: 256,
            axes_dims: [32, 32, 32, 32], theta: 2000.0,
        }
    }
    pub fn klein_9b() -> Self {
        Self {
            inner_dim: 4096, in_channels: 128, joint_attention_dim: 12288,
            num_double: 8, num_single: 24, num_heads: 32, head_dim: 128,
            mlp_hidden: 4096 * 3, timestep_dim: 256,
            axes_dims: [32, 32, 32, 32], theta: 2000.0,
        }
    }

    /// Auto-detect from a pre-loaded weight map (looks at `img_in.weight`,
    /// `txt_in.weight`, and counts `double_blocks.*` / `single_blocks.*`).
    /// Returns the inferred config; mirrors `inference-flame::KleinTransformer::from_weights`.
    pub fn from_weights(weights: &HashMap<String, Tensor>) -> Result<Self> {
        let img_in = weights.get("img_in.weight").ok_or_else(||
            crate::EriDiffusionError::Model("missing img_in.weight (Klein autodetect)".into()))?;
        let inner_dim = img_in.shape().dims()[0];
        let in_channels = img_in.shape().dims()[1];
        let joint = weights.get("txt_in.weight").ok_or_else(||
            crate::EriDiffusionError::Model("missing txt_in.weight (Klein autodetect)".into()))?;
        let joint_attention_dim = joint.shape().dims()[1];
        let mut num_double = 0;
        while weights.contains_key(&format!("double_blocks.{num_double}.img_attn.qkv.weight")) {
            num_double += 1;
        }
        let mut num_single = 0;
        while weights.contains_key(&format!("single_blocks.{num_single}.linear1.weight")) {
            num_single += 1;
        }
        let num_heads = inner_dim / 128;
        Ok(Self {
            inner_dim, in_channels, joint_attention_dim,
            num_double, num_single, num_heads, head_dim: 128,
            mlp_hidden: inner_dim * 3, timestep_dim: 256,
            axes_dims: [32, 32, 32, 32], theta: 2000.0,
        })
    }
}

const NORM_EPS: f32 = 1e-6;

// LoRA slot layout per double block (12 adapters).
// Audit fix KLEIN_VERIFY §H1.3 / SKEPTIC §H3: upstream Python wraps Q/K/V as 3
// separate `nn.Linear` modules per attention (`to_q`, `to_k`, `to_v` for
// the image stream, `add_q_proj`, `add_k_proj`, `add_v_proj` for the text
// stream — see diffusers `Flux2Attention.__init__` lines 526-543). A single
// fused `qkv` LoRA gives only rank-r capacity over the entire `3*inner`
// output; three separate Q/K/V LoRAs give 3r effective capacity. Matches
// upstream Python adapter granularity bit-for-bit.
//
//   0: img_attn.to_q (inner -> inner)
//   1: img_attn.to_k (inner -> inner)
//   2: img_attn.to_v (inner -> inner)
//   3: img_attn.proj (inner -> inner)
//   4: txt_attn.to_q (inner -> inner)   [maps to BFL fused txt_attn.qkv slice]
//   5: txt_attn.to_k (inner -> inner)
//   6: txt_attn.to_v (inner -> inner)
//   7: txt_attn.proj (inner -> inner)
//   8: img_mlp.0     (inner -> 2*mlp_hidden, gate+up fused — diffusers `ff.linear_in`)
//   9: img_mlp.2     (mlp_hidden -> inner,  diffusers `ff.linear_out`)
//  10: txt_mlp.0     (inner -> 2*mlp_hidden, diffusers `ff_context.linear_in`)
//  11: txt_mlp.2     (mlp_hidden -> inner,  diffusers `ff_context.linear_out`)
const DOUBLE_LORA_SLOTS: usize = 12;

// LoRA slot layout per single block (2 adapters):
//   0: linear1 (inner -> 3*inner + 2*mlp_hidden)
//     diffusers `attn.to_qkv_mlp_proj` is the same fused 5*inner Linear,
//     so 1:1 mapping — no Q/K/V split needed here.
//   1: linear2 (inner + mlp_hidden -> inner)  diffusers `attn.to_out`
const SINGLE_LORA_SLOTS: usize = 2;

const DOUBLE_LORA_KEYS: [&str; DOUBLE_LORA_SLOTS] = [
    "img_attn.to_q", "img_attn.to_k", "img_attn.to_v", "img_attn.proj",
    "txt_attn.to_q", "txt_attn.to_k", "txt_attn.to_v", "txt_attn.proj",
    "img_mlp.0", "img_mlp.2", "txt_mlp.0", "txt_mlp.2",
];
const SINGLE_LORA_KEYS: [&str; SINGLE_LORA_SLOTS] = ["linear1", "linear2"];

pub struct KleinModel {
    pub config: TrainConfig,
    pub kconfig: KleinConfig,
    pub device: Arc<CudaDevice>,
    pub weights: HashMap<String, Tensor>,
    /// `num_double * 8 + num_single * 2` adapters in stable order
    /// (all double blocks first, then all single blocks).
    pub lora_adapters: Vec<LoRALinear>,
    pub parameters: Vec<Parameter>,
    pub is_lora: bool,
    /// When Some, per-block weights are streamed from pinned host RAM into
    /// reusable GPU slots per block, per step via BlockOffloader.
    /// Unified index space: `0..num_double` → double_blocks.{i},
    /// `num_double..num_double+num_single` → single_blocks.{i}.
    pub offloader: Option<std::sync::Arc<std::sync::Mutex<crate::training::block_offload::BlockOffloader>>>,
}

impl KleinModel {
    pub fn load(
        paths: &[std::path::PathBuf],
        config: &TrainConfig,
        device: Arc<CudaDevice>,
    ) -> Result<Self> {
        let mut weights = HashMap::new();
        for p in paths {
            let part = flame_core::serialization::load_file(p, &device)?;
            for (k, v) in part {
                weights.insert(k, v.to_dtype(DType::BF16)?);
            }
        }
        let kconfig = KleinConfig::from_weights(&weights)?;
        log::info!(
            "Klein autodetect: inner={} joint={} double={} single={} heads={} (Klein {})",
            kconfig.inner_dim, kconfig.joint_attention_dim,
            kconfig.num_double, kconfig.num_single, kconfig.num_heads,
            if kconfig.inner_dim == 3072 { "4B" } else if kconfig.inner_dim == 4096 { "9B" } else { "?" },
        );
        Self::new(weights, kconfig, config, device)
    }

    /// Construct from already-loaded weights (used by sample_klein for the
    /// base-only path that wants to skip the F32 parameter copy).
    pub fn new(
        weights: HashMap<String, Tensor>,
        kconfig: KleinConfig,
        config: &TrainConfig,
        device: Arc<CudaDevice>,
    ) -> Result<Self> {
        let is_lora = config.is_lora();
        let mut lora_adapters = Vec::new();
        let mut parameters = Vec::new();
        if is_lora {
            let rank = config.lora_rank as usize;
            let alpha = config.lora_alpha as f32;
            let inner = kconfig.inner_dim;
            let mlp = kconfig.mlp_hidden;
            // Double blocks: 12 adapters each (split Q/K/V).
            for i in 0..kconfig.num_double {
                let s = 42u64 + i as u64 * 16;
                // 0/1/2: img_attn.to_q/to_k/to_v  (each inner -> inner)
                lora_adapters.push(LoRALinear::new(inner, inner, rank, alpha, device.clone(), s)?);
                lora_adapters.push(LoRALinear::new(inner, inner, rank, alpha, device.clone(), s + 1)?);
                lora_adapters.push(LoRALinear::new(inner, inner, rank, alpha, device.clone(), s + 2)?);
                // 3: img_attn.proj
                lora_adapters.push(LoRALinear::new(inner, inner, rank, alpha, device.clone(), s + 3)?);
                // 4/5/6: txt_attn.to_q/to_k/to_v
                lora_adapters.push(LoRALinear::new(inner, inner, rank, alpha, device.clone(), s + 4)?);
                lora_adapters.push(LoRALinear::new(inner, inner, rank, alpha, device.clone(), s + 5)?);
                lora_adapters.push(LoRALinear::new(inner, inner, rank, alpha, device.clone(), s + 6)?);
                // 7: txt_attn.proj
                lora_adapters.push(LoRALinear::new(inner, inner, rank, alpha, device.clone(), s + 7)?);
                // 8: img_mlp.0 (gate+up fused — diffusers ff.linear_in is also fused)
                lora_adapters.push(LoRALinear::new(inner, 2 * mlp, rank, alpha, device.clone(), s + 8)?);
                // 9: img_mlp.2 (down — diffusers ff.linear_out)
                lora_adapters.push(LoRALinear::new(mlp, inner, rank, alpha, device.clone(), s + 9)?);
                // 10: txt_mlp.0
                lora_adapters.push(LoRALinear::new(inner, 2 * mlp, rank, alpha, device.clone(), s + 10)?);
                // 11: txt_mlp.2
                lora_adapters.push(LoRALinear::new(mlp, inner, rank, alpha, device.clone(), s + 11)?);
            }
            // Single blocks: linear1 (5*inner fused) + linear2 (out projection).
            // Diffusers `attn.to_qkv_mlp_proj` is fused at the same granularity,
            // so 1:1 mapping — no Q/K/V split needed.
            for i in 0..kconfig.num_single {
                let s = 42u64 + (kconfig.num_double + i) as u64 * 16;
                // 0: linear1 (inner -> 3*inner + 2*mlp)
                lora_adapters.push(LoRALinear::new(inner, 3 * inner + 2 * mlp, rank, alpha, device.clone(), s)?);
                // 1: linear2 (inner + mlp -> inner)
                lora_adapters.push(LoRALinear::new(inner + mlp, inner, rank, alpha, device.clone(), s + 1)?);
            }
            for l in &lora_adapters { parameters.extend(l.parameters()); }
        } else {
            for (_, t) in &weights {
                parameters.push(Parameter::new(t.to_dtype(DType::F32)?.requires_grad_(true)));
            }
        }
        log::info!("Klein: {} tensors loaded, {} LoRA params (lora={})",
            weights.len(), parameters.len(), is_lora);
        Ok(Self {
            config: config.clone(), kconfig, device,
            weights, lora_adapters, parameters, is_lora,
            offloader: None,
        })
    }

    /// Enable per-block weight streaming via `BlockOffloader`. Drops
    /// `double_blocks.*`/`single_blocks.*` from VRAM; blocks are streamed from
    /// pinned host RAM into reusable GPU slots per block, per step.
    /// Works for both base and LoRA inference.
    pub fn enable_offload(&mut self, shards: Vec<std::path::PathBuf>) -> Result<()> {
        let num_double = self.kconfig.num_double;
        let num_single = self.kconfig.num_single;
        let to_drop: Vec<String> = self.weights.keys()
            .filter(|k| k.starts_with("double_blocks.") || k.starts_with("single_blocks."))
            .cloned()
            .collect();
        let n = to_drop.len();
        for k in to_drop { self.weights.remove(&k); }
        log::info!("Klein offload: dropped {} per-block weights", n);
        flame_core::cuda_alloc_pool::clear_pool_cache();
        flame_core::trim_cuda_mempool(0);

        struct KleinFacilitator { num_double: usize, num_single: usize }
        impl crate::training::block_offload::BlockFacilitator for KleinFacilitator {
            fn block_count(&self) -> usize { self.num_double + self.num_single }
            fn classify_key(&self, key: &str) -> Option<usize> {
                if let Some(rest) = key.strip_prefix("double_blocks.") {
                    let idx: usize = rest.split('.').next()?.parse().ok()?;
                    if idx < self.num_double { return Some(idx); }
                }
                if let Some(rest) = key.strip_prefix("single_blocks.") {
                    let idx: usize = rest.split('.').next()?.parse().ok()?;
                    if idx < self.num_single { return Some(self.num_double + idx); }
                }
                None
            }
        }
        let facilitator = KleinFacilitator { num_double, num_single };

        let shard_strs: Vec<String> = shards.iter()
            .map(|p| p.to_string_lossy().into_owned())
            .collect();
        let path_refs: Vec<&str> = shard_strs.iter().map(|s| s.as_str()).collect();

        let use_streaming = std::env::var("KLEIN_BLOCK_STREAMING")
            .ok()
            .map(|v| !matches!(v.as_str(), "0" | "" | "false" | "False"))
            .unwrap_or(true);

        let offloader = if use_streaming {
            log::info!("Klein BlockOffloader: streaming mode");
            crate::training::block_offload::BlockOffloader::load_streaming(
                &path_refs, &facilitator, self.device.clone(),
            )
        } else {
            log::info!("Klein BlockOffloader: pinned-RAM mode");
            crate::training::block_offload::BlockOffloader::load(
                &path_refs, &facilitator, self.device.clone(),
            )
        }
        // native_layout=true: leave 2D .weight tensors in on-disk [Cout, Cin] layout.
        // Klein model code calls `.transpose()` itself (via linear_3d) before matmul.
        .map(|o| o.with_native_layout(true))
        .map_err(|e| crate::EriDiffusionError::Model(format!("BlockOffloader: {e}")))?;

        self.offloader = Some(std::sync::Arc::new(std::sync::Mutex::new(offloader)));
        log::info!("Klein BlockOffloader ready ({} unified blocks)", num_double + num_single);
        Ok(())
    }

    fn w(&self, key: &str) -> Result<&Tensor> {
        self.weights.get(key).ok_or_else(||
            crate::EriDiffusionError::Model(format!("missing weight: {}", key)))
    }

    fn linear(&self, x: &Tensor, key: &str) -> Result<Tensor> {
        x.matmul(&self.w(key)?.transpose()?).map_err(Into::into)
    }
}

// ---------------------------------------------------------------------------
// Standalone helpers used inside `AutogradContext::checkpoint` closures.
// They take only owned Tensors / HashMaps so the closure can be `'static`.
// ---------------------------------------------------------------------------

/// `x @ w^T` for both `[B,N,C]` and `[M,C]` x. Returns flame Result.
fn linear_3d(x: &Tensor, w: &Tensor) -> flame_core::Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    if dims.len() == 2 {
        x.matmul(&w.transpose()?)
    } else {
        let m: usize = dims[..dims.len() - 1].iter().product();
        let c = *dims.last().unwrap();
        let x_2d = x.reshape(&[m, c])?;
        let out = x_2d.matmul(&w.transpose()?)?;
        let out_dim = *out.shape().dims().last().unwrap();
        let mut new_dims = dims.clone();
        *new_dims.last_mut().unwrap() = out_dim;
        out.reshape(&new_dims)
    }
}

/// `linear_3d` + LoRA delta if an adapter is present. Adapters are passed by
/// owned slice so the checkpoint closure can drop intermediates.
fn linear_with_lora(
    x: &Tensor,
    w: &Tensor,
    adapter: Option<&LoRALinear>,
) -> flame_core::Result<Tensor> {
    let base = linear_3d(x, w)?;
    match adapter {
        Some(a) => {
            let delta = a.forward_delta(x)
                .map_err(|e| flame_core::FlameError::InvalidInput(format!("lora delta: {e}")))?;
            base.add(&delta)
        }
        None => Ok(base),
    }
}

/// Fused-QKV linear with **three separate Q/K/V LoRAs** applied to the
/// matching slices of the output.
///
/// Background: the BFL on-disk weights store `img_attn.qkv.weight` and
/// `txt_attn.qkv.weight` as a single `[3*inner, inner]` matrix that produces
/// `[B, N, 3*inner]` (Q | K | V concatenated along the last axis). upstream Python
/// however wraps Q, K, V as 3 separate `nn.Linear` modules and so trains
/// 3 independent LoRA adapters per attention. To match that adapter
/// granularity bit-for-bit while keeping the fused base matmul, we compute
/// the LoRA deltas separately for each of Q/K/V (each `inner -> inner`)
/// and concatenate them along the last axis before adding to the base.
///
/// Audit fix KLEIN_VERIFY §H1.3 / SKEPTIC §H3.
fn linear_with_split_qkv_lora(
    x: &Tensor,
    w_fused_qkv: &Tensor,
    lora_q: Option<&LoRALinear>,
    lora_k: Option<&LoRALinear>,
    lora_v: Option<&LoRALinear>,
) -> flame_core::Result<Tensor> {
    let base = linear_3d(x, w_fused_qkv)?;
    if lora_q.is_none() && lora_k.is_none() && lora_v.is_none() {
        return Ok(base);
    }
    // Each delta is [B, N, inner]. If an adapter is missing for a slice,
    // we fall back to a zero tensor of the same shape (allocating only when
    // the partial-adapter case actually arises).
    let zeros_like_inner = || -> flame_core::Result<Tensor> {
        // Infer the per-slice inner dimension from the fused weight rows.
        let last = *base.shape().dims().last().unwrap();
        debug_assert!(last % 3 == 0, "linear_with_split_qkv_lora: fused QKV last dim {} not divisible by 3", last);
        let inner = last / 3;
        let mut shape = base.shape().dims().to_vec();
        *shape.last_mut().unwrap() = inner;
        Tensor::zeros_dtype(Shape::from_dims(&shape), base.dtype(), base.device().clone())
    };
    let dq = match lora_q {
        Some(a) => a.forward_delta(x)
            .map_err(|e| flame_core::FlameError::InvalidInput(format!("lora delta Q: {e}")))?,
        None => zeros_like_inner()?,
    };
    let dk = match lora_k {
        Some(a) => a.forward_delta(x)
            .map_err(|e| flame_core::FlameError::InvalidInput(format!("lora delta K: {e}")))?,
        None => zeros_like_inner()?,
    };
    let dv = match lora_v {
        Some(a) => a.forward_delta(x)
            .map_err(|e| flame_core::FlameError::InvalidInput(format!("lora delta V: {e}")))?,
        None => zeros_like_inner()?,
    };
    let delta = Tensor::cat(&[&dq, &dk, &dv], dq.shape().dims().len() - 1)?;
    base.add(&delta)
}

/// Sinusoidal timestep embedding (ComfyUI convention, time_factor=1000).
fn timestep_embedding(
    t: &Tensor, dim: usize, time_factor: f32,
    device: &Arc<CudaDevice>,
) -> flame_core::Result<Tensor> {
    let orig = t.dtype();
    let b = t.shape().dims()[0];
    let t_f32 = t.to_dtype(DType::F32)?;
    let t_scaled = t_f32.mul_scalar(time_factor)?;
    let half = dim / 2;
    let max_period: f32 = 10000.0;
    let freqs = Tensor::arange(0.0, half as f32, 1.0, device.clone())?;
    let freqs = freqs.mul_scalar(-max_period.ln() / half as f32)?.exp()?;
    let t_col = t_scaled.reshape(&[b, 1])?;
    let freqs_row = freqs.reshape(&[1, half])?;
    let args = t_col.mul(&freqs_row)?;
    let cos_part = args.cos()?;
    let sin_part = args.sin()?;
    let emb = Tensor::cat(&[&cos_part, &sin_part], 1)?;
    emb.to_dtype(orig)
}

/// Build 4-axis RoPE cos/sin tables. img_ids/txt_ids are `[N, 4]`.
/// Returns `(pe_cos, pe_sin)` each `[1, 1, n_total, head_dim]` (BF16),
/// where `head_dim = sum(axes_dims)` and the per-axis cos/sin table is
/// the same `[N, axis_dim/2]` concatenated across axes (matches inference-flame).
/// NOTE: matching inference-flame's `bf16_ops::rope_fused_bf16` expectations:
/// the helper inside Klein expects pe of length `head_dim/2` (half), but our
/// fully-elementwise port concatenates `cos,sin` for full `head_dim` length —
/// see `apply_rope_klein` for the math.
fn build_rope_klein(
    img_ids: &Tensor,
    txt_ids: &Tensor,
    axes_dims: &[usize; 4],
    theta: f32,
    device: &Arc<CudaDevice>,
) -> flame_core::Result<(Tensor, Tensor)> {
    // Concat: txt first, then img (matches inference-flame).
    let all_ids = Tensor::cat(&[txt_ids, img_ids], 0)?;
    let n_total = all_ids.shape().dims()[0];

    let mut cos_parts: Vec<Tensor> = Vec::new();
    let mut sin_parts: Vec<Tensor> = Vec::new();

    for (axis_idx, &dim) in axes_dims.iter().enumerate() {
        let half = dim / 2;
        let pos = all_ids
            .narrow(1, axis_idx, 1)?
            .squeeze(Some(1))?
            .to_dtype(DType::F32)?;
        let freq_idx = Tensor::arange(0.0, dim as f32, 2.0, device.clone())?;
        let log_freqs = freq_idx.mul_scalar(-theta.ln() / dim as f32)?.exp()?;
        let pos_col = pos.reshape(&[n_total, 1])?;
        let freq_row = log_freqs.reshape(&[1, half])?;
        let angles = pos_col.mul(&freq_row)?;
        cos_parts.push(angles.cos()?);
        sin_parts.push(angles.sin()?);
    }
    let cos_refs: Vec<&Tensor> = cos_parts.iter().collect();
    let sin_refs: Vec<&Tensor> = sin_parts.iter().collect();
    let pe_cos = Tensor::cat(&cos_refs, 1)?
        .unsqueeze(0)?.unsqueeze(0)?.to_dtype(DType::BF16)?;
    let pe_sin = Tensor::cat(&sin_refs, 1)?
        .unsqueeze(0)?.unsqueeze(0)?.to_dtype(DType::BF16)?;
    Ok((pe_cos, pe_sin))
}

/// Apply rotary embeddings (rotate-half, inference-flame parity).
/// `q`: `[B, H, N, D]`, `pe_cos`/`pe_sin`: `[1, 1, N, D/2]`.
/// Returns `[B, H, N, D]`.
fn apply_rope_klein(
    q: &Tensor, k: &Tensor, pe_cos: &Tensor, pe_sin: &Tensor,
) -> flame_core::Result<(Tensor, Tensor)> {
    // Use the verified flame-core fused kernel.
    let q_out = flame_core::bf16_ops::rope_fused_bf16(q, pe_cos, pe_sin)?;
    let k_out = flame_core::bf16_ops::rope_fused_bf16(k, pe_cos, pe_sin)?;
    Ok((q_out, k_out))
}

/// Per-head RMSNorm for query/key. `x`: `[B, H, N, D]`.
fn head_rms_norm_local(x: &Tensor, scale: &Tensor) -> flame_core::Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    let (b, h, n, d) = (dims[0], dims[1], dims[2], dims[3]);
    let flat = x.reshape(&[b * h * n, d])?;
    let normed = flame_core::norm::rms_norm(&flat, &[d], Some(scale), NORM_EPS)?;
    normed.reshape(&[b, h, n, d])
}

/// `(1 + scale) * LayerNorm(x) + shift` using flame-core's fused bf16 kernel.
fn modulate_pre_local(x: &Tensor, shift: &Tensor, scale: &Tensor) -> flame_core::Result<Tensor> {
    flame_core::bf16_ops::modulate_pre_fused_bf16(x, shift, scale, NORM_EPS)
}

/// Standalone double-block forward used inside `AutogradContext::checkpoint`.
#[allow(clippy::too_many_arguments)]
fn double_block_forward_standalone(
    img: Tensor,
    txt: Tensor,
    img_mods: [Tensor; 6],
    txt_mods: [Tensor; 6],
    pe_cos: Tensor,
    pe_sin: Tensor,
    layer_weights: HashMap<String, Tensor>,
    lora_adapters: Option<Vec<LoRALinear>>,
    block_idx: usize,
    num_heads: usize,
    head_dim: usize,
    inner_dim: usize,
) -> flame_core::Result<(Tensor, Tensor)> {
    let prefix = format!("double_blocks.{block_idx}");
    let h = num_heads;
    let d = head_dim;
    let w = |key: &str| -> flame_core::Result<&Tensor> {
        layer_weights.get(key).ok_or_else(||
            flame_core::FlameError::InvalidInput(format!("Klein double {block_idx}: missing {key}")))
    };
    let lin = |x: &Tensor, key_suffix: &str, lora_idx: usize| -> flame_core::Result<Tensor> {
        let key = format!("{prefix}.{key_suffix}.weight");
        let weight = w(&key)?;
        let adapter = lora_adapters.as_ref().and_then(|a| a.get(lora_idx));
        linear_with_lora(x, weight, adapter)
    };
    // Fused-QKV linear with 3 separate Q/K/V LoRAs (audit fix H1.3 / H3).
    let lin_qkv_split = |x: &Tensor, key_suffix: &str,
                         lora_q_idx: usize, lora_k_idx: usize, lora_v_idx: usize|
                         -> flame_core::Result<Tensor> {
        let key = format!("{prefix}.{key_suffix}.weight");
        let weight = w(&key)?;
        let lq = lora_adapters.as_ref().and_then(|a| a.get(lora_q_idx));
        let lk = lora_adapters.as_ref().and_then(|a| a.get(lora_k_idx));
        let lv = lora_adapters.as_ref().and_then(|a| a.get(lora_v_idx));
        linear_with_split_qkv_lora(x, weight, lq, lk, lv)
    };

    let (img_shift1, img_scale1, img_gate1) = (&img_mods[0], &img_mods[1], &img_mods[2]);
    let (img_shift2, img_scale2, img_gate2) = (&img_mods[3], &img_mods[4], &img_mods[5]);
    let (txt_shift1, txt_scale1, txt_gate1) = (&txt_mods[0], &txt_mods[1], &txt_mods[2]);
    let (txt_shift2, txt_scale2, txt_gate2) = (&txt_mods[3], &txt_mods[4], &txt_mods[5]);

    // Attention
    let img_normed = modulate_pre_local(&img, img_shift1, img_scale1)?;
    let txt_normed = modulate_pre_local(&txt, txt_shift1, txt_scale1)?;

    // Slot map per DOUBLE_LORA_KEYS (12 adapters/block):
    //  0/1/2 = img_attn.to_q/to_k/to_v ; 3 = img_attn.proj
    //  4/5/6 = txt_attn.to_q/to_k/to_v ; 7 = txt_attn.proj
    //  8/9   = img_mlp.0/.2            ; 10/11 = txt_mlp.0/.2
    let img_qkv = lin_qkv_split(&img_normed, "img_attn.qkv", 0, 1, 2)?;
    let txt_qkv = lin_qkv_split(&txt_normed, "txt_attn.qkv", 4, 5, 6)?;

    let n_img = img_qkv.shape().dims()[1];
    let n_txt = txt_qkv.shape().dims()[1];

    let (mut img_q, mut img_k, img_v) =
        flame_core::bf16_ops::qkv_split_permute_bf16(&img_qkv, h, d)?;
    let (mut txt_q, mut txt_k, txt_v) =
        flame_core::bf16_ops::qkv_split_permute_bf16(&txt_qkv, h, d)?;

    img_q = head_rms_norm_local(&img_q, w(&format!("{prefix}.img_attn.norm.query_norm.scale"))?)?;
    img_k = head_rms_norm_local(&img_k, w(&format!("{prefix}.img_attn.norm.key_norm.scale"))?)?;
    txt_q = head_rms_norm_local(&txt_q, w(&format!("{prefix}.txt_attn.norm.query_norm.scale"))?)?;
    txt_k = head_rms_norm_local(&txt_k, w(&format!("{prefix}.txt_attn.norm.key_norm.scale"))?)?;

    let q = Tensor::cat(&[&txt_q, &img_q], 2)?;
    let k = Tensor::cat(&[&txt_k, &img_k], 2)?;
    let v = Tensor::cat(&[&txt_v, &img_v], 2)?;

    let (q, k) = apply_rope_klein(&q, &k, &pe_cos, &pe_sin)?;

    let attn_out = flame_core::attention::sdpa(&q, &k, &v, None)?;
    let (txt_out, img_out) =
        flame_core::bf16_ops::attn_split_txt_img_bf16(&attn_out, n_txt, n_img)?;

    let img_proj = lin(&img_out, "img_attn.proj", 3)?;
    let txt_proj = lin(&txt_out, "txt_attn.proj", 7)?;

    let img = flame_core::bf16_ops::gate_residual_fused_bf16(&img, img_gate1, &img_proj)?;
    let txt = flame_core::bf16_ops::gate_residual_fused_bf16(&txt, txt_gate1, &txt_proj)?;
    let _ = inner_dim;

    // MLP (SwiGLU)
    let img_mlp_in = modulate_pre_local(&img, img_shift2, img_scale2)?;
    let txt_mlp_in = modulate_pre_local(&txt, txt_shift2, txt_scale2)?;

    // img_mlp: gate+up fused, then silu(gate)*up, then down
    let img_gu = lin(&img_mlp_in, "img_mlp.0", 8)?;
    let last_dim = *img_gu.shape().dims().last().unwrap();
    let half = last_dim / 2;
    let ndim = img_gu.shape().dims().len();
    let img_gate = img_gu.narrow(ndim - 1, 0, half)?;
    let img_up = img_gu.narrow(ndim - 1, half, half)?;
    let img_act = flame_core::bf16_ops::swiglu_fused_bf16(&img_gate, &img_up)?;
    let img_mlp_out = lin(&img_act, "img_mlp.2", 9)?;

    let txt_gu = lin(&txt_mlp_in, "txt_mlp.0", 10)?;
    let last_dim = *txt_gu.shape().dims().last().unwrap();
    let half = last_dim / 2;
    let ndim = txt_gu.shape().dims().len();
    let txt_gate = txt_gu.narrow(ndim - 1, 0, half)?;
    let txt_up = txt_gu.narrow(ndim - 1, half, half)?;
    let txt_act = flame_core::bf16_ops::swiglu_fused_bf16(&txt_gate, &txt_up)?;
    let txt_mlp_out = lin(&txt_act, "txt_mlp.2", 11)?;

    let img = flame_core::bf16_ops::gate_residual_fused_bf16(&img, img_gate2, &img_mlp_out)?;
    let txt = flame_core::bf16_ops::gate_residual_fused_bf16(&txt, txt_gate2, &txt_mlp_out)?;
    Ok((img, txt))
}

/// Standalone single-block forward used inside `AutogradContext::checkpoint`.
#[allow(clippy::too_many_arguments)]
fn single_block_forward_standalone(
    x: Tensor,
    mods: [Tensor; 3],
    pe_cos: Tensor,
    pe_sin: Tensor,
    layer_weights: HashMap<String, Tensor>,
    lora_adapters: Option<Vec<LoRALinear>>,
    block_idx: usize,
    num_heads: usize,
    head_dim: usize,
    inner_dim: usize,
    mlp_hidden: usize,
) -> flame_core::Result<Tensor> {
    let prefix = format!("single_blocks.{block_idx}");
    let h = num_heads;
    let d = head_dim;
    let w = |key: &str| -> flame_core::Result<&Tensor> {
        layer_weights.get(key).ok_or_else(||
            flame_core::FlameError::InvalidInput(format!("Klein single {block_idx}: missing {key}")))
    };
    let lin = |x: &Tensor, key_suffix: &str, lora_idx: usize| -> flame_core::Result<Tensor> {
        let key = format!("{prefix}.{key_suffix}.weight");
        let weight = w(&key)?;
        let adapter = lora_adapters.as_ref().and_then(|a| a.get(lora_idx));
        linear_with_lora(x, weight, adapter)
    };

    let (shift, scale, gate) = (&mods[0], &mods[1], &mods[2]);
    let x_normed = modulate_pre_local(&x, shift, scale)?;

    // Fused QKV + gate+up
    let qkv_mlp = lin(&x_normed, "linear1", 0)?;
    let qkv_dim = 3 * inner_dim;
    let qkv = qkv_mlp.narrow(2, 0, qkv_dim)?;
    let gate_up = qkv_mlp.narrow(2, qkv_dim, 2 * mlp_hidden)?;

    let dims = qkv.shape().dims();
    let (b, n) = (dims[0], dims[1]);

    let (q, k, v) = flame_core::bf16_ops::qkv_split_permute_bf16(&qkv, h, d)?;

    let q = head_rms_norm_local(&q, w(&format!("{prefix}.norm.query_norm.scale"))?)?;
    let k = head_rms_norm_local(&k, w(&format!("{prefix}.norm.key_norm.scale"))?)?;

    let (q, k) = apply_rope_klein(&q, &k, &pe_cos, &pe_sin)?;
    let attn_out = flame_core::attention::sdpa(&q, &k, &v, None)?;
    let attn_out = attn_out.permute(&[0, 2, 1, 3])?.reshape(&[b, n, h * d])?;

    let mlp_gate = gate_up.narrow(2, 0, mlp_hidden)?;
    let mlp_up = gate_up.narrow(2, mlp_hidden, mlp_hidden)?;
    let mlp_act = flame_core::bf16_ops::swiglu_fused_bf16(&mlp_gate, &mlp_up)?;

    let fused = Tensor::cat(&[&attn_out, &mlp_act], 2)?;
    let out = lin(&fused, "linear2", 1)?;

    flame_core::bf16_ops::gate_residual_fused_bf16(&x, gate, &out)
}

impl KleinModel {
    /// Klein forward.
    ///
    /// * `img`: `[B, in_channels, H, W]` packed VAE latents (post-patchify, post-scale).
    ///   Caller is responsible for `pack` (`[B,C,H,W]` → `[B, H*W, C]` via permute).
    /// * `txt`: `[B, T, joint_attention_dim]` text embeddings (Qwen3 stacked layers).
    /// * `timestep`: `[B]` continuous t in `[0, 1)` — i.e. `int_timestep / 1000`
    ///   per upstream Python `BaseFlux2Setup.py:144` (`timestep=timestep/1000`). The
    ///   `timestep_embedding` helper then multiplies by `time_factor=1000` so
    ///   the sin/cos arguments are `int_timestep * freq`. Matches inference-flame
    ///   klein euler (sigma fed directly from `get_schedule()` ∈ `[0, 1]`).
    ///
    /// Returns predicted velocity in **packed** `[B, in_channels, H, W]` matching `img` shape.
    pub fn forward(
        &mut self,
        img_packed_bchw: &Tensor,
        txt: &Tensor,
        timestep: &Tensor,
    ) -> Result<Tensor> {
        let dims = img_packed_bchw.shape().dims().to_vec();
        let (b, c, h_lat, w_lat) = (dims[0], dims[1], dims[2], dims[3]);
        if c != self.kconfig.in_channels {
            return Err(crate::EriDiffusionError::Model(format!(
                "Klein forward: expected {} channels, got {}", self.kconfig.in_channels, c)));
        }
        let n_img = h_lat * w_lat;
        let n_txt = txt.shape().dims()[1];
        let inner = self.kconfig.inner_dim;
        let mlp = self.kconfig.mlp_hidden;
        let in_ch = self.kconfig.in_channels;

        // Pack latent: [B, C, H, W] → [B, H*W, C]
        let img_packed = img_packed_bchw
            .permute(&[0, 2, 3, 1])?
            .contiguous()?
            .reshape(&[b, n_img, in_ch])?;

        // Build position IDs on the fly
        let mut img_ids_data = vec![0f32; n_img * 4];
        for r in 0..h_lat {
            for col in 0..w_lat {
                let idx = r * w_lat + col;
                img_ids_data[idx * 4 + 1] = r as f32;
                img_ids_data[idx * 4 + 2] = col as f32;
            }
        }
        let img_ids = Tensor::from_vec(
            img_ids_data, Shape::from_dims(&[n_img, 4]), self.device.clone()
        )?.to_dtype(DType::BF16)?;
        // upstream Python `Flux2Model.prepare_text_ids`:
        //   cartesian_prod(arange(1), arange(1), arange(1), arange(L))
        //   → row k = [0, 0, 0, k] for k in [0, L). The L-axis (column 3)
        // is the same axis that gets `axes_dims[3]=32` rotary frequencies,
        // so each text token receives a distinct RoPE phase.
        // Audit fix KLEIN_VERIFY §H2 / SKEPTIC §H2: previously all-zero,
        // which collapsed text positions and lost ordering information.
        let mut txt_ids_data = vec![0f32; n_txt * 4];
        for k in 0..n_txt {
            txt_ids_data[k * 4 + 3] = k as f32;
        }
        let txt_ids = Tensor::from_vec(
            txt_ids_data, Shape::from_dims(&[n_txt, 4]), self.device.clone(),
        )?.to_dtype(DType::BF16)?;

        // Input projections (NO bias)
        let img_proj = self.linear(&img_packed, "img_in.weight")?;
        let txt_proj = self.linear(txt, "txt_in.weight")?;

        // Timestep -> vec
        let t_emb = timestep_embedding(timestep, self.kconfig.timestep_dim, 1000.0, &self.device)?;
        let t_emb = t_emb.to_dtype(DType::BF16)?;
        let h1 = self.linear(&t_emb, "time_in.in_layer.weight")?.silu()?;
        let vec = self.linear(&h1, "time_in.out_layer.weight")?;

        // RoPE
        let (pe_cos, pe_sin) = build_rope_klein(
            &img_ids, &txt_ids, &self.kconfig.axes_dims, self.kconfig.theta, &self.device,
        )?;

        // Pre-compute shared modulations once
        let vec_silu = vec.silu()?;
        let img_mods_raw = self.linear(&vec_silu, "double_stream_modulation_img.lin.weight")?;
        let txt_mods_raw = self.linear(&vec_silu, "double_stream_modulation_txt.lin.weight")?;
        let single_mods_raw = self.linear(&vec_silu, "single_stream_modulation.lin.weight")?;

        let chunk_n = |t: &Tensor, n: usize| -> Result<Vec<Tensor>> {
            let last = *t.shape().dims().last().unwrap();
            let sz = last / n;
            let ndim = t.shape().dims().len();
            let mut chunks = Vec::with_capacity(n);
            for j in 0..n { chunks.push(t.narrow(ndim - 1, j * sz, sz)?); }
            Ok(chunks)
        };
        let img_mods_v = chunk_n(&img_mods_raw, 6)?;
        let txt_mods_v = chunk_n(&txt_mods_raw, 6)?;
        let single_mods_v = chunk_n(&single_mods_raw, 3)?;

        let to_arr6 = |mut v: Vec<Tensor>| -> [Tensor; 6] {
            let e5 = v.pop().unwrap(); let e4 = v.pop().unwrap();
            let e3 = v.pop().unwrap(); let e2 = v.pop().unwrap();
            let e1 = v.pop().unwrap(); let e0 = v.pop().unwrap();
            [e0, e1, e2, e3, e4, e5]
        };
        let to_arr3 = |mut v: Vec<Tensor>| -> [Tensor; 3] {
            let e2 = v.pop().unwrap(); let e1 = v.pop().unwrap(); let e0 = v.pop().unwrap();
            [e0, e1, e2]
        };
        let img_mods = to_arr6(img_mods_v);
        let txt_mods = to_arr6(txt_mods_v);
        let single_mods = to_arr3(single_mods_v);

        let use_checkpoint = std::env::var("KLEIN_GRAD_CHECKPOINT")
            .map(|v| v != "0").unwrap_or(true);

        // ---- Double blocks ----
        let mut img = img_proj;
        let mut txt = txt_proj;
        for i in 0..self.kconfig.num_double {
            let prefix = format!("double_blocks.{i}.");
            // BlockOffloader: stream block i from pinned host RAM into GPU slot.
            if let Some(ref off) = self.offloader {
                let arc = off.lock()
                    .map_err(|e| crate::EriDiffusionError::Model(format!("offloader lock: {e}")))?
                    .ensure_block(i)
                    .map_err(|e| crate::EriDiffusionError::Model(format!("offloader ensure_block({i}): {e}")))?;
                for (k, v) in arc.iter() {
                    self.weights.insert(k.clone(), v.clone());
                }
            }

            // Snapshot weights into a self-contained map for the closure.
            let mut layer_weights: HashMap<String, Tensor> = HashMap::new();
            for (k, v) in self.weights.iter() {
                if k.starts_with(&prefix) { layer_weights.insert(k.clone(), v.clone()); }
            }
            let lora_base = i * DOUBLE_LORA_SLOTS;
            let lora = if self.is_lora {
                Some(self.lora_adapters[lora_base..lora_base + DOUBLE_LORA_SLOTS].to_vec())
            } else { None };

            let img_in = img.clone();
            let txt_in = txt.clone();
            let img_mods_c = img_mods.clone();
            let txt_mods_c = txt_mods.clone();
            let pe_cos_c = pe_cos.clone();
            let pe_sin_c = pe_sin.clone();
            let nh = self.kconfig.num_heads;
            let hd = self.kconfig.head_dim;

            let (new_img, new_txt) = if use_checkpoint {
                // checkpoint takes one closure -> Result<Tensor>; for two
                // outputs we cat them and split after. To keep code simple
                // when checkpointing both streams, we run without checkpoint
                // here and instead checkpoint single blocks (which dominate
                // depth). Mirrors flame-diffusion's klein-trainer choice.
                double_block_forward_standalone(
                    img_in, txt_in, img_mods_c, txt_mods_c, pe_cos_c, pe_sin_c,
                    layer_weights, lora, i, nh, hd, inner,
                )?
            } else {
                double_block_forward_standalone(
                    img_in, txt_in, img_mods_c, txt_mods_c, pe_cos_c, pe_sin_c,
                    layer_weights, lora, i, nh, hd, inner,
                )?
            };
            img = new_img;
            txt = new_txt;
            // Evict double block i from weights and release GPU slot.
            if let Some(ref off) = self.offloader {
                self.weights.retain(|k, _| !k.starts_with(&prefix));
                off.lock()
                    .map_err(|e| crate::EriDiffusionError::Model(format!("offloader lock: {e}")))?
                    .evict_block();
            }
        }

        // ---- Single blocks (txt-then-img) ----
        let mut x = Tensor::cat(&[&txt, &img], 1)?;
        let txt_len = txt.shape().dims()[1];

        for i in 0..self.kconfig.num_single {
            let prefix = format!("single_blocks.{i}.");
            let unified_idx = self.kconfig.num_double + i;
            // BlockOffloader: stream single block (unified_idx) from pinned host RAM.
            if let Some(ref off) = self.offloader {
                let arc = off.lock()
                    .map_err(|e| crate::EriDiffusionError::Model(format!("offloader lock: {e}")))?
                    .ensure_block(unified_idx)
                    .map_err(|e| crate::EriDiffusionError::Model(format!("offloader ensure_block({unified_idx}): {e}")))?;
                for (k, v) in arc.iter() {
                    self.weights.insert(k.clone(), v.clone());
                }
            }
            let mut layer_weights: HashMap<String, Tensor> = HashMap::new();
            for (k, v) in self.weights.iter() {
                if k.starts_with(&prefix) { layer_weights.insert(k.clone(), v.clone()); }
            }
            let lora_base = self.kconfig.num_double * DOUBLE_LORA_SLOTS + i * SINGLE_LORA_SLOTS;
            let lora = if self.is_lora {
                Some(self.lora_adapters[lora_base..lora_base + SINGLE_LORA_SLOTS].to_vec())
            } else { None };

            let x_in = x.clone();
            let mods_c = single_mods.clone();
            let pe_cos_c = pe_cos.clone();
            let pe_sin_c = pe_sin.clone();
            let nh = self.kconfig.num_heads;
            let hd = self.kconfig.head_dim;

            x = if use_checkpoint {
                flame_core::autograd::AutogradContext::checkpoint(
                    &[x_in.clone()],
                    move || single_block_forward_standalone(
                        x_in.clone(), mods_c.clone(),
                        pe_cos_c.clone(), pe_sin_c.clone(),
                        layer_weights.clone(), lora.clone(),
                        i, nh, hd, inner, mlp,
                    ),
                )?
            } else {
                single_block_forward_standalone(
                    x_in, mods_c, pe_cos_c, pe_sin_c,
                    layer_weights, lora, i, nh, hd, inner, mlp,
                )?
            };
            // Evict single block from weights and release GPU slot.
            if let Some(ref off) = self.offloader {
                self.weights.retain(|k, _| !k.starts_with(&prefix));
                off.lock()
                    .map_err(|e| crate::EriDiffusionError::Model(format!("offloader lock: {e}")))?
                    .evict_block();
            }
        }

        // ---- Extract image tokens ----
        let total_len = x.shape().dims()[1];
        let img_only = x.narrow(1, txt_len, total_len - txt_len)?;

        // ---- Final layer: shift/scale + linear ----
        let final_mod = self.linear(&vec_silu, "final_layer.adaLN_modulation.1.weight")?;
        let last = *final_mod.shape().dims().last().unwrap();
        let half_mod = last / 2;
        let ndim = final_mod.shape().dims().len();
        let shift = final_mod.narrow(ndim - 1, 0, half_mod)?;
        let scale = final_mod.narrow(ndim - 1, half_mod, half_mod)?;

        let img_norm = modulate_pre_local(&img_only, &shift, &scale)?;
        let img_out = self.linear(&img_norm, "final_layer.linear.weight")?;
        // `img_out`: [B, N_img, in_channels] — unpack back to [B, C, H, W]
        let unpacked = img_out
            .reshape(&[b, h_lat, w_lat, in_ch])?
            .permute(&[0, 3, 1, 2])?
            .contiguous()?;
        Ok(unpacked)
    }
}

impl TrainableModel for KleinModel {
    fn forward(
        &mut self,
        noisy: &Tensor,
        timestep: &Tensor,
        context: &[Tensor],
        _pooled: Option<&Tensor>,
    ) -> Result<Tensor> {
        let txt = context.first().ok_or_else(||
            crate::EriDiffusionError::Model("Klein needs text embeddings".into()))?.clone();
        KleinModel::forward(self, noisy, &txt, timestep)
    }

    fn parameters(&self) -> Vec<Parameter> { self.parameters.clone() }
    fn post_optimizer_step(&mut self) {}

    fn save_weights(&self, path: &str) -> Result<()> {
        if !self.is_lora {
            return Err(crate::EriDiffusionError::Model(
                "save_weights for non-LoRA Klein not implemented".into()));
        }
        let mut out = HashMap::new();
        // Double blocks: 8 adapters each
        let mut k = 0;
        for i in 0..self.kconfig.num_double {
            for slot in 0..DOUBLE_LORA_SLOTS {
                let prefix = format!("double_blocks.{i}.{}", DOUBLE_LORA_KEYS[slot]);
                self.lora_adapters[k].save_tensors(&prefix, &mut out)?;
                k += 1;
            }
        }
        // Single blocks: 2 adapters each
        for i in 0..self.kconfig.num_single {
            for slot in 0..SINGLE_LORA_SLOTS {
                let prefix = format!("single_blocks.{i}.{}", SINGLE_LORA_KEYS[slot]);
                self.lora_adapters[k].save_tensors(&prefix, &mut out)?;
                k += 1;
            }
        }
        flame_core::serialization::save_file(&out, std::path::Path::new(path))
            .map_err(|e| crate::EriDiffusionError::Safetensors(format!("save_file: {e}")))?;
        log::info!("Klein LoRA saved to {} ({} tensors)", path, out.len());
        Ok(())
    }

    fn load_weights(&mut self, path: &str) -> Result<()> {
        if !self.is_lora {
            return Err(crate::EriDiffusionError::Model(
                "load_weights for non-LoRA Klein not implemented".into()));
        }
        let source = flame_core::serialization::load_file(
            std::path::Path::new(path), &self.device,
        ).map_err(|e| crate::EriDiffusionError::Safetensors(format!("load_file: {e}")))?;
        let mut k = 0;
        for i in 0..self.kconfig.num_double {
            for slot in 0..DOUBLE_LORA_SLOTS {
                let prefix = format!("double_blocks.{i}.{}", DOUBLE_LORA_KEYS[slot]);
                self.lora_adapters[k].load_tensors(&prefix, &source)?;
                k += 1;
            }
        }
        for i in 0..self.kconfig.num_single {
            for slot in 0..SINGLE_LORA_SLOTS {
                let prefix = format!("single_blocks.{i}.{}", SINGLE_LORA_KEYS[slot]);
                self.lora_adapters[k].load_tensors(&prefix, &source)?;
                k += 1;
            }
        }
        log::info!("Klein LoRA loaded from {} ({} tensors mapped)", path, k * 2);
        Ok(())
    }
}

impl KleinModel {
    /// Canonical (name, Parameter) pairs for full-checkpoint save/resume.
    /// Names mirror exactly what `<KleinModel as TrainableModel>::save_weights`
    /// writes (double_blocks.{i}.{suffix}.lora_{A,B}.weight then
    /// single_blocks.{i}.{suffix}.lora_{A,B}.weight). Iteration order is
    /// deterministic (block index ascending, slot ascending, A then B).
    pub fn named_parameters(&self) -> Vec<(String, Parameter)> {
        let mut out = Vec::with_capacity(self.lora_adapters.len() * 2);
        let mut k = 0;
        for i in 0..self.kconfig.num_double {
            for slot in 0..DOUBLE_LORA_SLOTS {
                let prefix = format!("double_blocks.{i}.{}", DOUBLE_LORA_KEYS[slot]);
                out.push((format!("{prefix}.lora_A.weight"), self.lora_adapters[k].lora_a().clone()));
                out.push((format!("{prefix}.lora_B.weight"), self.lora_adapters[k].lora_b().clone()));
                k += 1;
            }
        }
        for i in 0..self.kconfig.num_single {
            for slot in 0..SINGLE_LORA_SLOTS {
                let prefix = format!("single_blocks.{i}.{}", SINGLE_LORA_KEYS[slot]);
                out.push((format!("{prefix}.lora_A.weight"), self.lora_adapters[k].lora_a().clone()));
                out.push((format!("{prefix}.lora_B.weight"), self.lora_adapters[k].lora_b().clone()));
                k += 1;
            }
        }
        out
    }
}
