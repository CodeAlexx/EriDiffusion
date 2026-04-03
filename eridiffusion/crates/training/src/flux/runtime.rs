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

use std::{
    collections::HashMap,
    sync::{Arc, Mutex, Once, OnceLock},
    time::Instant,
};

use anyhow::{anyhow, ensure, Result};
use flame_core::{
    cuda_ops_bf16,
    ops::{
        attn::{streaming_attn_bf16_fp32, streaming_attn_bf16_fp32_smoke_test, StreamingAttnCfg},
        reduce::sum_dim_keepdim_as,
    },
    CudaDevice, DType, Error as FlameError, Shape, Tensor,
};

use crate::{
    flame_ctx,
    flux::lora::FluxBlockLora,
    tensor_utils::{add_inplace_same_dtype, broadcast_add, broadcast_mul, broadcast_to_as},
};

struct MaskPack {
    stream_mask: Tensor, // [B,H,S,S] same dtype as tokens
    query_mask: Tensor,  // [B,H,S,1] same dtype as tokens
    seq_mask: Tensor,    // [B,S] same dtype as tokens
}

#[derive(Clone, Debug)]
pub struct BlockMetadata {
    pub hidden: usize,
    pub mlp_inner: usize,
    pub image_cond: usize,
    pub text_cond: usize,
}

pub trait ExecutableBlock: Send + Sync {
    fn forward_image(
        &self,
        tokens: &Tensor,
        cond: &Tensor,
        lora: Option<&FluxBlockLora>,
    ) -> Result<Tensor>;
    fn forward_text(&self, tokens: &Tensor, cond: &Tensor) -> Result<Tensor>;
    fn hidden_dim(&self) -> usize;
    fn mlp_inner_dim(&self) -> usize;
    fn image_cond_dim(&self) -> usize;
    fn text_cond_dim(&self) -> usize;
}

pub struct LayerRegistry {
    pub blocks: Vec<Box<dyn ExecutableBlock>>,
}

impl LayerRegistry {
    pub fn new(blocks: Vec<Box<dyn ExecutableBlock>>) -> Self {
        Self { blocks }
    }

    pub fn forward(
        &self,
        img_tokens: &Tensor,
        img_cond: &Tensor,
        img_lora: Option<&[FluxBlockLora]>,
        txt_tokens: &Tensor,
        txt_cond: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let mut img = img_tokens.clone_result()?;
        let mut txt = txt_tokens.clone_result()?;
        for (idx, block) in self.blocks.iter().enumerate() {
            let lora_block = img_lora.and_then(|l| l.get(idx));
            img = block.forward_image(&img, img_cond, lora_block)?;
            txt = block.forward_text(&txt, txt_cond)?;
        }
        Ok((img, txt))
    }

    pub fn metadata(&self) -> Vec<BlockMetadata> {
        self.blocks
            .iter()
            .map(|b| BlockMetadata {
                hidden: b.hidden_dim(),
                mlp_inner: b.mlp_inner_dim(),
                image_cond: b.image_cond_dim(),
                text_cond: b.text_cond_dim(),
            })
            .collect()
    }
}

#[derive(Clone)]
struct ModStage {
    shift: Tensor,
    scale: Tensor,
    gate: Option<Tensor>,
}

#[derive(Clone)]
struct ModPair {
    attn: ModStage,
    mlp: ModStage,
}

enum Branch {
    Image,
    Text,
}

impl Branch {
    fn prefix(&self) -> &'static str {
        match self {
            Branch::Image => "image",
            Branch::Text => "text",
        }
    }

    fn cond_dim<'a>(&self, runtime: &'a FluxBlockRuntime) -> usize {
        match self {
            Branch::Image => runtime.image_cond_dim,
            Branch::Text => runtime.text_cond_dim,
        }
    }
}

const STREAM_ATTENTION_KERNEL_MAX_CHUNK: usize = 2048;

fn streaming_causal_text_enabled() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| {
        matches!(
            std::env::var("STREAM_ATTENTION_CAUSAL_TEXT")
                .ok()
                .map(|v| v.to_ascii_lowercase())
                .as_deref(),
            Some("1") | Some("true") | Some("on")
        )
    })
}

fn build_mask_pack(tokens: &Tensor, heads: usize) -> Result<MaskPack> {
    let dims = tokens.shape().dims().to_vec();
    ensure!(dims.len() == 3, "expected [B,Seq,Hidden], got {:?}", dims);
    let b = dims[0];
    let seq = dims[1];

    let target_dtype = tokens.dtype();
    let abs = tokens.abs()?;
    let sum = if target_dtype == DType::BF16 {
        sum_dim_keepdim_as(&abs, 2, DType::BF16)?
    } else {
        abs.sum_dim_keepdim(2)?
    };
    let sum = sum.reshape(&[b, seq])?;

    let eps = Tensor::zeros_dtype(Shape::from_dims(&[1, 1]), target_dtype, sum.device().clone())?
        .add_scalar(1e-6f32)?;
    let mask_seq = sum.gt(&eps)?;

    let mask_q = mask_seq.reshape(&[b, 1, seq, 1])?;
    let mask_k = mask_seq.reshape(&[b, 1, 1, seq])?;
    let mask_full = mask_q.mul(&mask_k)?;
    let mask_stream = mask_full.broadcast_to(&Shape::from_dims(&[b, heads, seq, seq]))?;
    let mask_query = mask_q.broadcast_to(&Shape::from_dims(&[b, heads, seq, 1]))?;

    Ok(MaskPack { stream_mask: mask_stream, query_mask: mask_query, seq_mask: mask_seq })
}

pub struct FluxBlockRuntime {
    params: HashMap<String, Tensor>,
    hidden: usize,
    heads: usize,
    head_dim: usize,
    image_cond_dim: usize,
    text_cond_dim: usize,
    mlp_inner: usize,
}

impl FluxBlockRuntime {
    pub fn from_params(mut params: HashMap<String, Tensor>) -> Result<Self> {
        let qkv_shape = params
            .get("image.qkv.weight")
            .ok_or_else(|| anyhow!("missing image.qkv.weight"))?
            .shape()
            .dims()
            .to_vec();
        ensure!(qkv_shape.len() == 2, "image.qkv.weight rank {:?}", qkv_shape);
        ensure!(qkv_shape[0] % 3 == 0, "image.qkv.weight rows {} not divisible by 3", qkv_shape[0]);
        let hidden = qkv_shape[1];
        ensure!(
            qkv_shape[0] == hidden * 3,
            "image.qkv.weight expected {} rows, got {}",
            hidden * 3,
            qkv_shape[0]
        );

        let norm_shape = params
            .get("image.k_norm.scale")
            .ok_or_else(|| anyhow!("missing image.k_norm.scale"))?
            .shape()
            .dims()
            .to_vec();
        ensure!(norm_shape.len() == 1, "image.k_norm.scale rank {:?}", norm_shape);
        let head_dim = norm_shape[0];
        ensure!(hidden % head_dim == 0, "hidden {} not divisible by head_dim {}", hidden, head_dim);
        let heads = hidden / head_dim;

        let image_mod_shape = params
            .get("image.mod.lin.weight")
            .ok_or_else(|| anyhow!("missing image.mod.lin.weight"))?
            .shape()
            .dims()
            .to_vec();
        ensure!(image_mod_shape.len() == 2, "image.mod.lin.weight rank {:?}", image_mod_shape);
        ensure!(
            image_mod_shape[0] % hidden == 0,
            "image.mod.lin.weight rows {} not multiple of hidden {}",
            image_mod_shape[0],
            hidden
        );
        let image_cond_dim = image_mod_shape[1];

        let text_mod_shape = params
            .get("text.mod.lin.weight")
            .ok_or_else(|| anyhow!("missing text.mod.lin.weight"))?
            .shape()
            .dims()
            .to_vec();
        ensure!(text_mod_shape.len() == 2, "text.mod.lin.weight rank {:?}", text_mod_shape);
        ensure!(
            text_mod_shape[0] % hidden == 0,
            "text.mod.lin.weight rows {} not multiple of hidden {}",
            text_mod_shape[0],
            hidden
        );
        let text_cond_dim = text_mod_shape[1];

        let fc1_shape = params
            .get("image.fc1.weight")
            .ok_or_else(|| anyhow!("missing image.fc1.weight"))?
            .shape()
            .dims()
            .to_vec();
        ensure!(fc1_shape.len() == 2, "image.fc1.weight rank {:?}", fc1_shape);
        ensure!(
            fc1_shape[1] == hidden,
            "image.fc1.weight expected in_dim {} got {}",
            hidden,
            fc1_shape[1]
        );
        let mlp_inner = fc1_shape[0];

        for tensor in params.values_mut() {
            if tensor.requires_grad() {
                *tensor = tensor.clone_result()?.requires_grad_(false).detach()?;
            }
        }

        Ok(Self { params, hidden, heads, head_dim, image_cond_dim, text_cond_dim, mlp_inner })
    }

    pub fn hidden(&self) -> usize {
        self.hidden
    }
    pub fn heads(&self) -> usize {
        self.heads
    }

    pub fn mlp_inner(&self) -> usize {
        self.mlp_inner
    }

    pub fn forward_image(
        &self,
        tokens: &Tensor,
        cond: &Tensor,
        lora: Option<&FluxBlockLora>,
    ) -> Result<Tensor> {
        self.forward_branch(tokens, cond, Branch::Image, lora)
    }

    pub fn forward_text(&self, tokens: &Tensor, cond: &Tensor) -> Result<Tensor> {
        self.forward_branch(tokens, cond, Branch::Text, None)
    }

    fn forward_branch(
        &self,
        tokens: &Tensor,
        cond: &Tensor,
        branch: Branch,
        lora: Option<&FluxBlockLora>,
    ) -> Result<Tensor> {
        self.verify_token_shape(tokens)?;
        self.verify_cond_shape(cond, branch.cond_dim(self))?;

        let target_dtype = tokens.dtype();
        let mods = self.modulation(cond, branch.prefix(), target_dtype)?;

        let norm1 = rms_norm(tokens)?;
        let mod_attn_in = apply_scale_shift(&norm1, &mods.attn, self.hidden)?;
        let attn = self.attention(&mod_attn_in, branch.prefix(), &branch, lora, target_dtype)?;
        let attn = apply_gate(&attn, &mods.attn, self.hidden)?;
        let mut resid1 = attn;
        add_inplace_same_dtype(&mut resid1, tokens)?;

        let norm2 = rms_norm(&resid1)?;
        let mod_mlp_in = apply_scale_shift(&norm2, &mods.mlp, self.hidden)?;
        let mlp = self.mlp(&mod_mlp_in, branch.prefix(), &branch, lora, target_dtype)?;
        let mlp = apply_gate(&mlp, &mods.mlp, self.hidden)?;
        let mut out = mlp;
        add_inplace_same_dtype(&mut out, &resid1)?;

        let out = if out.dtype() == target_dtype { out } else { out.to_dtype(target_dtype)? };
        Ok(out)
    }

    fn modulation(&self, cond: &Tensor, prefix: &str, target_dtype: DType) -> Result<ModPair> {
        let weight = self.param(&format!("{}.mod.lin.weight", prefix))?;
        let bias = self.params.get(&format!("{}.mod.lin.bias", prefix));

        let cond_act = cond.silu()?;
        let mut proj = cond_act.matmul(&weight.transpose_dims(0, 1)?)?;
        if let Some(bias) = bias {
            let proj_shape = proj.shape().dims().to_vec();
            let bias_view = broadcast_to_as(bias, &proj_shape, proj.dtype())?;
            add_inplace_same_dtype(&mut proj, &bias_view)?;
        }

        let dims = proj.shape().dims().to_vec();
        ensure!(dims.len() == 2, "modulation output shape {:?}", dims);
        let rows = dims[0];
        let cols = dims[1];
        ensure!(
            cols % self.hidden == 0,
            "modulation width {} not multiple of hidden {}",
            cols,
            self.hidden
        );
        let splits = cols / self.hidden;

        let mut parts = Vec::with_capacity(splits);
        for i in 0..splits {
            let part = proj
                .narrow(1, i * self.hidden, self.hidden)?
                .reshape(&[rows, self.hidden])?
                .to_dtype(target_dtype)?;
            parts.push(part);
        }

        let pair = match splits {
            2 => {
                let scale = parts[0].clone_result()?;
                let shift = parts[1].clone_result()?;
                let stage = ModStage { shift, scale, gate: None };
                ModPair { attn: stage.clone(), mlp: stage }
            }
            6 => ModPair {
                attn: ModStage {
                    shift: parts[0].clone_result()?,
                    scale: parts[1].clone_result()?,
                    gate: Some(parts[2].clone_result()?),
                },
                mlp: ModStage {
                    shift: parts[3].clone_result()?,
                    scale: parts[4].clone_result()?,
                    gate: Some(parts[5].clone_result()?),
                },
            },
            other => {
                return Err(anyhow!(
                    "unsupported modulation slice count {} (hidden={})",
                    other,
                    self.hidden
                ));
            }
        };

        Ok(pair)
    }

    fn attention(
        &self,
        tokens: &Tensor,
        prefix: &str,
        branch: &Branch,
        lora: Option<&FluxBlockLora>,
        target_dtype: DType,
    ) -> Result<Tensor> {
        let qkv_w = self.param(&format!("{}.qkv.weight", prefix))?;
        let qkv_b = self.params.get(&format!("{}.qkv.bias", prefix));
        let proj_w = self.param(&format!("{}.o.weight", prefix))?;
        let proj_b = self.params.get(&format!("{}.o.bias", prefix));
        let q_norm = self.params.get(&format!("{}.q_norm.scale", prefix));
        let k_norm = self.params.get(&format!("{}.k_norm.scale", prefix));

        let dims = tokens.shape().dims().to_vec();
        let b = dims[0];
        let seq = dims[1];

        let dense_env = std::env::var("FLUX_FORCE_DENSE_ATTENTION")
            .map(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "on"))
            .unwrap_or(false);

        let chunk_env_value =
            std::env::var("ATTN_CHUNK_SIZE").ok().and_then(|v| v.parse::<usize>().ok());
        let requested_stream_chunk = chunk_env_value.unwrap_or(2048usize);
        let chunk_upper = STREAM_ATTENTION_KERNEL_MAX_CHUNK.min(seq.max(1));
        let chunk_lower = if chunk_upper < 64 { chunk_upper } else { 64 };
        let stream_chunk = requested_stream_chunk.max(chunk_lower).min(chunk_upper).max(1usize);
        let proj_chunk = if dense_env { chunk_env_value.unwrap_or(512usize) } else { stream_chunk }
            .max(1usize)
            .min(seq.max(1));
        let chunk_divisor = proj_chunk.max(1);
        let debug_writes = std::env::var("STREAM_ATTENTION_DEBUG_WRITES")
            .map(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "on"))
            .unwrap_or(false);

        let weight_t = qkv_w.transpose_dims(0, 1)?;
        let bias = qkv_b;

        let device = tokens.device().clone();
        let bh = b * self.heads;
        let shape_bhsd = Shape::from_dims(&[b, self.heads, seq, self.head_dim]);
        let alloc_elems = b
            .checked_mul(self.heads)
            .and_then(|v| v.checked_mul(seq))
            .and_then(|v| v.checked_mul(self.head_dim))
            .unwrap_or(0);
        let bf16_bytes = alloc_elems * std::mem::size_of::<u16>();
        log_large_alloc("flux.attn.q_full", bf16_bytes);
        let q_full = Tensor::zeros_dtype(shape_bhsd.clone(), DType::BF16, device.clone())?;
        log_large_alloc("flux.attn.k_full", bf16_bytes);
        let k_full = Tensor::zeros_dtype(shape_bhsd.clone(), DType::BF16, device.clone())?;
        log_large_alloc("flux.attn.v_full", bf16_bytes);
        let v_full = Tensor::zeros_dtype(shape_bhsd.clone(), DType::BF16, device.clone())?;
        let q_base_ptr = if debug_writes {
            Some(q_full.as_device_ptr_bf16("streaming_q_full_ptr")? as usize)
        } else {
            None
        };
        let mut q_full_flat = q_full.reshape(&[bh, seq, self.head_dim])?;
        let mut k_full_flat = k_full.reshape(&[bh, seq, self.head_dim])?;
        let mut v_full_flat = v_full.reshape(&[bh, seq, self.head_dim])?;
        let scratch = streaming_workspace_buffer(&device, bh, proj_chunk, self.head_dim)?;
        let mut scratch_flat = scratch.reshape(&[bh, proj_chunk, self.head_dim])?;

        for start in (0..seq).step_by(proj_chunk) {
            let rows = (seq - start).min(proj_chunk);
            if rows == 0 {
                break;
            }

            let tokens_chunk = tokens.narrow(1, start, rows)?;
            let tokens_chunk_flat = tokens_chunk.reshape(&[b * rows, self.hidden])?;
            let mut qkv_chunk = tokens_chunk_flat.matmul(&weight_t)?;
            if let Some(bias) = &bias {
                let qkv_shape = vec![b * rows, self.hidden * 3];
                let bias_view = broadcast_to_as(bias, &qkv_shape, qkv_chunk.dtype())?;
                add_inplace_same_dtype(&mut qkv_chunk, &bias_view)?;
            }
            let qkv_chunk = qkv_chunk.reshape(&[b, rows, self.hidden * 3])?;
            let (mut q_chunk, mut k_chunk, mut v_chunk) =
                split_qkv(&qkv_chunk, self.heads, self.head_dim)?;

            if q_chunk.dtype() != target_dtype {
                q_chunk = q_chunk.to_dtype(target_dtype)?;
                k_chunk = k_chunk.to_dtype(target_dtype)?;
                v_chunk = v_chunk.to_dtype(target_dtype)?;
            }

            if matches!(branch, Branch::Image) {
                if let Some(l) = lora {
                    if let Some(lo) = &l.q {
                        let delta = lo.forward_delta(&tokens_chunk)?.reshape(&[
                            b,
                            rows,
                            self.heads,
                            self.head_dim,
                        ])?;
                        let delta = if delta.dtype() == target_dtype {
                            delta
                        } else {
                            delta.to_dtype(target_dtype)?
                        };
                        add_inplace_same_dtype(&mut q_chunk, &delta)?;
                    }
                    if let Some(lo) = &l.k {
                        let delta = lo.forward_delta(&tokens_chunk)?.reshape(&[
                            b,
                            rows,
                            self.heads,
                            self.head_dim,
                        ])?;
                        let delta = if delta.dtype() == target_dtype {
                            delta
                        } else {
                            delta.to_dtype(target_dtype)?
                        };
                        add_inplace_same_dtype(&mut k_chunk, &delta)?;
                    }
                    if let Some(lo) = &l.v {
                        let delta = lo.forward_delta(&tokens_chunk)?.reshape(&[
                            b,
                            rows,
                            self.heads,
                            self.head_dim,
                        ])?;
                        let delta = if delta.dtype() == target_dtype {
                            delta
                        } else {
                            delta.to_dtype(target_dtype)?
                        };
                        add_inplace_same_dtype(&mut v_chunk, &delta)?;
                    }
                }
            }

            q_chunk = apply_qk_norm(&q_chunk, q_norm, self.head_dim)?;
            k_chunk = apply_qk_norm(&k_chunk, k_norm, self.head_dim)?;

            let q_chunk_perm = q_chunk.permute(&[0, 2, 1, 3])?;
            let q_chunk_flat = q_chunk_perm.reshape(&[bh, rows, self.head_dim])?;
            let q_chunk_bf16 = if q_chunk_flat.dtype() == DType::BF16 {
                q_chunk_flat
            } else {
                q_chunk_flat.to_dtype(DType::BF16)?
            };
            write_chunk_bf16(&mut scratch_flat, &q_chunk_bf16, proj_chunk, 0)?;
            let scratch_q = scratch_flat.narrow(1, 0, rows)?;
            write_chunk_bf16(&mut q_full_flat, &scratch_q, seq, start)?;
            if debug_writes {
                let row_bytes = self.head_dim * std::mem::size_of::<u16>();
                let chunk_idx = start / chunk_divisor;
                let bytes_written = rows * row_bytes * bh;
                if let Some(base) = q_base_ptr {
                    let ptr = base + start * row_bytes;
                    println!(
                        "[streaming_attn_chunk] chunk={} start={} rows={} bh={} bytes={} ptr=0x{:x}",
                        chunk_idx, start, rows, bh, bytes_written, ptr
                    );
                } else {
                    println!(
                        "[streaming_attn_chunk] chunk={} start={} rows={} bh={} bytes={}",
                        chunk_idx, start, rows, bh, bytes_written
                    );
                }
            }

            let k_chunk_perm = k_chunk.permute(&[0, 2, 1, 3])?;
            let k_chunk_flat = k_chunk_perm.reshape(&[bh, rows, self.head_dim])?;
            let k_chunk_bf16 = if k_chunk_flat.dtype() == DType::BF16 {
                k_chunk_flat
            } else {
                k_chunk_flat.to_dtype(DType::BF16)?
            };
            write_chunk_bf16(&mut scratch_flat, &k_chunk_bf16, proj_chunk, 0)?;
            let scratch_k = scratch_flat.narrow(1, 0, rows)?;
            write_chunk_bf16(&mut k_full_flat, &scratch_k, seq, start)?;

            let v_chunk_perm = v_chunk.permute(&[0, 2, 1, 3])?;
            let v_chunk_flat = v_chunk_perm.reshape(&[bh, rows, self.head_dim])?;
            let v_chunk_bf16 = if v_chunk_flat.dtype() == DType::BF16 {
                v_chunk_flat
            } else {
                v_chunk_flat.to_dtype(DType::BF16)?
            };
            write_chunk_bf16(&mut scratch_flat, &v_chunk_bf16, proj_chunk, 0)?;
            let scratch_v = scratch_flat.narrow(1, 0, rows)?;
            write_chunk_bf16(&mut v_full_flat, &scratch_v, seq, start)?;
        }

        drop(q_full_flat);
        drop(k_full_flat);
        drop(v_full_flat);
        drop(scratch_flat);

        let mask_pack = if matches!(branch, Branch::Text) {
            Some(build_mask_pack(tokens, self.heads)?)
        } else {
            None
        };
        let streaming_causal = matches!(branch, Branch::Text) && streaming_causal_text_enabled();
        let dense_attention = dense_env || mask_pack.is_some();
        if !dense_env && mask_pack.is_some() {
            println!("[telemetry] dense_attention forced (streaming mask unsupported)");
        }

        let scale = 1.0f32 / (self.head_dim as f32).sqrt();
        let q_bhsd = &q_full;
        let k_bhsd = &k_full;
        let v_bhsd = &v_full;

        let mut attn_bf16 = if dense_attention {
            DENSE_ATTENTION_TELEMETRY.call_once(|| {
                println!("[telemetry] dense_attention enabled (streaming disabled)");
            });
            chunked_attention_bf16(
                q_bhsd,
                k_bhsd,
                v_bhsd,
                scale,
                proj_chunk,
                mask_pack.as_ref().map(|m| &m.stream_mask),
                streaming_causal,
            )?
        } else {
            println!(
                "[telemetry] streaming_attention launch chunk={} (requested {})",
                stream_chunk, requested_stream_chunk
            );

            STREAM_ATTENTION_TELEMETRY.call_once(|| {
                println!(
                    "[telemetry] streaming_attention enabled chunk={} heads={} head_dim={} hidden={} seq={}",
                    stream_chunk,
                    self.heads,
                    self.head_dim,
                    self.hidden,
                    seq
                );
            });

            let attn_cfg = StreamingAttnCfg {
                scale,
                chunk_size: stream_chunk,
                causal: streaming_causal,
                mask: mask_pack.as_ref().map(|m| &m.stream_mask),
            };

            let selftest_disabled = std::env::var("STREAM_ATTENTION_SELFTEST")
                .map(|v| matches!(v.to_ascii_lowercase().as_str(), "0" | "off" | "false"))
                .unwrap_or(false);
            let device_clone = q_bhsd.device().clone();
            let selftest_result = STREAM_ATTENTION_SELFTEST.get_or_init(move || {
                if selftest_disabled {
                    Ok(())
                } else {
                    match streaming_attn_bf16_fp32_smoke_test(device_clone.clone()) {
                        Ok(_) => {
                            println!(
                                "[telemetry] streaming_attention smoke test ok (B=1,H=1,S=128,D=64)"
                            );
                            Ok(())
                        }
                        Err(err) => Err(anyhow!(err)),
                    }
                }
            });
            if let Err(err) = selftest_result {
                return Err(anyhow!("streaming attention smoke test failed before launch: {err}"));
            }

            let attn_start = Instant::now();
            let out = streaming_attn_bf16_fp32(q_bhsd, k_bhsd, v_bhsd, attn_cfg)?;
            if std::env::var("ATTN_LOG_MS").ok().as_deref() == Some("1") {
                println!(
                    "[telemetry] streaming_attention_ms={:.2}",
                    attn_start.elapsed().as_secs_f32() * 1000.0
                );
            }
            out
        };

        if let Some(mask) = &mask_pack {
            attn_bf16 = attn_bf16.mul(&mask.query_mask)?;
        }
        let ctx = attn_bf16
            .permute(&[0, 2, 1, 3])?
            .reshape(&[b * seq, self.hidden])?;

        let mut out = ctx.matmul(&proj_w.transpose_dims(0, 1)?)?;
        if let Some(bias) = proj_b {
            let out_shape = out.shape().dims().to_vec();
            let bias_view = broadcast_to_as(bias, &out_shape, out.dtype())?;
            add_inplace_same_dtype(&mut out, &bias_view)?;
        }
        if matches!(branch, Branch::Image) {
            if let Some(l) = lora.and_then(|l| l.o.as_ref()) {
                let delta = l
                    .forward_delta(&ctx.reshape(&[b, seq, self.hidden])?)?
                    .reshape(&[b * seq, self.hidden])?;
                let delta = if delta.dtype() == out.dtype() {
                    delta
                } else {
                    delta.to_dtype(out.dtype())?
                };
                add_inplace_same_dtype(&mut out, &delta)?;
            }
        }
        Ok(out.reshape(&[b, seq, self.hidden])?.to_dtype(target_dtype)?)
    }

    fn mlp(
        &self,
        tokens: &Tensor,
        prefix: &str,
        branch: &Branch,
        lora: Option<&FluxBlockLora>,
        target_dtype: DType,
    ) -> Result<Tensor> {
        let fc1_w = self.param(&format!("{}.fc1.weight", prefix))?;
        let fc1_b = self.params.get(&format!("{}.fc1.bias", prefix));
        let fc2_w = self.param(&format!("{}.fc2.weight", prefix))?;
        let fc2_b = self.params.get(&format!("{}.fc2.bias", prefix));

        let dims = tokens.shape().dims().to_vec();
        let b = dims[0];
        let seq = dims[1];

        let tokens_flat = tokens.reshape(&[b * seq, self.hidden])?;
        let mut fc1 = tokens_flat.matmul(&fc1_w.transpose_dims(0, 1)?)?;
        if let Some(bias) = fc1_b {
            let fc1_shape = fc1.shape().dims().to_vec();
            let bias_view = broadcast_to_as(bias, &fc1_shape, fc1.dtype())?;
            add_inplace_same_dtype(&mut fc1, &bias_view)?;
        }
        if matches!(branch, Branch::Image) {
            if let Some(l) = lora.and_then(|l| l.fc1.as_ref()) {
                let delta = l.forward_delta(tokens)?.reshape(&[b * seq, fc1.shape().dims()[1]])?;
                let delta = if delta.dtype() == fc1.dtype() {
                    delta
                } else {
                    delta.to_dtype(fc1.dtype())?
                };
                add_inplace_same_dtype(&mut fc1, &delta)?;
            }
        }
        let fc1 = fc1.silu()?;
        let mut out = fc1.matmul(&fc2_w.transpose_dims(0, 1)?)?;
        if let Some(bias) = fc2_b {
            let out_shape = out.shape().dims().to_vec();
            let bias_view = broadcast_to_as(bias, &out_shape, out.dtype())?;
            add_inplace_same_dtype(&mut out, &bias_view)?;
        }
        if matches!(branch, Branch::Image) {
            if let Some(l) = lora.and_then(|l| l.fc2.as_ref()) {
                let delta = l
                    .forward_delta(&fc1.reshape(&[b, seq, fc1.shape().dims()[1]])?)?
                    .reshape(&[b * seq, self.hidden])?;
                let delta = if delta.dtype() == out.dtype() {
                    delta
                } else {
                    delta.to_dtype(out.dtype())?
                };
                add_inplace_same_dtype(&mut out, &delta)?;
            }
        }
        Ok(out.reshape(&[b, seq, self.hidden])?.to_dtype(target_dtype)?)
    }

    fn param(&self, key: &str) -> Result<&Tensor> {
        self.params.get(key).ok_or_else(|| anyhow!("missing tensor {}", key))
    }

    fn verify_token_shape(&self, tokens: &Tensor) -> Result<()> {
        let dims = tokens.shape().dims().to_vec();
        ensure!(
            dims.len() == 3 && dims[2] == self.hidden,
            "expected [B,T,{}], got {:?}",
            self.hidden,
            dims
        );
        Ok(())
    }

    fn verify_cond_shape(&self, cond: &Tensor, expected: usize) -> Result<()> {
        let dims = cond.shape().dims().to_vec();
        ensure!(
            dims.len() == 2 && dims[1] == expected,
            "expected cond [B,{}], got {:?}",
            expected,
            dims
        );
        Ok(())
    }
}

impl ExecutableBlock for FluxBlockRuntime {
    fn forward_image(
        &self,
        tokens: &Tensor,
        cond: &Tensor,
        lora: Option<&FluxBlockLora>,
    ) -> Result<Tensor> {
        self.forward_branch(tokens, cond, Branch::Image, lora)
    }

    fn forward_text(&self, tokens: &Tensor, cond: &Tensor) -> Result<Tensor> {
        self.forward_branch(tokens, cond, Branch::Text, None)
    }

    fn hidden_dim(&self) -> usize {
        self.hidden
    }

    fn mlp_inner_dim(&self) -> usize {
        self.mlp_inner
    }

    fn image_cond_dim(&self) -> usize {
        self.image_cond_dim
    }

    fn text_cond_dim(&self) -> usize {
        self.text_cond_dim
    }
}

fn rms_norm(x: &Tensor) -> Result<Tensor> {
    let target_dtype = x.dtype();
    let hidden = x.shape().dims()[2] as f32;

    let sq = x.mul(x)?;
    static RMS_NORM_DEBUG_ONCE: Once = Once::new();
    RMS_NORM_DEBUG_ONCE.call_once(|| {
        eprintln!(
            "[streaming_attn_debug] rms_norm dtypes: input={:?} mul={:?}",
            target_dtype,
            sq.dtype()
        );
    });
    let sum = if target_dtype == DType::BF16 {
        sum_dim_keepdim_as(&sq, 2, DType::BF16)?
    } else {
        sq.sum_dim_keepdim(2)?
    };
    let mean = sum.div_scalar(hidden)?;
    let inv = mean.add_scalar(1e-6)?.rsqrt()?;
    Ok(x.mul(&inv)?)
}

fn apply_scale_shift(x: &Tensor, stage: &ModStage, hidden: usize) -> Result<Tensor> {
    let b = x.shape().dims()[0];
    let scale = stage.scale.add_scalar(1.0)?.reshape(&[b, 1, hidden])?;
    let shift = stage.shift.reshape(&[b, 1, hidden])?;
    let scaled = flame_ctx!(broadcast_mul(x, &scale), "flux::runtime::apply_scale_shift.mul")?;
    flame_ctx!(broadcast_add(&scaled, &shift), "flux::runtime::apply_scale_shift.add")
}

fn apply_gate(x: &Tensor, stage: &ModStage, hidden: usize) -> Result<Tensor> {
    if let Some(gate) = &stage.gate {
        let b = x.shape().dims()[0];
        let gate = gate.reshape(&[b, 1, hidden])?;
        flame_ctx!(broadcast_mul(x, &gate), "flux::runtime::apply_gate")
    } else {
        Ok(x.clone_result()?)
    }
}

fn chunked_attention_bf16(
    q_bhsd: &Tensor,
    k_bhsd: &Tensor,
    v_bhsd: &Tensor,
    scale: f32,
    chunk_size: usize,
    mask: Option<&Tensor>,
    causal: bool,
) -> Result<Tensor> {
    match cuda_ops_bf16::sdpa_stream_bf16(
        q_bhsd,
        k_bhsd,
        v_bhsd,
        mask,
        chunk_size.max(1),
        causal,
        Some(scale),
    ) {
        Ok(out) => Ok(out),
        Err(FlameError::Unsupported(reason)) => {
            if causal {
                Err(anyhow!("dense attention unsupported with causal mask: {reason}"))
            } else {
                flame_core::sdpa::forward(q_bhsd, k_bhsd, v_bhsd, mask).map_err(Into::into)
            }
        }
        Err(err) => Err(anyhow!(err)),
    }
}

fn split_qkv(qkv: &Tensor, heads: usize, head_dim: usize) -> Result<(Tensor, Tensor, Tensor)> {
    let dims = qkv.shape().dims().to_vec();
    ensure!(dims.len() == 3 && dims[2] == heads * head_dim * 3, "qkv shape {:?}", dims);
    let q = qkv.narrow(2, 0, heads * head_dim)?.reshape(&[dims[0], dims[1], heads, head_dim])?;
    let k = qkv
        .narrow(2, heads * head_dim, heads * head_dim)?
        .reshape(&[dims[0], dims[1], heads, head_dim])?;
    let v = qkv
        .narrow(2, heads * head_dim * 2, heads * head_dim)?
        .reshape(&[dims[0], dims[1], heads, head_dim])?;
    Ok((q, k, v))
}

fn write_chunk_bf16(
    dst_flat: &mut Tensor,
    src_flat: &Tensor,
    seq: usize,
    offset: usize,
) -> Result<()> {
    let dims = src_flat.shape().dims().to_vec();
    ensure!(dims.len() == 3, "write_chunk_bf16 expects [BH, rows, head_dim], got {:?}", dims);
    let bh = dims[0];
    let rows = dims[1];
    let head_dim = dims[2];
    ensure!(
        offset + rows <= seq,
        "write_chunk_bf16 out of bounds offset={} rows={} seq={}",
        offset,
        rows,
        seq
    );
    let row_stride_src = rows * head_dim;
    let row_stride_dst = seq * head_dim;
    for row in 0..bh {
        let src_start = row * row_stride_src;
        let dst_start = row * row_stride_dst + offset * head_dim;
        dst_flat.copy_bf16_region_from(dst_start, src_flat, src_start, row_stride_src)?;
    }
    Ok(())
}

fn log_large_alloc(tag: &str, bytes: usize) {
    if bytes < (1 << 20) {
        return;
    }
    let enabled = std::env::var("STREAM_ATTENTION_LOG_ALLOCS")
        .map(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "on"))
        .unwrap_or(false);
    if enabled {
        let mib = bytes as f64 / (1024.0 * 1024.0);
        println!("[streaming_attn_alloc] tag={} bytes={} ({:.2} MiB)", tag, bytes, mib);
    }
}

#[derive(Hash, Eq, PartialEq, Clone, Debug)]
struct WorkspaceKey {
    device: usize,
    bh: usize,
    chunk: usize,
    head_dim: usize,
}

static STREAM_WORKSPACE: OnceLock<Mutex<HashMap<WorkspaceKey, Tensor>>> = OnceLock::new();

fn streaming_workspace_buffer(
    device: &Arc<CudaDevice>,
    bh: usize,
    chunk: usize,
    head_dim: usize,
) -> Result<Tensor> {
    let key = WorkspaceKey { device: device.ordinal(), bh, chunk, head_dim };
    let lock = STREAM_WORKSPACE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut map = lock.lock().expect("streaming workspace mutex poisoned");
    if let Some(existing) = map.get(&key) {
        return existing.clone_result().map_err(anyhow::Error::from);
    }
    let alloc_elems = bh.checked_mul(chunk).and_then(|v| v.checked_mul(head_dim)).unwrap_or(0);
    let alloc_bytes = alloc_elems * std::mem::size_of::<u16>();
    log_large_alloc("flux.attn.workspace", alloc_bytes);
    let tensor =
        Tensor::zeros_dtype(Shape::from_dims(&[bh, chunk, head_dim]), DType::BF16, device.clone())?;
    map.insert(key.clone(), tensor.clone_result().map_err(anyhow::Error::from)?);
    Ok(tensor)
}

fn apply_qk_norm(t: &Tensor, scale: Option<&Tensor>, head_dim: usize) -> Result<Tensor> {
    let target_dtype = t.dtype();

    let sq = t.mul(t)?;
    static QK_NORM_DEBUG_ONCE: Once = Once::new();
    QK_NORM_DEBUG_ONCE.call_once(|| {
        eprintln!(
            "[streaming_attn_debug] apply_qk_norm dtypes: input={:?} mul={:?} shape={:?}",
            target_dtype,
            sq.dtype(),
            sq.shape().dims()
        );
    });
    let dims = sq.shape().dims().to_vec();
    ensure!(
        dims.len() == 4 && dims[3] as usize == head_dim,
        "apply_qk_norm expects [B,T,H,head_dim], got {:?}",
        dims
    );
    let outer = (dims[0] * dims[1] * dims[2]) as usize;
    let reshaped = sq.reshape(&[outer, 1, head_dim])?;
    let sum = if target_dtype == DType::BF16 {
        sum_dim_keepdim_as(&reshaped, 2, DType::BF16)?
    } else {
        reshaped.sum_dim_keepdim(2)?
    };
    let mean = sum
        .reshape(&[dims[0], dims[1], dims[2], 1])?
        .div_scalar(head_dim as f32)?;
    let inv = mean.add_scalar(1e-6)?.rsqrt()?;
    let mut norm = t.mul(&inv)?;

    if let Some(scale) = scale {
        let scale = if scale.dtype() == target_dtype {
            scale.clone_result()?
        } else {
            scale.to_dtype(target_dtype)?
        };
        let scale = scale.reshape(&[1, 1, 1, head_dim])?;
        norm = norm.mul(&scale)?;
    }

    Ok(norm)
}
static STREAM_ATTENTION_TELEMETRY: Once = Once::new();
static DENSE_ATTENTION_TELEMETRY: Once = Once::new();
static STREAM_ATTENTION_SELFTEST: OnceLock<Result<()>> = OnceLock::new();
