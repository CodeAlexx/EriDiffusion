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

use std::collections::HashMap;

use anyhow::{anyhow, ensure, Result};
use eridiffusion_core::safe_ops::softmax_stable;
use flame_core::{DType, Tensor};

use crate::{
    flame_ctx,
    flux::lora::FluxBlockLora,
    tensor_utils::{broadcast_add, broadcast_mul},
};

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

        let tokens32 = tokens.to_dtype(DType::F32)?;
        let mods = self.modulation(cond, branch.prefix())?;

        let norm1 = rms_norm(&tokens32)?;
        let mod_attn_in = apply_scale_shift(&norm1, &mods.attn, self.hidden)?;
        let attn = self.attention(&mod_attn_in, branch.prefix(), &branch, lora)?;
        let attn = apply_gate(&attn, &mods.attn, self.hidden)?;
        let resid1 = tokens32.add(&attn)?;

        let norm2 = rms_norm(&resid1)?;
        let mod_mlp_in = apply_scale_shift(&norm2, &mods.mlp, self.hidden)?;
        let mlp = self.mlp(&mod_mlp_in, branch.prefix(), &branch, lora)?;
        let mlp = apply_gate(&mlp, &mods.mlp, self.hidden)?;
        let out = resid1.add(&mlp)?;

        Ok(out.to_dtype(DType::BF16)?)
    }

    fn modulation(&self, cond: &Tensor, prefix: &str) -> Result<ModPair> {
        let weight = self.param(&format!("{}.mod.lin.weight", prefix))?;
        let bias = self.params.get(&format!("{}.mod.lin.bias", prefix));

        let cond32 = cond.to_dtype(DType::F32)?.silu()?;
        let weight32 = weight.to_dtype(DType::F32)?;
        let mut proj = cond32.matmul(&weight32.transpose_dims(0, 1)?)?;
        if let Some(bias) = bias {
            proj = proj.add(&bias.to_dtype(DType::F32)?)?;
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
                .clone_result()?;
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

        let tokens_flat = tokens.reshape(&[b * seq, self.hidden])?;
        let weight32 = qkv_w.to_dtype(DType::F32)?;
        let mut qkv = tokens_flat.matmul(&weight32.transpose_dims(0, 1)?)?;
        if let Some(bias) = qkv_b {
            qkv = qkv.add(&bias.to_dtype(DType::F32)?)?;
        }
        let qkv = qkv.reshape(&[b, seq, self.hidden * 3])?;
        let (mut q, mut k, mut v) = split_qkv(&qkv, self.heads, self.head_dim)?;

        if matches!(branch, Branch::Image) {
            if let Some(l) = lora {
                if let Some(lo) = &l.q {
                    let delta =
                        lo.forward_delta(tokens)?.reshape(&[b, seq, self.heads, self.head_dim])?;
                    q = q.add(&delta)?;
                }
                if let Some(lo) = &l.k {
                    let delta =
                        lo.forward_delta(tokens)?.reshape(&[b, seq, self.heads, self.head_dim])?;
                    k = k.add(&delta)?;
                }
                if let Some(lo) = &l.v {
                    let delta =
                        lo.forward_delta(tokens)?.reshape(&[b, seq, self.heads, self.head_dim])?;
                    v = v.add(&delta)?;
                }
            }
        }

        q = apply_qk_norm(&q, q_norm, self.head_dim)?;
        k = apply_qk_norm(&k, k_norm, self.head_dim)?;

        let q = q.permute(&[0, 2, 1, 3])?.reshape(&[b * self.heads, seq, self.head_dim])?;
        let k = k.permute(&[0, 2, 1, 3])?.reshape(&[b * self.heads, seq, self.head_dim])?;
        let v = v.permute(&[0, 2, 1, 3])?.reshape(&[b * self.heads, seq, self.head_dim])?;

        let scale = 1.0f32 / (self.head_dim as f32).sqrt();
        let logits = q.matmul(&k.transpose_dims(1, 2)?)?.mul_scalar(scale)?;
        let weights = softmax_stable(&logits, 2)?;
        let ctx = weights.matmul(&v)?;
        let ctx = ctx
            .reshape(&[b, self.heads, seq, self.head_dim])?
            .permute(&[0, 2, 1, 3])?
            .reshape(&[b * seq, self.hidden])?;

        let proj_w32 = proj_w.to_dtype(DType::F32)?;
        let mut out = ctx.matmul(&proj_w32.transpose_dims(0, 1)?)?;
        if let Some(bias) = proj_b {
            out = out.add(&bias.to_dtype(DType::F32)?)?;
        }
        if matches!(branch, Branch::Image) {
            if let Some(l) = lora.and_then(|l| l.o.as_ref()) {
                let delta = l
                    .forward_delta(&ctx.reshape(&[b, seq, self.hidden])?)?
                    .reshape(&[b * seq, self.hidden])?;
                out = out.add(&delta)?;
            }
        }
        Ok(out.reshape(&[b, seq, self.hidden])?)
    }

    fn mlp(
        &self,
        tokens: &Tensor,
        prefix: &str,
        branch: &Branch,
        lora: Option<&FluxBlockLora>,
    ) -> Result<Tensor> {
        let fc1_w = self.param(&format!("{}.fc1.weight", prefix))?;
        let fc1_b = self.params.get(&format!("{}.fc1.bias", prefix));
        let fc2_w = self.param(&format!("{}.fc2.weight", prefix))?;
        let fc2_b = self.params.get(&format!("{}.fc2.bias", prefix));

        let dims = tokens.shape().dims().to_vec();
        let b = dims[0];
        let seq = dims[1];

        let tokens_flat = tokens.reshape(&[b * seq, self.hidden])?;
        let fc1_w32 = fc1_w.to_dtype(DType::F32)?;
        let mut fc1 = tokens_flat.matmul(&fc1_w32.transpose_dims(0, 1)?)?;
        if let Some(bias) = fc1_b {
            fc1 = fc1.add(&bias.to_dtype(DType::F32)?)?;
        }
        if matches!(branch, Branch::Image) {
            if let Some(l) = lora.and_then(|l| l.fc1.as_ref()) {
                let delta = l.forward_delta(tokens)?.reshape(&[b * seq, fc1.shape().dims()[1]])?;
                fc1 = fc1.add(&delta)?;
            }
        }
        let fc1 = fc1.silu()?;
        let fc2_w32 = fc2_w.to_dtype(DType::F32)?;
        let mut out = fc1.matmul(&fc2_w32.transpose_dims(0, 1)?)?;
        if let Some(bias) = fc2_b {
            out = out.add(&bias.to_dtype(DType::F32)?)?;
        }
        if matches!(branch, Branch::Image) {
            if let Some(l) = lora.and_then(|l| l.fc2.as_ref()) {
                let delta = l
                    .forward_delta(&fc1.reshape(&[b, seq, fc1.shape().dims()[1]])?)?
                    .reshape(&[b * seq, self.hidden])?;
                out = out.add(&delta)?;
            }
        }
        Ok(out.reshape(&[b, seq, self.hidden])?)
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
    let sq = x.mul(x)?;
    let mean = sq.sum_dim_keepdim(2)?.div_scalar(x.shape().dims()[2] as f32)?;
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

fn apply_qk_norm(t: &Tensor, scale: Option<&Tensor>, head_dim: usize) -> Result<Tensor> {
    let t32 = t.to_dtype(DType::F32)?;
    let sq = t32.mul(&t32)?;
    let mean = sq.sum_dim_keepdim(3)?.div_scalar(head_dim as f32)?;
    let mut norm = t32.mul(&mean.add_scalar(1e-6)?.rsqrt()?)?;
    if let Some(scale) = scale {
        let scale = scale.to_dtype(DType::F32)?.reshape(&[1, 1, 1, head_dim])?;
        norm = norm.mul(&scale)?;
    }
    Ok(norm)
}
