//! Pure-GPU `WanAttentionBlock` forward with LoRA injection.
//!
//! Port of `wan-trainer/src/forward_impl/block.rs`, adapted to use
//! EDv2's `Wan22LoraBundle` / `LoraTarget` (defined in the parent
//! `models::wan22` module). The archive's profile-instrumentation
//! sections are removed, and the block_idx==0 debug logs are
//! removed (re-add behind an env gate if needed for parity work).
//!
//! Python reference (`wan/modules/model.py`):
//! ```python
//! def forward(self, x, e, seq_lens, grid_sizes, freqs, context, hints):
//!     shift1, scale1, gate1, shift2, scale2, gate2 = e.chunk(6, dim=2)
//!
//!     y = self.self_attn(self.norm1(x) * (1 + scale1) + shift1,
//!                        seq_lens, grid_sizes, freqs)
//!     x = x + y * gate1
//!
//!     y = self.cross_attn(self.norm3(x), context, context_lens)
//!     x = x + y
//!     y = self.ffn(self.norm2(x) * (1 + scale2) + shift2)
//!     x = x + y * gate2
//!     return x
//! ```

use flame_core::{CudaDevice, DType, Error, Result, Tensor};
use std::{collections::HashMap, sync::Arc};

use super::rope::WanRope;
use super::super::wan22::{LoraTarget, Wan22LoraBundle};

/// Per-block weight lookup helper.
fn w<'a>(
    weights: &'a HashMap<String, Tensor>,
    block_idx: usize,
    suffix: &str,
) -> Result<&'a Tensor> {
    let key = format!("blocks.{block_idx}.{suffix}");
    weights.get(&key).ok_or_else(|| {
        Error::InvalidInput(format!("block_forward: missing weight '{key}'"))
    })
}

/// Layer norm without affine — F32 for parity with Python's
/// `WanLayerNorm.forward(x.float()).type_as(x)`.
fn layer_norm_no_affine(x: &Tensor, eps: f32) -> Result<Tensor> {
    let x_f32 = x.to_dtype(DType::F32)?;
    let dims = x_f32.shape().dims().to_vec();
    let hidden = *dims.last().unwrap();
    let batch: usize = dims[..dims.len() - 1].iter().product();

    let x_flat = x_f32.reshape(&[batch, hidden])?;
    let mean = x_flat.sum_dim_keepdim(1)?.mul_scalar(1.0 / hidden as f32)?;
    let centered = x_flat.sub(&mean)?;
    let var = centered.mul(&centered)?
        .sum_dim_keepdim(1)?
        .mul_scalar(1.0 / hidden as f32)?;
    let rstd = var.add_scalar(eps)?.sqrt()?.reciprocal()?;
    let normed = centered.mul(&rstd)?;
    normed.reshape(&dims)?.to_dtype(DType::BF16)
}

/// Layer norm with affine — F32 for parity.
fn layer_norm_affine(x: &Tensor, weight: &Tensor, bias: &Tensor, eps: f32) -> Result<Tensor> {
    let x_f32 = x.to_dtype(DType::F32)?;
    let w_f32 = weight.to_dtype(DType::F32)?;
    let b_f32 = bias.to_dtype(DType::F32)?;
    let dims = x_f32.shape().dims().to_vec();
    let hidden = *dims.last().unwrap();
    let batch: usize = dims[..dims.len() - 1].iter().product();

    let x_flat = x_f32.reshape(&[batch, hidden])?;
    let mean = x_flat.sum_dim_keepdim(1)?.mul_scalar(1.0 / hidden as f32)?;
    let centered = x_flat.sub(&mean)?;
    let var = centered.mul(&centered)?
        .sum_dim_keepdim(1)?
        .mul_scalar(1.0 / hidden as f32)?;
    let rstd = var.add_scalar(eps)?.sqrt()?.reciprocal()?;
    let normed = centered.mul(&rstd)?;
    let out = normed.mul(&w_f32)?.add(&b_f32)?;
    out.reshape(&dims)?.to_dtype(DType::BF16)
}

/// QK norm — F32 for parity with `WanRMSNorm(x.float()) * weight`.
fn rms_norm(x: &Tensor, scale: &Tensor, eps: f32) -> Result<Tensor> {
    let x_f32 = x.to_dtype(DType::F32)?;
    let dims = x_f32.shape().dims().to_vec();
    let hidden = *dims.last().unwrap();
    let batch: usize = dims[..dims.len() - 1].iter().product();
    let x_flat = x_f32.reshape(&[batch, hidden])?;
    let rms_sq = x_flat.mul(&x_flat)?
        .sum_dim_keepdim(1)?
        .mul_scalar(1.0 / hidden as f32)?;
    let inv_rms = rms_sq.add_scalar(eps)?.sqrt()?.reciprocal()?;
    let normed = x_flat.mul(&inv_rms)?;
    let out = normed.mul(&scale.to_dtype(DType::F32)?)?;
    out.reshape(&dims)?.to_dtype(DType::BF16)
}

/// Wan linear with bias: `y = x @ W^T + b`. Autograd-aware via Tensor::matmul.
///
/// `weight` is the `[Cout, Cin]` PyTorch-format tensor — we transpose on
/// the fly. (Archive used pre-transposed weights from BlockOffloader.
/// EDv2's `Wan22Model::weights` stores raw `[Cout, Cin]`, so an explicit
/// transpose here keeps the math identical.)
fn linear_bias(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    if dims.len() != 3 {
        return Err(Error::InvalidInput(format!(
            "linear_bias expects [B, N, C], got {dims:?}"
        )));
    }
    let (b, n, c) = (dims[0], dims[1], dims[2]);
    let weight_t = weight.transpose()?;
    let x_2d = x.reshape(&[b * n, c])?;
    let out_2d = x_2d.matmul(&weight_t)?;
    let out_cout = out_2d.shape().dims()[1];
    let out = out_2d.reshape(&[b, n, out_cout])?;
    out.add(bias)
}

/// Add a LoRA delta on top of a base linear output, if the bundle has
/// an adapter for `(block_idx, target)`.
fn add_lora_delta(
    base: Tensor,
    input: &Tensor,
    bundle: &Wan22LoraBundle,
    block_idx: usize,
    target: LoraTarget,
) -> Result<Tensor> {
    // Phase 2b: dispatch through `adapter_for` so LyCORIS adapters
    // (LoCon/LoHa/LoKr/Full/OFT) and the legacy plain-LoRA path both
    // contribute the delta. `adapter_for` checks `lycoris_adapters`
    // first, then falls back to the legacy `adapters` map; returns
    // `None` only when neither has an entry.
    if let Some(adapter) = bundle.adapter_for(block_idx, target) {
        let delta = adapter.forward_delta(input)?;
        return base.add(&delta);
    }
    Ok(base)
}

/// Build `(shift, scale, gate)` ×2 from the time-modulation tensor.
#[allow(clippy::type_complexity)]
fn build_modulation(
    e0: &Tensor,
    block_mod: &Tensor,
) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)> {
    let bm_f32 = block_mod.to_dtype(DType::F32)?.unsqueeze(1)?; // [1, 1, 6, dim]
    let mods = bm_f32.add(e0)?; // [1, seq, 6, dim] — F32 throughout

    let m0 = mods.narrow(2, 0, 1)?.squeeze(Some(2))?;
    let m1 = mods.narrow(2, 1, 1)?.squeeze(Some(2))?;
    let m2 = mods.narrow(2, 2, 1)?.squeeze(Some(2))?;
    let m3 = mods.narrow(2, 3, 1)?.squeeze(Some(2))?;
    let m4 = mods.narrow(2, 4, 1)?.squeeze(Some(2))?;
    let m5 = mods.narrow(2, 5, 1)?.squeeze(Some(2))?;

    Ok((m0, m1, m2, m3, m4, m5))
}

/// One Wan transformer block forward with LoRA injection.
///
/// `cross_attn_mask`: optional `[1, 1, sl, txt_len]` BF16 multiplicative
/// mask for the cross-attention (1.0 = real text token, 0.0 = padding).
/// When `None`, padding tokens still receive softmax weight, which dilutes
/// real-text conditioning — see audit H1.
#[allow(clippy::too_many_arguments)]
pub fn block_forward_with_lora(
    x: &Tensor,
    e0: &Tensor,
    txt: &Tensor,
    rope: &WanRope,
    eps: f32,
    num_heads: usize,
    head_dim: usize,
    block_weights: &HashMap<String, Tensor>,
    bundle: &Wan22LoraBundle,
    block_idx: usize,
    cross_attn_mask: Option<&Tensor>,
    _device: &Arc<CudaDevice>,
) -> Result<Tensor> {
    let dims = x.shape().dims();
    if dims.len() != 3 {
        return Err(Error::InvalidInput(format!(
            "block_forward expects [1, seq, dim], got {dims:?}"
        )));
    }
    let sl = dims[1];
    let dim = dims[2];

    // ── 1) Modulation ────────────────────────────────────────────────
    let (shift1, scale1, gate1, shift2, scale2, gate2) = {
        let block_mod = w(block_weights, block_idx, "modulation")?;
        build_modulation(e0, block_mod)?
    };

    // ── 2) Self-attention ───────────────────────────────────────────
    let sa_input = {
        // Python: torch.addcmul(shift1, self.norm1(x).float(), 1+scale1).to(org_dtype)
        let x_normed = layer_norm_no_affine(x, eps)?;
        let n = x_normed.to_dtype(DType::F32)?;
        n.mul(&scale1.add_scalar(1.0)?)?
            .add(&shift1)?
            .to_dtype(DType::BF16)?
    };

    let q_w = w(block_weights, block_idx, "self_attn.q.weight")?;
    let q_b = w(block_weights, block_idx, "self_attn.q.bias")?;
    let k_w = w(block_weights, block_idx, "self_attn.k.weight")?;
    let k_b = w(block_weights, block_idx, "self_attn.k.bias")?;
    let v_w = w(block_weights, block_idx, "self_attn.v.weight")?;
    let v_b = w(block_weights, block_idx, "self_attn.v.bias")?;
    let o_w = w(block_weights, block_idx, "self_attn.o.weight")?;
    let o_b = w(block_weights, block_idx, "self_attn.o.bias")?;
    let nq_w = w(block_weights, block_idx, "self_attn.norm_q.weight")?;
    let nk_w = w(block_weights, block_idx, "self_attn.norm_k.weight")?;

    let q = add_lora_delta(
        linear_bias(&sa_input, q_w, q_b)?,
        &sa_input,
        bundle,
        block_idx,
        LoraTarget::SelfQ,
    )?;
    let k = add_lora_delta(
        linear_bias(&sa_input, k_w, k_b)?,
        &sa_input,
        bundle,
        block_idx,
        LoraTarget::SelfK,
    )?;
    let v = add_lora_delta(
        linear_bias(&sa_input, v_w, v_b)?,
        &sa_input,
        bundle,
        block_idx,
        LoraTarget::SelfV,
    )?;

    let q = rms_norm(&q, nq_w, eps)?;
    let k = rms_norm(&k, nk_w, eps)?;

    // [1, sl, dim] → [1, sl, num_heads, head_dim] → [1, num_heads, sl, head_dim]
    let q = q
        .reshape(&[1, sl, num_heads, head_dim])?
        .permute(&[0, 2, 1, 3])?;
    let k = k
        .reshape(&[1, sl, num_heads, head_dim])?
        .permute(&[0, 2, 1, 3])?;
    let v = v
        .reshape(&[1, sl, num_heads, head_dim])?
        .permute(&[0, 2, 1, 3])?;

    // RoPE on q and k.
    let q = rope.apply(&q)?;
    let k = rope.apply(&k)?;

    let attn_out = flame_core::attention::sdpa(&q, &k, &v, None)?;
    let attn_out = attn_out.permute(&[0, 2, 1, 3])?.reshape(&[1, sl, dim])?;

    let sa_out = add_lora_delta(
        linear_bias(&attn_out, o_w, o_b)?,
        &attn_out,
        bundle,
        block_idx,
        LoraTarget::SelfO,
    )?;

    // gated_residual: x + sa_out * gate1 (F32 for parity)
    let x_after_sa = {
        let xf = x.to_dtype(DType::F32)?;
        let df = sa_out.to_dtype(DType::F32)?;
        let out = xf.add(&df.mul(&gate1)?)?;
        out.to_dtype(DType::BF16)?
    };

    // ── 3) Cross-attention ──────────────────────────────────────────
    let ca_input = match (
        block_weights.get(&format!("blocks.{block_idx}.norm3.weight")),
        block_weights.get(&format!("blocks.{block_idx}.norm3.bias")),
    ) {
        (Some(n3w), Some(n3b)) => layer_norm_affine(&x_after_sa, n3w, n3b, eps)?,
        _ => x_after_sa.clone(),
    };

    let caq_w = w(block_weights, block_idx, "cross_attn.q.weight")?;
    let caq_b = w(block_weights, block_idx, "cross_attn.q.bias")?;
    let cak_w = w(block_weights, block_idx, "cross_attn.k.weight")?;
    let cak_b = w(block_weights, block_idx, "cross_attn.k.bias")?;
    let cav_w = w(block_weights, block_idx, "cross_attn.v.weight")?;
    let cav_b = w(block_weights, block_idx, "cross_attn.v.bias")?;
    let cao_w = w(block_weights, block_idx, "cross_attn.o.weight")?;
    let cao_b = w(block_weights, block_idx, "cross_attn.o.bias")?;
    let canq_w = w(block_weights, block_idx, "cross_attn.norm_q.weight")?;
    let cank_w = w(block_weights, block_idx, "cross_attn.norm_k.weight")?;

    let ca_q = add_lora_delta(
        linear_bias(&ca_input, caq_w, caq_b)?,
        &ca_input,
        bundle,
        block_idx,
        LoraTarget::CrossQ,
    )?;
    let ca_k = add_lora_delta(
        linear_bias(txt, cak_w, cak_b)?,
        txt,
        bundle,
        block_idx,
        LoraTarget::CrossK,
    )?;
    let ca_v = add_lora_delta(
        linear_bias(txt, cav_w, cav_b)?,
        txt,
        bundle,
        block_idx,
        LoraTarget::CrossV,
    )?;

    let ca_q = rms_norm(&ca_q, canq_w, eps)?;
    let ca_k = rms_norm(&ca_k, cank_w, eps)?;

    let txt_len = txt.shape().dims()[1];
    let ca_q = ca_q
        .reshape(&[1, sl, num_heads, head_dim])?
        .permute(&[0, 2, 1, 3])?;
    let ca_k = ca_k
        .reshape(&[1, txt_len, num_heads, head_dim])?
        .permute(&[0, 2, 1, 3])?;
    let ca_v = ca_v
        .reshape(&[1, txt_len, num_heads, head_dim])?
        .permute(&[0, 2, 1, 3])?;

    let ca_attn = flame_core::attention::sdpa(&ca_q, &ca_k, &ca_v, cross_attn_mask)?;
    let ca_attn = ca_attn.permute(&[0, 2, 1, 3])?.reshape(&[1, sl, dim])?;

    let ca_out = add_lora_delta(
        linear_bias(&ca_attn, cao_w, cao_b)?,
        &ca_attn,
        bundle,
        block_idx,
        LoraTarget::CrossO,
    )?;

    // x + ca_out — no gate on cross-attn residual.
    let x_after_ca = x_after_sa.add(&ca_out)?;

    // ── 4) FFN ─────────────────────────────────────────────────────
    let ffn_input = {
        let x_ca_normed = layer_norm_no_affine(&x_after_ca, eps)?;
        let n = x_ca_normed.to_dtype(DType::F32)?;
        n.mul(&scale2.add_scalar(1.0)?)?
            .add(&shift2)?
            .to_dtype(DType::BF16)?
    };

    let ffn_w0 = w(block_weights, block_idx, "ffn.0.weight")?;
    let ffn_b0 = w(block_weights, block_idx, "ffn.0.bias")?;
    let ffn_w2 = w(block_weights, block_idx, "ffn.2.weight")?;
    let ffn_b2 = w(block_weights, block_idx, "ffn.2.bias")?;

    let ffn_out = {
        let hidden = linear_bias(&ffn_input, ffn_w0, ffn_b0)?;
        let hidden = hidden.gelu()?;
        linear_bias(&hidden, ffn_w2, ffn_b2)?
    };

    // gated_residual: x_after_ca + ffn_out * gate2 (F32 for parity)
    let xf = x_after_ca.to_dtype(DType::F32)?;
    let df = ffn_out.to_dtype(DType::F32)?;
    xf.add(&df.mul(&gate2)?)?.to_dtype(DType::BF16)
}
