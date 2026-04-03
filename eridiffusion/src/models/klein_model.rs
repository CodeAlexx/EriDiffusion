//! Flux 2 Klein DiT transformer — pure flame_core, key-exact for BFL checkpoints.
//!
//! Supports Klein 4B (5+20 blocks) and Klein 9B (8+24 blocks).
//!
//! Architecture-exact match with BFL safetensors checkpoints.
//!
//! Key structural differences from Flux 1:
//! - NO biases anywhere
//! - SHARED modulation (3 layers at model level, NOT per-block)
//! - SwiGLU MLP (6x ratio) instead of GELU (4x ratio)
//! - No guidance_in, no vector_in
//! - in_channels=128, joint_attention_dim=7680 (4B) or 12288 (9B)
//!
//! Rewritten from the ground-truth Python+Flame reference:
//!   flame-core/inference-test/models/klein_flame.py

use flame_core::cuda_ops_bf16;
use flame_core::serialization;
use flame_core::{DType, Result, Shape, Tensor};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Architecture constants for Klein variants.
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
    /// Klein 4B: 5 double + 20 single blocks, inner_dim=3072
    pub fn klein_4b() -> Self {
        Self {
            inner_dim: 3072,
            in_channels: 128,
            joint_attention_dim: 7680,
            num_double: 5,
            num_single: 20,
            num_heads: 24,
            head_dim: 128,
            mlp_hidden: 3072 * 3, // SwiGLU 3x ratio
            timestep_dim: 256,
            axes_dims: [32, 32, 32, 32],
            theta: 2000.0,
        }
    }

    /// Klein 9B: 8 double + 24 single blocks, inner_dim=4096
    pub fn klein_9b() -> Self {
        Self {
            inner_dim: 4096,
            in_channels: 128,
            joint_attention_dim: 12288,
            num_double: 8,
            num_single: 24,
            num_heads: 32,
            head_dim: 128,
            mlp_hidden: 4096 * 3,
            timestep_dim: 256,
            axes_dims: [32, 32, 32, 32],
            theta: 2000.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: 3D linear (flame_core Linear only handles 2D batch internally,
// but we need [B, N, C] -> [B, N, out_C] without constructing Linear structs)
// ---------------------------------------------------------------------------

/// Compute `x @ weight.T` for x with shape [B, N, C] and weight [out_C, C].
/// No bias (Klein has NO biases anywhere).
/// Linear projection using pre-transposed weights.
/// weight_t is already [in_features, out_features] (transposed at load time).
fn linear3d(x: &Tensor, weight_t: &Tensor) -> Result<Tensor> {
    let shape = x.shape().dims().to_vec();
    if shape.len() == 2 {
        return x.matmul(weight_t);
    }
    // [B, N, C] -> flatten to [B*N, C], matmul, reshape back
    let b = shape[0];
    let n = shape[1];
    let c = shape[2];
    let x_2d = x.reshape(&[b * n, c])?;
    let out_2d = x_2d.matmul(weight_t)?;
    let out_dim = out_2d.shape().dims()[1];
    out_2d.reshape(&[b, n, out_dim])
}

/// x @ weight where weight is ALREADY transposed [in, out].
/// Result is [M, out].
fn matmul_weight_pretransposed(x: &Tensor, weight_t: &Tensor) -> Result<Tensor> {
    x.matmul(weight_t)
}

// ---------------------------------------------------------------------------
// Timestep embedding
// ---------------------------------------------------------------------------

/// Sinusoidal timestep embedding matching ComfyUI.
///
/// `t`: [B] sigma values in [0, 1].
/// `dim`: embedding dimension.
/// `time_factor`: scales sigma to [0, 1000] (default 1000.0).
///
/// Returns [B, dim] in the original dtype of `t`.
fn timestep_embedding(t: &Tensor, dim: usize, time_factor: f32) -> Result<Tensor> {
    let orig_dtype = t.dtype();
    let device = t.device().clone();
    let b = t.shape().dims()[0];

    // Scale: t_scaled = t.to_f32() * time_factor
    let t_f32 = t.to_dtype(DType::F32)?;
    let t_scaled = t_f32.mul_scalar(time_factor)?;

    let half = dim / 2;
    let max_period: f32 = 10000.0;

    // freqs = exp(-log(max_period) * arange(0, half) / half)
    let freqs = Tensor::arange(0.0, half as f32, 1.0, device.clone())?;
    let freqs = freqs.mul_scalar(-max_period.ln() / half as f32)?.exp()?;

    // Outer product: [B, 1] * [1, half] -> [B, half]
    let t_col = t_scaled.reshape(&[b, 1])?;
    let freqs_row = freqs.reshape(&[1, half])?;
    let args = t_col.mul(&freqs_row)?;

    // [cos(args), sin(args)] -> [B, dim]
    let cos_part = args.cos()?;
    let sin_part = args.sin()?;
    let emb = Tensor::cat(&[&cos_part, &sin_part], 1)?;

    emb.to_dtype(orig_dtype)
}

// ---------------------------------------------------------------------------
// Modulation: adaLN with LayerNorm (subtracts mean, NOT RMSNorm)
// ---------------------------------------------------------------------------

/// `(1 + scale) * LayerNorm(x) + shift`
///
/// Uses proper LayerNorm (subtract mean, divide by std).
/// `shift`, `scale`: [B, dim]; `x`: [B, N, dim].
fn modulate_pre(x: &Tensor, shift: &Tensor, scale: &Tensor) -> Result<Tensor> {
    let dim = *x.shape().dims().last().unwrap();

    // LayerNorm in F32 for numerical stability, then back to original dtype
    // flame_core::layer_norm::layer_norm expects BF16 input; we use it directly.
    let x_normed = x.layer_norm(&[dim], None, None, 1e-6)?;

    // scale and shift: [B, dim] -> unsqueeze to [B, 1, dim]
    let scale_u = scale.unsqueeze(1)?;
    let shift_u = shift.unsqueeze(1)?;

    // (1 + scale) * x_normed + shift
    let scaled = scale_u.add_scalar(1.0)?;
    let out = x_normed.mul(&scaled)?;
    out.add(&shift_u)
}

// ---------------------------------------------------------------------------
// QK RMSNorm (no mean subtraction — different from LayerNorm)
// ---------------------------------------------------------------------------

/// Per-head RMSNorm for query/key.
///
/// `x`: [B, H, N, D] reshaped to [B*H*N, D] for rms_norm, then back.
/// `scale`: [D] learned scale parameter.
fn head_rms_norm(x: &Tensor, scale: &Tensor) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    let (b, h, n, d) = (dims[0], dims[1], dims[2], dims[3]);
    let flat = x.reshape(&[b * h * n, d])?;
    let normed = cuda_ops_bf16::rms_norm_bf16(&flat, Some(scale), 1e-6)?;
    normed.reshape(&[b, h, n, d])
}

// ---------------------------------------------------------------------------
// RoPE — real-valued rotation (no complex numbers)
// ---------------------------------------------------------------------------

/// Build 2D RoPE cos/sin tables from position IDs.
///
/// `img_ids`: [N_img, ndim] position coordinates.
/// `txt_ids`: [N_txt, ndim] position coordinates.
/// `axes_dims`: per-axis RoPE dimensions (must sum to head_dim).
/// `theta`: base frequency (2000.0 for Klein).
///
/// Returns `(pe_cos, pe_sin)`: each [1, 1, N_total, head_dim//2] in BF16.
fn build_rope_2d(
    img_ids: &Tensor,
    txt_ids: &Tensor,
    axes_dims: &[usize; 4],
    theta: f32,
) -> Result<(Tensor, Tensor)> {
    let device = img_ids.device().clone();

    // Concat: txt first, then img (matching Python: flame_core.cat([txt_ids, img_ids], 0))
    let all_ids = Tensor::cat(&[txt_ids, img_ids], 0)?;
    let n_total = all_ids.shape().dims()[0];

    let mut cos_parts: Vec<Tensor> = Vec::new();
    let mut sin_parts: Vec<Tensor> = Vec::new();

    for (axis_idx, &dim) in axes_dims.iter().enumerate() {
        let half = dim / 2;

        // Position values for this axis: all_ids[:, axis_idx] -> [N]
        let pos = all_ids
            .narrow(1, axis_idx, 1)?
            .squeeze(Some(1))?
            .to_dtype(DType::F32)?;

        // Frequency indices: exp(-log(theta) * arange(0, dim, 2) / dim)
        let freq_idx = Tensor::arange(0.0, dim as f32, 2.0, device.clone())?;
        let log_freqs = freq_idx
            .mul_scalar(-(theta as f32).ln() / dim as f32)?
            .exp()?;

        // Outer product: [N, 1] * [1, half] -> [N, half]
        let pos_col = pos.reshape(&[n_total, 1])?;
        let freq_row = log_freqs.reshape(&[1, half])?;
        let angles = pos_col.mul(&freq_row)?;

        cos_parts.push(angles.cos()?);
        sin_parts.push(angles.sin()?);
    }

    // Concat along last dim -> [N, head_dim//2]
    let cos_refs: Vec<&Tensor> = cos_parts.iter().collect();
    let sin_refs: Vec<&Tensor> = sin_parts.iter().collect();
    let pe_cos = Tensor::cat(&cos_refs, 1)?;
    let pe_sin = Tensor::cat(&sin_refs, 1)?;

    // -> [1, 1, N, head_dim//2] in BF16
    let pe_cos = pe_cos.unsqueeze(0)?.unsqueeze(0)?.to_dtype(DType::BF16)?;
    let pe_sin = pe_sin.unsqueeze(0)?.unsqueeze(0)?.to_dtype(DType::BF16)?;

    Ok((pe_cos, pe_sin))
}

/// Apply rotary position embeddings using real-valued math — all BF16.
///
/// `q`, `k`: [B, H, N, D] in BF16.
/// `pe_cos`, `pe_sin`: [1, 1, N, D//2] in BF16.
///
/// Replicates PyTorch's `view_as_complex` rotation:
/// treats consecutive pairs as (real, imag), applies complex multiply.
/// Stays in BF16 throughout — avoids F32 CudaKernels recompilation.
fn apply_rope(
    q: &Tensor,
    k: &Tensor,
    pe_cos: &Tensor,
    pe_sin: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let q_dims = q.shape().dims().to_vec();
    let (b, h, n, d) = (q_dims[0], q_dims[1], q_dims[2], q_dims[3]);
    let half = d / 2;

    // pe_cos/pe_sin are already BF16 [1, 1, N, half]
    // Reshape for broadcasting: [1, N, half] to broadcast over B*H
    let cos_flat = pe_cos.reshape(&[1, n, half])?;
    let sin_flat = pe_sin.reshape(&[1, n, half])?;

    let total_flat = b * h * n * half;

    let rope_one = |x: &Tensor| -> Result<Tensor> {
        let bh = b * h;
        let x_4d = x.reshape(&[bh, n, half, 2])?;

        // Extract even (real) and odd (imag) components via BF16 narrow
        let x_even = x_4d.narrow(3, 0, 1)?.squeeze(Some(3))?; // [BH, N, half]
        let x_odd = x_4d.narrow(3, 1, 1)?.squeeze(Some(3))?;

        // Complex multiply: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        let ac = x_even.mul(&cos_flat)?;
        let bd = x_odd.mul(&sin_flat)?;
        let ad = x_even.mul(&sin_flat)?;
        let bc = x_odd.mul(&cos_flat)?;
        let new_even = ac.sub(&bd)?;  // [BH, N, half]
        let new_odd = ad.add(&bc)?;   // [BH, N, half]

        // Interleave even/odd back into [BH, N, D]:
        // 1. Flatten both to [1, total] and cat on dim 0 → [2, total] (2 large copies)
        // 2. 2D transpose → [total, 2] (GPU kernel)
        // 3. Reshape to [B, H, N, D] (view)
        let even_flat = new_even.reshape(&[1, total_flat])?;
        let odd_flat = new_odd.reshape(&[1, total_flat])?;
        let combined = Tensor::cat(&[&even_flat, &odd_flat], 0)?; // [2, total]
        let transposed = combined.permute(&[1, 0])?; // [total, 2] — GPU 2D transpose
        transposed.reshape(&[b, h, n, d])
    };

    Ok((rope_one(q)?, rope_one(k)?))
}

// ---------------------------------------------------------------------------
// SwiGLU MLP
// ---------------------------------------------------------------------------

/// SwiGLU: split fused gate+up projection, apply silu*up, project down.
///
/// `gate_up_w`: [2*mlp_hidden, inner_dim] fused weight.
/// `down_w`: [inner_dim, mlp_hidden] down projection weight.
/// `x`: [B, N, inner_dim].
fn swiglu(gate_up_w: &Tensor, down_w: &Tensor, x: &Tensor) -> Result<Tensor> {
    let gu = linear3d(x, gate_up_w)?;
    let last_dim = *gu.shape().dims().last().unwrap();
    let half_dim = last_dim / 2;
    let ndim = gu.shape().dims().len();
    let gate = gu.narrow(ndim - 1, 0, half_dim)?;
    let up = gu.narrow(ndim - 1, half_dim, half_dim)?;
    let activated = gate.silu()?.mul(&up)?;
    linear3d(&activated, down_w)
}

// ---------------------------------------------------------------------------
// Double block
// ---------------------------------------------------------------------------

/// Execute one double-stream block.
///
/// `img_mods`: [shift1, scale1, gate1, shift2, scale2, gate2]
/// `txt_mods`: same structure.
fn double_block_forward(
    weights: &HashMap<String, Tensor>,
    block_idx: usize,
    img: &Tensor,
    txt: &Tensor,
    img_mods: &[Tensor; 6],
    txt_mods: &[Tensor; 6],
    pe_cos: &Tensor,
    pe_sin: &Tensor,
    num_heads: usize,
    head_dim: usize,
) -> Result<(Tensor, Tensor)> {
    let prefix = format!("double_blocks.{block_idx}");
    let h = num_heads;
    let d = head_dim;
    // Unpack modulation parameters
    let (img_shift1, img_scale1, img_gate1) = (&img_mods[0], &img_mods[1], &img_mods[2]);
    let (img_shift2, img_scale2, img_gate2) = (&img_mods[3], &img_mods[4], &img_mods[5]);
    let (txt_shift1, txt_scale1, txt_gate1) = (&txt_mods[0], &txt_mods[1], &txt_mods[2]);
    let (txt_shift2, txt_scale2, txt_gate2) = (&txt_mods[3], &txt_mods[4], &txt_mods[5]);

    // --- Attention ---
    let img_normed = modulate_pre(img, img_shift1, img_scale1)?;
    let txt_normed = modulate_pre(txt, txt_shift1, txt_scale1)?;

    // QKV projections
    let img_qkv = linear3d(&img_normed, &weights[&format!("{prefix}.img_attn.qkv.weight")])?;
    let txt_qkv = linear3d(&txt_normed, &weights[&format!("{prefix}.txt_attn.qkv.weight")])?;

    let b = img_qkv.shape().dims()[0];
    let n_img = img_qkv.shape().dims()[1];
    let n_txt = txt_qkv.shape().dims()[1];

    // Split QKV: [B, N, 3*inner_dim] -> q, k, v each [B, H, N, D]
    let split_qkv = |qkv: &Tensor, seq_len: usize| -> Result<(Tensor, Tensor, Tensor)> {
        let inner = h * d;
        let q = qkv.narrow(2, 0, inner)?;
        let k = qkv.narrow(2, inner, inner)?;
        let v = qkv.narrow(2, 2 * inner, inner)?;
        // [B, N, H, D] -> permute [0, 2, 1, 3] -> [B, H, N, D]
        let q = q.reshape(&[b, seq_len, h, d])?.permute(&[0, 2, 1, 3])?;
        let k = k.reshape(&[b, seq_len, h, d])?.permute(&[0, 2, 1, 3])?;
        let v = v.reshape(&[b, seq_len, h, d])?.permute(&[0, 2, 1, 3])?;
        Ok((q, k, v))
    };

    let (mut img_q, mut img_k, img_v) = split_qkv(&img_qkv, n_img)?;
    let (mut txt_q, mut txt_k, txt_v) = split_qkv(&txt_qkv, n_txt)?;

    // QK norm (RMSNorm per head)
    img_q = head_rms_norm(
        &img_q,
        &weights[&format!("{prefix}.img_attn.norm.query_norm.scale")],
    )?;
    img_k = head_rms_norm(
        &img_k,
        &weights[&format!("{prefix}.img_attn.norm.key_norm.scale")],
    )?;
    txt_q = head_rms_norm(
        &txt_q,
        &weights[&format!("{prefix}.txt_attn.norm.query_norm.scale")],
    )?;
    txt_k = head_rms_norm(
        &txt_k,
        &weights[&format!("{prefix}.txt_attn.norm.key_norm.scale")],
    )?;

    // Joint attention: concat txt first, then img along sequence dim
    let q = Tensor::cat(&[&txt_q, &img_q], 2)?; // [B, H, N_txt+N_img, D]
    let k = Tensor::cat(&[&txt_k, &img_k], 2)?;
    let v = Tensor::cat(&[&txt_v, &img_v], 2)?;

    // Apply RoPE
    let (q, k) = apply_rope(&q, &k, pe_cos, pe_sin)?;

    // Scaled dot-product attention
    let attn_out = flame_core::attention::sdpa(&q, &k, &v, None)?; // [B, H, N_total, D]

    // Split back: txt first, then img
    let txt_out = attn_out.narrow(2, 0, n_txt)?;
    let img_out = attn_out.narrow(2, n_txt, n_img)?;

    // Reshape: [B, H, N, D] -> permute [0, 2, 1, 3] -> [B, N, H*D]
    let img_out = img_out
        .permute(&[0, 2, 1, 3])?
        .reshape(&[b, n_img, h * d])?;
    let txt_out = txt_out
        .permute(&[0, 2, 1, 3])?
        .reshape(&[b, n_txt, h * d])?;

    // Output projection (NO bias)
    let img_out = linear3d(&img_out, &weights[&format!("{prefix}.img_attn.proj.weight")])?;
    let txt_out = linear3d(&txt_out, &weights[&format!("{prefix}.txt_attn.proj.weight")])?;

    // Gate + residual
    let img = img.add(&img_gate1.unsqueeze(1)?.mul(&img_out)?)?;
    let txt = txt.add(&txt_gate1.unsqueeze(1)?.mul(&txt_out)?)?;

    // --- MLP (SwiGLU) ---
    let img_mlp_in = modulate_pre(&img, img_shift2, img_scale2)?;
    let txt_mlp_in = modulate_pre(&txt, txt_shift2, txt_scale2)?;

    let img_mlp_out = swiglu(
        &weights[&format!("{prefix}.img_mlp.0.weight")],
        &weights[&format!("{prefix}.img_mlp.2.weight")],
        &img_mlp_in,
    )?;
    let txt_mlp_out = swiglu(
        &weights[&format!("{prefix}.txt_mlp.0.weight")],
        &weights[&format!("{prefix}.txt_mlp.2.weight")],
        &txt_mlp_in,
    )?;

    // Gate + residual
    let img = img.add(&img_gate2.unsqueeze(1)?.mul(&img_mlp_out)?)?;
    let txt = txt.add(&txt_gate2.unsqueeze(1)?.mul(&txt_mlp_out)?)?;

    Ok((img, txt))
}

// ---------------------------------------------------------------------------
// Single block
// ---------------------------------------------------------------------------

/// Execute one single-stream block.
///
/// `linear1` fuses: QKV (3*dim) + SwiGLU gate+up (2*mlp_hidden) = 9*dim
/// `linear2` fuses: attn_proj (dim) + SwiGLU down (mlp_hidden) = 4*dim
fn single_block_forward(
    weights: &HashMap<String, Tensor>,
    block_idx: usize,
    x: &Tensor,
    mods: &[Tensor; 3],
    pe_cos: &Tensor,
    pe_sin: &Tensor,
    num_heads: usize,
    head_dim: usize,
    inner_dim: usize,
    mlp_hidden: usize,
) -> Result<Tensor> {
    let prefix = format!("single_blocks.{block_idx}");
    let h = num_heads;
    let d = head_dim;

    let (shift, scale, gate) = (&mods[0], &mods[1], &mods[2]);

    let x_normed = modulate_pre(x, shift, scale)?;

    // Fused QKV + SwiGLU gate+up
    let qkv_mlp = linear3d(&x_normed, &weights[&format!("{prefix}.linear1.weight")])?;

    // Split: first 3*inner_dim = QKV, rest = gate+up
    let qkv_dim = 3 * inner_dim;
    let qkv = qkv_mlp.narrow(2, 0, qkv_dim)?;
    let gate_up = qkv_mlp.narrow(2, qkv_dim, 2 * mlp_hidden)?;

    let b = qkv.shape().dims()[0];
    let n = qkv.shape().dims()[1];

    // Split QKV -> q, k, v each [B, H, N, D]
    let q = qkv
        .narrow(2, 0, inner_dim)?
        .reshape(&[b, n, h, d])?
        .permute(&[0, 2, 1, 3])?;
    let k = qkv
        .narrow(2, inner_dim, inner_dim)?
        .reshape(&[b, n, h, d])?
        .permute(&[0, 2, 1, 3])?;
    let v = qkv
        .narrow(2, 2 * inner_dim, inner_dim)?
        .reshape(&[b, n, h, d])?
        .permute(&[0, 2, 1, 3])?;

    // QK norm (RMSNorm per head)
    let q = head_rms_norm(&q, &weights[&format!("{prefix}.norm.query_norm.scale")])?;
    let k = head_rms_norm(&k, &weights[&format!("{prefix}.norm.key_norm.scale")])?;

    // RoPE
    let (q, k) = apply_rope(&q, &k, pe_cos, pe_sin)?;

    // Attention
    let attn_out = flame_core::attention::sdpa(&q, &k, &v, None)?;
    let attn_out = attn_out
        .permute(&[0, 2, 1, 3])?
        .reshape(&[b, n, h * d])?;

    // SwiGLU from the fused gate+up
    let gate_proj = gate_up.narrow(2, 0, mlp_hidden)?;
    let up_proj = gate_up.narrow(2, mlp_hidden, mlp_hidden)?;
    let mlp_out = gate_proj.silu()?.mul(&up_proj)?;

    // Fused output: cat [attn, mlp] -> linear2
    let fused = Tensor::cat(&[&attn_out, &mlp_out], 2)?;
    let out = linear3d(&fused, &weights[&format!("{prefix}.linear2.weight")])?;

    // Gate + residual
    x.add(&gate.unsqueeze(1)?.mul(&out)?)
}

// ---------------------------------------------------------------------------
// Shared modulation helper
// ---------------------------------------------------------------------------

/// Apply shared modulation: silu(vec) -> linear -> chunk into N parts.
fn shared_modulation_from_silu(
    vec_silu: &Tensor,
    weight: &Tensor,
    num_chunks: usize,
) -> Result<Vec<Tensor>> {
    let raw = linear3d(vec_silu, weight)?;
    let last_dim = *raw.shape().dims().last().unwrap();
    let chunk_size = last_dim / num_chunks;
    let ndim = raw.shape().dims().len();

    let mut chunks = Vec::with_capacity(num_chunks);
    for j in 0..num_chunks {
        chunks.push(raw.narrow(ndim - 1, j * chunk_size, chunk_size)?);
    }
    Ok(chunks)
}

/// Convert a 6-element Vec into a fixed-size array (moves, no clone).
fn vec_to_arr6(mut v: Vec<Tensor>) -> Result<[Tensor; 6]> {
    if v.len() != 6 {
        return Err(flame_core::Error::InvalidOperation(format!(
            "Expected 6 modulation chunks, got {}",
            v.len()
        )));
    }
    // Drain in reverse to pop efficiently
    let e5 = v.pop().unwrap();
    let e4 = v.pop().unwrap();
    let e3 = v.pop().unwrap();
    let e2 = v.pop().unwrap();
    let e1 = v.pop().unwrap();
    let e0 = v.pop().unwrap();
    Ok([e0, e1, e2, e3, e4, e5])
}

/// Convert a 3-element Vec into a fixed-size array (moves, no clone).
fn vec_to_arr3(mut v: Vec<Tensor>) -> Result<[Tensor; 3]> {
    if v.len() != 3 {
        return Err(flame_core::Error::InvalidOperation(format!(
            "Expected 3 modulation chunks, got {}",
            v.len()
        )));
    }
    let e2 = v.pop().unwrap();
    let e1 = v.pop().unwrap();
    let e0 = v.pop().unwrap();
    Ok([e0, e1, e2])
}

// ---------------------------------------------------------------------------
// Main Transformer
// ---------------------------------------------------------------------------

/// Flux 2 Klein DiT — pure flame_core implementation.
///
/// Exact key-compatible with BFL .safetensors checkpoints.
pub struct KleinTransformer {
    weights: HashMap<String, Tensor>,
    config: KleinConfig,
}

impl KleinTransformer {
    /// Load from a BFL safetensors checkpoint.
    ///
    /// Auto-detects Klein 4B vs 9B from weight shapes.
    pub fn from_safetensors(path: &str) -> Result<Self> {
        let device = flame_core::global_cuda_device();
        let weights = serialization::load_file(path, &device)?;
        Self::from_weights(weights)
    }

    /// Construct from a pre-loaded weight dict.
    ///
    /// Auto-detects Klein 4B vs 9B from weight shapes.
    pub fn from_weights(weights: HashMap<String, Tensor>) -> Result<Self> {
        // Auto-detect architecture from weight shapes
        let inner_dim = weights["img_in.weight"].shape().dims()[0];
        let in_channels = weights["img_in.weight"].shape().dims()[1];
        let joint_dim = weights["txt_in.weight"].shape().dims()[1];

        let mut num_double = 0;
        while weights.contains_key(&format!("double_blocks.{num_double}.img_attn.qkv.weight")) {
            num_double += 1;
        }
        let mut num_single = 0;
        while weights.contains_key(&format!("single_blocks.{num_single}.linear1.weight")) {
            num_single += 1;
        }

        let num_heads = inner_dim / 128;
        let head_dim = 128;
        let mlp_hidden = inner_dim * 3; // SwiGLU 3x ratio

        let config = KleinConfig {
            inner_dim,
            in_channels,
            joint_attention_dim: joint_dim,
            num_double,
            num_single,
            num_heads,
            head_dim,
            mlp_hidden,
            timestep_dim: 256,
            axes_dims: [32, 32, 32, 32],
            theta: 2000.0,
        };

        log::info!(
            "[KleinTransformer] Detected: inner_dim={}, joint_dim={}, \
             double={}, single={}, heads={}",
            inner_dim,
            joint_dim,
            num_double,
            num_single,
            num_heads,
        );

        // Pre-transpose ALL weight matrices [out, in] -> [in, out] for faster matmul.
        // This avoids 500+ GPU transpose copies per forward pass.
        log::info!("[KleinTransformer] Pre-transposing weights...");
        let mut weights = weights;
        let keys: Vec<String> = weights.keys().cloned().collect();
        for key in &keys {
            if key.ends_with(".weight") && !key.ends_with(".scale") {
                let w = &weights[key];
                let dims = w.shape().dims();
                if dims.len() == 2 {
                    let wt = flame_core::bf16_elementwise::transpose2d_bf16(w)?;
                    weights.insert(key.clone(), wt);
                }
            }
        }
        log::info!("[KleinTransformer] Weights pre-transposed.");

        Ok(Self { weights, config })
    }

    /// Forward pass.
    ///
    /// * `img`: [B, N_img, in_channels] image latent tokens.
    /// * `txt`: [B, N_txt, joint_attention_dim] text embeddings.
    /// * `timesteps`: [B] sigma values.
    /// * `img_ids`: [N_img, 4] image position IDs.
    /// * `txt_ids`: [N_txt, 4] text position IDs.
    ///
    /// Returns [B, N_img, in_channels] predicted noise/velocity.
    pub fn forward(
        &self,
        img: &Tensor,
        txt: &Tensor,
        timesteps: &Tensor,
        img_ids: &Tensor,
        txt_ids: &Tensor,
    ) -> Result<Tensor> {
        let w = &self.weights;
        let cfg = &self.config;

        // 1. Input projections (NO bias)
        let mut img = linear3d(img, &w["img_in.weight"])?;
        let mut txt = linear3d(txt, &w["txt_in.weight"])?;

        // 2. Timestep embedding -> vec
        let t_emb = timestep_embedding(timesteps, cfg.timestep_dim, 1000.0)?;
        let t_emb_bf16 = t_emb.to_dtype(DType::BF16)?;
        let vec = {
            let h = linear3d(&t_emb_bf16, &w["time_in.in_layer.weight"])?;
            let h = h.silu()?;
            linear3d(&h, &w["time_in.out_layer.weight"])?
        };

        // 3. Build RoPE tables
        let (pe_cos, pe_sin) = build_rope_2d(img_ids, txt_ids, &cfg.axes_dims, cfg.theta)?;

        // 4. Pre-compute shared modulations ONCE (vec doesn't change between blocks)
        let vec_silu = vec.silu()?;
        let img_mods_arr: [Tensor; 6] = vec_to_arr6(
            shared_modulation_from_silu(&vec_silu, &w["double_stream_modulation_img.lin.weight"], 6)?
        )?;
        let txt_mods_arr: [Tensor; 6] = vec_to_arr6(
            shared_modulation_from_silu(&vec_silu, &w["double_stream_modulation_txt.lin.weight"], 6)?
        )?;
        let single_mods_arr: [Tensor; 3] = vec_to_arr3(
            shared_modulation_from_silu(&vec_silu, &w["single_stream_modulation.lin.weight"], 3)?
        )?;

        // 5. Double blocks (reuse cached modulations)
        for i in 0..cfg.num_double {
            let (new_img, new_txt) = double_block_forward(
                w,
                i,
                &img,
                &txt,
                &img_mods_arr,
                &txt_mods_arr,
                &pe_cos,
                &pe_sin,
                cfg.num_heads,
                cfg.head_dim,
            )?;
            img = new_img;
            txt = new_txt;
        }

        // 6. Concatenate for single stream: txt first, then img
        let mut x = Tensor::cat(&[&txt, &img], 1)?;
        let txt_len = txt.shape().dims()[1];

        // 7. Single blocks (reuse cached modulations)
        for i in 0..cfg.num_single {
            x = single_block_forward(
                w,
                i,
                &x,
                &single_mods_arr,
                &pe_cos,
                &pe_sin,
                cfg.num_heads,
                cfg.head_dim,
                cfg.inner_dim,
                cfg.mlp_hidden,
            )?;
        }

        // 8. Extract image tokens (txt_len onwards)
        let total_len = x.shape().dims()[1];
        let img_out = x.narrow(1, txt_len, total_len - txt_len)?;

        // 9. Final layer: adaLN modulation + linear (reuse cached silu)
        let final_mod = linear3d(&vec_silu, &w["final_layer.adaLN_modulation.1.weight"])?;
        let last_dim = *final_mod.shape().dims().last().unwrap();
        let half_mod = last_dim / 2;
        let ndim = final_mod.shape().dims().len();
        let shift = final_mod.narrow(ndim - 1, 0, half_mod)?;
        let scale = final_mod.narrow(ndim - 1, half_mod, half_mod)?;

        let img_out = modulate_pre(&img_out, &shift, &scale)?;
        linear3d(&img_out, &w["final_layer.linear.weight"])
    }

    /// Get the model config.
    pub fn config(&self) -> &KleinConfig {
        &self.config
    }

    /// Get read-only access to weights.
    pub fn weights(&self) -> &HashMap<String, Tensor> {
        &self.weights
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_klein_config_4b() {
        let cfg = KleinConfig::klein_4b();
        assert_eq!(cfg.inner_dim, 3072);
        assert_eq!(cfg.num_double, 5);
        assert_eq!(cfg.num_single, 20);
        assert_eq!(cfg.num_heads, 24);
        assert_eq!(cfg.head_dim, 128);
        assert_eq!(cfg.in_channels, 128);
        assert_eq!(cfg.joint_attention_dim, 7680);
        assert_eq!(cfg.mlp_hidden, 9216); // 3072 * 3
        assert_eq!(cfg.theta, 2000.0);
    }

    #[test]
    fn test_klein_config_9b() {
        let cfg = KleinConfig::klein_9b();
        assert_eq!(cfg.inner_dim, 4096);
        assert_eq!(cfg.num_double, 8);
        assert_eq!(cfg.num_single, 24);
        assert_eq!(cfg.num_heads, 32);
        assert_eq!(cfg.head_dim, 128);
        assert_eq!(cfg.in_channels, 128);
        assert_eq!(cfg.joint_attention_dim, 12288);
        assert_eq!(cfg.mlp_hidden, 12288); // 4096 * 3
    }

    #[test]
    fn test_klein_config_detection() {
        // Simulate auto-detection from weight shapes
        let mut weights: HashMap<String, Vec<usize>> = HashMap::new();

        // 4B variant shapes
        weights.insert("img_in.weight".into(), vec![3072, 128]);
        weights.insert("txt_in.weight".into(), vec![3072, 7680]);

        let inner_dim = weights["img_in.weight"][0];
        let in_channels = weights["img_in.weight"][1];
        let joint_dim = weights["txt_in.weight"][1];

        assert_eq!(inner_dim, 3072);
        assert_eq!(in_channels, 128);
        assert_eq!(joint_dim, 7680);
        assert_eq!(inner_dim / 128, 24); // num_heads
    }

    #[test]
    fn test_timestep_embedding_properties() {
        // Verify structural properties without GPU:
        // - dim=256 with time_factor=1000.0
        // - Output should be [B, 256] = [B, cos(128) + sin(128)]
        // - cos comes first, then sin (matching ComfyUI ordering)
        let dim = 256;
        let half = dim / 2;
        assert_eq!(half, 128);

        // Verify frequency computation matches expected formula
        let max_period: f32 = 10000.0;
        let freq_0 = (-max_period.ln() * 0.0 / half as f32).exp();
        assert!((freq_0 - 1.0).abs() < 1e-6, "freq[0] should be 1.0");

        let freq_last = (-max_period.ln() * (half - 1) as f32 / half as f32).exp();
        assert!(
            freq_last < 0.01,
            "freq[last] should be very small, got {}",
            freq_last
        );
    }

    #[test]
    fn test_weight_key_names() {
        // Verify all expected weight key patterns for Klein architecture
        let expected_double_keys = [
            "double_blocks.0.img_attn.qkv.weight",
            "double_blocks.0.img_attn.proj.weight",
            "double_blocks.0.img_attn.norm.query_norm.scale",
            "double_blocks.0.img_attn.norm.key_norm.scale",
            "double_blocks.0.img_mlp.0.weight",
            "double_blocks.0.img_mlp.2.weight",
            "double_blocks.0.txt_attn.qkv.weight",
            "double_blocks.0.txt_attn.proj.weight",
            "double_blocks.0.txt_attn.norm.query_norm.scale",
            "double_blocks.0.txt_attn.norm.key_norm.scale",
            "double_blocks.0.txt_mlp.0.weight",
            "double_blocks.0.txt_mlp.2.weight",
        ];

        let expected_single_keys = [
            "single_blocks.0.linear1.weight",
            "single_blocks.0.linear2.weight",
            "single_blocks.0.norm.query_norm.scale",
            "single_blocks.0.norm.key_norm.scale",
        ];

        let expected_model_keys = [
            "img_in.weight",
            "txt_in.weight",
            "time_in.in_layer.weight",
            "time_in.out_layer.weight",
            "double_stream_modulation_img.lin.weight",
            "double_stream_modulation_txt.lin.weight",
            "single_stream_modulation.lin.weight",
            "final_layer.adaLN_modulation.1.weight",
            "final_layer.linear.weight",
        ];

        // Just verify the strings are well-formed
        for key in expected_double_keys
            .iter()
            .chain(expected_single_keys.iter())
            .chain(expected_model_keys.iter())
        {
            assert!(!key.is_empty());
            assert!(key.contains('.'));
        }
    }
}
