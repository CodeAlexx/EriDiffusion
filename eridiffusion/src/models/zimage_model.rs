//! ZImage NextDiT transformer — pure flame_core, key-exact for safetensors.
//!
//! Architecture: Lumina2/NextDiT with joint attention, 3D RoPE, SwiGLU FFN.
//! - 30 main layers, 2 noise refiners, 2 context refiners
//! - dim=3840, 30 heads, head_dim=128
//! - Qwen3 4B text (cap_feat_dim=2560)
//! - adaLN modulation with tanh gates, min_mod=256
//! - Patchify 2x2 -> Linear(64, 3840), NOT Conv2d
//! - Model returns negated velocity: -img
//!
//! Block pattern (matches ComfyUI):
//! - norm1 = PRE-norm (before attention/FFN)
//! - norm2 = POST-norm (after attention/FFN output, before gate+residual)
//! - Modulation applies to ALL tokens (text + image), not just image
//! - Single sequence processing: text+image concatenated before block
//!
//! Rewritten from the ground-truth Python+Flame reference:
//!   flame-core/inference-test/models/zimage_dit.py

use flame_core::cuda_ops_bf16;
use flame_core::serialization;
use flame_core::{DType, Result, Shape, Tensor};
use std::collections::HashMap;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Architecture constants for ZImage NextDiT.
#[derive(Debug, Clone)]
pub struct ZImageConfig {
    pub dim: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub num_layers: usize,
    pub num_noise_refiner: usize,
    pub num_context_refiner: usize,
    pub cap_feat_dim: usize,
    pub mlp_hidden: usize,
    pub min_mod: usize,
    pub t_embedder_hidden: usize,
    pub patch_size: usize,
    pub in_channels: usize,
    pub axes_dims_rope: [usize; 3],
    pub rope_theta: f32,
    pub time_scale: f32,
    pub pad_tokens_multiple: usize,
}

impl Default for ZImageConfig {
    fn default() -> Self {
        Self {
            dim: 3840,
            num_heads: 30,
            head_dim: 128,
            num_layers: 30,
            num_noise_refiner: 2,
            num_context_refiner: 2,
            cap_feat_dim: 2560,
            mlp_hidden: 10240,
            min_mod: 256,
            t_embedder_hidden: 1024,
            patch_size: 2,
            in_channels: 16,
            axes_dims_rope: [32, 48, 48],
            rope_theta: 256.0,
            time_scale: 1000.0,
            pad_tokens_multiple: 32,
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers: 3D linear and element-wise ops
// ---------------------------------------------------------------------------

/// `x @ weight_t` where weight_t is ALREADY pre-transposed [in, out].
fn linear3d(x: &Tensor, weight_t: &Tensor) -> Result<Tensor> {
    let shape = x.shape().dims().to_vec();
    if shape.len() == 2 {
        return x.matmul(weight_t);
    }
    let b = shape[0];
    let n = shape[1];
    let c = shape[2];
    let x_2d = x.reshape(&[b * n, c])?;
    let out_2d = x_2d.matmul(weight_t)?;
    let out_dim = out_2d.shape().dims()[1];
    out_2d.reshape(&[b, n, out_dim])
}

/// `x @ weight_t + bias` for x [B, N, C], weight_t pre-transposed [in, out].
fn linear3d_bias(x: &Tensor, weight_t: &Tensor, bias: &Tensor) -> Result<Tensor> {
    let shape = x.shape().dims().to_vec();
    let b = shape[0];
    let n = shape[1];
    let c = shape[2];
    let x_2d = x.reshape(&[b * n, c])?;
    let out_2d = x_2d.matmul(weight_t)?;
    let out_dim = out_2d.shape().dims()[1];
    let bias_row = bias.reshape(&[1, out_dim])?;
    let out_2d = out_2d.add(&bias_row)?;
    out_2d.reshape(&[b, n, out_dim])
}

/// `x @ weight_t + bias` for x [B, C], weight_t pre-transposed [in, out].
fn linear2d_bias(x: &Tensor, weight_t: &Tensor, bias: &Tensor) -> Result<Tensor> {
    let out = x.matmul(weight_t)?;
    let out_dim = out.shape().dims()[1];
    let bias_row = bias.reshape(&[1, out_dim])?;
    out.add(&bias_row)
}

/// Per-head RMSNorm: x [B, N, H, D] -> flatten to [B*N*H, D], rms_norm, reshape back.
fn head_rms_norm(x: &Tensor, scale: &Tensor) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    let (b, n, h, d) = (dims[0], dims[1], dims[2], dims[3]);
    let flat = x.reshape(&[b * n * h, d])?;
    let normed = cuda_ops_bf16::rms_norm_bf16(&flat, Some(scale), 1e-6)?;
    normed.reshape(&[b, n, h, d])
}

/// RMSNorm for a 2D or 3D tensor on the last dimension.
fn rms_norm(x: &Tensor, weight: &Tensor) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    let last_dim = *dims.last().unwrap();
    let leading: usize = dims.iter().product::<usize>() / last_dim;
    let flat = x.reshape(&[leading, last_dim])?;
    let normed = cuda_ops_bf16::rms_norm_bf16(&flat, Some(weight), 1e-6)?;
    normed.reshape(&dims)
}

/// LayerNorm without affine parameters (for final layer).
fn layer_norm_no_affine(x: &Tensor, dim: usize) -> Result<Tensor> {
    x.layer_norm(&[dim], None, None, 1e-6)
}

// ---------------------------------------------------------------------------
// Timestep embedding
// ---------------------------------------------------------------------------

/// Sinusoidal timestep embedding.
///
/// `t`: [B] values. `dim`: embedding dimension (min_mod = 256).
///
/// Returns [B, dim] in BF16.
fn sinusoidal_embedding(t: &Tensor, dim: usize) -> Result<Tensor> {
    let orig_dtype = t.dtype();
    let device = t.device().clone();
    let b = t.shape().dims()[0];

    let t_f32 = t.to_dtype(DType::F32)?;
    let half = dim / 2;
    let max_period: f32 = 10000.0;

    // freqs = exp(-log(max_period) * arange(0, half) / half)
    let freqs = Tensor::arange(0.0, half as f32, 1.0, device.clone())?;
    let freqs = freqs.mul_scalar(-max_period.ln() / half as f32)?.exp()?;

    // Outer product: [B, 1] * [1, half] -> [B, half]
    let t_col = t_f32.reshape(&[b, 1])?;
    let freqs_row = freqs.reshape(&[1, half])?;
    let args = t_col.mul(&freqs_row)?;

    let cos_part = args.cos()?;
    let sin_part = args.sin()?;
    let emb = Tensor::cat(&[&cos_part, &sin_part], 1)?;

    emb.to_dtype(orig_dtype)
}

/// Timestep embedder: sinusoidal(min_mod) -> MLP(min_mod -> hidden -> min_mod).
///
/// Key structure:
///   t_embedder.mlp.0.weight   (hidden, freq_dim)
///   t_embedder.mlp.0.bias     (hidden,)
///   t_embedder.mlp.2.weight   (freq_dim, hidden)
///   t_embedder.mlp.2.bias     (freq_dim,)
fn timestep_embed(
    t: &Tensor,
    weights: &HashMap<String, Tensor>,
    config: &ZImageConfig,
) -> Result<Tensor> {
    let emb = sinusoidal_embedding(t, config.min_mod)?;

    // MLP: Linear(min_mod, hidden) -> SiLU -> Linear(hidden, min_mod)
    let w0 = &weights["t_embedder.mlp.0.weight"];
    let b0 = &weights["t_embedder.mlp.0.bias"];
    let w2 = &weights["t_embedder.mlp.2.weight"];
    let b2 = &weights["t_embedder.mlp.2.bias"];

    let h = linear2d_bias(&emb, w0, b0)?;
    let h = h.silu()?;
    linear2d_bias(&h, w2, b2)
}

// ---------------------------------------------------------------------------
// 3D RoPE (real-valued rotation, not complex)
// ---------------------------------------------------------------------------

/// Build 3D RoPE cos/sin from position IDs.
///
/// `pos_ids`: [N, 3] — (t, h, w) for each token.
/// `axes_dims`: frequency dimensions per axis, e.g. [32, 48, 48]. Sum = head_dim.
/// `theta`: RoPE base frequency (256.0 for ZImage).
///
/// Returns (cos, sin) each [N, head_dim/2] in F32.
fn build_rope_3d(
    pos_ids: &Tensor,
    axes_dims: &[usize; 3],
    theta: f32,
) -> Result<(Tensor, Tensor)> {
    let device = pos_ids.device().clone();
    let n = pos_ids.shape().dims()[0];

    let mut cos_parts = Vec::new();
    let mut sin_parts = Vec::new();

    for (axis_idx, &axis_dim) in axes_dims.iter().enumerate() {
        let half_dim = axis_dim / 2;

        // Extract positions for this axis: pos_ids[:, axis_idx]
        let positions = pos_ids.narrow(1, axis_idx, 1)?.reshape(&[n])?;
        let positions = positions.to_dtype(DType::F32)?;

        // freqs = 1.0 / (theta ^ (arange(0, half_dim) / half_dim))
        //       = theta ^ (-(arange(0, half_dim) / half_dim))
        //       = exp(-(arange / half_dim) * ln(theta))
        let indices = Tensor::arange(0.0, half_dim as f32, 1.0, device.clone())?;
        let log_theta = (theta as f32).ln();
        let neg_exponents = indices.mul_scalar(-log_theta / half_dim as f32)?;
        let freqs = neg_exponents.exp()?;

        // angles = positions[:, None] * freqs[None, :]
        let pos_col = positions.reshape(&[n, 1])?;
        let freqs_row = freqs.reshape(&[1, half_dim])?;
        let angles = pos_col.mul(&freqs_row)?;

        cos_parts.push(angles.cos()?);
        sin_parts.push(angles.sin()?);
    }

    // Concatenate all axis parts: [N, sum(half_dims)] = [N, head_dim/2]
    let cos_refs: Vec<&Tensor> = cos_parts.iter().collect();
    let sin_refs: Vec<&Tensor> = sin_parts.iter().collect();
    let cos = Tensor::cat(&cos_refs, 1)?;
    let sin = Tensor::cat(&sin_refs, 1)?;

    Ok((cos, sin))
}

/// Apply rotary position embedding via interleaved complex rotation — BF16.
///
/// `x`: [B, N, H, D] in BF16.
/// `cos`, `sin`: [N, D/2] in F32 (converted to BF16 internally).
///
/// Python uses `view_as_complex(x.reshape(..., -1, 2))` which treats consecutive
/// pairs (x[2i], x[2i+1]) as (real, imag). We match this exactly.
///
/// Uses Klein's proven pattern: transpose to [B,H,N,D] (GPU permute [0,2,1,3]),
/// reshape to [BH,N,D], do all ops with dim-0 broadcast only, transpose back.
///
/// Returns [B, N, H, D] in BF16.
fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    let (b, n, h, d) = (dims[0], dims[1], dims[2], dims[3]);
    let half_d = d / 2;
    let bh = b * h;
    let total_flat = bh * n * half_d;

    // Transpose [B, N, H, D] -> [B, H, N, D] (GPU permute [0,2,1,3])
    let x_t = x.permute(&[0, 2, 1, 3])?;

    // Reshape to [BH, N, D/2, 2] to access interleaved pairs
    let x_pairs = x_t.reshape(&[bh, n, half_d, 2])?;
    let x_even = x_pairs.narrow(3, 0, 1)?.squeeze(Some(3))?; // [BH, N, D/2]
    let x_odd = x_pairs.narrow(3, 1, 1)?.squeeze(Some(3))?;  // [BH, N, D/2]

    // cos/sin: [N, D/2] -> [1, N, D/2] for dim-0 broadcast (proven in Klein)
    let cos_flat = cos.to_dtype(DType::BF16)?.reshape(&[1, n, half_d])?;
    let sin_flat = sin.to_dtype(DType::BF16)?.reshape(&[1, n, half_d])?;

    // Complex multiply: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
    let ac = x_even.mul(&cos_flat)?;
    let bd = x_odd.mul(&sin_flat)?;
    let ad = x_even.mul(&sin_flat)?;
    let bc = x_odd.mul(&cos_flat)?;
    let new_even = ac.sub(&bd)?; // [BH, N, D/2]
    let new_odd = ad.add(&bc)?;  // [BH, N, D/2]

    // Interleave even/odd back into [BH, N, D]:
    // Klein trick: flatten to [1, total], cat on dim 0 -> [2, total],
    // 2D transpose -> [total, 2] (GPU kernel), reshape.
    let even_flat = new_even.reshape(&[1, total_flat])?;
    let odd_flat = new_odd.reshape(&[1, total_flat])?;
    let combined = Tensor::cat(&[&even_flat, &odd_flat], 0)?; // [2, total]
    let transposed = combined.permute(&[1, 0])?; // [total, 2] — GPU 2D transpose

    // Reshape to [B, H, N, D] then transpose back to [B, N, H, D]
    let result_bhnd = transposed.reshape(&[b, h, n, d])?;
    result_bhnd.permute(&[0, 2, 1, 3]) // [B, N, H, D]
}

// ---------------------------------------------------------------------------
// SwiGLU feed-forward
// ---------------------------------------------------------------------------

/// SwiGLU FFN: w2(silu(w1(x)) * w3(x))
///
/// Key structure (prefix = "{block_prefix}.feed_forward"):
///   .w1.weight   (mlp_hidden, dim)
///   .w2.weight   (dim, mlp_hidden)
///   .w3.weight   (mlp_hidden, dim)
fn swiglu(
    x: &Tensor,
    w1: &Tensor,
    w2: &Tensor,
    w3: &Tensor,
) -> Result<Tensor> {
    let h1 = linear3d(x, w1)?;
    let h3 = linear3d(x, w3)?;
    let h1_act = h1.silu()?;
    let gated = h1_act.mul(&h3)?;
    linear3d(&gated, w2)
}

// ---------------------------------------------------------------------------
// Joint Attention
// ---------------------------------------------------------------------------

/// Self-attention with fused QKV and RMSNorm QK normalization.
///
/// Key structure (prefix = "{block_prefix}.attention"):
///   .qkv.weight      (3*dim, dim)
///   .out.weight       (dim, dim)
///   .q_norm.weight    (head_dim,)
///   .k_norm.weight    (head_dim,)
fn joint_attention(
    x: &Tensor,
    weights: &HashMap<String, Tensor>,
    prefix: &str,
    config: &ZImageConfig,
    rope_cos: &Tensor,
    rope_sin: &Tensor,
) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    let b = dims[0];
    let seq_len = dims[1];

    let qkv_w = &weights[&format!("{prefix}.attention.qkv.weight")];
    let out_w = &weights[&format!("{prefix}.attention.out.weight")];
    let q_norm_w = &weights[&format!("{prefix}.attention.q_norm.weight")];
    let k_norm_w = &weights[&format!("{prefix}.attention.k_norm.weight")];

    // QKV projection: [B, N, dim] -> [B, N, 3*dim]
    let qkv = linear3d(x, qkv_w)?;

    // Split into Q, K, V: each [B, N, dim]
    let dim = config.dim;
    let q = qkv.narrow(2, 0, dim)?;
    let k = qkv.narrow(2, dim, dim)?;
    let v = qkv.narrow(2, dim * 2, dim)?;

    // Reshape to [B, N, H, D]
    let h = config.num_heads;
    let d = config.head_dim;
    let q = q.reshape(&[b, seq_len, h, d])?;
    let k = k.reshape(&[b, seq_len, h, d])?;
    let v = v.reshape(&[b, seq_len, h, d])?;

    // Per-head RMSNorm on Q and K
    let q = head_rms_norm(&q, q_norm_w)?;
    let k = head_rms_norm(&k, k_norm_w)?;

    // Apply RoPE
    let q = apply_rope(&q, rope_cos, rope_sin)?;
    let k = apply_rope(&k, rope_cos, rope_sin)?;

    // Transpose to [B, H, N, D] for SDPA
    let q = q.transpose_dims(1, 2)?;
    let k = k.transpose_dims(1, 2)?;
    let v = v.transpose_dims(1, 2)?;

    // Scaled dot-product attention
    let attn_out = flame_core::sdpa::forward(&q, &k, &v, None)?;

    // Transpose back to [B, N, H, D] and reshape to [B, N, dim]
    let attn_out = attn_out.transpose_dims(1, 2)?;
    let attn_out = attn_out.reshape(&[b, seq_len, h * d])?;

    // Output projection
    linear3d(&attn_out, out_w)
}

// ---------------------------------------------------------------------------
// Transformer Block
// ---------------------------------------------------------------------------

/// JointTransformerBlock forward pass.
///
/// Key structure for conditioned blocks (prefix = "layers.N" or "noise_refiner.N"):
///   {prefix}.adaLN_modulation.0.weight   (4*dim, min_mod)
///   {prefix}.adaLN_modulation.0.bias     (4*dim,)
///   {prefix}.attention_norm1.weight      (dim,)
///   {prefix}.attention_norm2.weight      (dim,)
///   {prefix}.attention.qkv.weight        (3*dim, dim)
///   {prefix}.attention.out.weight        (dim, dim)
///   {prefix}.attention.q_norm.weight     (head_dim,)
///   {prefix}.attention.k_norm.weight     (head_dim,)
///   {prefix}.ffn_norm1.weight            (dim,)
///   {prefix}.ffn_norm2.weight            (dim,)
///   {prefix}.feed_forward.w1.weight      (mlp_hidden, dim)
///   {prefix}.feed_forward.w2.weight      (dim, mlp_hidden)
///   {prefix}.feed_forward.w3.weight      (mlp_hidden, dim)
///
/// Unconditioned blocks (context_refiner) have no adaLN_modulation.
fn transformer_block(
    x: &Tensor,
    weights: &HashMap<String, Tensor>,
    prefix: &str,
    config: &ZImageConfig,
    rope_cos: &Tensor,
    rope_sin: &Tensor,
    t_cond: Option<&Tensor>,
    conditioned: bool,
) -> Result<Tensor> {
    let dim = config.dim;

    // Compute modulation if conditioned
    let (scale_msa, gate_msa, scale_mlp, gate_mlp) = if conditioned {
        if let Some(t) = t_cond {
            let mod_w = &weights[&format!("{prefix}.adaLN_modulation.0.weight")];
            let mod_b = &weights[&format!("{prefix}.adaLN_modulation.0.bias")];
            // t_cond: [B, min_mod] -> [B, 4*dim]
            let mod_out = linear2d_bias(t, mod_w, mod_b)?;
            // Split into 4 chunks of [B, dim]
            let s_msa = mod_out.narrow(1, 0, dim)?;
            let g_msa = mod_out.narrow(1, dim, dim)?;
            let s_mlp = mod_out.narrow(1, dim * 2, dim)?;
            let g_mlp = mod_out.narrow(1, dim * 3, dim)?;
            (Some(s_msa), Some(g_msa), Some(s_mlp), Some(g_mlp))
        } else {
            (None, None, None, None)
        }
    } else {
        (None, None, None, None)
    };

    // --- Attention path ---
    // Pre-norm (RMSNorm)
    let attn_norm1_w = &weights[&format!("{prefix}.attention_norm1.weight")];
    let mut x_norm = rms_norm(x, attn_norm1_w)?;

    // Modulate: x_norm * (1 + scale_msa)
    if let Some(ref scale) = scale_msa {
        let scale_u = scale.unsqueeze(1)?; // [B, 1, dim]
        let one_plus_scale = scale_u.add_scalar(1.0)?;
        x_norm = x_norm.mul(&one_plus_scale)?;
    }

    // Attention
    let attn_out = joint_attention(&x_norm, weights, prefix, config, rope_cos, rope_sin)?;

    // Post-norm (RMSNorm)
    let attn_norm2_w = &weights[&format!("{prefix}.attention_norm2.weight")];
    let attn_out = rms_norm(&attn_out, attn_norm2_w)?;

    // Gate + residual: x = x + tanh(gate_msa) * attn_out
    let mut x = if let Some(ref gate) = gate_msa {
        let gate_u = gate.unsqueeze(1)?; // [B, 1, dim]
        let gate_tanh = gate_u.tanh()?;
        let gated = gate_tanh.mul(&attn_out)?;
        x.add(&gated)?
    } else {
        x.add(&attn_out)?
    };

    // --- FFN path ---
    // Pre-norm (RMSNorm)
    let ffn_norm1_w = &weights[&format!("{prefix}.ffn_norm1.weight")];
    let mut x_norm = rms_norm(&x, ffn_norm1_w)?;

    // Modulate: x_norm * (1 + scale_mlp)
    if let Some(ref scale) = scale_mlp {
        let scale_u = scale.unsqueeze(1)?;
        let one_plus_scale = scale_u.add_scalar(1.0)?;
        x_norm = x_norm.mul(&one_plus_scale)?;
    }

    // SwiGLU FFN
    let w1 = &weights[&format!("{prefix}.feed_forward.w1.weight")];
    let w2 = &weights[&format!("{prefix}.feed_forward.w2.weight")];
    let w3 = &weights[&format!("{prefix}.feed_forward.w3.weight")];
    let ff_out = swiglu(&x_norm, w1, w2, w3)?;

    // Post-norm (RMSNorm)
    let ffn_norm2_w = &weights[&format!("{prefix}.ffn_norm2.weight")];
    let ff_out = rms_norm(&ff_out, ffn_norm2_w)?;

    // Gate + residual: x = x + tanh(gate_mlp) * ff_out
    x = if let Some(ref gate) = gate_mlp {
        let gate_u = gate.unsqueeze(1)?;
        let gate_tanh = gate_u.tanh()?;
        let gated = gate_tanh.mul(&ff_out)?;
        x.add(&gated)?
    } else {
        x.add(&ff_out)?
    };

    Ok(x)
}

// ---------------------------------------------------------------------------
// Final Layer
// ---------------------------------------------------------------------------

/// Final layer: adaLN scale-only modulation + linear projection.
///
/// Key structure:
///   final_layer.adaLN_modulation.1.weight   (dim, min_mod)
///   final_layer.adaLN_modulation.1.bias     (dim,)
///   final_layer.linear.weight               (patch_dim, dim)
///   final_layer.linear.bias                 (patch_dim,)
fn final_layer(
    x: &Tensor,
    t_cond: &Tensor,
    weights: &HashMap<String, Tensor>,
    config: &ZImageConfig,
) -> Result<Tensor> {
    let dim = config.dim;

    // adaLN modulation: SiLU -> Linear
    let mod_w = &weights["final_layer.adaLN_modulation.1.weight"];
    let mod_b = &weights["final_layer.adaLN_modulation.1.bias"];
    let t_silu = t_cond.silu()?;
    let scale = linear2d_bias(&t_silu, mod_w, mod_b)?; // [B, dim]
    let scale_u = scale.unsqueeze(1)?; // [B, 1, dim]

    // LayerNorm (no affine) then scale: x * (1 + scale)
    let x_normed = layer_norm_no_affine(x, dim)?;
    let one_plus_scale = scale_u.add_scalar(1.0)?;
    let x_mod = x_normed.mul(&one_plus_scale)?;

    // Linear projection to patch_dim
    let lin_w = &weights["final_layer.linear.weight"];
    let lin_b = &weights["final_layer.linear.bias"];
    linear3d_bias(&x_mod, lin_w, lin_b)
}

// ---------------------------------------------------------------------------
// Patchify / Unpatchify
// ---------------------------------------------------------------------------

/// (B, C, H, W) -> (B, N, patch_dim) where N = (H/p)*(W/p), patch_dim = p*p*C.
/// Returns (patches, ph, pw).
///
/// Uses fused GPU kernel — no 6D permute, no F32 round-trip, stays in BF16.
fn patchify(x: &Tensor, patch_size: usize) -> Result<(Tensor, usize, usize)> {
    flame_core::bf16_elementwise::patchify_bf16(x, patch_size)
}

/// (B, N, patch_dim) -> (B, C, H, W).
///
/// Uses fused GPU kernel — no 6D permute, no F32 round-trip, stays in BF16.
fn unpatchify(
    x: &Tensor,
    ph: usize,
    pw: usize,
    patch_size: usize,
    in_channels: usize,
) -> Result<Tensor> {
    flame_core::bf16_elementwise::unpatchify_bf16(x, ph, pw, patch_size, in_channels)
}

/// Pad token sequence to next multiple. Returns (padded, pad_len).
fn pad_to_multiple(
    tokens: &Tensor,
    pad_token: &Tensor,
    multiple: usize,
) -> Result<(Tensor, usize)> {
    let dims = tokens.shape().dims().to_vec();
    let b = dims[0];
    let seq_len = dims[1];
    let dim = dims[2];

    let remainder = seq_len % multiple;
    let pad_len = if remainder == 0 { 0 } else { multiple - remainder };

    if pad_len == 0 {
        return Ok((tokens.clone(), 0));
    }

    // pad_token: [1, dim] -> [B, pad_len, dim]
    let pad = pad_token
        .reshape(&[1, 1, dim])?
        .broadcast_to(&Shape::from_dims(&[b, pad_len, dim]))?;

    let padded = Tensor::cat(&[tokens, &pad], 1)?;
    Ok((padded, pad_len))
}

// ---------------------------------------------------------------------------
// Position ID construction
// ---------------------------------------------------------------------------

/// Build position IDs for caption + image tokens.
///
/// Caption: t=1..cap_len, h=0, w=0.
/// Image: t=cap_len+1, h=0..ph-1, w=0..pw-1.
/// Image padding: zeros.
fn build_position_ids(
    cap_len: usize,
    ph: usize,
    pw: usize,
    img_pad_len: usize,
    device: Arc<flame_core::CudaDevice>,
) -> Result<Tensor> {
    let img_len = ph * pw;
    let total = cap_len + img_len + img_pad_len;

    // Build on CPU as F32, then upload
    let mut pos_data = vec![0.0f32; total * 3];

    // Caption positions: (t=1..cap_len, h=0, w=0)
    for i in 0..cap_len {
        pos_data[i * 3] = (i + 1) as f32; // t
        // h and w stay 0
    }

    // Image positions: (t=cap_len+1, h=row, w=col)
    let t_val = (cap_len + 1) as f32;
    for row in 0..ph {
        for col in 0..pw {
            let idx = cap_len + row * pw + col;
            pos_data[idx * 3] = t_val;
            pos_data[idx * 3 + 1] = row as f32;
            pos_data[idx * 3 + 2] = col as f32;
        }
    }

    // Image padding positions: all zeros (already zeroed)

    Tensor::from_vec_dtype(
        pos_data,
        Shape::from_dims(&[total, 3]),
        device,
        DType::F32,
    )
}

// ---------------------------------------------------------------------------
// Main Transformer
// ---------------------------------------------------------------------------

/// ZImage NextDiT transformer — pure flame_core implementation.
///
/// Exact key-compatible with ZImage .safetensors checkpoints.
pub struct ZImageTransformer {
    weights: HashMap<String, Tensor>,
    config: ZImageConfig,
}

impl ZImageTransformer {
    /// Load from a safetensors checkpoint.
    ///
    /// Auto-detects architecture from weight shapes.
    pub fn from_safetensors(path: &str) -> Result<Self> {
        let device = flame_core::global_cuda_device();
        let weights = serialization::load_file(path, &device)?;
        Self::from_weights(weights)
    }

    /// Construct from a pre-loaded weight dict.
    ///
    /// Auto-detects config from weight shapes.
    pub fn from_weights(weights: HashMap<String, Tensor>) -> Result<Self> {
        // Auto-detect from weight shapes
        let dim = weights["x_embedder.weight"].shape().dims()[0];
        let patch_dim = weights["x_embedder.weight"].shape().dims()[1];
        let cap_feat_dim = weights["cap_embedder.1.weight"].shape().dims()[1];
        let mlp_hidden = weights["layers.0.feed_forward.w1.weight"].shape().dims()[0];
        let min_mod = weights["layers.0.adaLN_modulation.0.weight"].shape().dims()[1];
        let t_hidden = weights["t_embedder.mlp.0.weight"].shape().dims()[0];
        let head_dim = weights["layers.0.attention.q_norm.weight"].shape().dims()[0];
        let num_heads = dim / head_dim;

        // Count layers
        let mut num_layers = 0;
        while weights.contains_key(&format!("layers.{num_layers}.attention.qkv.weight")) {
            num_layers += 1;
        }
        let mut num_noise_refiner = 0;
        while weights
            .contains_key(&format!("noise_refiner.{num_noise_refiner}.attention.qkv.weight"))
        {
            num_noise_refiner += 1;
        }
        let mut num_context_refiner = 0;
        while weights.contains_key(&format!(
            "context_refiner.{num_context_refiner}.attention.qkv.weight"
        )) {
            num_context_refiner += 1;
        }

        let in_channels = 16;
        let patch_size = ((patch_dim as f32 / in_channels as f32).sqrt()) as usize;

        // Auto-detect RoPE params based on model dim
        let (axes_dims_rope, rope_theta, time_scale) = if dim == 3840 {
            // ZImage
            ([32usize, 48, 48], 256.0f32, 1000.0f32)
        } else if dim == 2304 {
            // Lumina2
            ([32usize, 32, 32], 10000.0f32, 1.0f32)
        } else {
            ([32usize, 48, 48], 256.0f32, 1000.0f32)
        };

        let config = ZImageConfig {
            dim,
            num_heads,
            head_dim,
            num_layers,
            num_noise_refiner,
            num_context_refiner,
            cap_feat_dim,
            mlp_hidden,
            min_mod,
            t_embedder_hidden: t_hidden,
            patch_size,
            in_channels,
            axes_dims_rope,
            rope_theta,
            time_scale,
            pad_tokens_multiple: 32,
        };

        log::info!(
            "ZImage loaded: dim={}, heads={}, layers={}, noise_refiner={}, context_refiner={}, \
             cap_feat_dim={}, mlp_hidden={}, min_mod={}, patch_size={}, keys={}",
            dim,
            num_heads,
            num_layers,
            num_noise_refiner,
            num_context_refiner,
            cap_feat_dim,
            mlp_hidden,
            min_mod,
            patch_size,
            weights.len(),
        );

        // Pre-transpose all 2D weight matrices [out, in] -> [in, out] for faster matmul.
        log::info!("[ZImage] Pre-transposing weights...");
        let mut weights = weights;
        let keys: Vec<String> = weights.keys().cloned().collect();
        for key in &keys {
            if key.ends_with(".weight") && !key.contains("norm") {
                let w = &weights[key];
                let dims = w.shape().dims();
                if dims.len() == 2 {
                    let wt = flame_core::bf16_elementwise::transpose2d_bf16(w)?;
                    weights.insert(key.clone(), wt);
                }
            }
        }
        log::info!("[ZImage] Weights pre-transposed.");

        Ok(Self { weights, config })
    }

    /// Get the model config.
    pub fn config(&self) -> &ZImageConfig {
        &self.config
    }

    /// Get a reference to the weight map.
    pub fn weights(&self) -> &HashMap<String, Tensor> {
        &self.weights
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// * `x` - Spatial latents [B, C, H, W] in BF16.
    /// * `timestep` - Sigma values [B] in [0, 1].
    /// * `cap_feats` - Text features [B, seq, cap_feat_dim] from Qwen3.
    ///
    /// # Returns
    /// Negated velocity [B, C, H, W] in BF16.
    pub fn forward(
        &self,
        x: &Tensor,
        timestep: &Tensor,
        cap_feats: &Tensor,
    ) -> Result<Tensor> {
        let config = &self.config;
        let weights = &self.weights;
        let b = x.shape().dims()[0];
        let pad_mult = config.pad_tokens_multiple;

        // Invert timestep (ComfyUI convention) and scale
        let t = timestep.neg()?.add_scalar(1.0)?; // 1.0 - timestep
        let t = t.mul_scalar(config.time_scale)?;
        let t_cond = timestep_embed(&t, weights, config)?; // [B, min_mod]

        // Patchify and embed image -> [B, N_img, dim]
        let (patches, ph, pw) = patchify(x, config.patch_size)?;
        let x_emb_w = &weights["x_embedder.weight"];
        let x_emb_b = &weights["x_embedder.bias"];
        let img_tokens = linear3d_bias(&patches, x_emb_w, x_emb_b)?;
        let img_len = img_tokens.shape().dims()[1];

        // Embed captions: RMSNorm(cap_feat_dim) -> Linear(cap_feat_dim, dim)
        let cap_norm_w = &weights["cap_embedder.0.weight"];
        let cap_proj_w = &weights["cap_embedder.1.weight"];
        let cap_proj_b = &weights["cap_embedder.1.bias"];
        let cap_normed = rms_norm(cap_feats, cap_norm_w)?;
        let cap_tokens = linear3d_bias(&cap_normed, cap_proj_w, cap_proj_b)?;

        // Pad caption to multiple of pad_tokens_multiple
        let cap_pad_token = &weights["cap_pad_token"]; // [1, dim]
        let (cap_tokens, _cap_pad_len) =
            pad_to_multiple(&cap_tokens, cap_pad_token, pad_mult)?;
        let cap_len = cap_tokens.shape().dims()[1];

        // Pad image to multiple of pad_tokens_multiple
        let x_pad_token = &weights["x_pad_token"]; // [1, dim]
        let (img_tokens, img_pad_len) =
            pad_to_multiple(&img_tokens, x_pad_token, pad_mult)?;

        // Build position IDs and RoPE
        let device = x.device().clone();
        let pos_ids = build_position_ids(cap_len, ph, pw, img_pad_len, device)?;
        let (rope_cos, rope_sin) =
            build_rope_3d(&pos_ids, &config.axes_dims_rope, config.rope_theta)?;

        // Split RoPE into caption and image portions
        let rope_cos_cap = rope_cos.narrow(0, 0, cap_len)?;
        let rope_sin_cap = rope_sin.narrow(0, 0, cap_len)?;
        let rope_cos_img = rope_cos.narrow(0, cap_len, img_len + img_pad_len)?;
        let rope_sin_img = rope_sin.narrow(0, cap_len, img_len + img_pad_len)?;

        // Context refiner: text self-attention (unconditioned)
        let mut c = cap_tokens;
        for i in 0..config.num_context_refiner {
            let prefix = format!("context_refiner.{i}");
            c = transformer_block(
                &c,
                weights,
                &prefix,
                config,
                &rope_cos_cap,
                &rope_sin_cap,
                None,
                false,
            )?;
        }

        // Noise refiner: image-only self-attention (conditioned)
        let mut img = img_tokens;
        for i in 0..config.num_noise_refiner {
            let prefix = format!("noise_refiner.{i}");
            img = transformer_block(
                &img,
                weights,
                &prefix,
                config,
                &rope_cos_img,
                &rope_sin_img,
                Some(&t_cond),
                true,
            )?;
        }

        // Concatenate text + image for main layers
        let mut xc = Tensor::cat(&[&c, &img], 1)?;

        // Main transformer layers: joint text+image attention
        for i in 0..config.num_layers {
            let prefix = format!("layers.{i}");
            xc = transformer_block(
                &xc,
                weights,
                &prefix,
                config,
                &rope_cos,
                &rope_sin,
                Some(&t_cond),
                true,
            )?;
        }

        // Extract image tokens (skip text, remove padding)
        let img_out = xc.narrow(1, cap_len, img_len)?;

        // Final layer
        let img_out = final_layer(&img_out, &t_cond, weights, config)?;

        // Unpatchify -> [B, C, H, W]
        let img_out = unpatchify(&img_out, ph, pw, config.patch_size, config.in_channels)?;

        // ZImage convention: return negated velocity
        img_out.neg()
    }
}
