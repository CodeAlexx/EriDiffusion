//! LTX-2.3 (Lightricks 18.88B) Video+Audio Transformer — pure flame_core.
//!
//! Architecture matches `diffusers.LTX2VideoTransformer3DModel` exactly.
//! Key-exact for Lightricks/LTX-2 safetensors checkpoints.
//!
//! This is a dual-stream (video + audio) transformer with:
//! - Video stream: inner_dim = num_heads * head_dim (default 32 * 128 = 4096)
//! - Audio stream: audio_inner_dim = audio_heads * audio_head_dim (default 32 * 64 = 2048)
//! - 48 transformer blocks with self-attn, cross-attn, a2v/v2a cross-attn, FFN
//! - RoPE positional embeddings (3D for video, 1D for audio)
//! - AdaLN-Single timestep conditioning with 6 mod params per block
//! - GELU-approximate FeedForward (4x expansion)
//! - RMSNorm QK normalization across heads
//!
//! At 18.88B parameters, this model REQUIRES block-level CPU offloading
//! (Stagehand-style) for inference on 24GB GPUs.
//!
//! Weight key format (diffusers):
//!   proj_in.{weight,bias}
//!   audio_proj_in.{weight,bias}
//!   caption_projection.linear_{1,2}.{weight,bias}
//!   audio_caption_projection.linear_{1,2}.{weight,bias}
//!   time_embed.emb.timestep_embedder.linear_{1,2}.{weight,bias}
//!   time_embed.linear.{weight,bias}
//!   audio_time_embed.emb.timestep_embedder.linear_{1,2}.{weight,bias}
//!   audio_time_embed.linear.{weight,bias}
//!   av_cross_attn_video_scale_shift.{emb,linear}.* (similarly for other cross attn mods)
//!   scale_shift_table, audio_scale_shift_table
//!   transformer_blocks.{i}.norm1.weight (RMSNorm, no affine by default)
//!   transformer_blocks.{i}.attn1.{to_q,to_k,to_v}.{weight,bias}
//!   transformer_blocks.{i}.attn1.{norm_q,norm_k}.weight
//!   transformer_blocks.{i}.attn1.to_out.0.{weight,bias}
//!   transformer_blocks.{i}.ff.net.0.proj.{weight,bias}   (GELU-approximate)
//!   transformer_blocks.{i}.ff.net.2.{weight,bias}        (output projection)
//!   transformer_blocks.{i}.scale_shift_table              [6, dim]
//!   ... (audio_* mirrors for each block)
//!   norm_out.{weight,bias} (LayerNorm, no affine)
//!   proj_out.{weight,bias}
//!   audio_norm_out.{weight,bias}
//!   audio_proj_out.{weight,bias}

use flame_core::serialization;
use flame_core::{DType, Result, Shape, Tensor};
use std::collections::HashMap;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Architecture config for LTX-2 variants.
#[derive(Debug, Clone)]
pub struct LTX2Config {
    // Video stream
    pub in_channels: usize,
    pub out_channels: usize,
    pub patch_size: usize,
    pub patch_size_t: usize,
    pub num_attention_heads: usize,
    pub attention_head_dim: usize,
    pub cross_attention_dim: usize,

    // Audio stream
    pub audio_in_channels: usize,
    pub audio_out_channels: usize,
    pub audio_patch_size: usize,
    pub audio_patch_size_t: usize,
    pub audio_num_attention_heads: usize,
    pub audio_attention_head_dim: usize,
    pub audio_cross_attention_dim: usize,
    pub audio_scale_factor: usize,
    pub audio_sampling_rate: usize,
    pub audio_hop_length: usize,

    // Shared
    pub num_layers: usize,
    pub caption_channels: usize,
    pub norm_eps: f32,
    pub attention_bias: bool,

    // RoPE
    pub vae_scale_factors: [usize; 3],
    pub pos_embed_max_pos: usize,
    pub base_height: usize,
    pub base_width: usize,
    pub rope_theta: f64,
    pub causal_offset: usize,
    pub timestep_scale_multiplier: f64,
    pub cross_attn_timestep_scale_multiplier: f64,
}

impl Default for LTX2Config {
    fn default() -> Self {
        Self {
            in_channels: 128,
            out_channels: 128,
            patch_size: 1,
            patch_size_t: 1,
            num_attention_heads: 32,
            attention_head_dim: 128,
            cross_attention_dim: 4096,

            audio_in_channels: 128,
            audio_out_channels: 128,
            audio_patch_size: 1,
            audio_patch_size_t: 1,
            audio_num_attention_heads: 32,
            audio_attention_head_dim: 64,
            audio_cross_attention_dim: 2048,
            audio_scale_factor: 4,
            audio_sampling_rate: 16000,
            audio_hop_length: 160,

            num_layers: 48,
            caption_channels: 3840,
            norm_eps: 1e-6,
            attention_bias: true,

            vae_scale_factors: [8, 32, 32],
            pos_embed_max_pos: 20,
            base_height: 2048,
            base_width: 2048,
            rope_theta: 10000.0,
            causal_offset: 1,
            timestep_scale_multiplier: 1000.0,
            cross_attn_timestep_scale_multiplier: 1000.0,
        }
    }
}

impl LTX2Config {
    /// Video inner dimension: num_heads * head_dim
    pub fn inner_dim(&self) -> usize {
        self.num_attention_heads * self.attention_head_dim
    }

    /// Audio inner dimension: audio_heads * audio_head_dim
    pub fn audio_inner_dim(&self) -> usize {
        self.audio_num_attention_heads * self.audio_attention_head_dim
    }

    /// FFN hidden dimension (4x expansion, GELU-approximate)
    pub fn ffn_hidden(&self) -> usize {
        self.inner_dim() * 4
    }

    /// Audio FFN hidden dimension
    pub fn audio_ffn_hidden(&self) -> usize {
        self.audio_inner_dim() * 4
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute `x @ weight.T + bias` for x=[B, N, C], weight=[out, C], bias=[out].
fn linear3d(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
    let shape = x.shape().dims().to_vec();
    let (b, n, c) = if shape.len() == 3 {
        (shape[0], shape[1], shape[2])
    } else if shape.len() == 2 {
        (1, shape[0], shape[1])
    } else {
        return Err(flame_core::Error::InvalidShape(format!(
            "linear3d expects rank 2 or 3, got {:?}",
            shape
        )));
    };

    let x_2d = x.reshape(&[b * n, c])?;
    let out_2d = matmul_weight_t(&x_2d, weight)?;
    let out_dim = out_2d.shape().dims()[1];
    let mut result = out_2d.reshape(&[b, n, out_dim])?;

    if let Some(b_tensor) = bias {
        result = result.add(&b_tensor.reshape(&[1, 1, out_dim])?)?;
    }
    Ok(result)
}

/// Compute `x @ weight.T` for 2D tensors. BF16-accelerated when available.
fn matmul_weight_t(x: &Tensor, weight: &Tensor) -> Result<Tensor> {
    // weight is [out_features, in_features], we need x @ weight.T
    let wt = weight.transpose()?;
    x.matmul(&wt)
}

/// SiLU activation: x * sigmoid(x)
fn silu(x: &Tensor) -> Result<Tensor> {
    x.silu()
}

/// GELU (tanh approximation): 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
fn gelu_approximate(x: &Tensor) -> Result<Tensor> {
    x.gelu()
}

/// RMSNorm: x / sqrt(mean(x^2) + eps), optionally scaled by weight
fn rms_norm(x: &Tensor, weight: Option<&Tensor>, eps: f32) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    let last_dim = dims[dims.len() - 1];

    // Compute in F32 for stability
    let x_f32 = x.to_dtype(DType::F32)?;
    let x_sq = x_f32.mul(&x_f32)?;
    let mean_sq = x_sq.mean_along_dims(&[dims.len() - 1], true)?;

    // Reshape mean to match input for broadcast
    let mut bc_shape = dims.clone();
    bc_shape[dims.len() - 1] = 1;
    let mean_sq = mean_sq.reshape(&bc_shape)?;

    let rsqrt = mean_sq.add_scalar(eps)?.rsqrt()?;
    let normed = x_f32.mul(&rsqrt)?.to_dtype(DType::BF16)?;

    if let Some(w) = weight {
        let w_bc = w.reshape(&vec![1; dims.len() - 1].into_iter().chain(std::iter::once(last_dim)).collect::<Vec<_>>())?;
        normed.mul(&w_bc)
    } else {
        Ok(normed)
    }
}

/// LayerNorm without affine parameters: (x - mean) / sqrt(var + eps)
fn layer_norm_no_affine(x: &Tensor, eps: f32) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    let rank = dims.len();

    let x_f32 = x.to_dtype(DType::F32)?;

    // Construct broadcast shape: [..., 1]
    let mut bc_shape = dims.clone();
    bc_shape[rank - 1] = 1;

    let mean = x_f32.mean_along_dims(&[rank - 1], true)?.reshape(&bc_shape)?;
    let centered = x_f32.sub(&mean)?;
    let var = centered.mul(&centered)?.mean_along_dims(&[rank - 1], true)?.reshape(&bc_shape)?;
    let rstd = var.add_scalar(eps)?.rsqrt()?;
    centered.mul(&rstd)?.to_dtype(DType::BF16)
}

// ---------------------------------------------------------------------------
// Sinusoidal Timestep Embedding
// ---------------------------------------------------------------------------

/// Get sinusoidal timestep embedding, matching diffusers' `get_timestep_embedding`
/// with flip_sin_to_cos=True, downscale_freq_shift=0.
fn timestep_embedding(timesteps: &Tensor, dim: usize) -> Result<Tensor> {
    let half_dim = dim / 2;
    let device = timesteps.device().clone();

    // Build frequencies: exp(-i * log(10000) / half_dim)
    let mut freq_data = vec![0.0f32; half_dim];
    for i in 0..half_dim {
        freq_data[i] = (-(i as f32) * (10000f32.ln()) / (half_dim as f32)).exp();
    }
    let freqs = Tensor::from_vec(freq_data, Shape::from_dims(&[1, half_dim]), device)?;

    // timesteps: [B] -> [B, 1]
    let t = timesteps.to_dtype(DType::F32)?.unsqueeze(1)?;

    // [B, 1] * [1, half_dim] -> [B, half_dim]
    let args = t.mul(&freqs)?;

    let cos_part = args.cos()?.to_dtype(DType::BF16)?;
    let sin_part = args.sin()?.to_dtype(DType::BF16)?;

    // flip_sin_to_cos=True -> [cos, sin]
    Tensor::cat(&[&cos_part, &sin_part], 1)
}

// ---------------------------------------------------------------------------
// Rotary Position Embedding (3D Video / 1D Audio)
// ---------------------------------------------------------------------------

/// Compute interleaved RoPE frequencies for video (3D) or audio (1D) coordinates.
///
/// Returns (cos_freqs, sin_freqs) tensors suitable for apply_rotary_emb.
///
/// `coords` shape: [B, num_dims, num_patches, 2] where last dim is [start, end)
/// For video: num_dims=3 (frame, height, width)
/// For audio: num_dims=1 (temporal)
pub fn compute_rope_frequencies(
    coords: &Tensor,
    dim: usize,
    max_positions: &[f64],  // base normalization for each dim
    theta: f64,
    num_heads: usize,       // for split-rope head reshaping
) -> Result<(Tensor, Tensor)> {
    let device = coords.device().clone();
    let cdims = coords.shape().dims().to_vec();
    let batch_size = cdims[0];
    let num_pos_dims = cdims[1];
    let num_patches = cdims[2];

    // Midpoint of [start, end) boundaries
    // coords: [B, num_pos_dims, num_patches, 2]
    let coords_f32 = coords.to_dtype(DType::F32)?;
    let starts = coords_f32.narrow(3, 0, 1)?;  // [B, D, P, 1]
    let ends = coords_f32.narrow(3, 1, 1)?;    // [B, D, P, 1]
    let midpoints = starts.add(&ends)?.mul_scalar(0.5)?;
    let midpoints = midpoints.squeeze_dim(3)?;  // [B, D, P]

    // Normalize to fraction of base shape
    // grid[d] = midpoints[:, d, :] / max_positions[d]
    // Then stack -> [B, P, num_pos_dims]
    let mut grid_parts = Vec::with_capacity(num_pos_dims);
    for d in 0..num_pos_dims {
        let dim_slice = midpoints.narrow(1, d, 1)?.squeeze_dim(1)?; // [B, P]
        let normed = dim_slice.mul_scalar(1.0 / max_positions[d] as f32)?;
        grid_parts.push(normed);
    }

    // Stack along last dim: [B, P, num_pos_dims]
    // We'll do this by unsqueezing each to [B, P, 1] and cat
    let mut grid_expanded = Vec::new();
    for part in &grid_parts {
        grid_expanded.push(part.unsqueeze(2)?);
    }
    let grid = Tensor::cat(&grid_expanded.iter().collect::<Vec<_>>(), 2)?; // [B, P, num_pos_dims]

    // Number of RoPE elements per dimension
    let num_rope_elems = num_pos_dims * 2;

    // Frequency vector: theta^(linspace(0, 1, dim // num_rope_elems)) * pi / 2
    let freq_count = dim / num_rope_elems;
    let mut freq_data = Vec::with_capacity(freq_count);
    for i in 0..freq_count {
        let t = i as f64 / (freq_count as f64 - 1.0).max(1.0);
        let val = theta.powf(t) * PI / 2.0;
        freq_data.push(val as f32);
    }
    let freqs_vec = Tensor::from_vec(
        freq_data,
        Shape::from_dims(&[1, 1, 1, freq_count]),
        device.clone(),
    )?; // [1, 1, 1, freq_count]

    // grid: [B, P, num_pos_dims] -> [B, P, num_pos_dims, 1]
    let grid_4d = grid.unsqueeze(3)?;

    // (grid * 2 - 1) * freqs -> [B, P, num_pos_dims, freq_count]
    let scaled = grid_4d.mul_scalar(2.0)?.add_scalar(-1.0)?;
    let angles = scaled.mul(&freqs_vec.expand(&[batch_size, num_patches, num_pos_dims, freq_count])?)?;

    // Transpose and flatten: [B, P, freq_count, num_pos_dims] -> [B, P, freq_count * num_pos_dims]
    let angles_t = angles.permute(&[0, 1, 3, 2])?;
    let half_dim_rope = freq_count * num_pos_dims;
    let angles_flat = angles_t.reshape(&[batch_size, num_patches, half_dim_rope])?;

    // Interleaved cos/sin with repeat_interleave(2)
    let cos_raw = angles_flat.cos()?.to_dtype(DType::BF16)?;
    let sin_raw = angles_flat.sin()?.to_dtype(DType::BF16)?;

    // repeat_interleave(2, dim=-1): each value duplicated
    // [B, P, half] -> [B, P, half*2]
    let cos_freqs = repeat_interleave_last(&cos_raw, 2)?;
    let sin_freqs = repeat_interleave_last(&sin_raw, 2)?;

    // Pad if dim % num_rope_elems != 0
    let current_size = cos_freqs.shape().dims()[2];
    if current_size < dim {
        let pad_size = dim - current_size;
        let cos_pad = Tensor::ones_dtype(
            Shape::from_dims(&[batch_size, num_patches, pad_size]),
            DType::BF16,
            device.clone(),
        )?;
        let sin_pad = Tensor::zeros_dtype(
            Shape::from_dims(&[batch_size, num_patches, pad_size]),
            DType::BF16,
            device.clone(),
        )?;
        let cos_freqs = Tensor::cat(&[&cos_pad, &cos_freqs], 2)?;
        let sin_freqs = Tensor::cat(&[&sin_pad, &sin_freqs], 2)?;
        return Ok((cos_freqs, sin_freqs));
    }

    Ok((cos_freqs, sin_freqs))
}

/// Repeat each element along the last dimension `n` times.
/// [B, S, D] -> [B, S, D*n]
fn repeat_interleave_last(x: &Tensor, n: usize) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    let rank = dims.len();
    let last = dims[rank - 1];

    // Unsqueeze last dim: [..., D, 1], expand to [..., D, n], flatten
    let expanded = x.unsqueeze(rank)?; // [..., D, 1]
    let mut target_shape = dims.clone();
    target_shape.push(n);
    let expanded = expanded.expand(&target_shape)?;

    // Flatten last two dims
    let mut out_shape = dims[..rank - 1].to_vec();
    out_shape.push(last * n);
    expanded.reshape(&out_shape)
}

/// Apply interleaved rotary embedding to query/key tensor.
/// x: [B, S, inner_dim], freqs: (cos [B, S, inner_dim], sin [B, S, inner_dim])
pub fn apply_rotary_emb(x: &Tensor, cos_freqs: &Tensor, sin_freqs: &Tensor) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    let last = dims[dims.len() - 1];

    // Split into pairs: x_real, x_imag from interleaved [r0, i0, r1, i1, ...]
    // x -> [B, S, D/2, 2] -> unbind last -> x_real, x_imag
    let mut shape_pairs = dims[..dims.len() - 1].to_vec();
    shape_pairs.push(last / 2);
    shape_pairs.push(2);
    let x_pairs = x.reshape(&shape_pairs)?;

    let x_real = x_pairs.narrow(dims.len(), 0, 1)?.squeeze_dim(dims.len())?; // [B, S, D/2]
    let x_imag = x_pairs.narrow(dims.len(), 1, 1)?.squeeze_dim(dims.len())?; // [B, S, D/2]

    // Rotated: [-x_imag, x_real] interleaved
    let neg_imag = x_imag.mul_scalar(-1.0)?;
    // Stack [-x_imag, x_real] along last dim and flatten
    let neg_imag_u = neg_imag.unsqueeze(dims.len())?;
    let x_real_u = x_real.unsqueeze(dims.len())?;
    let rotated = Tensor::cat(&[&neg_imag_u, &x_real_u], dims.len())?;
    let mut rot_shape = dims[..dims.len() - 1].to_vec();
    rot_shape.push(last);
    let x_rotated = rotated.reshape(&rot_shape)?;

    // out = x * cos + x_rotated * sin
    let x_f32 = x.to_dtype(DType::F32)?;
    let cos_f32 = cos_freqs.to_dtype(DType::F32)?;
    let sin_f32 = sin_freqs.to_dtype(DType::F32)?;
    let rot_f32 = x_rotated.to_dtype(DType::F32)?;

    let out = x_f32.mul(&cos_f32)?.add(&rot_f32.mul(&sin_f32)?)?;
    out.to_dtype(DType::BF16)
}

// ---------------------------------------------------------------------------
// Sub-modules
// ---------------------------------------------------------------------------

/// Timestep embedding MLP: sinusoidal -> linear -> SiLU -> linear
struct TimestepEmbedderMLP {
    linear_1_weight: Tensor,
    linear_1_bias: Tensor,
    linear_2_weight: Tensor,
    linear_2_bias: Tensor,
}

impl TimestepEmbedderMLP {
    fn forward(&self, timesteps: &Tensor) -> Result<Tensor> {
        // Sinusoidal embedding (256 channels)
        let emb = timestep_embedding(timesteps, 256)?;

        // MLP: linear -> silu -> linear
        let h = linear3d(&emb, &self.linear_1_weight, Some(&self.linear_1_bias))?;
        let h = silu(&h)?;
        linear3d(&h, &self.linear_2_weight, Some(&self.linear_2_bias))
    }
}

/// AdaLN-Single: timestep_embedding -> silu -> linear (produces num_mod_params * dim modulation values)
struct AdaLayerNormSingle {
    /// TimestepEmbedderMLP inside `emb`
    emb: TimestepEmbedderMLP,
    /// Modulation projection: linear(dim -> num_mod_params * dim)
    linear_weight: Tensor,
    linear_bias: Tensor,
    num_mod_params: usize,
}

impl AdaLayerNormSingle {
    /// Forward: returns (mod_params, embedded_timestep)
    fn forward(&self, timestep: &Tensor) -> Result<(Tensor, Tensor)> {
        let embedded = self.emb.forward(timestep)?;
        let h = silu(&embedded)?;
        let mod_params = linear3d(&h, &self.linear_weight, Some(&self.linear_bias))?;
        Ok((mod_params, embedded))
    }
}

/// Caption projection (PixArtAlphaTextProjection): linear -> GELU(tanh) -> linear
struct CaptionProjection {
    linear_1_weight: Tensor,
    linear_1_bias: Tensor,
    linear_2_weight: Tensor,
    linear_2_bias: Tensor,
}

impl CaptionProjection {
    fn forward(&self, caption: &Tensor) -> Result<Tensor> {
        let h = linear3d(caption, &self.linear_1_weight, Some(&self.linear_1_bias))?;
        let h = gelu_approximate(&h)?;
        linear3d(&h, &self.linear_2_weight, Some(&self.linear_2_bias))
    }
}

/// GELU-approximate FeedForward: GELU(linear(x)) -> linear -> output
struct FeedForward {
    /// net.0.proj: linear -> gelu
    gelu_proj_weight: Tensor,
    gelu_proj_bias: Tensor,
    /// net.2: output projection
    out_weight: Tensor,
    out_bias: Tensor,
}

impl FeedForward {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = linear3d(x, &self.gelu_proj_weight, Some(&self.gelu_proj_bias))?;
        let h = gelu_approximate(&h)?;
        linear3d(&h, &self.out_weight, Some(&self.out_bias))
    }
}

/// LTX2 Attention block: Q/K/V projection, RMSNorm on Q/K, optional RoPE, SDPA, output projection.
struct LTX2Attention {
    to_q_weight: Tensor,
    to_q_bias: Tensor,
    to_k_weight: Tensor,
    to_k_bias: Tensor,
    to_v_weight: Tensor,
    to_v_bias: Tensor,
    norm_q_weight: Tensor,  // RMSNorm across heads
    norm_k_weight: Tensor,
    to_out_weight: Tensor,  // to_out.0
    to_out_bias: Tensor,
    num_heads: usize,
    head_dim: usize,
    eps: f32,
}

impl LTX2Attention {
    /// Forward with optional separate query/key RoPE.
    fn forward(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        query_rope: Option<(&Tensor, &Tensor)>,
        key_rope: Option<(&Tensor, &Tensor)>,
    ) -> Result<Tensor> {
        let kv_input = encoder_hidden_states.unwrap_or(hidden_states);

        let q = linear3d(hidden_states, &self.to_q_weight, Some(&self.to_q_bias))?;
        let k = linear3d(kv_input, &self.to_k_weight, Some(&self.to_k_bias))?;
        let v = linear3d(kv_input, &self.to_v_weight, Some(&self.to_v_bias))?;

        // RMSNorm across heads on Q and K
        let q = rms_norm(&q, Some(&self.norm_q_weight), self.eps)?;
        let k = rms_norm(&k, Some(&self.norm_k_weight), self.eps)?;

        // Apply rotary embeddings
        let q = if let Some((cos, sin)) = query_rope {
            apply_rotary_emb(&q, cos, sin)?
        } else {
            q
        };
        let k = if let Some((cos, sin)) = key_rope.or(query_rope) {
            apply_rotary_emb(&k, cos, sin)?
        } else {
            k
        };

        // Reshape to [B, num_heads, S, head_dim] for SDPA
        let q_dims = q.shape().dims().to_vec();
        let (b, s_q) = (q_dims[0], q_dims[1]);
        let k_dims = k.shape().dims().to_vec();
        let s_kv = k_dims[1];

        let q = q.reshape(&[b, s_q, self.num_heads, self.head_dim])?.permute(&[0, 2, 1, 3])?;
        let k = k.reshape(&[b, s_kv, self.num_heads, self.head_dim])?.permute(&[0, 2, 1, 3])?;
        let v = v.reshape(&[b, s_kv, self.num_heads, self.head_dim])?.permute(&[0, 2, 1, 3])?;

        // Scaled dot-product attention
        let attn_out = flame_core::sdpa::forward(&q, &k, &v, attention_mask)?;

        // Reshape back: [B, heads, S, head_dim] -> [B, S, inner_dim]
        let attn_out = attn_out.permute(&[0, 2, 1, 3])?;
        let inner_dim = self.num_heads * self.head_dim;
        let attn_out = attn_out.reshape(&[b, s_q, inner_dim])?;

        // Output projection
        linear3d(&attn_out, &self.to_out_weight, Some(&self.to_out_bias))
    }
}

/// A single LTX2 transformer block: dual-stream (video + audio) with
/// self-attention, cross-attention, a2v/v2a cross-attention, and FFN.
pub struct LTX2TransformerBlock {
    // Video self-attention
    norm1_weight: Option<Tensor>,  // RMSNorm (may be None if no elementwise_affine)
    attn1: LTX2Attention,

    // Audio self-attention
    audio_norm1_weight: Option<Tensor>,
    audio_attn1: LTX2Attention,

    // Video cross-attention (with text)
    norm2_weight: Option<Tensor>,
    attn2: LTX2Attention,

    // Audio cross-attention (with text)
    audio_norm2_weight: Option<Tensor>,
    audio_attn2: LTX2Attention,

    // Audio-to-Video cross-attention: Q=video, KV=audio
    audio_to_video_norm_weight: Option<Tensor>,
    audio_to_video_attn: LTX2Attention,

    // Video-to-Audio cross-attention: Q=audio, KV=video
    video_to_audio_norm_weight: Option<Tensor>,
    video_to_audio_attn: LTX2Attention,

    // Video FFN
    norm3_weight: Option<Tensor>,
    ff: FeedForward,

    // Audio FFN
    audio_norm3_weight: Option<Tensor>,
    audio_ff: FeedForward,

    // AdaLN-Zero modulation tables
    scale_shift_table: Tensor,                    // [6, dim] for video
    audio_scale_shift_table: Tensor,              // [6, audio_dim]
    video_a2v_cross_attn_scale_shift_table: Tensor, // [5, dim]
    audio_a2v_cross_attn_scale_shift_table: Tensor, // [5, audio_dim]

    eps: f32,
}

impl LTX2TransformerBlock {
    /// Video-only forward: self-attn → cross-attn (text) → FFN.
    /// Skips all audio paths, A2V/V2A cross-attention.
    pub fn forward_video_only(
        &self,
        hidden_states: &Tensor,        // [B, N, dim]
        encoder_hidden_states: &Tensor, // [B, seq, dim]
        temb: &Tensor,                 // [B, N, 6*dim]
        video_rotary_emb: Option<(&Tensor, &Tensor)>,
        encoder_attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let b = hidden_states.shape().dims()[0];
        let dim = hidden_states.shape().dims()[2];

        // Extract 6 AdaLN-Zero params: [shift_sa, scale_sa, gate_sa, shift_ff, scale_ff, gate_ff]
        let (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) =
            self.compute_ada_params(&self.scale_shift_table, temb, b, dim)?;

        // 1. Self-Attention with AdaLN-Zero
        let norm_h = rms_norm(hidden_states, self.norm1_weight.as_ref(), self.eps)?;
        let mod_h = norm_h.mul(&scale_msa.add_scalar(1.0)?)?.add(&shift_msa)?;
        let attn_out = self.attn1.forward(&mod_h, None, None, video_rotary_emb, None)?;
        let mut hs = hidden_states.add(&attn_out.mul(&gate_msa)?)?;

        // 2. Cross-Attention (text)
        let norm_h2 = rms_norm(&hs, self.norm2_weight.as_ref(), self.eps)?;
        let ca_out = self.attn2.forward(
            &norm_h2, Some(encoder_hidden_states), encoder_attention_mask, None, None,
        )?;
        hs = hs.add(&ca_out)?;

        // 3. FeedForward with AdaLN-Zero
        let norm_ff = rms_norm(&hs, self.norm3_weight.as_ref(), self.eps)?;
        let mod_ff = norm_ff.mul(&scale_mlp.add_scalar(1.0)?)?.add(&shift_mlp)?;
        let ff_out = self.ff.forward(&mod_ff)?;
        hs = hs.add(&ff_out.mul(&gate_mlp)?)?;

        Ok(hs)
    }

    /// Forward pass for one block. Returns (video_hidden, audio_hidden).
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        audio_hidden_states: &Tensor,
        encoder_hidden_states: &Tensor,
        audio_encoder_hidden_states: &Tensor,
        temb: &Tensor,                    // [B, 1, 6*dim]
        temb_audio: &Tensor,              // [B, 1, 6*audio_dim]
        temb_ca_scale_shift: &Tensor,     // [B, 1, 4*dim]
        temb_ca_audio_scale_shift: &Tensor, // [B, 1, 4*audio_dim]
        temb_ca_gate: &Tensor,            // [B, 1, dim]
        temb_ca_audio_gate: &Tensor,      // [B, 1, audio_dim]
        video_rotary_emb: Option<(&Tensor, &Tensor)>,
        audio_rotary_emb: Option<(&Tensor, &Tensor)>,
        ca_video_rotary_emb: Option<(&Tensor, &Tensor)>,
        ca_audio_rotary_emb: Option<(&Tensor, &Tensor)>,
        encoder_attention_mask: Option<&Tensor>,
        audio_encoder_attention_mask: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let b = hidden_states.shape().dims()[0];
        let video_dim = hidden_states.shape().dims()[2];
        let audio_dim = audio_hidden_states.shape().dims()[2];

        // ---- 1. Video Self-Attention with AdaLN-Zero ----
        let norm_h = rms_norm(hidden_states, self.norm1_weight.as_ref(), self.eps)?;

        // Compute 6 modulation params from scale_shift_table + temb
        let (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) =
            self.compute_ada_params(&self.scale_shift_table, temb, b, video_dim)?;

        // Modulate: norm * (1 + scale) + shift
        let mod_h = norm_h.mul(&scale_msa.add_scalar(1.0)?)?.add(&shift_msa)?;
        let attn_out = self.attn1.forward(&mod_h, None, None, video_rotary_emb, None)?;
        let mut hidden_states = hidden_states.add(&attn_out.mul(&gate_msa)?)?;

        // ---- Audio Self-Attention with AdaLN-Zero ----
        let norm_a = rms_norm(audio_hidden_states, self.audio_norm1_weight.as_ref(), self.eps)?;
        let (a_shift_msa, a_scale_msa, a_gate_msa, a_shift_mlp, a_scale_mlp, a_gate_mlp) =
            self.compute_ada_params(&self.audio_scale_shift_table, temb_audio, b, audio_dim)?;
        let mod_a = norm_a.mul(&a_scale_msa.add_scalar(1.0)?)?.add(&a_shift_msa)?;
        let attn_a_out = self.audio_attn1.forward(&mod_a, None, None, audio_rotary_emb, None)?;
        let mut audio_hidden_states = audio_hidden_states.add(&attn_a_out.mul(&a_gate_msa)?)?;

        // ---- 2. Video/Audio Cross-Attention with text ----
        let norm_h2 = rms_norm(&hidden_states, self.norm2_weight.as_ref(), self.eps)?;
        let ca_out = self.attn2.forward(&norm_h2, Some(encoder_hidden_states), encoder_attention_mask, None, None)?;
        hidden_states = hidden_states.add(&ca_out)?;

        let norm_a2 = rms_norm(&audio_hidden_states, self.audio_norm2_weight.as_ref(), self.eps)?;
        let ca_a_out = self.audio_attn2.forward(&norm_a2, Some(audio_encoder_hidden_states), audio_encoder_attention_mask, None, None)?;
        audio_hidden_states = audio_hidden_states.add(&ca_a_out)?;

        // ---- 3. Audio-to-Video / Video-to-Audio Cross-Attention ----
        let norm_a2v = rms_norm(&hidden_states, self.audio_to_video_norm_weight.as_ref(), self.eps)?;
        let norm_v2a = rms_norm(&audio_hidden_states, self.video_to_audio_norm_weight.as_ref(), self.eps)?;

        // Compute per-layer cross-attention modulation
        let (a2v_gate, v2a_gate, video_a2v_mod, video_v2a_mod, audio_a2v_mod, audio_v2a_mod) =
            self.compute_cross_attn_params(
                temb_ca_scale_shift, temb_ca_audio_scale_shift,
                temb_ca_gate, temb_ca_audio_gate,
                b, video_dim, audio_dim,
            )?;

        // A2V: Q=video, KV=audio
        let mod_video_a2v = norm_a2v.mul(&video_a2v_mod.0.add_scalar(1.0)?)?.add(&video_a2v_mod.1)?;
        let mod_audio_a2v = norm_v2a.mul(&audio_a2v_mod.0.add_scalar(1.0)?)?.add(&audio_a2v_mod.1)?;

        let a2v_out = self.audio_to_video_attn.forward(
            &mod_video_a2v, Some(&mod_audio_a2v), None,
            ca_video_rotary_emb, ca_audio_rotary_emb,
        )?;
        hidden_states = hidden_states.add(&a2v_out.mul(&a2v_gate)?)?;

        // V2A: Q=audio, KV=video
        let mod_video_v2a = norm_a2v.mul(&video_v2a_mod.0.add_scalar(1.0)?)?.add(&video_v2a_mod.1)?;
        let mod_audio_v2a = norm_v2a.mul(&audio_v2a_mod.0.add_scalar(1.0)?)?.add(&audio_v2a_mod.1)?;

        let v2a_out = self.video_to_audio_attn.forward(
            &mod_audio_v2a, Some(&mod_video_v2a), None,
            ca_audio_rotary_emb, ca_video_rotary_emb,
        )?;
        audio_hidden_states = audio_hidden_states.add(&v2a_out.mul(&v2a_gate)?)?;

        // ---- 4. FeedForward ----
        let norm_ff = rms_norm(&hidden_states, self.norm3_weight.as_ref(), self.eps)?;
        let mod_ff = norm_ff.mul(&scale_mlp.add_scalar(1.0)?)?.add(&shift_mlp)?;
        let ff_out = self.ff.forward(&mod_ff)?;
        hidden_states = hidden_states.add(&ff_out.mul(&gate_mlp)?)?;

        let norm_aff = rms_norm(&audio_hidden_states, self.audio_norm3_weight.as_ref(), self.eps)?;
        let mod_aff = norm_aff.mul(&a_scale_mlp.add_scalar(1.0)?)?.add(&a_shift_mlp)?;
        let aff_out = self.audio_ff.forward(&mod_aff)?;
        audio_hidden_states = audio_hidden_states.add(&aff_out.mul(&a_gate_mlp)?)?;

        Ok((hidden_states, audio_hidden_states))
    }

    /// Compute 6 AdaLN-Zero modulation parameters from scale_shift_table + temb.
    /// Returns (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp).
    fn compute_ada_params(
        &self,
        table: &Tensor,  // [6, dim]
        temb: &Tensor,   // [B, 1, 6*dim]
        batch_size: usize,
        dim: usize,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)> {
        // table[None, None] + temb.reshape(B, 1, 6, dim)
        let table_bc = table.unsqueeze(0)?.unsqueeze(0)?; // [1, 1, 6, dim]
        let temb_4d = temb.reshape(&[batch_size, 1, 6, dim])?;
        let ada = table_bc.add(&temb_4d)?; // [B, 1, 6, dim]

        // Unbind dim=2 -> 6 tensors of [B, 1, dim]
        let shift_msa = ada.narrow(2, 0, 1)?.squeeze_dim(2)?;
        let scale_msa = ada.narrow(2, 1, 1)?.squeeze_dim(2)?;
        let gate_msa = ada.narrow(2, 2, 1)?.squeeze_dim(2)?;
        let shift_mlp = ada.narrow(2, 3, 1)?.squeeze_dim(2)?;
        let scale_mlp = ada.narrow(2, 4, 1)?.squeeze_dim(2)?;
        let gate_mlp = ada.narrow(2, 5, 1)?.squeeze_dim(2)?;

        Ok((shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp))
    }

    /// Compute cross-attention modulation parameters from per-layer tables + global temb.
    /// Returns (a2v_gate, v2a_gate, (video_a2v_scale, video_a2v_shift),
    ///          (video_v2a_scale, video_v2a_shift), (audio_a2v_scale, audio_a2v_shift),
    ///          (audio_v2a_scale, audio_v2a_shift))
    #[allow(clippy::type_complexity)]
    fn compute_cross_attn_params(
        &self,
        temb_ca_ss: &Tensor,           // [B, 1, 4*video_dim]
        temb_ca_audio_ss: &Tensor,     // [B, 1, 4*audio_dim]
        temb_ca_gate: &Tensor,         // [B, 1, video_dim]
        temb_ca_audio_gate: &Tensor,   // [B, 1, audio_dim]
        b: usize,
        video_dim: usize,
        audio_dim: usize,
    ) -> Result<(
        Tensor,                  // a2v_gate
        Tensor,                  // v2a_gate
        (Tensor, Tensor),       // (video_a2v_scale, video_a2v_shift)
        (Tensor, Tensor),       // (video_v2a_scale, video_v2a_shift)
        (Tensor, Tensor),       // (audio_a2v_scale, audio_a2v_shift)
        (Tensor, Tensor),       // (audio_v2a_scale, audio_v2a_shift)
    )> {
        // Video per-layer: first 4 rows = scale/shift, 5th row = gate contribution
        let v_table_ss = self.video_a2v_cross_attn_scale_shift_table.narrow(0, 0, 4)?; // [4, dim]
        let v_table_gate = self.video_a2v_cross_attn_scale_shift_table.narrow(0, 4, 1)?; // [1, dim]

        // Combine global + per-layer video cross-attn scale/shift
        let v_ss_bc = v_table_ss.unsqueeze(0)?; // [1, 4, dim]
        let temb_ss_4d = temb_ca_ss.reshape(&[b, 1, 4, video_dim])?;
        let v_combined = v_ss_bc.to_dtype(temb_ss_4d.dtype())?.add(&temb_ss_4d)?;

        let video_a2v_scale = v_combined.narrow(2, 0, 1)?.squeeze_dim(2)?;
        let video_a2v_shift = v_combined.narrow(2, 1, 1)?.squeeze_dim(2)?;
        let video_v2a_scale = v_combined.narrow(2, 2, 1)?.squeeze_dim(2)?;
        let video_v2a_shift = v_combined.narrow(2, 3, 1)?.squeeze_dim(2)?;

        // a2v gate: per_layer + global
        let v_gate_bc = v_table_gate.unsqueeze(0)?;
        let temb_gate_4d = temb_ca_gate.reshape(&[b, 1, 1, video_dim])?;
        let a2v_gate = v_gate_bc.to_dtype(temb_gate_4d.dtype())?.add(&temb_gate_4d)?.squeeze_dim(2)?;

        // Audio per-layer
        let a_table_ss = self.audio_a2v_cross_attn_scale_shift_table.narrow(0, 0, 4)?;
        let a_table_gate = self.audio_a2v_cross_attn_scale_shift_table.narrow(0, 4, 1)?;

        let a_ss_bc = a_table_ss.unsqueeze(0)?;
        let temb_a_ss_4d = temb_ca_audio_ss.reshape(&[b, 1, 4, audio_dim])?;
        let a_combined = a_ss_bc.to_dtype(temb_a_ss_4d.dtype())?.add(&temb_a_ss_4d)?;

        let audio_a2v_scale = a_combined.narrow(2, 0, 1)?.squeeze_dim(2)?;
        let audio_a2v_shift = a_combined.narrow(2, 1, 1)?.squeeze_dim(2)?;
        let audio_v2a_scale = a_combined.narrow(2, 2, 1)?.squeeze_dim(2)?;
        let audio_v2a_shift = a_combined.narrow(2, 3, 1)?.squeeze_dim(2)?;

        let a_gate_bc = a_table_gate.unsqueeze(0)?;
        let temb_a_gate_4d = temb_ca_audio_gate.reshape(&[b, 1, 1, audio_dim])?;
        let v2a_gate = a_gate_bc.to_dtype(temb_a_gate_4d.dtype())?.add(&temb_a_gate_4d)?.squeeze_dim(2)?;

        Ok((
            a2v_gate, v2a_gate,
            (video_a2v_scale, video_a2v_shift),
            (video_v2a_scale, video_v2a_shift),
            (audio_a2v_scale, audio_a2v_shift),
            (audio_v2a_scale, audio_v2a_shift),
        ))
    }
}

// ---------------------------------------------------------------------------
// Top-level model
// ---------------------------------------------------------------------------

/// LTX-2.3 Video+Audio Transformer (18.88B parameters).
///
/// This is the main model struct. Blocks are stored as a Vec for
/// Stagehand-style block-level CPU offloading during inference.
pub struct LTX2Model {
    pub config: LTX2Config,

    // Patchification
    proj_in_weight: Tensor,
    proj_in_bias: Tensor,
    audio_proj_in_weight: Tensor,
    audio_proj_in_bias: Tensor,

    // Caption projections
    caption_projection: CaptionProjection,
    audio_caption_projection: CaptionProjection,

    // Timestep embeddings and modulation
    time_embed: AdaLayerNormSingle,
    audio_time_embed: AdaLayerNormSingle,

    // Cross-attention global modulation
    av_cross_attn_video_scale_shift: AdaLayerNormSingle,
    av_cross_attn_audio_scale_shift: AdaLayerNormSingle,
    av_cross_attn_video_a2v_gate: AdaLayerNormSingle,
    av_cross_attn_audio_v2a_gate: AdaLayerNormSingle,

    // Output scale/shift tables
    scale_shift_table: Tensor,       // [2, inner_dim]
    audio_scale_shift_table: Tensor, // [2, audio_inner_dim]

    // Transformer blocks (stored separately for block-level offloading)
    pub blocks: Vec<LTX2TransformerBlock>,

    // Output layers
    // norm_out: LayerNorm (no affine)
    proj_out_weight: Tensor,
    proj_out_bias: Tensor,
    // audio_norm_out: LayerNorm (no affine)
    audio_proj_out_weight: Tensor,
    audio_proj_out_bias: Tensor,
}

impl LTX2Model {
    /// Total number of transformer blocks.
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// * `hidden_states` - [B, num_video_tokens, in_channels] patchified video latents
    /// * `audio_hidden_states` - [B, num_audio_tokens, audio_in_channels] patchified audio latents
    /// * `encoder_hidden_states` - [B, text_seq_len, caption_channels] text embeddings for video
    /// * `audio_encoder_hidden_states` - [B, text_seq_len, caption_channels] text embeddings for audio
    /// * `timestep` - [B, num_video_tokens] scaled timestep
    /// * `audio_timestep` - [B, num_audio_tokens] scaled timestep (or same as timestep)
    /// * `video_coords` - [B, 3, num_video_tokens, 2] RoPE coordinate bounds
    /// * `audio_coords` - [B, 1, num_audio_tokens, 2] RoPE coordinate bounds
    /// * `encoder_attention_mask` - optional [B, text_seq_len]
    /// * `audio_encoder_attention_mask` - optional [B, text_seq_len]
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        audio_hidden_states: &Tensor,
        encoder_hidden_states: &Tensor,
        audio_encoder_hidden_states: &Tensor,
        timestep: &Tensor,
        audio_timestep: &Tensor,
        video_coords: &Tensor,
        audio_coords: &Tensor,
        encoder_attention_mask: Option<&Tensor>,
        audio_encoder_attention_mask: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let batch_size = hidden_states.shape().dims()[0];
        let inner_dim = self.config.inner_dim();
        let audio_inner_dim = self.config.audio_inner_dim();

        // 1. Compute RoPE positional embeddings
        let video_max_pos = [
            self.config.pos_embed_max_pos as f64,
            self.config.base_height as f64,
            self.config.base_width as f64,
        ];
        let audio_max_pos = [self.config.pos_embed_max_pos as f64];

        let (v_cos, v_sin) = compute_rope_frequencies(
            video_coords, inner_dim, &video_max_pos,
            self.config.rope_theta, self.config.num_attention_heads,
        )?;
        let (a_cos, a_sin) = compute_rope_frequencies(
            audio_coords, audio_inner_dim, &audio_max_pos,
            self.config.rope_theta, self.config.audio_num_attention_heads,
        )?;

        // Cross-attention RoPE: temporal only (first dim of video coords)
        let video_temporal_coords = video_coords.narrow(1, 0, 1)?; // [B, 1, P, 2]
        let ca_audio_dim = self.config.audio_cross_attention_dim;
        let ca_video_max_pos = [self.config.pos_embed_max_pos as f64];

        let (ca_v_cos, ca_v_sin) = compute_rope_frequencies(
            &video_temporal_coords, ca_audio_dim, &ca_video_max_pos,
            self.config.rope_theta, self.config.num_attention_heads,
        )?;
        let audio_temporal_coords = audio_coords.narrow(1, 0, 1)?;
        let (ca_a_cos, ca_a_sin) = compute_rope_frequencies(
            &audio_temporal_coords, ca_audio_dim, &ca_video_max_pos,
            self.config.rope_theta, self.config.audio_num_attention_heads,
        )?;

        // 2. Convert encoder attention masks to bias
        let enc_mask = encoder_attention_mask.map(|m| -> Result<Tensor> {
            let m_f = m.to_dtype(DType::BF16)?;
            let one = Tensor::ones_dtype(m_f.shape().clone(), DType::BF16, m_f.device().clone())?;
            let inverted = one.sub(&m_f)?;
            let biased = inverted.mul_scalar(-10000.0)?;
            biased.unsqueeze(1)
        }).transpose()?;

        let audio_enc_mask = audio_encoder_attention_mask.map(|m| -> Result<Tensor> {
            let m_f = m.to_dtype(DType::BF16)?;
            let one = Tensor::ones_dtype(m_f.shape().clone(), DType::BF16, m_f.device().clone())?;
            let inverted = one.sub(&m_f)?;
            let biased = inverted.mul_scalar(-10000.0)?;
            biased.unsqueeze(1)
        }).transpose()?;

        // 3. Patchify input projections
        let mut hs = linear3d(hidden_states, &self.proj_in_weight, Some(&self.proj_in_bias))?;
        let mut ahs = linear3d(audio_hidden_states, &self.audio_proj_in_weight, Some(&self.audio_proj_in_bias))?;

        // 4. Timestep embeddings and global modulation params
        let ts_flat = timestep.reshape(&[batch_size * timestep.shape().dims()[1]])?;
        let (temb, embedded_ts) = self.time_embed.forward(&ts_flat)?;
        let temb = temb.reshape(&[batch_size, 1, temb.shape().dims()[temb.shape().rank() - 1]])?;
        let embedded_ts = embedded_ts.reshape(&[batch_size, 1, embedded_ts.shape().dims()[embedded_ts.shape().rank() - 1]])?;

        let ats_flat = audio_timestep.reshape(&[batch_size * audio_timestep.shape().dims()[1]])?;
        let (temb_audio, audio_embedded_ts) = self.audio_time_embed.forward(&ats_flat)?;
        let temb_audio = temb_audio.reshape(&[batch_size, 1, temb_audio.shape().dims()[temb_audio.shape().rank() - 1]])?;
        let audio_embedded_ts = audio_embedded_ts.reshape(&[batch_size, 1, audio_embedded_ts.shape().dims()[audio_embedded_ts.shape().rank() - 1]])?;

        // 4.2 Global cross-attention modulation params
        let cross_gate_scale = (self.config.cross_attn_timestep_scale_multiplier
            / self.config.timestep_scale_multiplier) as f32;

        let (v_ca_ss, _) = self.av_cross_attn_video_scale_shift.forward(&ts_flat)?;
        let v_ca_ss = v_ca_ss.reshape(&[batch_size, 1, v_ca_ss.shape().dims()[v_ca_ss.shape().rank() - 1]])?;

        let scaled_ts = ts_flat.mul_scalar(cross_gate_scale)?;
        let (v_ca_gate, _) = self.av_cross_attn_video_a2v_gate.forward(&scaled_ts)?;
        let v_ca_gate = v_ca_gate.reshape(&[batch_size, 1, v_ca_gate.shape().dims()[v_ca_gate.shape().rank() - 1]])?;

        let (a_ca_ss, _) = self.av_cross_attn_audio_scale_shift.forward(&ats_flat)?;
        let a_ca_ss = a_ca_ss.reshape(&[batch_size, 1, a_ca_ss.shape().dims()[a_ca_ss.shape().rank() - 1]])?;

        let scaled_ats = ats_flat.mul_scalar(cross_gate_scale)?;
        let (a_ca_gate, _) = self.av_cross_attn_audio_v2a_gate.forward(&scaled_ats)?;
        let a_ca_gate = a_ca_gate.reshape(&[batch_size, 1, a_ca_gate.shape().dims()[a_ca_gate.shape().rank() - 1]])?;

        // 5. Prepare prompt embeddings
        let enc_hs = self.caption_projection.forward(encoder_hidden_states)?;
        let enc_hs = enc_hs.reshape(&[batch_size, enc_hs.shape().dims()[1], inner_dim])?;

        let audio_enc_hs = self.audio_caption_projection.forward(audio_encoder_hidden_states)?;
        let audio_enc_hs = audio_enc_hs.reshape(&[batch_size, audio_enc_hs.shape().dims()[1], audio_inner_dim])?;

        // 6. Run transformer blocks
        for block in &self.blocks {
            let (new_hs, new_ahs) = block.forward(
                &hs, &ahs,
                &enc_hs, &audio_enc_hs,
                &temb, &temb_audio,
                &v_ca_ss, &a_ca_ss,
                &v_ca_gate, &a_ca_gate,
                Some((&v_cos, &v_sin)),
                Some((&a_cos, &a_sin)),
                Some((&ca_v_cos, &ca_v_sin)),
                Some((&ca_a_cos, &ca_a_sin)),
                enc_mask.as_ref(),
                audio_enc_mask.as_ref(),
            )?;
            hs = new_hs;
            ahs = new_ahs;
        }

        // 7. Output layers
        // Video: norm -> scale/shift -> proj_out
        let shift_scale = self.scale_shift_table.unsqueeze(0)?.unsqueeze(0)?; // [1, 1, 2, dim]
        let embedded_ts_4d = embedded_ts.unsqueeze(2)?; // [B, 1, 1, dim]
        let final_ss = shift_scale.add(&embedded_ts_4d)?;
        let shift = final_ss.narrow(2, 0, 1)?.squeeze_dim(2)?;
        let scale = final_ss.narrow(2, 1, 1)?.squeeze_dim(2)?;

        let normed = layer_norm_no_affine(&hs, 1e-6)?;
        let output = normed.mul(&scale.add_scalar(1.0)?)?.add(&shift)?;
        let output = linear3d(&output, &self.proj_out_weight, Some(&self.proj_out_bias))?;

        // Audio: norm -> scale/shift -> proj_out
        let a_shift_scale = self.audio_scale_shift_table.unsqueeze(0)?.unsqueeze(0)?;
        let a_embedded_ts_4d = audio_embedded_ts.unsqueeze(2)?;
        let a_final_ss = a_shift_scale.add(&a_embedded_ts_4d)?;
        let a_shift = a_final_ss.narrow(2, 0, 1)?.squeeze_dim(2)?;
        let a_scale = a_final_ss.narrow(2, 1, 1)?.squeeze_dim(2)?;

        let a_normed = layer_norm_no_affine(&ahs, 1e-6)?;
        let a_output = a_normed.mul(&a_scale.add_scalar(1.0)?)?.add(&a_shift)?;
        let audio_output = linear3d(&a_output, &self.audio_proj_out_weight, Some(&self.audio_proj_out_bias))?;

        Ok((output, audio_output))
    }
}

// ---------------------------------------------------------------------------
// Weight Loading
// ---------------------------------------------------------------------------

/// Load an AdaLayerNormSingle from a weight map with the given prefix.
fn load_ada_layer_norm_single(
    weights: &HashMap<String, Tensor>,
    prefix: &str,
    num_mod_params: usize,
) -> Result<AdaLayerNormSingle> {
    let get = |key: &str| -> Result<Tensor> {
        weights.get(key).cloned().ok_or_else(|| {
            flame_core::Error::InvalidInput(format!("Missing weight: {}", key))
        })
    };

    Ok(AdaLayerNormSingle {
        emb: TimestepEmbedderMLP {
            linear_1_weight: get(&format!("{prefix}.emb.timestep_embedder.linear_1.weight"))?,
            linear_1_bias: get(&format!("{prefix}.emb.timestep_embedder.linear_1.bias"))?,
            linear_2_weight: get(&format!("{prefix}.emb.timestep_embedder.linear_2.weight"))?,
            linear_2_bias: get(&format!("{prefix}.emb.timestep_embedder.linear_2.bias"))?,
        },
        linear_weight: get(&format!("{prefix}.linear.weight"))?,
        linear_bias: get(&format!("{prefix}.linear.bias"))?,
        num_mod_params,
    })
}

/// Load a CaptionProjection from a weight map.
fn load_caption_projection(
    weights: &HashMap<String, Tensor>,
    prefix: &str,
) -> Result<CaptionProjection> {
    let get = |key: &str| -> Result<Tensor> {
        weights.get(key).cloned().ok_or_else(|| {
            flame_core::Error::InvalidInput(format!("Missing weight: {}", key))
        })
    };

    Ok(CaptionProjection {
        linear_1_weight: get(&format!("{prefix}.linear_1.weight"))?,
        linear_1_bias: get(&format!("{prefix}.linear_1.bias"))?,
        linear_2_weight: get(&format!("{prefix}.linear_2.weight"))?,
        linear_2_bias: get(&format!("{prefix}.linear_2.bias"))?,
    })
}

/// Load a FeedForward from a weight map.
fn load_feed_forward(
    weights: &HashMap<String, Tensor>,
    prefix: &str,
) -> Result<FeedForward> {
    let get = |key: &str| -> Result<Tensor> {
        weights.get(key).cloned().ok_or_else(|| {
            flame_core::Error::InvalidInput(format!("Missing weight: {}", key))
        })
    };

    Ok(FeedForward {
        gelu_proj_weight: get(&format!("{prefix}.net.0.proj.weight"))?,
        gelu_proj_bias: get(&format!("{prefix}.net.0.proj.bias"))?,
        out_weight: get(&format!("{prefix}.net.2.weight"))?,
        out_bias: get(&format!("{prefix}.net.2.bias"))?,
    })
}

/// Load an LTX2Attention from a weight map.
fn load_attention(
    weights: &HashMap<String, Tensor>,
    prefix: &str,
    num_heads: usize,
    head_dim: usize,
    eps: f32,
) -> Result<LTX2Attention> {
    let get = |key: &str| -> Result<Tensor> {
        weights.get(key).cloned().ok_or_else(|| {
            flame_core::Error::InvalidInput(format!("Missing weight: {}", key))
        })
    };

    Ok(LTX2Attention {
        to_q_weight: get(&format!("{prefix}.to_q.weight"))?,
        to_q_bias: get(&format!("{prefix}.to_q.bias"))?,
        to_k_weight: get(&format!("{prefix}.to_k.weight"))?,
        to_k_bias: get(&format!("{prefix}.to_k.bias"))?,
        to_v_weight: get(&format!("{prefix}.to_v.weight"))?,
        to_v_bias: get(&format!("{prefix}.to_v.bias"))?,
        norm_q_weight: get(&format!("{prefix}.norm_q.weight"))
            .or_else(|_| get(&format!("{prefix}.q_norm.weight")))?,
        norm_k_weight: get(&format!("{prefix}.norm_k.weight"))
            .or_else(|_| get(&format!("{prefix}.k_norm.weight")))?,
        to_out_weight: get(&format!("{prefix}.to_out.0.weight"))?,
        to_out_bias: get(&format!("{prefix}.to_out.0.bias"))?,
        num_heads,
        head_dim,
        eps,
    })
}

/// Load a single transformer block.
fn load_transformer_block(
    weights: &HashMap<String, Tensor>,
    block_idx: usize,
    config: &LTX2Config,
) -> Result<LTX2TransformerBlock> {
    let prefix = format!("transformer_blocks.{block_idx}");
    let get = |key: &str| -> Result<Tensor> {
        weights.get(key).cloned().ok_or_else(|| {
            flame_core::Error::InvalidInput(format!("Missing weight: {}", key))
        })
    };
    let get_opt = |key: &str| -> Option<Tensor> {
        weights.get(key).cloned()
    };

    let eps = config.norm_eps;
    let num_heads = config.num_attention_heads;
    let head_dim = config.attention_head_dim;
    let audio_heads = config.audio_num_attention_heads;
    let audio_head_dim = config.audio_attention_head_dim;

    Ok(LTX2TransformerBlock {
        // Video self-attention
        norm1_weight: get_opt(&format!("{prefix}.norm1.weight")),
        attn1: load_attention(weights, &format!("{prefix}.attn1"), num_heads, head_dim, eps)?,

        // Audio self-attention
        audio_norm1_weight: get_opt(&format!("{prefix}.audio_norm1.weight")),
        audio_attn1: load_attention(weights, &format!("{prefix}.audio_attn1"), audio_heads, audio_head_dim, eps)?,

        // Video cross-attention
        norm2_weight: get_opt(&format!("{prefix}.norm2.weight")),
        attn2: load_attention(weights, &format!("{prefix}.attn2"), num_heads, head_dim, eps)?,

        // Audio cross-attention
        audio_norm2_weight: get_opt(&format!("{prefix}.audio_norm2.weight")),
        audio_attn2: load_attention(weights, &format!("{prefix}.audio_attn2"), audio_heads, audio_head_dim, eps)?,

        // A2V cross-attention
        audio_to_video_norm_weight: get_opt(&format!("{prefix}.audio_to_video_norm.weight")),
        audio_to_video_attn: load_attention(weights, &format!("{prefix}.audio_to_video_attn"), audio_heads, audio_head_dim, eps)?,

        // V2A cross-attention
        video_to_audio_norm_weight: get_opt(&format!("{prefix}.video_to_audio_norm.weight")),
        video_to_audio_attn: load_attention(weights, &format!("{prefix}.video_to_audio_attn"), audio_heads, audio_head_dim, eps)?,

        // Video FFN
        norm3_weight: get_opt(&format!("{prefix}.norm3.weight")),
        ff: load_feed_forward(weights, &format!("{prefix}.ff"))?,

        // Audio FFN
        audio_norm3_weight: get_opt(&format!("{prefix}.audio_norm3.weight")),
        audio_ff: load_feed_forward(weights, &format!("{prefix}.audio_ff"))?,

        // Modulation tables
        scale_shift_table: get(&format!("{prefix}.scale_shift_table"))?,
        audio_scale_shift_table: get(&format!("{prefix}.audio_scale_shift_table"))?,
        video_a2v_cross_attn_scale_shift_table: get(&format!("{prefix}.video_a2v_cross_attn_scale_shift_table"))?,
        audio_a2v_cross_attn_scale_shift_table: get(&format!("{prefix}.audio_a2v_cross_attn_scale_shift_table"))?,

        eps,
    })
}

/// Load the full LTX-2 model from a safetensors weight map.
///
/// The weight map should contain all keys matching the diffusers
/// `LTX2VideoTransformer3DModel` state dict format.
///
/// # Block-Level Offloading
///
/// After loading, individual blocks can be moved to CPU and loaded
/// back on-demand using the `blocks` Vec. This is essential for
/// fitting the 18.88B model in 24GB VRAM.
pub fn load_ltx2_model(
    weights: &HashMap<String, Tensor>,
    config: &LTX2Config,
) -> Result<LTX2Model> {
    let get = |key: &str| -> Result<Tensor> {
        weights.get(key).cloned().ok_or_else(|| {
            flame_core::Error::InvalidInput(format!("Missing weight: {}", key))
        })
    };

    // Load all transformer blocks
    let mut blocks = Vec::with_capacity(config.num_layers);
    for i in 0..config.num_layers {
        blocks.push(load_transformer_block(weights, i, config)?);
    }

    Ok(LTX2Model {
        config: config.clone(),

        proj_in_weight: get("proj_in.weight")?,
        proj_in_bias: get("proj_in.bias")?,
        audio_proj_in_weight: get("audio_proj_in.weight")?,
        audio_proj_in_bias: get("audio_proj_in.bias")?,

        caption_projection: load_caption_projection(weights, "caption_projection")?,
        audio_caption_projection: load_caption_projection(weights, "audio_caption_projection")?,

        time_embed: load_ada_layer_norm_single(weights, "time_embed", 6)?,
        audio_time_embed: load_ada_layer_norm_single(weights, "audio_time_embed", 6)?,

        av_cross_attn_video_scale_shift: load_ada_layer_norm_single(weights, "av_cross_attn_video_scale_shift", 4)?,
        av_cross_attn_audio_scale_shift: load_ada_layer_norm_single(weights, "av_cross_attn_audio_scale_shift", 4)?,
        av_cross_attn_video_a2v_gate: load_ada_layer_norm_single(weights, "av_cross_attn_video_a2v_gate", 1)?,
        av_cross_attn_audio_v2a_gate: load_ada_layer_norm_single(weights, "av_cross_attn_audio_v2a_gate", 1)?,

        scale_shift_table: get("scale_shift_table")?,
        audio_scale_shift_table: get("audio_scale_shift_table")?,

        blocks,

        proj_out_weight: get("proj_out.weight")?,
        proj_out_bias: get("proj_out.bias")?,
        audio_proj_out_weight: get("audio_proj_out.weight")?,
        audio_proj_out_bias: get("audio_proj_out.bias")?,
    })
}

// ---------------------------------------------------------------------------
// Block-level offloading API (Stagehand-style)
// ---------------------------------------------------------------------------

impl LTX2Model {
    /// Number of parameters in a single transformer block (approximate).
    /// Useful for VRAM budget calculations.
    pub fn block_param_count(&self) -> usize {
        let d = self.config.inner_dim();
        let ad = self.config.audio_inner_dim();
        let ffn = self.config.ffn_hidden();
        let affn = self.config.audio_ffn_hidden();

        // Per-block: 6 attention modules + 2 FFN + modulation tables
        // Each attention: Q/K/V/Out projections + 2 norms
        let video_self_attn = 4 * d * d + 4 * d + 2 * d;  // Q,K,V,Out weights+biases + QK norms
        let audio_self_attn = 4 * ad * ad + 4 * ad + 2 * ad;
        let video_cross_attn = video_self_attn;  // same structure
        let audio_cross_attn = audio_self_attn;
        let a2v_attn = 2 * d * ad + 2 * ad * ad + (d + ad) * 2;  // mixed dims
        let v2a_attn = a2v_attn;
        let video_ffn = d * ffn + ffn + ffn * d + d;
        let audio_ffn = ad * affn + affn + affn * ad + ad;
        let tables = 6 * d + 6 * ad + 5 * d + 5 * ad;

        video_self_attn + audio_self_attn + video_cross_attn + audio_cross_attn
            + a2v_attn + v2a_attn + video_ffn + audio_ffn + tables
    }

    /// Estimated VRAM for a single block in BF16 (bytes).
    pub fn block_vram_bytes(&self) -> usize {
        self.block_param_count() * 2  // BF16 = 2 bytes per param
    }

    /// Estimated total model VRAM in BF16 (bytes).
    pub fn total_vram_bytes(&self) -> usize {
        self.block_vram_bytes() * self.config.num_layers
            + self.non_block_param_count() * 2
    }

    /// Non-block parameter count (input/output projections, embeddings, etc.)
    fn non_block_param_count(&self) -> usize {
        let d = self.config.inner_dim();
        let ad = self.config.audio_inner_dim();
        let cc = self.config.caption_channels;

        // proj_in, proj_out, audio variants
        let proj = 2 * (self.config.in_channels * d + d)
            + 2 * (self.config.audio_in_channels * ad + ad);

        // caption projections (2x: linear1 + linear2 for video and audio)
        let caption = 2 * (cc * d + d + d * d + d)
            + 2 * (cc * ad + ad + ad * ad + ad);

        // time embeddings (6 AdaLayerNormSingle modules)
        // Each: timestep_embedder(256->dim, dim->dim) + silu + linear(dim->n*dim)
        let time_emb = |dim: usize, n: usize| -> usize {
            (256 * dim + dim + dim * dim + dim) + (dim * n * dim + n * dim)
        };
        let te = time_emb(d, 6) + time_emb(ad, 6)
            + time_emb(d, 4) + time_emb(ad, 4)
            + time_emb(d, 1) + time_emb(ad, 1);

        // Output tables
        let tables = 2 * d + 2 * ad;

        proj + caption + te + tables
    }
}

// ---------------------------------------------------------------------------
// Video-Only Forward + Loader
// ---------------------------------------------------------------------------

impl LTX2Model {
    /// Video-only forward pass. Skips all audio paths.
    ///
    /// # Arguments
    /// * `x` - Spatial video latents [B, C, F, H, W] in BF16
    /// * `timestep` - Sigma values [B] in [0, 1] (NOT pre-scaled)
    /// * `context` - Text embeddings [B, seq_len, caption_channels] from Gemma
    /// * `frame_rate` - Video frame rate (default 25.0), affects temporal RoPE scaling
    ///
    /// # Returns
    /// Model velocity output [B, C, F, H, W] in BF16.
    pub fn forward_video_only(
        &self,
        x: &Tensor,
        timestep: &Tensor,
        context: &Tensor,
        frame_rate: f32,
    ) -> Result<Tensor> {
        let x_dims = x.shape().dims().to_vec();
        let (batch_size, channels, num_frames, height, width) =
            (x_dims[0], x_dims[1], x_dims[2], x_dims[3], x_dims[4]);
        let inner_dim = self.config.inner_dim();
        let num_tokens = num_frames * height * width;

        // 1. Patchify: [B, C, F, H, W] → [B, F*H*W, C]
        // For patch_size=1 this is just a reshape
        let x_flat = x.reshape(&[batch_size, channels, num_tokens])?
            .permute(&[0, 2, 1])?; // [B, N, C]

        // 2. Build coordinate grid [B, 3, N, 2] for RoPE
        let device = x.device().clone();
        let vae_sf = &self.config.vae_scale_factors;
        let coords = build_video_coords(
            batch_size, num_frames, height, width,
            vae_sf, self.config.causal_offset, frame_rate, device.clone(),
        )?;

        // 3. Compute RoPE
        let max_pos = [
            self.config.pos_embed_max_pos as f64,
            self.config.base_height as f64,
            self.config.base_width as f64,
        ];
        let (v_cos, v_sin) = compute_rope_frequencies(
            &coords, inner_dim, &max_pos,
            self.config.rope_theta, self.config.num_attention_heads,
        )?;

        // 4. Input projection: [B, N, C] → [B, N, inner_dim]
        let mut hs = linear3d(&x_flat, &self.proj_in_weight, Some(&self.proj_in_bias))?;

        // 5. Timestep conditioning
        // Expand sigma to per-token: [B] → [B, N]
        let ts_expanded = timestep.unsqueeze(1)?
            .expand(&[batch_size, num_tokens])?;
        let ts_scaled = ts_expanded.mul_scalar(self.config.timestep_scale_multiplier as f32)?;
        let ts_flat = ts_scaled.reshape(&[batch_size * num_tokens])?;
        let (v_timestep, v_embedded) = self.time_embed.forward(&ts_flat)?;
        // Reshape: [B*N, 6*dim] → [B, N, 6*dim]
        let v_timestep = v_timestep.reshape(&[batch_size, num_tokens, 6 * inner_dim])?;
        let v_embedded = v_embedded.reshape(&[batch_size, num_tokens, inner_dim])?;

        // 6. Caption projection: [B, seq, 3840] → [B, seq, inner_dim]
        let enc_hs = self.caption_projection.forward(context)?;

        // 7. Run transformer blocks (video-only)
        for block in &self.blocks {
            hs = block.forward_video_only(
                &hs, &enc_hs, &v_timestep,
                Some((&v_cos, &v_sin)),
                None, // encoder_attention_mask
            )?;
        }

        // 8. Output: norm → scale/shift → proj_out
        let shift_scale = self.scale_shift_table.unsqueeze(0)?.unsqueeze(0)?; // [1, 1, 2, dim]
        let emb_4d = v_embedded.unsqueeze(2)?; // [B, N, 1, dim]
        let final_ss = shift_scale.add(&emb_4d)?;
        let shift = final_ss.narrow(2, 0, 1)?.squeeze_dim(2)?;
        let scale = final_ss.narrow(2, 1, 1)?.squeeze_dim(2)?;

        let normed = layer_norm_no_affine(&hs, self.config.norm_eps)?;
        let output = normed.mul(&scale.add_scalar(1.0)?)?.add(&shift)?;
        let output = linear3d(&output, &self.proj_out_weight, Some(&self.proj_out_bias))?;

        // 9. Unpatchify: [B, N, C] → [B, C, F, H, W]
        let output = output.permute(&[0, 2, 1])?; // [B, C, N]
        output.reshape(&[batch_size, channels, num_frames, height, width])
    }
}

/// Build 3D video coordinate grid for RoPE.
///
/// Returns [B, 3, N, 2] tensor where N = F*H*W, last dim is [start, end).
/// Coordinates are in pixel space (scaled by VAE factors) with causal fix
/// and frame_rate scaling applied.
fn build_video_coords(
    batch_size: usize,
    num_frames: usize,
    height: usize,
    width: usize,
    vae_scale_factors: &[usize; 3],
    causal_offset: usize,
    frame_rate: f32,
    device: std::sync::Arc<flame_core::CudaDevice>,
) -> Result<Tensor> {
    let num_tokens = num_frames * height * width;

    // Build meshgrid: (f, h, w) for each token
    let mut coords_data = vec![0.0f32; batch_size * 3 * num_tokens * 2];

    for b in 0..batch_size {
        for f in 0..num_frames {
            for h in 0..height {
                for w in 0..width {
                    let token_idx = f * height * width + h * width + w;
                    let base = b * 3 * num_tokens * 2;

                    // Latent coords: [start, end) = [idx, idx+1]
                    // Pixel coords = latent * vae_scale_factor
                    let f_start = (f * vae_scale_factors[0]) as f32;
                    let f_end = ((f + 1) * vae_scale_factors[0]) as f32;
                    let h_start = (h * vae_scale_factors[1]) as f32;
                    let h_end = ((h + 1) * vae_scale_factors[1]) as f32;
                    let w_start = (w * vae_scale_factors[2]) as f32;
                    let w_end = ((w + 1) * vae_scale_factors[2]) as f32;

                    // Causal fix: temporal coords = (coords + causal_offset - vae_t).clamp(min=0)
                    let vae_t = vae_scale_factors[0] as f32;
                    let f_start_causal = (f_start + causal_offset as f32 - vae_t).max(0.0);
                    let f_end_causal = (f_end + causal_offset as f32 - vae_t).max(0.0);

                    // Frame-rate scaling on temporal dim
                    let f_start_scaled = f_start_causal / frame_rate;
                    let f_end_scaled = f_end_causal / frame_rate;

                    // dim 0 = temporal
                    coords_data[base + 0 * num_tokens * 2 + token_idx * 2] = f_start_scaled;
                    coords_data[base + 0 * num_tokens * 2 + token_idx * 2 + 1] = f_end_scaled;
                    // dim 1 = height
                    coords_data[base + 1 * num_tokens * 2 + token_idx * 2] = h_start;
                    coords_data[base + 1 * num_tokens * 2 + token_idx * 2 + 1] = h_end;
                    // dim 2 = width
                    coords_data[base + 2 * num_tokens * 2 + token_idx * 2] = w_start;
                    coords_data[base + 2 * num_tokens * 2 + token_idx * 2 + 1] = w_end;
                }
            }
        }
    }

    Tensor::from_vec_dtype(
        coords_data,
        Shape::from_dims(&[batch_size, 3, num_tokens, 2]),
        device,
        DType::F32,
    )
}

/// LTX-2 model with only global params loaded (blocks streamed from disk).
///
/// This struct holds the non-block parameters on GPU (~400MB) and loads
/// each of the 48 blocks from the safetensors file on demand during
/// forward pass. This is essential for fitting 33GB of video-only
/// weights into 24GB VRAM.
pub struct LTX2StreamingModel {
    pub config: LTX2Config,
    checkpoint_path: String,

    // Global params on GPU
    proj_in_weight: Tensor,
    proj_in_bias: Tensor,
    caption_projection: CaptionProjection,
    time_embed: AdaLayerNormSingle,
    scale_shift_table: Tensor,  // [2, inner_dim]
    proj_out_weight: Tensor,
    proj_out_bias: Tensor,
}

impl LTX2StreamingModel {
    /// Load global (non-block) params from checkpoint. Blocks are NOT loaded.
    ///
    /// Auto-detects key prefix (`model.diffusion_model.` for ComfyUI format)
    /// and strips it. Supports both diffusers and ComfyUI checkpoints.
    pub fn load_globals(path: &str, config: &LTX2Config) -> Result<Self> {
        let device = flame_core::global_cuda_device();
        log::info!("[LTX2] Loading global params (non-block)...");

        // Detect key prefix by checking for known global keys
        let prefix = detect_key_prefix(path)?;
        log::info!("[LTX2] Detected key prefix: '{}'", prefix);

        let globals = flame_core::serialization::load_file_filtered(
            path, &device,
            |key| {
                // Strip prefix for matching
                let k = key.strip_prefix(&prefix).unwrap_or(key);
                !k.contains("audio")
                    && !k.starts_with("transformer_blocks.")
            },
        )?;

        // Remap keys: strip prefix
        let globals: HashMap<String, Tensor> = globals.into_iter()
            .map(|(k, v)| {
                let stripped = k.strip_prefix(&prefix).unwrap_or(&k).to_string();
                (stripped, v)
            })
            .collect();

        let get = |key: &str| -> Result<Tensor> {
            globals.get(key).cloned()
                .ok_or_else(|| flame_core::Error::InvalidInput(format!("Missing: {}", key)))
        };

        log::info!("[LTX2] Global params loaded: {} keys", globals.len());

        // Support both diffusers and ComfyUI key names
        let get_alt = |k1: &str, k2: &str| -> Result<Tensor> {
            get(k1).or_else(|_| get(k2))
        };

        // proj_in / patchify_proj
        let proj_in_w = get_alt("proj_in.weight", "patchify_proj.weight")?;
        let proj_in_b = get_alt("proj_in.bias", "patchify_proj.bias")?;

        // caption_projection / prompt_adaln_single (different structure in 2.3)
        let caption_proj = load_caption_projection(&globals, "caption_projection")
            .or_else(|_| {
                // LTX-2.3 uses different caption projection structure
                // For now, try the prompt_adaln_single path
                load_caption_projection(&globals, "prompt_projection")
            })?;

        // time_embed / adaln_single
        let time_emb = load_ada_layer_norm_single(&globals, "time_embed", 6)
            .or_else(|_| load_ada_layer_norm_single(&globals, "adaln_single", 6))?;

        Ok(Self {
            config: config.clone(),
            checkpoint_path: path.to_string(),
            proj_in_weight: proj_in_w,
            proj_in_bias: proj_in_b,
            caption_projection: caption_proj,
            time_embed: time_emb,
            scale_shift_table: get("scale_shift_table")?,
            proj_out_weight: get("proj_out.weight")?,
            proj_out_bias: get("proj_out.bias")?,
        })
    }

    /// Load a single block's video-only weights from checkpoint.
    fn load_block(&self, block_idx: usize) -> Result<LTX2TransformerBlock> {
        let device = flame_core::global_cuda_device();
        let key_prefix = detect_key_prefix(&self.checkpoint_path)?;
        let prefix = format!("{key_prefix}transformer_blocks.{block_idx}.");

        let raw_weights = flame_core::serialization::load_file_filtered(
            &self.checkpoint_path, &device,
            |key| key.starts_with(&prefix) && !key.contains("audio"),
        )?;

        // Strip key prefix so load_attention/load_feed_forward find keys
        let block_weights: HashMap<String, Tensor> = raw_weights.into_iter()
            .map(|(k, v)| {
                let stripped = k.strip_prefix(&key_prefix).unwrap_or(&k).to_string();
                (stripped, v)
            })
            .collect();

        let eps = self.config.norm_eps;
        let num_heads = self.config.num_attention_heads;
        let head_dim = self.config.attention_head_dim;
        let ad = self.config.audio_inner_dim();

        let get_opt = |key: &str| -> Option<Tensor> { block_weights.get(key).cloned() };

        // Dummy tensors for audio fields
        let dummy_1d = |sz: usize| Tensor::zeros_dtype(
            Shape::from_dims(&[sz]), DType::BF16, device.clone(),
        );
        let dummy_2d = |r: usize, c: usize| Tensor::zeros_dtype(
            Shape::from_dims(&[r, c]), DType::BF16, device.clone(),
        );

        let dummy_attn = || -> Result<LTX2Attention> {
            Ok(LTX2Attention {
                to_q_weight: dummy_2d(ad, ad)?, to_q_bias: dummy_1d(ad)?,
                to_k_weight: dummy_2d(ad, ad)?, to_k_bias: dummy_1d(ad)?,
                to_v_weight: dummy_2d(ad, ad)?, to_v_bias: dummy_1d(ad)?,
                norm_q_weight: dummy_1d(ad)?, norm_k_weight: dummy_1d(ad)?,
                to_out_weight: dummy_2d(ad, ad)?, to_out_bias: dummy_1d(ad)?,
                num_heads: self.config.audio_num_attention_heads,
                head_dim: self.config.audio_attention_head_dim, eps,
            })
        };

        let dummy_ff = || -> Result<FeedForward> {
            Ok(FeedForward {
                gelu_proj_weight: dummy_2d(self.config.audio_ffn_hidden(), ad)?,
                gelu_proj_bias: dummy_1d(self.config.audio_ffn_hidden())?,
                out_weight: dummy_2d(ad, self.config.audio_ffn_hidden())?,
                out_bias: dummy_1d(ad)?,
            })
        };

        let pfx = format!("transformer_blocks.{block_idx}");
        Ok(LTX2TransformerBlock {
            norm1_weight: get_opt(&format!("{pfx}.norm1.weight")),
            attn1: load_attention(&block_weights, &format!("{pfx}.attn1"), num_heads, head_dim, eps)?,
            audio_norm1_weight: None,
            audio_attn1: dummy_attn()?,
            norm2_weight: get_opt(&format!("{pfx}.norm2.weight")),
            attn2: load_attention(&block_weights, &format!("{pfx}.attn2"), num_heads, head_dim, eps)?,
            audio_norm2_weight: None,
            audio_attn2: dummy_attn()?,
            audio_to_video_norm_weight: None,
            audio_to_video_attn: dummy_attn()?,
            video_to_audio_norm_weight: None,
            video_to_audio_attn: dummy_attn()?,
            norm3_weight: get_opt(&format!("{pfx}.norm3.weight")),
            ff: load_feed_forward(&block_weights, &format!("{pfx}.ff"))?,
            audio_norm3_weight: None,
            audio_ff: dummy_ff()?,
            scale_shift_table: block_weights.get(&format!("{pfx}.scale_shift_table"))
                .cloned()
                .ok_or_else(|| flame_core::Error::InvalidInput(format!("Missing: {pfx}.scale_shift_table")))?,
            audio_scale_shift_table: dummy_2d(6, ad)?,
            video_a2v_cross_attn_scale_shift_table: dummy_2d(5, self.config.inner_dim())?,
            audio_a2v_cross_attn_scale_shift_table: dummy_2d(5, ad)?,
            eps,
        })
    }

    /// Video-only forward with block streaming from disk.
    ///
    /// Each block is loaded from the safetensors checkpoint, executed,
    /// and immediately dropped to free GPU memory. Only 1 block + global
    /// params are on GPU at any time (~1GB total).
    pub fn forward_video_only(
        &self,
        x: &Tensor,
        timestep: &Tensor,
        context: &Tensor,
        frame_rate: f32,
    ) -> Result<Tensor> {
        let x_dims = x.shape().dims().to_vec();
        let (batch_size, channels, num_frames, height, width) =
            (x_dims[0], x_dims[1], x_dims[2], x_dims[3], x_dims[4]);
        let inner_dim = self.config.inner_dim();
        let num_tokens = num_frames * height * width;

        // 1. Patchify: [B, C, F, H, W] → [B, F*H*W, C]
        let x_flat = x.reshape(&[batch_size, channels, num_tokens])?
            .permute(&[0, 2, 1])?;

        // 2. Build coordinate grid for RoPE
        let device = x.device().clone();
        let vae_sf = &self.config.vae_scale_factors;
        let coords = build_video_coords(
            batch_size, num_frames, height, width,
            vae_sf, self.config.causal_offset, frame_rate, device.clone(),
        )?;

        // 3. Compute RoPE
        let max_pos = [
            self.config.pos_embed_max_pos as f64,
            self.config.base_height as f64,
            self.config.base_width as f64,
        ];
        let (v_cos, v_sin) = compute_rope_frequencies(
            &coords, inner_dim, &max_pos,
            self.config.rope_theta, self.config.num_attention_heads,
        )?;

        // 4. Input projection
        let mut hs = linear3d(&x_flat, &self.proj_in_weight, Some(&self.proj_in_bias))?;

        // 5. Timestep conditioning
        let ts_expanded = timestep.unsqueeze(1)?.expand(&[batch_size, num_tokens])?;
        let ts_scaled = ts_expanded.mul_scalar(self.config.timestep_scale_multiplier as f32)?;
        let ts_flat = ts_scaled.reshape(&[batch_size * num_tokens])?;
        let (v_timestep, v_embedded) = self.time_embed.forward(&ts_flat)?;
        let v_timestep = v_timestep.reshape(&[batch_size, num_tokens, 6 * inner_dim])?;
        let v_embedded = v_embedded.reshape(&[batch_size, num_tokens, inner_dim])?;

        // 6. Caption projection
        let enc_hs = self.caption_projection.forward(context)?;

        // 7. Stream blocks: load → run → drop
        for i in 0..self.config.num_layers {
            let block = self.load_block(i)?;
            hs = block.forward_video_only(
                &hs, &enc_hs, &v_timestep,
                Some((&v_cos, &v_sin)),
                None,
            )?;
            drop(block);  // Free ~536MB immediately

            if (i + 1) % 12 == 0 || i + 1 == self.config.num_layers {
                log::info!("[LTX2] Block {}/{} done", i + 1, self.config.num_layers);
            }
        }

        // 8. Output
        let shift_scale = self.scale_shift_table.unsqueeze(0)?.unsqueeze(0)?;
        let emb_4d = v_embedded.unsqueeze(2)?;
        let final_ss = shift_scale.add(&emb_4d)?;
        let shift = final_ss.narrow(2, 0, 1)?.squeeze_dim(2)?;
        let scale = final_ss.narrow(2, 1, 1)?.squeeze_dim(2)?;

        let normed = layer_norm_no_affine(&hs, self.config.norm_eps)?;
        let output = normed.mul(&scale.add_scalar(1.0)?)?.add(&shift)?;
        let output = linear3d(&output, &self.proj_out_weight, Some(&self.proj_out_bias))?;

        // 9. Unpatchify
        let output = output.permute(&[0, 2, 1])?;
        output.reshape(&[batch_size, channels, num_frames, height, width])
    }
}

/// Detect key prefix in a safetensors checkpoint.
///
/// Returns `"model.diffusion_model."` for ComfyUI format, `""` for diffusers.
fn detect_key_prefix(path: &str) -> Result<String> {
    use serde_json::Value;
    use std::io::Read;

    let file = std::fs::File::open(path)
        .map_err(|e| flame_core::Error::Io(format!("Failed to open: {}", e)))?;
    let mmap = unsafe { memmap2::Mmap::map(&file) }
        .map_err(|e| flame_core::Error::Io(format!("mmap: {}", e)))?;

    let header_size = u64::from_le_bytes(mmap[..8].try_into().unwrap()) as usize;
    let metadata: Value = serde_json::from_slice(&mmap[8..8 + header_size])
        .map_err(|e| flame_core::Error::Io(format!("json: {}", e)))?;

    if let Some(obj) = metadata.as_object() {
        for key in obj.keys() {
            if key.starts_with("model.diffusion_model.") {
                return Ok("model.diffusion_model.".to_string());
            }
            if key.starts_with("transformer_blocks.") || key == "proj_in.weight" {
                return Ok(String::new());
            }
        }
    }
    Ok(String::new())
}

/// Load LTX-2 model with video-only weights (skip all audio keys).
///
/// This loads ~16.5B params instead of 22B, saving ~13GB of VRAM.
/// Audio fields are initialized with zero-sized dummy tensors.
pub fn load_ltx2_model_video_only(
    weights: &HashMap<String, Tensor>,
    config: &LTX2Config,
) -> Result<LTX2Model> {
    let get = |key: &str| -> Result<Tensor> {
        weights.get(key).cloned().ok_or_else(|| {
            flame_core::Error::InvalidInput(format!("Missing weight: {}", key))
        })
    };

    let device = weights.values().next()
        .map(|t| t.device().clone())
        .ok_or_else(|| flame_core::Error::InvalidInput("Empty weights".into()))?;

    // Create zero-filled dummy tensors for audio fields
    let dummy_1d = |size: usize| -> Result<Tensor> {
        Tensor::zeros_dtype(Shape::from_dims(&[size]), DType::BF16, device.clone())
    };
    let dummy_2d = |r: usize, c: usize| -> Result<Tensor> {
        Tensor::zeros_dtype(Shape::from_dims(&[r, c]), DType::BF16, device.clone())
    };

    let ad = config.audio_inner_dim();
    let ahd = config.audio_attention_head_dim;
    let anh = config.audio_num_attention_heads;

    let dummy_attention = |_prefix: &str| -> Result<LTX2Attention> {
        Ok(LTX2Attention {
            to_q_weight: dummy_2d(ad, ad)?,
            to_q_bias: dummy_1d(ad)?,
            to_k_weight: dummy_2d(ad, ad)?,
            to_k_bias: dummy_1d(ad)?,
            to_v_weight: dummy_2d(ad, ad)?,
            to_v_bias: dummy_1d(ad)?,
            norm_q_weight: dummy_1d(ad)?,
            norm_k_weight: dummy_1d(ad)?,
            to_out_weight: dummy_2d(ad, ad)?,
            to_out_bias: dummy_1d(ad)?,
            num_heads: anh,
            head_dim: ahd,
            eps: config.norm_eps,
        })
    };

    let dummy_ff = || -> Result<FeedForward> {
        Ok(FeedForward {
            gelu_proj_weight: dummy_2d(config.audio_ffn_hidden(), ad)?,
            gelu_proj_bias: dummy_1d(config.audio_ffn_hidden())?,
            out_weight: dummy_2d(ad, config.audio_ffn_hidden())?,
            out_bias: dummy_1d(ad)?,
        })
    };

    let dummy_adaln = || -> Result<AdaLayerNormSingle> {
        Ok(AdaLayerNormSingle {
            emb: TimestepEmbedderMLP {
                linear_1_weight: dummy_2d(ad, 256)?,
                linear_1_bias: dummy_1d(ad)?,
                linear_2_weight: dummy_2d(ad, ad)?,
                linear_2_bias: dummy_1d(ad)?,
            },
            linear_weight: dummy_2d(ad, ad)?,
            linear_bias: dummy_1d(ad)?,
            num_mod_params: 1,
        })
    };

    let dummy_caption = || -> Result<CaptionProjection> {
        Ok(CaptionProjection {
            linear_1_weight: dummy_2d(ad, config.caption_channels)?,
            linear_1_bias: dummy_1d(ad)?,
            linear_2_weight: dummy_2d(ad, ad)?,
            linear_2_bias: dummy_1d(ad)?,
        })
    };

    // Load video-only transformer blocks
    let eps = config.norm_eps;
    let num_heads = config.num_attention_heads;
    let head_dim = config.attention_head_dim;

    let mut blocks = Vec::with_capacity(config.num_layers);
    for i in 0..config.num_layers {
        let prefix = format!("transformer_blocks.{i}");
        let get_opt = |key: &str| -> Option<Tensor> { weights.get(key).cloned() };

        blocks.push(LTX2TransformerBlock {
            // Video self-attention (real weights)
            norm1_weight: get_opt(&format!("{prefix}.norm1.weight")),
            attn1: load_attention(weights, &format!("{prefix}.attn1"), num_heads, head_dim, eps)?,

            // Audio self-attention (dummy)
            audio_norm1_weight: None,
            audio_attn1: dummy_attention("")?,

            // Video cross-attention (real weights)
            norm2_weight: get_opt(&format!("{prefix}.norm2.weight")),
            attn2: load_attention(weights, &format!("{prefix}.attn2"), num_heads, head_dim, eps)?,

            // Audio cross-attention (dummy)
            audio_norm2_weight: None,
            audio_attn2: dummy_attention("")?,

            // A2V/V2A cross-attention (dummy)
            audio_to_video_norm_weight: None,
            audio_to_video_attn: dummy_attention("")?,
            video_to_audio_norm_weight: None,
            video_to_audio_attn: dummy_attention("")?,

            // Video FFN (real weights)
            norm3_weight: get_opt(&format!("{prefix}.norm3.weight")),
            ff: load_feed_forward(weights, &format!("{prefix}.ff"))?,

            // Audio FFN (dummy)
            audio_norm3_weight: None,
            audio_ff: dummy_ff()?,

            // Video modulation table (real), audio (dummy)
            scale_shift_table: get(&format!("{prefix}.scale_shift_table"))?,
            audio_scale_shift_table: dummy_2d(6, ad)?,
            video_a2v_cross_attn_scale_shift_table: dummy_2d(5, config.inner_dim())?,
            audio_a2v_cross_attn_scale_shift_table: dummy_2d(5, ad)?,

            eps,
        });

        if (i + 1) % 12 == 0 {
            log::info!("[LTX2] Loaded block {}/{}", i + 1, config.num_layers);
        }
    }

    log::info!("[LTX2] All {} blocks loaded (video-only)", config.num_layers);

    Ok(LTX2Model {
        config: config.clone(),

        proj_in_weight: get("proj_in.weight")?,
        proj_in_bias: get("proj_in.bias")?,
        audio_proj_in_weight: dummy_2d(ad, config.audio_in_channels)?,
        audio_proj_in_bias: dummy_1d(ad)?,

        caption_projection: load_caption_projection(weights, "caption_projection")?,
        audio_caption_projection: dummy_caption()?,

        time_embed: load_ada_layer_norm_single(weights, "time_embed", 6)?,
        audio_time_embed: dummy_adaln()?,

        av_cross_attn_video_scale_shift: dummy_adaln()?,
        av_cross_attn_audio_scale_shift: dummy_adaln()?,
        av_cross_attn_video_a2v_gate: dummy_adaln()?,
        av_cross_attn_audio_v2a_gate: dummy_adaln()?,

        scale_shift_table: get("scale_shift_table")?,
        audio_scale_shift_table: dummy_2d(2, ad)?,

        blocks,

        proj_out_weight: get("proj_out.weight")?,
        proj_out_bias: get("proj_out.bias")?,
        audio_proj_out_weight: dummy_2d(config.audio_out_channels, ad)?,
        audio_proj_out_bias: dummy_1d(config.audio_out_channels)?,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let cfg = LTX2Config::default();
        assert_eq!(cfg.inner_dim(), 4096);          // 32 * 128
        assert_eq!(cfg.audio_inner_dim(), 2048);     // 32 * 64
        assert_eq!(cfg.ffn_hidden(), 16384);          // 4096 * 4
        assert_eq!(cfg.audio_ffn_hidden(), 8192);     // 2048 * 4
        assert_eq!(cfg.num_layers, 48);
    }

    #[test]
    fn test_vram_estimate() {
        let cfg = LTX2Config::default();
        // Rough sanity check: 18.88B params * 2 bytes = ~37.76 GB
        // Our estimate won't be exact but should be in the right ballpark
        let model_estimate_gb = {
            let d = cfg.inner_dim();
            let ad = cfg.audio_inner_dim();
            // Very rough: each block has ~6 attention modules + 2 FFN
            let per_block = 6 * (4 * d * d) + 6 * (4 * ad * ad) + 2 * (d * d * 4 * 2) + 2 * (ad * ad * 4 * 2);
            let total_params = per_block * cfg.num_layers;
            (total_params * 2) as f64 / (1024.0 * 1024.0 * 1024.0)
        };
        // Should be roughly 30-40 GB in BF16
        assert!(model_estimate_gb > 10.0, "Model should be > 10 GB, got {:.1} GB", model_estimate_gb);
        assert!(model_estimate_gb < 80.0, "Model should be < 80 GB, got {:.1} GB", model_estimate_gb);
    }
}
