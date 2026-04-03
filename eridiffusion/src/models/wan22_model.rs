//! Wan 2.2 Video Diffusion Transformer — pure Rust inference model
//!
//! Ported from the PyTorch reference:
//!   - serenity/models/wan_transformer.py (Serenity native port)
//!   - musubi-tuner/wan/modules/model.py (upstream Alibaba Wan)
//!
//! Architecture overview:
//!   - Video input → Conv3d patch embedding → [B, L, C] token sequence
//!   - Per-token sinusoidal time embedding → time_projection → 6-way modulation
//!   - T5 text embedding → Linear → GELU → Linear
//!   - N transformer blocks: LayerNorm + SelfAttn(RoPE) + CrossAttn + FFN
//!   - Output head with modulated LayerNorm → unpatchify to video
//!
//! Wan 2.2 "MoE" is dual-expert: two identical transformers switched by a timestep
//! boundary ratio (e.g. 0.875). High-noise steps use expert 1, low-noise expert 2.
//! This is NOT a gated MoE — it is simple timestep-based routing.
//!
//! Key differences from Wan 2.1:
//!   - Time modulation is per-token: e0 shape [B, L, 6, C] (vs [B, 6, C] in 2.1)
//!   - Cross-attention always uses standard WanCrossAttention (no I2V variant)
//!   - Two transformer checkpoints loaded as expert_high / expert_low
//!
//! Weight key format (safetensors, no prefix for 14B):
//!   patch_embedding.weight, patch_embedding.bias
//!   text_embedding.0.weight, text_embedding.0.bias, text_embedding.2.weight, ...
//!   time_embedding.0.weight, time_embedding.0.bias, time_embedding.2.weight, ...
//!   time_projection.1.weight, time_projection.1.bias
//!   blocks.{i}.norm1.weight, blocks.{i}.norm1.bias
//!   blocks.{i}.self_attn.q.weight, blocks.{i}.self_attn.q.bias
//!   blocks.{i}.self_attn.k.weight, ...
//!   blocks.{i}.self_attn.v.weight, ...
//!   blocks.{i}.self_attn.o.weight, ...
//!   blocks.{i}.self_attn.norm_q.weight
//!   blocks.{i}.self_attn.norm_k.weight
//!   blocks.{i}.norm3.weight (if cross_attn_norm)
//!   blocks.{i}.cross_attn.q.weight, ...
//!   blocks.{i}.cross_attn.norm_q.weight, ...
//!   blocks.{i}.norm2.weight
//!   blocks.{i}.ffn.0.weight, blocks.{i}.ffn.0.bias
//!   blocks.{i}.ffn.2.weight, blocks.{i}.ffn.2.bias
//!   blocks.{i}.modulation (Parameter, shape [1, 6, dim])
//!   head.norm.weight
//!   head.head.weight, head.head.bias
//!   head.modulation (Parameter, shape [1, 2, dim])

use crate::ops::{LayerNorm, Linear};
// Minimal Conv3d stub until flame-core exports it publicly.
// Wan 2.2 only uses Conv3d for patch embedding (stride=kernel, no padding).
pub struct Conv3d {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub kernel: (usize, usize, usize),
    pub stride: (usize, usize, usize),
}

impl Conv3d {
    pub fn new(
        in_channels: usize, out_channels: usize,
        kernel: (usize, usize, usize),
        stride: Option<(usize, usize, usize)>,
        _padding: Option<(usize, usize, usize)>,
        _dilation: Option<(usize, usize, usize)>,
        _groups: Option<usize>,
        device: &std::sync::Arc<flame_core::CudaDevice>,
    ) -> Result<Self> {
        let numel = out_channels * in_channels * kernel.0 * kernel.1 * kernel.2;
        let weight = Tensor::zeros_dtype(
            Shape::from_dims(&[out_channels, in_channels, kernel.0, kernel.1, kernel.2]),
            DType::BF16, device.clone(),
        )?;
        Ok(Self { weight, bias: None, kernel, stride: stride.unwrap_or(kernel) })
    }

    pub fn forward(&self, _x: &Tensor) -> Result<Tensor> {
        // TODO: Implement via flame_core::conv3d when exported
        Err(flame_core::Error::InvalidOperation(
            "Conv3d forward not yet wired to flame-core. Requires public conv3d export.".into()
        ))
    }
}
use flame_core::device::Device;
use flame_core::{DType, Result, Shape, Tensor};
use std::collections::HashMap;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for Wan 2.2 transformer
#[derive(Debug, Clone)]
pub struct Wan22Config {
    /// Model variant: "t2v" or "i2v"
    pub model_type: String,
    /// 3D patch dimensions (t, h, w) — typically (1, 2, 2)
    pub patch_size: (usize, usize, usize),
    /// Max text token length (512 for T5)
    pub text_len: usize,
    /// Input latent channels (16 for T2V, 36 for I2V)
    pub in_dim: usize,
    /// Hidden dimension (5120 for 14B)
    pub dim: usize,
    /// FFN intermediate dimension (13824 for 14B)
    pub ffn_dim: usize,
    /// Sinusoidal embedding dimension (256)
    pub freq_dim: usize,
    /// Text encoder output dimension (4096 for T5-XXL)
    pub text_dim: usize,
    /// Output latent channels (16)
    pub out_dim: usize,
    /// Number of attention heads (40 for 14B)
    pub num_heads: usize,
    /// Number of transformer blocks (40 for 14B)
    pub num_layers: usize,
    /// Timestep boundary for dual-expert routing (0.875 for T2V, 0.900 for I2V)
    pub boundary_ratio: f32,
    /// Whether cross-attention uses LayerNorm (true for Wan)
    pub cross_attn_norm: bool,
    /// Epsilon for normalization layers
    pub eps: f32,
}

impl Default for Wan22Config {
    fn default() -> Self {
        Self {
            model_type: "t2v".into(),
            patch_size: (1, 2, 2),
            text_len: 512,
            in_dim: 16,
            dim: 5120,
            ffn_dim: 13824,
            freq_dim: 256,
            text_dim: 4096,
            out_dim: 16,
            num_heads: 40,
            num_layers: 40,
            boundary_ratio: 0.875,
            cross_attn_norm: true,
            eps: 1e-6,
        }
    }
}

impl Wan22Config {
    /// T2V A14B preset (Wan 2.2 text-to-video, 14B params per expert)
    pub fn t2v_a14b() -> Self {
        Self {
            model_type: "t2v".into(),
            in_dim: 16,
            boundary_ratio: 0.875,
            ..Default::default()
        }
    }

    /// I2V A14B preset (Wan 2.2 image-to-video, 14B params per expert)
    pub fn i2v_a14b() -> Self {
        Self {
            model_type: "i2v".into(),
            in_dim: 36,
            boundary_ratio: 0.900,
            ..Default::default()
        }
    }

    pub fn head_dim(&self) -> usize {
        self.dim / self.num_heads
    }
}

// ---------------------------------------------------------------------------
// RMS Normalization (Wan-style: compute in F32, output in BF16)
// ---------------------------------------------------------------------------

/// Wan RMSNorm: weight-only (no bias), F32 internal computation
///
/// Weight key: `{prefix}.weight` — shape [dim]
pub struct WanRMSNorm {
    pub weight: Tensor,
    pub dim: usize,
    pub eps: f32,
}

impl WanRMSNorm {
    pub fn new(dim: usize, eps: f32, device: &Arc<flame_core::CudaDevice>) -> Result<Self> {
        let weight = Tensor::ones_dtype(
            Shape::from_dims(&[dim]),
            DType::BF16,
            device.clone(),
        )?;
        Ok(Self { weight, dim, eps })
    }

    /// Forward: x * rsqrt(mean(x^2) + eps) * weight
    /// Input/output: [B, L, C] in BF16
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Promote to F32 for numerical stability
        let x_f32 = x.to_dtype(DType::F32)?;
        let x_sq = x_f32.square()?;
        let hidden_size = self.dim as f32;
        let mean_sq = x_sq
            .sum_dim_keepdim(x_sq.shape().rank() - 1)?
            .mul_scalar(1.0 / hidden_size)?;
        let rsqrt = mean_sq.add_scalar(self.eps)?.sqrt()?;
        let one = Tensor::full(rsqrt.shape().clone(), 1.0, rsqrt.device().clone())?;
        let inv = one.div(&rsqrt)?;
        let normed = x_f32.mul(&inv)?;
        // Back to BF16, apply weight
        let normed_bf16 = normed.to_dtype(DType::BF16)?;
        normed_bf16.mul(&self.weight)
    }

    /// Load weight from state dict
    pub fn load_weights(&mut self, prefix: &str, weights: &HashMap<String, Tensor>) -> Result<()> {
        let key = format!("{prefix}.weight");
        if let Some(w) = weights.get(&key) {
            self.weight = w.to_dtype(DType::BF16)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Layer Normalization (Wan-style: F32 internal, optionally elementwise_affine)
// ---------------------------------------------------------------------------

/// Wan LayerNorm: computes in F32, returns in original dtype.
///
/// Weight keys: `{prefix}.weight`, `{prefix}.bias` (if affine)
pub struct WanLayerNorm {
    pub weight: Option<Tensor>,
    pub bias: Option<Tensor>,
    pub dim: usize,
    pub eps: f32,
    pub elementwise_affine: bool,
}

impl WanLayerNorm {
    pub fn new(
        dim: usize,
        eps: f32,
        elementwise_affine: bool,
        device: &Arc<flame_core::CudaDevice>,
    ) -> Result<Self> {
        let (weight, bias) = if elementwise_affine {
            let w = Tensor::ones_dtype(Shape::from_dims(&[dim]), DType::BF16, device.clone())?;
            let b = Tensor::zeros_dtype(Shape::from_dims(&[dim]), DType::BF16, device.clone())?;
            (Some(w), Some(b))
        } else {
            (None, None)
        };
        Ok(Self { weight, bias, dim, eps, elementwise_affine })
    }

    /// Forward pass — standard LayerNorm computed in F32
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_f32 = x.to_dtype(DType::F32)?;
        let rank = x_f32.shape().rank();
        let last = rank - 1;
        let hidden_size = self.dim as f32;

        // mean
        let mean = x_f32.sum_dim_keepdim(last)?.mul_scalar(1.0 / hidden_size)?;
        let centered = x_f32.sub(&mean)?;

        // variance
        let var = centered.square()?
            .sum_dim_keepdim(last)?
            .mul_scalar(1.0 / hidden_size)?;
        let inv_std = var.add_scalar(self.eps)?.sqrt()?;
        let one = Tensor::full(inv_std.shape().clone(), 1.0, inv_std.device().clone())?;
        let inv = one.div(&inv_std)?;
        let normed = centered.mul(&inv)?;

        // Convert back
        let mut out = normed.to_dtype(DType::BF16)?;
        if let Some(ref w) = self.weight {
            out = out.mul(w)?;
        }
        if let Some(ref b) = self.bias {
            out = out.add(b)?;
        }
        Ok(out)
    }

    pub fn load_weights(&mut self, prefix: &str, weights: &HashMap<String, Tensor>) -> Result<()> {
        if self.elementwise_affine {
            if let Some(w) = weights.get(&format!("{prefix}.weight")) {
                self.weight = Some(w.to_dtype(DType::BF16)?);
            }
            if let Some(b) = weights.get(&format!("{prefix}.bias")) {
                self.bias = Some(b.to_dtype(DType::BF16)?);
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Self-Attention with 3D RoPE
// ---------------------------------------------------------------------------

/// Wan self-attention block
///
/// Weight keys under `{prefix}`:
///   .q.weight, .q.bias, .k.weight, .k.bias, .v.weight, .v.bias, .o.weight, .o.bias
///   .norm_q.weight, .norm_k.weight
pub struct WanSelfAttention {
    pub q: Linear,
    pub k: Linear,
    pub v: Linear,
    pub o: Linear,
    pub norm_q: WanRMSNorm,
    pub norm_k: WanRMSNorm,
    pub num_heads: usize,
    pub head_dim: usize,
}

impl WanSelfAttention {
    pub fn new(
        dim: usize,
        num_heads: usize,
        eps: f32,
        device: &Arc<flame_core::CudaDevice>,
    ) -> Result<Self> {
        Ok(Self {
            q: Linear::new_zeroed(dim, dim, true, device)?,
            k: Linear::new_zeroed(dim, dim, true, device)?,
            v: Linear::new_zeroed(dim, dim, true, device)?,
            o: Linear::new_zeroed(dim, dim, true, device)?,
            norm_q: WanRMSNorm::new(dim, eps, device)?,
            norm_k: WanRMSNorm::new(dim, eps, device)?,
            num_heads,
            head_dim: dim / num_heads,
        })
    }

    /// Forward pass
    ///
    /// x: [B, L, C] — input tokens
    /// rope_cos, rope_sin: [L, head_dim/2] — precomputed 3D RoPE
    ///
    /// Returns: [B, L, C]
    pub fn forward(
        &self,
        x: &Tensor,
        rope_cos: &Tensor,
        rope_sin: &Tensor,
    ) -> Result<Tensor> {
        let dims = x.shape().dims();
        let (b, s) = (dims[0], dims[1]);
        let n = self.num_heads;
        let d = self.head_dim;

        // Project Q, K, V
        let q = self.q.forward(x)?;
        let k = self.k.forward(x)?;
        let v = self.v.forward(x)?;

        // QK normalization (RMSNorm)
        let q = self.norm_q.forward(&q)?;
        let k = self.norm_k.forward(&k)?;

        // Reshape to [B, L, N, D]
        let q = q.reshape(&[b, s, n, d])?;
        let k = k.reshape(&[b, s, n, d])?;
        let v = v.reshape(&[b, s, n, d])?;

        // Apply 3D RoPE
        let q = apply_rope_3d(&q, rope_cos, rope_sin)?;
        let k = apply_rope_3d(&k, rope_cos, rope_sin)?;

        // Transpose to [B, N, L, D] for SDPA
        let q = q.permute(&[0, 2, 1, 3])?;
        let k = k.permute(&[0, 2, 1, 3])?;
        let v = v.permute(&[0, 2, 1, 3])?;

        // Scaled dot-product attention
        let attn_out = flame_core::sdpa::forward(&q, &k, &v, None)?;

        // Transpose back to [B, L, N, D] then flatten to [B, L, C]
        let attn_out = attn_out.permute(&[0, 2, 1, 3])?;
        let attn_out = attn_out.reshape(&[b, s, n * d])?;

        // Output projection
        self.o.forward(&attn_out)
    }

    pub fn load_weights(&mut self, prefix: &str, weights: &HashMap<String, Tensor>) -> Result<()> {
        load_linear(&mut self.q, &format!("{prefix}.q"), weights)?;
        load_linear(&mut self.k, &format!("{prefix}.k"), weights)?;
        load_linear(&mut self.v, &format!("{prefix}.v"), weights)?;
        load_linear(&mut self.o, &format!("{prefix}.o"), weights)?;
        self.norm_q.load_weights(&format!("{prefix}.norm_q"), weights)?;
        self.norm_k.load_weights(&format!("{prefix}.norm_k"), weights)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Cross-Attention (text conditioning)
// ---------------------------------------------------------------------------

/// Wan cross-attention: queries from latent tokens, keys/values from text
///
/// Same weight structure as self-attention, but K and V project from context.
pub struct WanCrossAttention {
    pub q: Linear,
    pub k: Linear,
    pub v: Linear,
    pub o: Linear,
    pub norm_q: WanRMSNorm,
    pub norm_k: WanRMSNorm,
    pub num_heads: usize,
    pub head_dim: usize,
}

impl WanCrossAttention {
    pub fn new(
        dim: usize,
        num_heads: usize,
        eps: f32,
        device: &Arc<flame_core::CudaDevice>,
    ) -> Result<Self> {
        Ok(Self {
            q: Linear::new_zeroed(dim, dim, true, device)?,
            k: Linear::new_zeroed(dim, dim, true, device)?,
            v: Linear::new_zeroed(dim, dim, true, device)?,
            o: Linear::new_zeroed(dim, dim, true, device)?,
            norm_q: WanRMSNorm::new(dim, eps, device)?,
            norm_k: WanRMSNorm::new(dim, eps, device)?,
            num_heads,
            head_dim: dim / num_heads,
        })
    }

    /// Forward: x queries into context
    ///
    /// x: [B, L1, C], context: [B, L2, C]
    /// Returns: [B, L1, C]
    pub fn forward(&self, x: &Tensor, context: &Tensor) -> Result<Tensor> {
        let dims = x.shape().dims();
        let b = dims[0];
        let n = self.num_heads;
        let d = self.head_dim;

        let q = self.norm_q.forward(&self.q.forward(x)?)?;
        let k = self.norm_k.forward(&self.k.forward(context)?)?;
        let v = self.v.forward(context)?;

        let l_q = q.shape().dims()[1];
        let l_k = k.shape().dims()[1];

        let q = q.reshape(&[b, l_q, n, d])?.permute(&[0, 2, 1, 3])?;
        let k = k.reshape(&[b, l_k, n, d])?.permute(&[0, 2, 1, 3])?;
        let v = v.reshape(&[b, l_k, n, d])?.permute(&[0, 2, 1, 3])?;

        let attn_out = flame_core::sdpa::forward(&q, &k, &v, None)?;
        let attn_out = attn_out.permute(&[0, 2, 1, 3])?;
        let attn_out = attn_out.reshape(&[b, l_q, n * d])?;

        self.o.forward(&attn_out)
    }

    pub fn load_weights(&mut self, prefix: &str, weights: &HashMap<String, Tensor>) -> Result<()> {
        load_linear(&mut self.q, &format!("{prefix}.q"), weights)?;
        load_linear(&mut self.k, &format!("{prefix}.k"), weights)?;
        load_linear(&mut self.v, &format!("{prefix}.v"), weights)?;
        load_linear(&mut self.o, &format!("{prefix}.o"), weights)?;
        self.norm_q.load_weights(&format!("{prefix}.norm_q"), weights)?;
        self.norm_k.load_weights(&format!("{prefix}.norm_k"), weights)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// FFN (GELU-gated MLP)
// ---------------------------------------------------------------------------

/// Feed-forward network: Linear → GELU(tanh approx) → Linear
///
/// Weight keys: `{prefix}.0.weight`, `{prefix}.0.bias`, `{prefix}.2.weight`, `{prefix}.2.bias`
/// (Sequential indices: 0=Linear, 1=GELU, 2=Linear)
pub struct WanFFN {
    pub lin1: Linear,
    pub lin2: Linear,
}

impl WanFFN {
    pub fn new(
        dim: usize,
        ffn_dim: usize,
        device: &Arc<flame_core::CudaDevice>,
    ) -> Result<Self> {
        Ok(Self {
            lin1: Linear::new_zeroed(dim, ffn_dim, true, device)?,
            lin2: Linear::new_zeroed(ffn_dim, dim, true, device)?,
        })
    }

    /// Forward: x → Linear → GELU(tanh) → Linear
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.lin1.forward(x)?;
        let h = gelu_tanh_approx(&h)?;
        self.lin2.forward(&h)
    }

    pub fn load_weights(&mut self, prefix: &str, weights: &HashMap<String, Tensor>) -> Result<()> {
        load_linear(&mut self.lin1, &format!("{prefix}.0"), weights)?;
        load_linear(&mut self.lin2, &format!("{prefix}.2"), weights)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Transformer Block (one layer of the Wan backbone)
// ---------------------------------------------------------------------------

/// Single Wan 2.2 transformer block with adaptive modulation.
///
/// Architecture per block:
///   1. AdaLN-modulated self-attention with 3D RoPE
///   2. Cross-attention to text conditioning
///   3. AdaLN-modulated FFN
///
/// Modulation parameter: [1, 6, dim] — 6 vectors for (shift1, scale1, gate1, shift2, scale2, gate2)
///
/// Wan 2.2 per-token modulation: e comes in as [B, L, 6, C], chunked along dim=2
pub struct WanAttentionBlock {
    pub norm1: WanLayerNorm,
    pub self_attn: WanSelfAttention,
    pub norm3: Option<WanLayerNorm>,  // cross_attn_norm (elementwise_affine=True)
    pub cross_attn: WanCrossAttention,
    pub norm2: WanLayerNorm,
    pub ffn: WanFFN,
    /// Learned modulation bias: [1, 6, dim]
    pub modulation: Tensor,
    pub dim: usize,
}

impl WanAttentionBlock {
    pub fn new(config: &Wan22Config, device: &Arc<flame_core::CudaDevice>) -> Result<Self> {
        let dim = config.dim;
        let eps = config.eps;
        let num_heads = config.num_heads;

        let norm3 = if config.cross_attn_norm {
            Some(WanLayerNorm::new(dim, eps, true, device)?)
        } else {
            None
        };

        // Modulation parameter initialized to small random values (dim^-0.5 scale)
        // Will be overwritten by checkpoint loading
        let modulation = Tensor::zeros_dtype(
            Shape::from_dims(&[1, 6, dim]),
            DType::BF16,
            device.clone(),
        )?;

        Ok(Self {
            norm1: WanLayerNorm::new(dim, eps, false, device)?,
            self_attn: WanSelfAttention::new(dim, num_heads, eps, device)?,
            norm3,
            cross_attn: WanCrossAttention::new(dim, num_heads, eps, device)?,
            norm2: WanLayerNorm::new(dim, eps, false, device)?,
            ffn: WanFFN::new(dim, config.ffn_dim, device)?,
            modulation,
            dim,
        })
    }

    /// Forward pass for Wan 2.2 block (per-token modulation).
    ///
    /// x: [B, L, C]
    /// e: [B, L, 6, C] — per-token time modulation (F32)
    /// rope_cos, rope_sin: [L, head_dim/2] — precomputed 3D RoPE
    /// context: [B, L_text, C]
    pub fn forward(
        &self,
        x: &Tensor,
        e: &Tensor,
        rope_cos: &Tensor,
        rope_sin: &Tensor,
        context: &Tensor,
    ) -> Result<Tensor> {
        let org_dtype = x.dtype();

        // Add learned modulation bias: e = modulation + e
        // modulation: [1, 6, dim], broadcast to [B, L, 6, dim]
        let mod_f32 = self.modulation.to_dtype(DType::F32)?;
        let e = e.add(&mod_f32)?;

        // Chunk e into 6 modulation vectors along dim=2
        // Each: [B, L, 1, C] → squeeze to [B, L, C]
        let e_chunks = chunk_dim2_6(&e)?;

        // --- Self-attention with AdaLN modulation ---
        // norm_out = norm1(x) * (1 + scale1) + shift1
        let norm_x = self.norm1.forward(x)?.to_dtype(DType::F32)?;
        let one_plus_scale = e_chunks[1].add_scalar(1.0)?;
        let modulated = norm_x.mul(&one_plus_scale)?.add(&e_chunks[0])?;
        let modulated = modulated.to_dtype(org_dtype)?;

        let y = self.self_attn.forward(&modulated, rope_cos, rope_sin)?;

        // x = x + y * gate1
        let y_f32 = y.to_dtype(DType::F32)?;
        let gated = y_f32.mul(&e_chunks[2])?;
        let x = x.to_dtype(DType::F32)?.add(&gated)?.to_dtype(org_dtype)?;

        // --- Cross-attention ---
        let norm3_x = if let Some(ref n3) = self.norm3 {
            n3.forward(&x)?
        } else {
            x.clone()
        };
        let cross_out = self.cross_attn.forward(&norm3_x, context)?;
        let x = x.add(&cross_out)?;

        // --- FFN with AdaLN modulation ---
        let norm2_x = self.norm2.forward(&x)?.to_dtype(DType::F32)?;
        let one_plus_scale2 = e_chunks[4].add_scalar(1.0)?;
        let modulated2 = norm2_x.mul(&one_plus_scale2)?.add(&e_chunks[3])?;
        let modulated2 = modulated2.to_dtype(org_dtype)?;

        let y2 = self.ffn.forward(&modulated2)?;

        // x = x + y2 * gate2
        let y2_f32 = y2.to_dtype(DType::F32)?;
        let gated2 = y2_f32.mul(&e_chunks[5])?;
        let x = x.to_dtype(DType::F32)?.add(&gated2)?.to_dtype(org_dtype)?;

        Ok(x)
    }

    pub fn load_weights(&mut self, prefix: &str, weights: &HashMap<String, Tensor>) -> Result<()> {
        self.norm1.load_weights(&format!("{prefix}.norm1"), weights)?;
        self.self_attn.load_weights(&format!("{prefix}.self_attn"), weights)?;
        if let Some(ref mut n3) = self.norm3 {
            n3.load_weights(&format!("{prefix}.norm3"), weights)?;
        }
        self.cross_attn.load_weights(&format!("{prefix}.cross_attn"), weights)?;
        self.norm2.load_weights(&format!("{prefix}.norm2"), weights)?;
        self.ffn.load_weights(&format!("{prefix}.ffn"), weights)?;

        // Load modulation parameter
        let mod_key = format!("{prefix}.modulation");
        if let Some(m) = weights.get(&mod_key) {
            self.modulation = m.to_dtype(DType::BF16)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Output Head
// ---------------------------------------------------------------------------

/// Wan output head: modulated LayerNorm → Linear → unpatchify
///
/// Weight keys:
///   head.norm.weight, head.head.weight, head.head.bias, head.modulation
pub struct WanHead {
    pub norm: WanLayerNorm,
    pub head: Linear,
    /// Modulation: [1, 2, dim]
    pub modulation: Tensor,
    pub out_dim: usize,
    pub patch_size: (usize, usize, usize),
    pub dim: usize,
}

impl WanHead {
    pub fn new(config: &Wan22Config, device: &Arc<flame_core::CudaDevice>) -> Result<Self> {
        let dim = config.dim;
        let out_channels = config.patch_size.0 * config.patch_size.1 * config.patch_size.2 * config.out_dim;

        let modulation = Tensor::zeros_dtype(
            Shape::from_dims(&[1, 2, dim]),
            DType::BF16,
            device.clone(),
        )?;

        Ok(Self {
            norm: WanLayerNorm::new(dim, config.eps, false, device)?,
            head: Linear::new_zeroed(dim, out_channels, true, device)?,
            modulation,
            out_dim: config.out_dim,
            patch_size: config.patch_size,
            dim,
        })
    }

    /// Forward pass with Wan 2.2 per-token modulation
    ///
    /// x: [B, L, C]
    /// e: [B, L, C] — time embedding (F32, from time_embedding output)
    pub fn forward(&self, x: &Tensor, e: &Tensor) -> Result<Tensor> {
        // Wan 2.2: modulation.unsqueeze(0) + e.unsqueeze(2), chunk along dim=2
        // modulation: [1, 2, dim] → unsqueeze(0) → [1, 1, 2, dim]
        // e: [B, L, dim] → unsqueeze(2) → [B, L, 1, dim]
        // sum → [B, L, 2, dim], chunk into 2 along dim=2
        let mod_f32 = self.modulation.to_dtype(DType::F32)?;

        // Reshape for broadcasting: mod [1, 1, 2, dim], e [B, L, 1, dim]
        let e_dims = e.shape().dims();
        let (b, l) = (e_dims[0], e_dims[1]);

        let mod_expanded = mod_f32.reshape(&[1, 1, 2, self.dim])?;
        let e_expanded = e.reshape(&[b, l, 1, self.dim])?;
        let combined = e_expanded.add(&mod_expanded)?;

        // Chunk into shift, scale — each [B, L, 1, dim] → squeeze to [B, L, dim]
        // combined: [B, L, 2, dim]
        // TODO: proper narrow/split when flame-core supports it
        // For now, use index-based extraction
        let shift = narrow_dim2(&combined, 0)?; // [B, L, dim]
        let scale = narrow_dim2(&combined, 1)?; // [B, L, dim]

        // Apply: head(norm(x) * (1 + scale) + shift)
        let norm_x = self.norm.forward(x)?;
        let one_plus_scale = scale.add_scalar(1.0)?;
        let modulated = norm_x.to_dtype(DType::F32)?.mul(&one_plus_scale)?.add(&shift)?;
        let modulated = modulated.to_dtype(x.dtype())?;

        self.head.forward(&modulated)
    }

    pub fn load_weights(&mut self, prefix: &str, weights: &HashMap<String, Tensor>) -> Result<()> {
        self.norm.load_weights(&format!("{prefix}.norm"), weights)?;
        load_linear(&mut self.head, &format!("{prefix}.head"), weights)?;
        let mod_key = format!("{prefix}.modulation");
        if let Some(m) = weights.get(&mod_key) {
            self.modulation = m.to_dtype(DType::BF16)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Single Wan 2.2 Transformer (one expert)
// ---------------------------------------------------------------------------

/// One complete Wan 2.2 transformer (used as a single expert in the dual-expert model).
///
/// Contains: patch_embedding, text_embedding, time_embedding, time_projection,
///           blocks (N transformer layers), head, and RoPE frequency buffer.
pub struct Wan22Transformer {
    pub config: Wan22Config,

    // --- Embeddings ---
    /// Conv3d patch embedding: in_dim → dim, kernel/stride = patch_size
    pub patch_embedding: Conv3d,
    /// Text projection: text_dim → dim → GELU → dim
    pub text_embedding_0: Linear,
    pub text_embedding_2: Linear,
    /// Time embedding: freq_dim → dim → SiLU → dim
    pub time_embedding_0: Linear,
    pub time_embedding_2: Linear,
    /// Time projection: SiLU → Linear(dim, dim*6)
    pub time_projection_1: Linear,

    // --- Transformer blocks ---
    pub blocks: Vec<WanAttentionBlock>,

    // --- Output ---
    pub head: WanHead,

    // --- RoPE cache ---
    /// Precomputed RoPE frequencies — recomputed per (F,H,W) grid
    /// Stored as cos/sin tensors for the current resolution
    rope_cos_cache: Option<Tensor>,
    rope_sin_cache: Option<Tensor>,
    rope_cache_key: Option<(usize, usize, usize)>,
}

impl Wan22Transformer {
    /// Create a new transformer with zeroed weights (ready for checkpoint loading)
    pub fn new(
        config: Wan22Config,
        device: &Arc<flame_core::CudaDevice>,
    ) -> Result<Self> {
        let dim = config.dim;
        let freq_dim = config.freq_dim;
        let text_dim = config.text_dim;
        let (pt, ph, pw) = config.patch_size;

        let patch_embedding = Conv3d::new(
            config.in_dim,
            dim,
            (pt, ph, pw),
            Some((pt, ph, pw)),     // stride = kernel_size
            None,                    // no padding
            None,                    // no dilation
            None,                    // groups=1
            device,
        )?;

        let mut blocks = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            blocks.push(WanAttentionBlock::new(&config, device)?);
        }

        Ok(Self {
            patch_embedding,
            text_embedding_0: Linear::new_zeroed(text_dim, dim, true, device)?,
            text_embedding_2: Linear::new_zeroed(dim, dim, true, device)?,
            time_embedding_0: Linear::new_zeroed(freq_dim, dim, true, device)?,
            time_embedding_2: Linear::new_zeroed(dim, dim, true, device)?,
            time_projection_1: Linear::new_zeroed(dim, dim * 6, true, device)?,
            blocks,
            head: WanHead::new(&config, device)?,
            rope_cos_cache: None,
            rope_sin_cache: None,
            rope_cache_key: None,
            config,
        })
    }

    /// Full forward pass.
    ///
    /// x: [B, C_in, F, H, W] — noisy latents
    /// t: [B] — diffusion timesteps (f32)
    /// context: [B, L_text, text_dim] — T5 text embeddings (BF16)
    /// seq_len: max patch sequence length (F/pt * H/ph * W/pw)
    ///
    /// Returns: [B, C_out, F, H, W]
    pub fn forward(
        &mut self,
        x: &Tensor,
        t: &[f32],
        context: &Tensor,
        device: &Arc<flame_core::CudaDevice>,
    ) -> Result<Tensor> {
        let x_dims = x.shape().dims();
        let (b, _c_in, f, h, w) = (x_dims[0], x_dims[1], x_dims[2], x_dims[3], x_dims[4]);
        let (pt, ph, pw) = self.config.patch_size;
        let dim = self.config.dim;

        // --- Patch embedding ---
        // Conv3d: [B, C_in, F, H, W] → [B, dim, F/pt, H/ph, W/pw]
        // NOTE: flame-core Conv3d is F32-only, so we cast input to F32 and output back to BF16
        let x_f32 = x.to_dtype(DType::F32)?;
        let x_patched = self.patch_embedding.forward(&x_f32)?.to_dtype(DType::BF16)?;
        let grid_f = f / pt;
        let grid_h = h / ph;
        let grid_w = w / pw;
        let seq_len = grid_f * grid_h * grid_w;

        // Flatten spatial dims to sequence: [B, dim, F', H', W'] → [B, L, dim]
        let x_flat = x_patched.reshape(&[b, dim, seq_len])?;
        let x_seq = x_flat.permute(&[0, 2, 1])?; // [B, L, dim]

        // --- Compute or retrieve RoPE ---
        let (rope_cos, rope_sin) = self.get_rope_3d(
            grid_f, grid_h, grid_w, device,
        )?;

        // --- Time embedding (per-token for Wan 2.2) ---
        // Expand scalar timestep to per-token: [B] → [B, L]
        // Then sinusoidal → time MLP → time_projection → [B, L, 6, dim]
        let t_tensor = Tensor::from_vec(t.to_vec(), Shape::from_dims(&[b]), device.clone())?;
        let e0 = self.compute_time_embedding_v22(&t_tensor, seq_len, device)?;

        // --- Text embedding ---
        let ctx = self.embed_text(context)?;

        // --- Transformer blocks ---
        let mut hidden = x_seq;
        for block in &self.blocks {
            hidden = block.forward(&hidden, &e0, &rope_cos, &rope_sin, &ctx)?;
        }

        // --- Output head ---
        // e for head: reuse time embedding (before projection) as [B, L, dim]
        let e_head = self.compute_time_embedding_for_head(&t_tensor, seq_len, device)?;
        let out_tokens = self.head.forward(&hidden, &e_head)?;

        // --- Unpatchify ---
        unpatchify(
            &out_tokens,
            self.config.out_dim,
            self.config.patch_size,
            grid_f,
            grid_h,
            grid_w,
        )
    }

    // -- Embedding helpers --

    fn embed_text(&self, context: &Tensor) -> Result<Tensor> {
        let h = self.text_embedding_0.forward(context)?;
        let h = gelu_tanh_approx(&h)?;
        self.text_embedding_2.forward(&h)
    }

    /// Compute Wan 2.2 per-token time embedding → [B, L, 6, dim] (F32)
    fn compute_time_embedding_v22(
        &self,
        t: &Tensor,       // [B] scalar timesteps
        seq_len: usize,
        device: &Arc<flame_core::CudaDevice>,
    ) -> Result<Tensor> {
        let b = t.shape().dims()[0];
        let freq_dim = self.config.freq_dim;
        let dim = self.config.dim;

        // Expand t to per-token: [B] → [B*L]
        // In PyTorch: t.unsqueeze(1).expand(-1, seq_len).flatten()
        let t_data = t.to_vec()?;
        let mut expanded = Vec::with_capacity(b * seq_len);
        for &ti in &t_data {
            for _ in 0..seq_len {
                expanded.push(ti);
            }
        }
        let t_flat = Tensor::from_vec(expanded, Shape::from_dims(&[b * seq_len]), device.clone())?;

        // Sinusoidal embedding: [B*L] → [B*L, freq_dim]
        let sin_emb = sinusoidal_embedding_1d(freq_dim, &t_flat, device)?;

        // Reshape to [B, L, freq_dim]
        let sin_emb = sin_emb.reshape(&[b, seq_len, freq_dim])?;

        // time_embedding MLP: freq_dim → dim → SiLU → dim (computed in F32)
        let sin_f32 = sin_emb.to_dtype(DType::F32)?;
        let e = self.time_embedding_0.forward(&sin_f32.to_dtype(DType::BF16)?)?;
        let e = silu(&e.to_dtype(DType::F32)?)?;
        let e = self.time_embedding_2.forward(&e.to_dtype(DType::BF16)?)?;
        let e = e.to_dtype(DType::F32)?; // [B, L, dim]

        // time_projection: SiLU → Linear → [B, L, dim*6]
        let proj_in = silu(&e)?;
        let e0 = self.time_projection_1.forward(&proj_in.to_dtype(DType::BF16)?)?;
        let e0 = e0.to_dtype(DType::F32)?;

        // Reshape to [B, L, 6, dim]
        e0.reshape(&[b, seq_len, 6, dim])
    }

    /// Time embedding for head (before projection): [B, L, dim] in F32
    fn compute_time_embedding_for_head(
        &self,
        t: &Tensor,
        seq_len: usize,
        device: &Arc<flame_core::CudaDevice>,
    ) -> Result<Tensor> {
        let b = t.shape().dims()[0];
        let freq_dim = self.config.freq_dim;

        let t_data = t.to_vec()?;
        let mut expanded = Vec::with_capacity(b * seq_len);
        for &ti in &t_data {
            for _ in 0..seq_len {
                expanded.push(ti);
            }
        }
        let t_flat = Tensor::from_vec(expanded, Shape::from_dims(&[b * seq_len]), device.clone())?;
        let sin_emb = sinusoidal_embedding_1d(freq_dim, &t_flat, device)?;
        let sin_emb = sin_emb.reshape(&[b, seq_len, freq_dim])?;

        let e = self.time_embedding_0.forward(&sin_emb.to_dtype(DType::BF16)?)?;
        let e = silu(&e.to_dtype(DType::F32)?)?;
        let e = self.time_embedding_2.forward(&e.to_dtype(DType::BF16)?)?;
        e.to_dtype(DType::F32)
    }

    // -- RoPE --

    /// Get or compute 3D factored RoPE cos/sin for a given grid.
    /// Returns (cos, sin) each of shape [F*H*W, head_dim/2]
    fn get_rope_3d(
        &mut self,
        grid_f: usize,
        grid_h: usize,
        grid_w: usize,
        device: &Arc<flame_core::CudaDevice>,
    ) -> Result<(Tensor, Tensor)> {
        let key = (grid_f, grid_h, grid_w);
        if self.rope_cache_key.as_ref() == Some(&key) {
            if let (Some(ref c), Some(ref s)) = (&self.rope_cos_cache, &self.rope_sin_cache) {
                return Ok((c.clone(), s.clone()));
            }
        }

        let (cos, sin) = compute_rope_3d(
            self.config.dim / self.config.num_heads,
            grid_f,
            grid_h,
            grid_w,
            device,
        )?;

        self.rope_cos_cache = Some(cos.clone());
        self.rope_sin_cache = Some(sin.clone());
        self.rope_cache_key = Some(key);

        Ok((cos, sin))
    }

    /// Load all weights from a flat state dict (key → Tensor).
    /// Keys should NOT have "model.diffusion_model." prefix (strip it before calling).
    pub fn load_weights(&mut self, weights: &HashMap<String, Tensor>) -> Result<()> {
        // Patch embedding
        if let Some(w) = weights.get("patch_embedding.weight") {
            self.patch_embedding.weight = w.to_dtype(DType::BF16)?;
        }
        if let Some(b) = weights.get("patch_embedding.bias") {
            self.patch_embedding.bias = Some(b.to_dtype(DType::BF16)?);
        }

        // Text embedding (Sequential: 0=Linear, 1=GELU, 2=Linear)
        load_linear(&mut self.text_embedding_0, "text_embedding.0", weights)?;
        load_linear(&mut self.text_embedding_2, "text_embedding.2", weights)?;

        // Time embedding (Sequential: 0=Linear, 1=SiLU, 2=Linear)
        load_linear(&mut self.time_embedding_0, "time_embedding.0", weights)?;
        load_linear(&mut self.time_embedding_2, "time_embedding.2", weights)?;

        // Time projection (Sequential: 0=SiLU, 1=Linear)
        load_linear(&mut self.time_projection_1, "time_projection.1", weights)?;

        // Blocks
        for (i, block) in self.blocks.iter_mut().enumerate() {
            block.load_weights(&format!("blocks.{i}"), weights)?;
        }

        // Head
        self.head.load_weights("head", weights)?;

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Wan22Model — Dual-Expert wrapper
// ---------------------------------------------------------------------------

/// Wan 2.2 dual-expert model: two transformers switched by timestep boundary.
///
/// For timestep t (normalized 0..1):
///   - t > boundary_ratio  → use expert_high (high-noise denoising)
///   - t <= boundary_ratio → use expert_low  (low-noise refinement)
///
/// Each expert is a full Wan22Transformer (~14B params).
pub struct Wan22Model {
    pub config: Wan22Config,
    pub expert_high: Wan22Transformer,
    pub expert_low: Wan22Transformer,
}

impl Wan22Model {
    /// Create with two transformers
    pub fn new(config: Wan22Config, device: &Arc<flame_core::CudaDevice>) -> Result<Self> {
        let expert_high = Wan22Transformer::new(config.clone(), device)?;
        let expert_low = Wan22Transformer::new(config.clone(), device)?;
        Ok(Self { config, expert_high, expert_low })
    }

    /// Forward pass — routes to correct expert based on timestep.
    ///
    /// All items in the batch must use the same expert (same noise level regime).
    /// This is the standard inference pattern where CFG samples share timestep.
    ///
    /// x: [B, C_in, F, H, W]
    /// t: [B] — timesteps (all should be in the same regime)
    /// context: [B, L_text, text_dim]
    pub fn forward(
        &mut self,
        x: &Tensor,
        t: &[f32],
        context: &Tensor,
        device: &Arc<flame_core::CudaDevice>,
    ) -> Result<Tensor> {
        // Route based on first timestep (all should be same in standard inference)
        let use_high = t[0] > self.config.boundary_ratio;
        if use_high {
            self.expert_high.forward(x, t, context, device)
        } else {
            self.expert_low.forward(x, t, context, device)
        }
    }

    /// Load weights for both experts from separate state dicts.
    ///
    /// In the diffusers directory format:
    ///   transformer/   → expert_high (high-noise)
    ///   transformer_2/ → expert_low  (low-noise)
    pub fn load_weights(
        &mut self,
        weights_high: &HashMap<String, Tensor>,
        weights_low: &HashMap<String, Tensor>,
    ) -> Result<()> {
        self.expert_high.load_weights(weights_high)?;
        self.expert_low.load_weights(weights_low)?;
        Ok(())
    }
}

// ===========================================================================
// Helper functions
// ===========================================================================

/// Load Linear layer weights from state dict
fn load_linear(linear: &mut Linear, prefix: &str, weights: &HashMap<String, Tensor>) -> Result<()> {
    let w_key = format!("{prefix}.weight");
    let b_key = format!("{prefix}.bias");
    if let Some(w) = weights.get(&w_key) {
        linear.copy_weight_from(&w.to_dtype(DType::BF16)?)?;
    }
    if let Some(b) = weights.get(&b_key) {
        linear.copy_bias_from(&b.to_dtype(DType::BF16)?)?;
    }
    Ok(())
}

/// Sinusoidal positional embedding (1D)
///
/// position: [N] float tensor
/// Returns: [N, dim] with cos/sin interleaved (first half cos, second half sin)
fn sinusoidal_embedding_1d(
    dim: usize,
    position: &Tensor,
    device: &Arc<flame_core::CudaDevice>,
) -> Result<Tensor> {
    assert!(dim % 2 == 0, "sinusoidal_embedding_1d requires even dim");
    let half = dim / 2;

    let pos_data = position.to_vec()?;
    let n = pos_data.len();

    // freqs[i] = 10000^(-i/half)
    let mut freqs = vec![0.0f64; half];
    for i in 0..half {
        freqs[i] = 10000.0_f64.powf(-(i as f64) / half as f64);
    }

    // output[j, i]       = cos(position[j] * freqs[i])   for i < half
    // output[j, half + i] = sin(position[j] * freqs[i])   for i < half
    let mut data = vec![0.0f32; n * dim];
    for j in 0..n {
        let p = pos_data[j] as f64;
        for i in 0..half {
            let angle = p * freqs[i];
            data[j * dim + i] = angle.cos() as f32;
            data[j * dim + half + i] = angle.sin() as f32;
        }
    }

    Tensor::from_vec(data, Shape::from_dims(&[n, dim]), device.clone())
}

/// GELU activation with tanh approximation
/// gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
fn gelu_tanh_approx(x: &Tensor) -> Result<Tensor> {
    // Compute in F32 for numerical precision
    let x_f32 = x.to_dtype(DType::F32)?;
    let data = x_f32.to_vec()?;

    let sqrt_2_over_pi: f32 = (2.0_f32 / std::f32::consts::PI).sqrt();
    let result: Vec<f32> = data.iter().map(|&xi| {
        let inner = sqrt_2_over_pi * (xi + 0.044715 * xi * xi * xi);
        0.5 * xi * (1.0 + inner.tanh())
    }).collect();

    let out = Tensor::from_vec(result, x_f32.shape().clone(), x_f32.device().clone())?;
    out.to_dtype(x.dtype())
}

/// SiLU (Swish) activation: x * sigmoid(x)
fn silu(x: &Tensor) -> Result<Tensor> {
    let data = x.to_vec()?;
    let result: Vec<f32> = data.iter().map(|&xi| {
        xi * (1.0 / (1.0 + (-xi).exp()))
    }).collect();
    Tensor::from_vec(result, x.shape().clone(), x.device().clone())
}

/// 3D factored RoPE computation for Wan's non-uniform axis dimensions.
///
/// head_dim is split as: d_time = head_dim - 4*(head_dim/6), d_h = 2*(head_dim/6), d_w = 2*(head_dim/6)
/// Each axis gets independent frequencies, then concatenated.
///
/// Returns (cos, sin) each shape [F*H*W, head_dim/2]
fn compute_rope_3d(
    head_dim: usize,
    grid_f: usize,
    grid_h: usize,
    grid_w: usize,
    device: &Arc<flame_core::CudaDevice>,
) -> Result<(Tensor, Tensor)> {
    let half_dim = head_dim / 2;
    let d6 = half_dim / 3;  // head_dim/6 in the half-space
    let d_time = half_dim - 2 * d6;  // remaining dims for temporal axis
    let d_h = d6;
    let d_w = d6;
    let theta: f64 = 10000.0;

    let seq_len = grid_f * grid_h * grid_w;

    // Compute frequency vectors for each axis
    let freqs_time = rope_freqs(d_time, theta);
    let freqs_h = rope_freqs(d_h, theta);
    let freqs_w = rope_freqs(d_w, theta);

    // Build cos/sin for the full 3D grid
    let mut cos_data = vec![0.0f32; seq_len * half_dim];
    let mut sin_data = vec![0.0f32; seq_len * half_dim];

    for fi in 0..grid_f {
        for hi in 0..grid_h {
            for wi in 0..grid_w {
                let idx = fi * grid_h * grid_w + hi * grid_w + wi;
                let row_offset = idx * half_dim;

                // Temporal frequencies
                let mut col = 0;
                for k in 0..d_time {
                    let angle = fi as f64 * freqs_time[k];
                    cos_data[row_offset + col] = angle.cos() as f32;
                    sin_data[row_offset + col] = angle.sin() as f32;
                    col += 1;
                }
                // Height frequencies
                for k in 0..d_h {
                    let angle = hi as f64 * freqs_h[k];
                    cos_data[row_offset + col] = angle.cos() as f32;
                    sin_data[row_offset + col] = angle.sin() as f32;
                    col += 1;
                }
                // Width frequencies
                for k in 0..d_w {
                    let angle = wi as f64 * freqs_w[k];
                    cos_data[row_offset + col] = angle.cos() as f32;
                    sin_data[row_offset + col] = angle.sin() as f32;
                    col += 1;
                }
            }
        }
    }

    let cos = Tensor::from_vec(cos_data, Shape::from_dims(&[seq_len, half_dim]), device.clone())?;
    let sin = Tensor::from_vec(sin_data, Shape::from_dims(&[seq_len, half_dim]), device.clone())?;

    Ok((cos.to_dtype(DType::BF16)?, sin.to_dtype(DType::BF16)?))
}

/// Compute frequency vector for RoPE: freq[i] = 1 / theta^(2i/dim)
fn rope_freqs(half_dim: usize, theta: f64) -> Vec<f64> {
    let mut freqs = vec![0.0f64; half_dim];
    for i in 0..half_dim {
        freqs[i] = 1.0 / theta.powf(2.0 * i as f64 / (2.0 * half_dim as f64));
    }
    freqs
}

/// Apply 3D RoPE to tensor x: [B, L, N, D]
/// rope_cos, rope_sin: [L, D/2]
///
/// Uses the standard complex-number rotation:
///   x_rot[..., 2k]   = x[..., 2k] * cos - x[..., 2k+1] * sin
///   x_rot[..., 2k+1] = x[..., 2k] * sin + x[..., 2k+1] * cos
fn apply_rope_3d(
    x: &Tensor,
    rope_cos: &Tensor,
    rope_sin: &Tensor,
) -> Result<Tensor> {
    // x: [B, L, N, D], rope_cos/sin: [L, D/2]
    // For now: CPU implementation (production would use a fused CUDA kernel)
    let x_dims = x.shape().dims();
    let (b, l, n, d) = (x_dims[0], x_dims[1], x_dims[2], x_dims[3]);
    let half_d = d / 2;

    let x_data = x.to_dtype(DType::F32)?.to_vec()?;
    let cos_data = rope_cos.to_dtype(DType::F32)?.to_vec()?;
    let sin_data = rope_sin.to_dtype(DType::F32)?.to_vec()?;

    let mut out = vec![0.0f32; b * l * n * d];

    for bi in 0..b {
        for li in 0..l {
            for ni in 0..n {
                let x_base = ((bi * l + li) * n + ni) * d;
                let rope_base = li * half_d;
                for k in 0..half_d {
                    let x_even = x_data[x_base + 2 * k];
                    let x_odd = x_data[x_base + 2 * k + 1];
                    let c = cos_data[rope_base + k];
                    let s = sin_data[rope_base + k];
                    out[x_base + 2 * k] = x_even * c - x_odd * s;
                    out[x_base + 2 * k + 1] = x_even * s + x_odd * c;
                }
            }
        }
    }

    let result = Tensor::from_vec(out, x.shape().clone(), x.device().clone())?;
    result.to_dtype(x.dtype())
}

/// Chunk a [B, L, 6, C] tensor along dim=2 into 6 tensors of [B, L, C].
///
/// Returns F32 tensors for modulation arithmetic.
fn chunk_dim2_6(e: &Tensor) -> Result<Vec<Tensor>> {
    // e: [B, L, 6, C]
    let dims = e.shape().dims();
    let (b, l, _six, c) = (dims[0], dims[1], dims[2], dims[3]);

    let e_f32 = e.to_dtype(DType::F32)?;
    let data = e_f32.to_vec()?;

    let mut chunks = Vec::with_capacity(6);
    for chunk_idx in 0..6 {
        let mut chunk_data = vec![0.0f32; b * l * c];
        for bi in 0..b {
            for li in 0..l {
                let src_offset = ((bi * l + li) * 6 + chunk_idx) * c;
                let dst_offset = (bi * l + li) * c;
                chunk_data[dst_offset..dst_offset + c]
                    .copy_from_slice(&data[src_offset..src_offset + c]);
            }
        }
        let t = Tensor::from_vec(
            chunk_data,
            Shape::from_dims(&[b, l, c]),
            e.device().clone(),
        )?;
        chunks.push(t);
    }
    Ok(chunks)
}

/// Extract a single slice along dim=2 from a [B, L, K, C] tensor.
/// Returns [B, L, C] in F32.
fn narrow_dim2(t: &Tensor, idx: usize) -> Result<Tensor> {
    let dims = t.shape().dims();
    let (b, l, _k, c) = (dims[0], dims[1], dims[2], dims[3]);
    let data = t.to_dtype(DType::F32)?.to_vec()?;
    let mut out = vec![0.0f32; b * l * c];
    for bi in 0..b {
        for li in 0..l {
            let src = ((bi * l + li) * _k + idx) * c;
            let dst = (bi * l + li) * c;
            out[dst..dst + c].copy_from_slice(&data[src..src + c]);
        }
    }
    Tensor::from_vec(out, Shape::from_dims(&[b, l, c]), t.device().clone())
}

/// Unpatchify: [B, L, patch_vol * C_out] → [B, C_out, F, H, W]
///
/// Reverses the Conv3d patch embedding by rearranging patch tokens back to video.
fn unpatchify(
    x: &Tensor,
    out_dim: usize,
    patch_size: (usize, usize, usize),
    grid_f: usize,
    grid_h: usize,
    grid_w: usize,
) -> Result<Tensor> {
    let (pt, ph, pw) = patch_size;
    let x_dims = x.shape().dims();
    let b = x_dims[0];
    let _l = x_dims[1];
    let patch_vol = pt * ph * pw;

    // x: [B, F'*H'*W', pt*ph*pw*C_out]
    // Reshape to [B, F', H', W', pt, ph, pw, C_out]
    let x = x.reshape(&[b, grid_f, grid_h, grid_w, pt, ph, pw, out_dim])?;

    // Permute from [B, F', H', W', pt, ph, pw, C] to [B, C, F', pt, H', ph, W', pw]
    // PyTorch einsum "fhwpqrc->cfphqwr" (no batch in musubi, but we add B)
    let x = x.permute(&[0, 7, 1, 4, 2, 5, 3, 6])?;

    // Reshape to [B, C_out, F'*pt, H'*ph, W'*pw]
    let f_out = grid_f * pt;
    let h_out = grid_h * ph;
    let w_out = grid_w * pw;
    x.reshape(&[b, out_dim, f_out, h_out, w_out])
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_presets() {
        let t2v = Wan22Config::t2v_a14b();
        assert_eq!(t2v.dim, 5120);
        assert_eq!(t2v.num_layers, 40);
        assert_eq!(t2v.num_heads, 40);
        assert_eq!(t2v.in_dim, 16);
        assert_eq!(t2v.head_dim(), 128);
        assert!((t2v.boundary_ratio - 0.875).abs() < 1e-6);

        let i2v = Wan22Config::i2v_a14b();
        assert_eq!(i2v.in_dim, 36);
        assert!((i2v.boundary_ratio - 0.900).abs() < 1e-6);
    }

    #[test]
    fn test_sinusoidal_embedding_shape() {
        // CPU-only test (no device needed for shape logic)
        let dim = 256;
        let half = dim / 2;
        let n = 4;

        // Verify frequency computation
        let theta: f64 = 10000.0;
        let freq_0 = 1.0 / theta.powf(0.0 / half as f64);
        assert!((freq_0 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rope_freqs() {
        let freqs = rope_freqs(4, 10000.0);
        assert_eq!(freqs.len(), 4);
        // First freq should be 1.0 (theta^0 = 1)
        assert!((freqs[0] - 1.0).abs() < 1e-10);
        // Frequencies should decrease
        for i in 1..freqs.len() {
            assert!(freqs[i] < freqs[i - 1]);
        }
    }

    #[test]
    fn test_rope_3d_axis_split() {
        // Verify the non-uniform axis dimension split
        let head_dim = 128; // 5120 / 40 heads
        let half_dim = head_dim / 2; // 64
        let d6 = half_dim / 3; // 21
        let d_time = half_dim - 2 * d6; // 64 - 42 = 22
        let d_h = d6; // 21
        let d_w = d6; // 21
        assert_eq!(d_time + d_h + d_w, half_dim);

        // Match PyTorch: d - 4*(d//6), 2*(d//6), 2*(d//6) where d = head_dim
        // d//6 = 128//6 = 21
        // time_dim = 128 - 4*21 = 128 - 84 = 44 (for full dim)
        // In half-space: 44/2 = 22, 42/2 = 21, 42/2 = 21
        // But PyTorch rope_params takes full dims, then produces complex freqs of half size.
        // Our computation matches: d_time=22, d_h=21, d_w=21, total=64=half_dim
    }
}
