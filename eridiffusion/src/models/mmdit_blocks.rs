//! MMDiT (Multimodal Diffusion Transformer) blocks for SD3.5
//!
//! This module provides the core blocks for the SD3.5 diffusion model,
//! which uses a multimodal architecture processing image and text jointly.

use crate::ops::{GroupNorm, LayerNorm, Linear, RMSNorm};
use flame_core::device::Device;
use flame_core::{
    autograd::{AutogradContext, Op},
    DType, Error, Result, Shape, Tensor, TensorId,
};
use std::sync::Arc;

/// Configuration for MMDiT model
#[derive(Clone)]
pub struct MMDiTConfig {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub depth: usize,
    pub mlp_ratio: f32,
    pub qkv_bias: bool,
    pub qk_norm: bool,
    pub pos_embed_max_size: usize,
}

impl Default for MMDiTConfig {
    fn default() -> Self {
        Self {
            hidden_size: 1536, // SD3.5 Large
            num_heads: 24,
            depth: 38,
            mlp_ratio: 4.0,
            qkv_bias: false,
            qk_norm: true,
            pos_embed_max_size: 192, // Max 192x192 patches
        }
    }
}

/// QK Normalization for stable attention
pub struct QKNorm {
    pub norm_q: LayerNorm,
    pub norm_k: LayerNorm,
}

impl QKNorm {
    pub fn new(head_dim: usize, device: &Device) -> Result<Self> {
        Ok(Self {
            norm_q: LayerNorm::new(vec![head_dim], 1e-6, device.cuda_device().clone())?,
            norm_k: LayerNorm::new(vec![head_dim], 1e-6, device.cuda_device().clone())?,
        })
    }

    pub fn forward(&self, q: &Tensor, k: &Tensor) -> Result<(Tensor, Tensor)> {
        let q_normed = self.norm_q.forward(q)?;
        let k_normed = self.norm_k.forward(k)?;
        Ok((q_normed, k_normed))
    }
}

/// Adaptive Layer Normalization with modulation
pub struct AdaLayerNorm {
    pub norm: LayerNorm,
    pub linear: Linear,
}

impl AdaLayerNorm {
    pub fn new(hidden_size: usize, cond_dim: usize, device: &Device) -> Result<Self> {
        Ok(Self {
            norm: LayerNorm::new(vec![hidden_size], 1e-6, device.cuda_device().clone())?,
            linear: Linear::new(cond_dim, hidden_size * 2, true, &device.cuda_device())?,
        })
    }

    pub fn forward(&self, x: &Tensor, c: &Tensor) -> Result<Tensor> {
        // Normalize input
        let normed = self.norm.forward(x)?;

        // Get modulation parameters from conditioning
        let params = self.linear.forward(c)?;
        let chunks = params.chunk(2, params.shape().rank() - 1)?;
        let scale = chunks[0].add_scalar(1.0)?;
        let shift = &chunks[1];

        // Apply modulation: scale * normed + shift
        normed.mul(&scale)?.add(shift)
    }
}

/// Joint attention module for MMDiT
/// Processes concatenated image and text sequences
pub struct JointAttention {
    pub num_heads: usize,
    pub head_dim: usize,
    pub qkv: Linear,
    pub proj: Linear,
    pub qk_norm: Option<QKNorm>,
}

impl JointAttention {
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        qkv_bias: bool,
        qk_norm: bool,
        device: &Device,
    ) -> Result<Self> {
        let head_dim = hidden_size / num_heads;

        Ok(Self {
            num_heads,
            head_dim,
            qkv: Linear::new(hidden_size, hidden_size * 3, qkv_bias, &device.cuda_device())?,
            proj: Linear::new(hidden_size, hidden_size, true, &device.cuda_device())?,
            qk_norm: if qk_norm { Some(QKNorm::new(head_dim, device)?) } else { None },
        })
    }

    pub fn forward(&self, x: &Tensor, context: &Tensor) -> Result<(Tensor, Tensor)> {
        let b = x.shape().dims()[0];
        let n_img = x.shape().dims()[1];
        let n_txt = context.shape().dims()[1];
        let c = x.shape().dims()[2];

        // Concatenate image and text
        let x_concat = Tensor::cat(&[x, context], 1)?;

        // Compute QKV
        let qkv = self.qkv.forward(&x_concat)?;
        let qkv = qkv.reshape(&[b, n_img + n_txt, 3, self.num_heads, self.head_dim])?;
        let qkv = qkv.permute(&[2, 0, 3, 1, 4])?;

        let q = qkv.slice(&[
            (0, 1),
            (0, b),
            (0, self.num_heads),
            (0, n_img + n_txt),
            (0, self.head_dim),
        ])?;
        let k = qkv.slice(&[
            (1, 2),
            (0, b),
            (0, self.num_heads),
            (0, n_img + n_txt),
            (0, self.head_dim),
        ])?;
        let v = qkv.slice(&[
            (2, 3),
            (0, b),
            (0, self.num_heads),
            (0, n_img + n_txt),
            (0, self.head_dim),
        ])?;

        // Apply QK normalization if enabled
        let (q, k) =
            if let Some(qk_norm) = &self.qk_norm { qk_norm.forward(&q, &k)? } else { (q, k) };

        // Attention backend toggle: flash or sdpa
        let impl_sel = std::env::var("ATTENTION_IMPL").unwrap_or_else(|_| "sdpa".into());
        let out_bt = if impl_sel == "flash" {
            let fa = flame_core::flash_attention::FlashAttention::new()
                .with_scale((self.head_dim as f32).sqrt().recip());
            match fa.forward(&q, &k, &v, None) {
                Ok(ctx4) => ctx4.permute(&[0, 2, 1, 3])?.reshape(&[b, n_img + n_txt, c])?,
                Err(_) => {
                    let scale = (self.head_dim as f32).sqrt();
                    let scores = q.matmul(&k.transpose_dims(2, 3)?)?.div_scalar(scale)?;
                    let attn_weights = scores.softmax(3)?;
                    attn_weights.matmul(&v)?.permute(&[0, 2, 1, 3])?.reshape(&[
                        b,
                        n_img + n_txt,
                        c,
                    ])?
                }
            }
        } else {
            let scale = (self.head_dim as f32).sqrt();
            let scores = q.matmul(&k.transpose_dims(2, 3)?)?.div_scalar(scale)?;
            let attn_weights = scores.softmax(3)?;
            attn_weights.matmul(&v)?.permute(&[0, 2, 1, 3])?.reshape(&[b, n_img + n_txt, c])?
        };

        // Project output
        let out = self.proj.forward(&out_bt)?;

        // Split back to image and text
        let out_img = out.slice(&[(0, b), (0, n_img), (0, c)])?;
        let out_txt = out.slice(&[(0, b), (n_img, n_img + n_txt), (0, c)])?;

        Ok((out_img, out_txt))
    }
}

/// MLP module for MMDiT
pub struct MLP {
    pub fc1: Linear,
    pub fc2: Linear,
}

impl MLP {
    pub fn new(hidden_size: usize, mlp_ratio: f32, device: &Device) -> Result<Self> {
        let hidden_dim = (hidden_size as f32 * mlp_ratio) as usize;

        Ok(Self {
            fc1: Linear::new(hidden_size, hidden_dim, true, &device.cuda_device())?,
            fc2: Linear::new(hidden_dim, hidden_size, true, &device.cuda_device())?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.fc1.forward(x)?;
        let x = x.gelu()?;
        self.fc2.forward(&x)
    }
}

/// Joint Transformer Block for MMDiT
pub struct JointTransformerBlock {
    pub norm1_context: RMSNorm,
    pub norm1: AdaLayerNorm,
    pub attn: JointAttention,
    pub norm2_context: RMSNorm,
    pub norm2: AdaLayerNorm,
    pub mlp: MLP,
    pub mlp_context: MLP,
}

impl JointTransformerBlock {
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        mlp_ratio: f32,
        qkv_bias: bool,
        qk_norm: bool,
        cond_dim: usize,
        device: &Device,
    ) -> Result<Self> {
        Ok(Self {
            norm1_context: RMSNorm::new(
                vec![hidden_size],
                1e-6,
                true,
                device.cuda_device().clone(),
            )?,
            norm1: AdaLayerNorm::new(hidden_size, cond_dim, &device)?,
            attn: JointAttention::new(hidden_size, num_heads, qkv_bias, qk_norm, device)?,
            norm2_context: RMSNorm::new(
                vec![hidden_size],
                1e-6,
                true,
                device.cuda_device().clone(),
            )?,
            norm2: AdaLayerNorm::new(hidden_size, cond_dim, &device)?,
            mlp: MLP::new(hidden_size, mlp_ratio, device)?,
            mlp_context: MLP::new(hidden_size, mlp_ratio, device)?,
        })
    }

    pub fn forward(&self, x: &Tensor, context: &Tensor, c: &Tensor) -> Result<(Tensor, Tensor)> {
        // Self-attention with adaptive norm
        let x_norm = self.norm1.forward(x, c)?;
        let context_norm = self.norm1_context.forward(context)?;

        let (x_attn, context_attn) = self.attn.forward(&x_norm, &context_norm)?;

        let x = x.add(&x_attn)?;
        let context = context.add(&context_attn)?;

        // MLP with adaptive norm
        let x_norm = self.norm2.forward(&x, c)?;
        let context_norm = self.norm2_context.forward(&context)?;

        let x_mlp = self.mlp.forward(&x_norm)?;
        let context_mlp = self.mlp_context.forward(&context_norm)?;

        let x = x.add(&x_mlp)?;
        let context = context.add(&context_mlp)?;

        Ok((x, context))
    }
}

/// RoPE 2D position embeddings
pub struct RoPE2D {
    pub freqs: Tensor,
}

impl RoPE2D {
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        max_size: usize,
        device: &Device,
    ) -> Result<Self> {
        let head_dim = hidden_size / num_heads;
        let theta = 10000.0;

        // Generate frequency grid
        let freqs = Self::compute_freqs_grid(head_dim, max_size, theta, device)?;

        Ok(Self { freqs })
    }

    fn compute_freqs_grid(
        head_dim: usize,
        max_size: usize,
        theta: f32,
        device: &Device,
    ) -> Result<Tensor> {
        // Compute frequencies for 2D RoPE
        let dim_indices = flame_core::Tensor::arange(
            0.0,
            (head_dim / 2) as f32,
            1.0,
            device.cuda_device().clone(),
        )?;
        let freqs = dim_indices.mul_scalar(2.0 as f32)?.div_scalar(head_dim as f32)?;
        let freqs = freqs.mul_scalar(-theta.ln())?.exp()?;

        // Create 2D position grid
        let pos_h =
            flame_core::Tensor::arange(0.0, max_size as f32, 1.0, device.cuda_device().clone())?;
        let pos_w =
            flame_core::Tensor::arange(0.0, max_size as f32, 1.0, device.cuda_device().clone())?;

        // Combine into 2D frequencies
        let freqs_h = pos_h.unsqueeze(1)?.matmul(&freqs.unsqueeze(0)?)?;
        let freqs_w = pos_w.unsqueeze(1)?.matmul(&freqs.unsqueeze(0)?)?;

        // Stack for 2D positions
        Tensor::stack(&[freqs_h.clone(), freqs_w.clone()], 2)
    }

    pub fn forward(&self, x: &Tensor, positions: &Tensor) -> Result<Tensor> {
        // Apply 2D RoPE to input tensor
        // This is a simplified version - full implementation would include
        // proper sine/cosine application based on positions
        Ok(x.clone())
    }
}

/// Complete MMDiT model
pub struct MMDiT {
    pub config: MMDiTConfig,
    pub blocks: Vec<JointTransformerBlock>,
    pub norm_out: AdaLayerNorm,
    pub proj_out: Linear,
    pub pos_embed: RoPE2D,
}

impl MMDiT {
    pub fn new(config: MMDiTConfig, cond_dim: usize, device: &Device) -> Result<Self> {
        let mut blocks = Vec::new();

        for _ in 0..config.depth {
            blocks.push(JointTransformerBlock::new(
                config.hidden_size,
                config.num_heads,
                config.mlp_ratio,
                config.qkv_bias,
                config.qk_norm,
                cond_dim,
                device,
            )?);
        }

        Ok(Self {
            config: config.clone(),
            blocks,
            norm_out: AdaLayerNorm::new(config.hidden_size, cond_dim, device)?,
            proj_out: Linear::new(
                config.hidden_size,
                config.hidden_size,
                true,
                &device.cuda_device(),
            )?,
            pos_embed: RoPE2D::new(
                config.hidden_size,
                config.num_heads,
                config.pos_embed_max_size,
                device,
            )?,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        context: &Tensor,
        c: &Tensor,
        positions: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        // Apply position embeddings
        let x = self.pos_embed.forward(x, positions)?;

        let mut x = x;
        let mut context = context.clone();

        // Process through transformer blocks
        for block in &self.blocks {
            let (new_x, new_context) = block.forward(&x, &context, c)?;
            x = new_x;
            context = new_context;
        }

        // Final normalization and projection
        let x = self.norm_out.forward(&x, c)?;
        let x = self.proj_out.forward(&x)?;

        Ok((x, context))
    }
}
