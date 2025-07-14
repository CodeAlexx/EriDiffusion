//! Attention with RoPE support for Flux
//! 
//! Extends the basic AttentionWithLoRA to include rotary position embeddings

use candle_core::{Tensor, Module, Result, Device, DType, D};
use candle_nn::VarBuilder;
use super::lora_layers::AttentionWithLoRA;
use super::lora_config::LoRALayerConfig;
use crate::ops::RotaryEmbedding;
use std::sync::Arc;

/// Attention layer with LoRA and RoPE support
pub struct AttentionWithLoRAAndRoPE {
    base_attention: AttentionWithLoRA,
    rope: Option<Arc<RotaryEmbedding>>,
    use_rope: bool,
}

impl AttentionWithLoRAAndRoPE {
    pub fn new(
        dim: usize,
        num_heads: usize,
        lora_config: Option<&LoRALayerConfig>,
        use_rope: bool,
        max_seq_len: usize,
        theta_base: f32,
        vb: VarBuilder,
    ) -> Result<Self> {
        let device = vb.device().clone();
        let base_attention = AttentionWithLoRA::new(dim, num_heads, lora_config, vb)?;
        
        let rope = if use_rope {
            let head_dim = dim / num_heads;
            Some(Arc::new(RotaryEmbedding::new(
                head_dim,
                max_seq_len,
                theta_base,
                &device,
            )?))
        } else {
            None
        };
        
        Ok(Self {
            base_attention,
            rope,
            use_rope,
        })
    }
    
    /// Forward pass with RoPE and optional context
    pub fn forward_with_rope(
        &self,
        x: &Tensor,
        positions: Option<&Tensor>,
        context: Option<&Tensor>,
        mask: Option<&Tensor>,
        is_2d: bool,
    ) -> Result<Tensor> {
        let (batch_size, seq_len, dim) = x.dims3()?;
        
        // Compute Q from input
        let q = self.base_attention.to_q.forward(x)?;
        
        // Compute K,V from context (or input if no context)
        let kv_input = context.unwrap_or(x);
        let k = self.base_attention.to_k.forward(kv_input)?;
        let v = self.base_attention.to_v.forward(kv_input)?;
        
        // Reshape for multi-head attention
        let q = self.reshape_for_attention(&q)?;
        let k = self.reshape_for_attention(&k)?;
        let v = self.reshape_for_attention(&v)?;
        
        // Apply RoPE if enabled and positions provided
        let (q, k) = if self.use_rope && positions.is_some() && self.rope.is_some() {
            let rope = self.rope.as_ref().unwrap();
            crate::ops::apply_rotary_emb(&q, &k, positions.unwrap(), rope, is_2d)?
        } else {
            (q, k)
        };
        
        // Compute attention
        let attn_out = self.attention(&q, &k, &v, mask)?;
        
        // Reshape back
        let attn_out = attn_out
            .transpose(1, 2)?
            .reshape(&[batch_size, seq_len, dim])?;
        
        // Output projection
        self.base_attention.to_out.forward(&attn_out)
    }
    
    /// Get trainable LoRA parameters
    pub fn trainable_parameters(&self) -> Vec<&candle_core::Var> {
        self.base_attention.trainable_parameters()
    }
    
    /// Helper methods delegated to base attention
    fn reshape_for_attention(&self, x: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, _) = x.dims3()?;
        x.reshape(&[batch_size, seq_len, self.base_attention.num_heads, self.base_attention.head_dim])?
            .transpose(1, 2)
    }
    
    fn attention(&self, q: &Tensor, k: &Tensor, v: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let scores = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)?
            .affine(self.base_attention.scale, 0.0)?;
        
        let scores = if let Some(mask) = mask {
            scores.broadcast_add(mask)?
        } else {
            scores
        };
        
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        
        // Note: Dropout not implemented in inference mode
        let probs = probs;
        
        probs.matmul(v)
    }
}

impl Module for AttentionWithLoRAAndRoPE {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Default forward without RoPE
        self.forward_with_rope(x, None, None, None, false)
    }
}