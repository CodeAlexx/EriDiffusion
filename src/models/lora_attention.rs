//! LoRA-enabled attention layers for SDXL
//! This module provides attention mechanisms with integrated LoRA support

use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn as nn;
use std::collections::HashMap;
use super::sdxl_lora_layer::LoRALinear;

/// Configuration for LoRA in attention layers
#[derive(Debug, Clone)]
pub struct LoRAAttentionConfig {
    pub rank: usize,
    pub alpha: f32,
    pub dropout: f32,
    pub target_modules: Vec<String>,
}

/// CrossAttention with LoRA support
/// This properly applies LoRA transformations to Q, K, V projections
pub struct CrossAttentionWithLoRA {
    // Base layers (frozen during training)
    to_q_base: nn::Linear,
    to_k_base: nn::Linear,
    to_v_base: nn::Linear,
    to_out_base: nn::Linear,
    
    // LoRA layers (trainable)
    to_q_lora: Option<LoRALinear>,
    to_k_lora: Option<LoRALinear>,
    to_v_lora: Option<LoRALinear>,
    to_out_lora: Option<LoRALinear>,
    
    heads: usize,
    scale: f64,
    slice_size: Option<usize>,
    use_flash_attn: bool,
    device: Device,
    dtype: DType,
}

impl CrossAttentionWithLoRA {
    pub fn new(
        vs: nn::VarBuilder,
        query_dim: usize,
        context_dim: Option<usize>,
        heads: usize,
        dim_head: usize,
        slice_size: Option<usize>,
        use_flash_attn: bool,
        lora_config: Option<&LoRAAttentionConfig>,
        device: Device,
        dtype: DType,
    ) -> Result<Self> {
        let inner_dim = dim_head * heads;
        let context_dim = context_dim.unwrap_or(query_dim);
        let scale = 1.0 / f64::sqrt(dim_head as f64);
        
        // Load base layers (pre-trained weights)
        let to_q_base = nn::linear_no_bias(query_dim, inner_dim, vs.pp("to_q"))?;
        let to_k_base = nn::linear_no_bias(context_dim, inner_dim, vs.pp("to_k"))?;
        let to_v_base = nn::linear_no_bias(context_dim, inner_dim, vs.pp("to_v"))?;
        let to_out_base = nn::linear(inner_dim, query_dim, vs.pp("to_out.0"))?;
        
        // Create LoRA layers if config provided
        let (to_q_lora, to_k_lora, to_v_lora, to_out_lora) = if let Some(config) = lora_config {
            let mut q_lora = None;
            let mut k_lora = None;
            let mut v_lora = None;
            let mut out_lora = None;
            
            for target in &config.target_modules {
                match target.as_str() {
                    "to_q" => {
                        q_lora = Some(LoRALinear::new_without_base(
                            &vs.pp("lora_q"),
                            query_dim,
                            inner_dim,
                            config.rank,
                            config.alpha,
                            config.dropout,
                        )?);
                    }
                    "to_k" => {
                        k_lora = Some(LoRALinear::new_without_base(
                            &vs.pp("lora_k"),
                            context_dim,
                            inner_dim,
                            config.rank,
                            config.alpha,
                            config.dropout,
                        )?);
                    }
                    "to_v" => {
                        v_lora = Some(LoRALinear::new_without_base(
                            &vs.pp("lora_v"),
                            context_dim,
                            inner_dim,
                            config.rank,
                            config.alpha,
                            config.dropout,
                        )?);
                    }
                    "to_out" | "to_out.0" => {
                        out_lora = Some(LoRALinear::new_without_base(
                            &vs.pp("lora_out"),
                            inner_dim,
                            query_dim,
                            config.rank,
                            config.alpha,
                            config.dropout,
                        )?);
                    }
                    _ => {}
                }
            }
            
            (q_lora, k_lora, v_lora, out_lora)
        } else {
            (None, None, None, None)
        };
        
        Ok(Self {
            to_q_base,
            to_k_base,
            to_v_base,
            to_out_base,
            to_q_lora,
            to_k_lora,
            to_v_lora,
            to_out_lora,
            heads,
            scale,
            slice_size,
            use_flash_attn,
            device,
            dtype,
        })
    }
    
    /// Apply base layer and optionally add LoRA transformation
    fn apply_with_lora(
        &self,
        base_layer: &nn::Linear,
        lora_layer: &Option<LoRALinear>,
        input: &Tensor,
    ) -> Result<Tensor> {
        // Get base output
        let base_output = base_layer.forward(input)?;
        
        // Add LoRA if available
        if let Some(lora) = lora_layer {
            // LoRA forward computes: base + (input @ A @ B) * scale
            let lora_output = lora.forward(input)?;
            base_output.add(&lora_output)
        } else {
            Ok(base_output)
        }
    }
    
    fn reshape_heads_to_batch_dim(&self, xs: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, dim) = xs.dims3()?;
        xs.reshape((batch_size, seq_len, self.heads, dim / self.heads))?
            .transpose(1, 2)?
            .reshape((batch_size * self.heads, seq_len, dim / self.heads))
    }
    
    fn reshape_batch_dim_to_heads(&self, xs: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, dim) = xs.dims3()?;
        xs.reshape((batch_size / self.heads, self.heads, seq_len, dim))?
            .transpose(1, 2)?
            .reshape((batch_size / self.heads, seq_len, dim * self.heads))
    }
    
    fn attention(&self, query: &Tensor, key: &Tensor, value: &Tensor) -> Result<Tensor> {
        if self.use_flash_attn {
            #[cfg(feature = "flash-attn")]
            {
                let init_dtype = query.dtype();
                let q = query
                    .to_dtype(candle_core::DType::F16)?
                    .unsqueeze(0)?
                    .transpose(1, 2)?;
                let k = key
                    .to_dtype(candle_core::DType::F16)?
                    .unsqueeze(0)?
                    .transpose(1, 2)?;
                let v = value
                    .to_dtype(candle_core::DType::F16)?
                    .unsqueeze(0)?
                    .transpose(1, 2)?;
                candle_flash_attn::flash_attn(&q, &k, &v, self.scale as f32, false)?
                    .transpose(1, 2)?
                    .squeeze(0)?
                    .to_dtype(init_dtype)
            }
            #[cfg(not(feature = "flash-attn"))]
            {
                return Err(candle_core::Error::Msg("flash-attn feature not enabled".to_string()));
            }
        } else {
            let in_dtype = query.dtype();
            let query = query.to_dtype(DType::F32)?;
            let key = key.to_dtype(DType::F32)?;
            let value = value.to_dtype(DType::F32)?;
            
            let scores = query.matmul(&(key.t()? * self.scale)?)?;
            let probs = nn::ops::softmax_last_dim(&scores)?;
            probs.matmul(&value)?.to_dtype(in_dtype)
        }
    }
    
    pub fn forward(&self, xs: &Tensor, context: Option<&Tensor>) -> Result<Tensor> {
        // Compute Q with base + LoRA
        let query = self.apply_with_lora(&self.to_q_base, &self.to_q_lora, xs)?;
        
        // Use context if provided, otherwise self-attention
        let context = context.unwrap_or(xs);
        
        // Compute K and V with base + LoRA
        let key = self.apply_with_lora(&self.to_k_base, &self.to_k_lora, context)?;
        let value = self.apply_with_lora(&self.to_v_base, &self.to_v_lora, context)?;
        
        // Reshape for multi-head attention
        let query = self.reshape_heads_to_batch_dim(&query)?;
        let key = self.reshape_heads_to_batch_dim(&key)?;
        let value = self.reshape_heads_to_batch_dim(&value)?;
        
        // Compute attention
        let attn_output = self.attention(&query, &key, &value)?;
        let attn_output = self.reshape_batch_dim_to_heads(&attn_output)?;
        
        // Final projection with base + LoRA
        self.apply_with_lora(&self.to_out_base, &self.to_out_lora, &attn_output)
    }
    
    /// Get all LoRA layers for training
    pub fn get_lora_layers(&self) -> Vec<(&str, &LoRALinear)> {
        let mut layers = Vec::new();
        
        if let Some(ref lora) = self.to_q_lora {
            layers.push(("to_q", lora));
        }
        if let Some(ref lora) = self.to_k_lora {
            layers.push(("to_k", lora));
        }
        if let Some(ref lora) = self.to_v_lora {
            layers.push(("to_v", lora));
        }
        if let Some(ref lora) = self.to_out_lora {
            layers.push(("to_out", lora));
        }
        
        layers
    }
}

/// BasicTransformerBlock with LoRA support
pub struct BasicTransformerBlockWithLoRA {
    attn1: CrossAttentionWithLoRA,
    attn2: CrossAttentionWithLoRA,
    ff: FeedForward,
    norm1: nn::LayerNorm,
    norm2: nn::LayerNorm,
    norm3: nn::LayerNorm,
}

/// Feed-forward layer (no LoRA needed here typically)
struct FeedForward {
    net: Vec<Box<dyn Module>>,
}

impl FeedForward {
    fn new(vs: nn::VarBuilder, dim: usize, dim_out: Option<usize>, mult: usize) -> Result<Self> {
        let inner_dim = dim * mult;
        let dim_out = dim_out.unwrap_or(dim);
        let vs = vs.pp("net");
        
        // Create GEGLU activation
        let w1 = nn::linear(dim, inner_dim * 2, vs.pp("0").pp("proj"))?;
        let w2 = nn::linear(inner_dim, dim_out, vs.pp("2"))?;
        
        Ok(Self {
            net: vec![Box::new(w1), Box::new(w2)],
        })
    }
}

impl Module for FeedForward {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // First layer outputs twice the inner dimension
        let hidden = self.net[0].forward(xs)?;
        let chunks = hidden.chunk(2, D::Minus1)?;
        
        // GELU activation on second half, multiply with first half
        let hidden = (&chunks[0] * chunks[1].gelu()?)?;
        
        // Final projection
        self.net[1].forward(&hidden)
    }
}

impl BasicTransformerBlockWithLoRA {
    pub fn new(
        vs: nn::VarBuilder,
        dim: usize,
        n_heads: usize,
        d_head: usize,
        context_dim: Option<usize>,
        sliced_attention_size: Option<usize>,
        use_flash_attn: bool,
        lora_config: Option<&LoRAAttentionConfig>,
        device: Device,
        dtype: DType,
    ) -> Result<Self> {
        // Self-attention (no context)
        let attn1 = CrossAttentionWithLoRA::new(
            vs.pp("attn1"),
            dim,
            None,
            n_heads,
            d_head,
            sliced_attention_size,
            use_flash_attn,
            lora_config,
            device.clone(),
            dtype,
        )?;
        
        // Cross-attention (with context)
        let attn2 = CrossAttentionWithLoRA::new(
            vs.pp("attn2"),
            dim,
            context_dim,
            n_heads,
            d_head,
            sliced_attention_size,
            use_flash_attn,
            lora_config,
            device,
            dtype,
        )?;
        
        let ff = FeedForward::new(vs.pp("ff"), dim, None, 4)?;
        let norm1 = nn::layer_norm(dim, 1e-5, vs.pp("norm1"))?;
        let norm2 = nn::layer_norm(dim, 1e-5, vs.pp("norm2"))?;
        let norm3 = nn::layer_norm(dim, 1e-5, vs.pp("norm3"))?;
        
        Ok(Self {
            attn1,
            attn2,
            ff,
            norm1,
            norm2,
            norm3,
        })
    }
    
    pub fn forward(&self, xs: &Tensor, context: Option<&Tensor>) -> Result<Tensor> {
        // Self-attention block
        let residual = xs;
        let xs = self.norm1.forward(xs)?;
        let xs = self.attn1.forward(&xs, None)?;
        let xs = (xs + residual)?;
        
        // Cross-attention block
        let residual = &xs;
        let xs_norm = self.norm2.forward(&xs)?;
        let xs = self.attn2.forward(&xs_norm, context)?;
        let xs = (xs + residual)?;
        
        // Feed-forward block
        let residual = &xs;
        let xs_norm = self.norm3.forward(&xs)?;
        let xs = self.ff.forward(&xs_norm)?;
        xs + residual
    }
    
    /// Get all LoRA layers from both attention blocks
    pub fn get_lora_layers(&self) -> Vec<(&str, &LoRALinear)> {
        let mut layers = Vec::new();
        
        // Get from self-attention
        for (name, layer) in self.attn1.get_lora_layers() {
            layers.push((name, layer));
        }
        
        // Get from cross-attention
        for (name, layer) in self.attn2.get_lora_layers() {
            layers.push((name, layer));
        }
        
        layers
    }
}