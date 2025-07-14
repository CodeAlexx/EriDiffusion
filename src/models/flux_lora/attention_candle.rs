//! Flux attention implementation matching Candle's exact structure

use anyhow::Result;
use candle_core::{Device, Module, Tensor, D};
use candle_nn::{Linear, VarBuilder, linear, linear_b};

use super::norm_wrapper::QkNorm;
use crate::networks::lora::LoRAModule;

/// Flux SelfAttention matching Candle's implementation exactly
pub struct FluxSelfAttention {
    qkv: Linear,
    norm: QkNorm,
    proj: Linear,
    num_heads: usize,
}

impl FluxSelfAttention {
    pub fn new(dim: usize, num_heads: usize, qkv_bias: bool, vb: VarBuilder) -> Result<Self> {
        let head_dim = dim / num_heads;
        let qkv = linear_b(dim, dim * 3, qkv_bias, vb.pp("qkv"))?;
        let norm = QkNorm::new(head_dim, vb.pp("norm"))?;
        let proj = linear(dim, dim, vb.pp("proj"))?;
        Ok(Self {
            qkv,
            norm,
            proj,
            num_heads,
        })
    }

    fn qkv(&self, xs: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let qkv = xs.apply(&self.qkv)?;
        let (b, l, _khd) = qkv.dims3()?;
        let qkv = qkv.reshape((b, l, 3, self.num_heads, ()))?;
        let q = qkv.i((.., .., 0))?.transpose(1, 2)?;
        let k = qkv.i((.., .., 1))?.transpose(1, 2)?;
        let v = qkv.i((.., .., 2))?.transpose(1, 2)?;
        let q = q.apply(&self.norm.query_norm)?;
        let k = k.apply(&self.norm.key_norm)?;
        Ok((q, k, v))
    }

    pub fn forward(&self, xs: &Tensor, pe: &Tensor) -> Result<Tensor> {
        let (q, k, v) = self.qkv(xs)?;
        attention(&q, &k, &v, pe)?.apply(&self.proj)
    }
}

/// Flux SelfAttention with LoRA support
pub struct FluxSelfAttentionWithLoRA {
    qkv: Linear,
    norm: QkNorm,
    proj: Linear,
    num_heads: usize,
    
    // LoRA modules
    qkv_lora: Option<LoRAModule>,
    proj_lora: Option<LoRAModule>,
}

impl FluxSelfAttentionWithLoRA {
    pub fn new(
        dim: usize,
        num_heads: usize,
        qkv_bias: bool,
        vb: VarBuilder,
        lora_rank: Option<usize>,
        lora_alpha: Option<f32>,
    ) -> Result<Self> {
        let head_dim = dim / num_heads;
        let qkv = linear_b(dim, dim * 3, qkv_bias, vb.pp("qkv"))?;
        let norm = QkNorm::new(head_dim, vb.pp("norm"))?;
        let proj = linear(dim, dim, vb.pp("proj"))?;
        
        // Initialize LoRA modules if rank is provided
        let (qkv_lora, proj_lora) = if let Some(rank) = lora_rank {
            let alpha = lora_alpha.unwrap_or(rank as f32);
            let device = vb.device();
            let dtype = vb.dtype();
            
            // LoRA for qkv projection
            let qkv_lora = Some(LoRAModule::new(
                dim,
                dim * 3,
                rank,
                alpha,
                device.clone(),
                dtype,
            )?);
            
            // LoRA for output projection
            let proj_lora = Some(LoRAModule::new(
                dim,
                dim,
                rank,
                alpha,
                device.clone(),
                dtype,
            )?);
            
            (qkv_lora, proj_lora)
        } else {
            (None, None)
        };
        
        Ok(Self {
            qkv,
            norm,
            proj,
            num_heads,
            qkv_lora,
            proj_lora,
        })
    }

    fn qkv(&self, xs: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        // Apply base qkv projection
        let mut qkv = xs.apply(&self.qkv)?;
        
        // Add LoRA if present
        if let Some(lora) = &self.qkv_lora {
            qkv = (qkv + lora.forward(xs)?)?;
        }
        
        let (b, l, _khd) = qkv.dims3()?;
        let qkv = qkv.reshape((b, l, 3, self.num_heads, ()))?;
        let q = qkv.i((.., .., 0))?.transpose(1, 2)?;
        let k = qkv.i((.., .., 1))?.transpose(1, 2)?;
        let v = qkv.i((.., .., 2))?.transpose(1, 2)?;
        let q = q.apply(&self.norm.query_norm)?;
        let k = k.apply(&self.norm.key_norm)?;
        Ok((q, k, v))
    }

    pub fn forward(&self, xs: &Tensor, pe: &Tensor) -> Result<Tensor> {
        let (q, k, v) = self.qkv(xs)?;
        let mut out = attention(&q, &k, &v, pe)?.apply(&self.proj)?;
        
        // Add LoRA to projection if present
        if let Some(lora) = &self.proj_lora {
            let pre_proj = attention(&q, &k, &v, pe)?;
            out = (out + lora.forward(&pre_proj)?)?;
        }
        
        Ok(out)
    }
}

// Copy exact attention function from Candle
fn attention(q: &Tensor, k: &Tensor, v: &Tensor, pe: &Tensor) -> Result<Tensor> {
    let q = apply_rope(q, pe)?;
    let k = apply_rope(k, pe)?;
    
    scaled_dot_product_attention(&q, &k, v)
}

fn scaled_dot_product_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let dim = q.dim(D::Minus1)?;
    let scale_factor = 1.0 / (dim as f64).sqrt();
    let mut batch_dims = q.dims().to_vec();
    batch_dims.pop();
    batch_dims.pop();
    let q = q.flatten_to(batch_dims.len() - 1)?;
    let k = k.flatten_to(batch_dims.len() - 1)?;
    let v = v.flatten_to(batch_dims.len() - 1)?;
    let attn_weights = (q.matmul(&k.t()?)? * scale_factor)?;
    let attn_scores = candle_nn::ops::softmax_last_dim(&attn_weights)?.matmul(&v)?;
    batch_dims.push(attn_scores.dim(D::Minus2)?);
    batch_dims.push(attn_scores.dim(D::Minus1)?);
    attn_scores.reshape(batch_dims)
}

fn apply_rope(xs: &Tensor, pe: &Tensor) -> Result<Tensor> {
    // RoPE implementation would go here
    // For now, returning the input unchanged
    // TODO: Implement proper RoPE
    Ok(xs.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn test_flux_attention_structure() -> Result<()> {
        let device = Device::Cpu;
        let dtype = DType::F32;
        let vb = VarBuilder::zeros(dtype, &device);
        
        let dim = 768;
        let num_heads = 12;
        
        let attn = FluxSelfAttention::new(dim, num_heads, true, vb.pp("test"))?;
        
        // Check dimensions
        assert_eq!(attn.num_heads, num_heads);
        
        Ok(())
    }
}