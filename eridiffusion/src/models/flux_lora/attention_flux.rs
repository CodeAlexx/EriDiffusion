//! Flux-compatible attention implementation

use candle_core::{Tensor, Module, Result, D};
use candle_nn::{VarBuilder, Linear, linear, linear_b};
use super::lora_config::LoRALayerConfig;

/// QK-Norm for Flux attention
pub struct QkNorm {
    query_norm: candle_nn::RmsNorm,
    key_norm: candle_nn::RmsNorm,
}

impl QkNorm {
    fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let query_norm = vb.get(dim, "query_norm.scale")?;
        let query_norm = candle_nn::RmsNorm::new(query_norm, 1e-6);
        let key_norm = vb.get(dim, "key_norm.scale")?;
        let key_norm = candle_nn::RmsNorm::new(key_norm, 1e-6);
        Ok(Self {
            query_norm,
            key_norm,
        })
    }
}

/// Flux Self-Attention that matches the model structure
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
    
    pub fn qkv(&self, xs: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let qkv = xs.apply(&self.qkv)?;
        let (b, l, _khd) = qkv.dims3()?;
        let qkv = qkv.reshape((b, l, 3, self.num_heads, ()))?;
        let q = qkv.narrow(2, 0, 1)?.squeeze(2)?.transpose(1, 2)?;
        let k = qkv.narrow(2, 1, 1)?.squeeze(2)?.transpose(1, 2)?;
        let v = qkv.narrow(2, 2, 1)?.squeeze(2)?.transpose(1, 2)?;
        let q = q.apply(&self.norm.query_norm)?;
        let k = k.apply(&self.norm.key_norm)?;
        Ok((q, k, v))
    }
    
    pub fn forward(&self, xs: &Tensor, pe: &Tensor) -> Result<Tensor> {
        let (q, k, v) = self.qkv(xs)?;
        crate::ops::attention(&q, &k, &v, pe)?.apply(&self.proj)
    }
}

/// Flux attention with LoRA support
pub struct FluxAttentionWithLoRA {
    base: FluxSelfAttention,
    lora_config: Option<LoRALayerConfig>,
    // LoRA adapters would go here
}

impl FluxAttentionWithLoRA {
    pub fn new(
        dim: usize,
        num_heads: usize,
        qkv_bias: bool,
        lora_config: Option<&LoRALayerConfig>,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            base: FluxSelfAttention::new(dim, num_heads, qkv_bias, vb)?,
            lora_config: lora_config.cloned(),
        })
    }
    
    pub fn qkv(&self, xs: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        self.base.qkv(xs)
    }
    
    pub fn proj(&self) -> &Linear {
        &self.base.proj
    }
    
    pub fn trainable_parameters(&self) -> Vec<&candle_core::Var> {
        // Would return LoRA parameters when implemented
        vec![]
    }
    
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // For now, just delegate to base
        // TODO: Add RoPE support
        let dummy_pe = Tensor::zeros((1, 1, 1, 1), xs.dtype(), xs.device())?;
        self.base.forward(xs, &dummy_pe)
    }
}