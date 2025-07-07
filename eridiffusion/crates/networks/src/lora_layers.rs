//! LoRA-aware layer implementations for Flux and other models
//! 
//! This module provides base building blocks that support LoRA adaptation natively,
//! working within Candle's constraints while enabling proper gradient flow.

use candle_core::{Tensor, Module, Var, Result, Device, DType, D};
use candle_nn::VarBuilder;
use serde::{Serialize, Deserialize};
use std::sync::Arc;

/// Configuration for LoRA layers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoRALayerConfig {
    pub rank: usize,
    pub alpha: f32,
    pub dropout: f32,
    pub use_bias: bool,
}

impl Default for LoRALayerConfig {
    fn default() -> Self {
        Self {
            rank: 32,
            alpha: 32.0,
            dropout: 0.0,
            use_bias: false,
        }
    }
}

/// Linear layer with optional LoRA adaptation
/// 
/// This layer combines a frozen base linear transformation with optional
/// trainable LoRA matrices for parameter-efficient fine-tuning.
pub struct LinearWithLoRA {
    // Base layer parameters (frozen during training)
    weight: Tensor,
    bias: Option<Tensor>,
    
    // LoRA parameters (trainable)
    lora_a: Option<Var>,  // Down projection: [rank, in_features]
    lora_b: Option<Var>,  // Up projection: [out_features, rank]
    
    // Configuration
    in_features: usize,
    out_features: usize,
    rank: Option<usize>,
    alpha: f32,
    dropout: f32,
    enabled: bool,
}

impl LinearWithLoRA {
    /// Create a new LinearWithLoRA layer
    pub fn new(
        weight: Tensor,
        bias: Option<Tensor>,
        rank: Option<usize>,
        alpha: f32,
        device: &Device,
    ) -> Result<Self> {
        let shape = weight.shape();
        let out_features = shape.dims()[0];
        let in_features = shape.dims()[1];
        
        let (lora_a, lora_b) = if let Some(r) = rank {
            // Initialize LoRA weights following the paper's recommendations
            // A: Normal initialization with std = 1/sqrt(in_features)
            // B: Zero initialization for stability
            let std_dev = (1.0 / in_features as f64).sqrt();
            
            let a = Var::from_tensor(
                &Tensor::randn(0.0, std_dev, &[r, in_features], device)?.to_dtype(weight.dtype())?
            )?;
            
            let b = Var::from_tensor(
                &Tensor::zeros(&[out_features, r], weight.dtype(), device)?
            )?;
            
            (Some(a), Some(b))
        } else {
            (None, None)
        };
        
        Ok(Self {
            weight,
            bias,
            lora_a,
            lora_b,
            in_features,
            out_features,
            rank,
            alpha,
            dropout: 0.0,
            enabled: true,
        })
    }
    
    /// Create from a VarBuilder with optional LoRA
    pub fn from_vb(
        vb: VarBuilder,
        in_features: usize,
        out_features: usize,
        config: Option<&LoRALayerConfig>,
    ) -> Result<Self> {
        let weight = vb.get((out_features, in_features), "weight")?;
        let bias = vb.get(out_features, "bias").ok();
        
        let (rank, alpha) = config
            .map(|c| (Some(c.rank), c.alpha))
            .unwrap_or((None, 1.0));
        
        Self::new(weight, bias, rank, alpha, vb.device())
    }
    
    /// Get trainable LoRA parameters
    pub fn trainable_parameters(&self) -> Vec<&Var> {
        let mut params = Vec::new();
        if let Some(a) = &self.lora_a {
            params.push(a);
        }
        if let Some(b) = &self.lora_b {
            params.push(b);
        }
        params
    }
    
    /// Enable or disable LoRA adaptation
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
    
    /// Get the effective rank (0 if no LoRA)
    pub fn effective_rank(&self) -> usize {
        self.rank.unwrap_or(0)
    }
    
    /// Apply dropout during training (if configured)
    fn apply_dropout(&self, x: &Tensor, training: bool) -> Result<Tensor> {
        if training && self.dropout > 0.0 {
            // Create dropout mask
            let keep_prob = 1.0 - self.dropout;
            let random = Tensor::rand(0f32, 1f32, x.shape(), x.device())?;
            let mask = random.ge(self.dropout)?;
            let scale = 1.0 / keep_prob;
            
            // Apply dropout with scaling
            x.broadcast_mul(&mask)?.affine(scale as f64, 0.0)
        } else {
            Ok(x.clone())
        }
    }
}

impl Module for LinearWithLoRA {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Base forward pass: x @ W^T + bias
        let mut output = x.matmul(&self.weight.t()?)?;
        
        // Add LoRA adaptation if enabled and present
        if self.enabled {
            if let (Some(a), Some(b)) = (&self.lora_a, &self.lora_b) {
                // LoRA forward: x @ A @ B
                // Where A: [in_features, rank] and B: [rank, out_features]
                let lora_out = x
                    .matmul(&a.as_tensor())?      // x @ A
                    .matmul(&b.as_tensor())?;     // (x @ A) @ B
                
                // Scale by alpha/rank as per LoRA paper
                let scale = self.alpha / self.rank.unwrap_or(1) as f32;
                output = output.add(&lora_out.affine(scale as f64, 0.0)?)?;
            }
        }
        
        // Add bias if present
        if let Some(bias) = &self.bias {
            output = output.broadcast_add(bias)?;
        }
        
        Ok(output)
    }
}

/// Multi-head attention layer with LoRA support
pub struct AttentionWithLoRA {
    // Query, Key, Value projections with LoRA
    to_q: LinearWithLoRA,
    to_k: LinearWithLoRA,
    to_v: LinearWithLoRA,
    to_out: LinearWithLoRA,
    
    // Attention configuration
    num_heads: usize,
    head_dim: usize,
    scale: f64,
    dropout: f32,
}

impl AttentionWithLoRA {
    /// Create new attention layer with optional LoRA on Q,K,V,O projections
    pub fn new(
        dim: usize,
        num_heads: usize,
        config: Option<&LoRALayerConfig>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let head_dim = dim / num_heads;
        let scale = (head_dim as f64).powf(-0.5);
        
        Ok(Self {
            to_q: LinearWithLoRA::from_vb(vb.pp("to_q"), dim, dim, config)?,
            to_k: LinearWithLoRA::from_vb(vb.pp("to_k"), dim, dim, config)?,
            to_v: LinearWithLoRA::from_vb(vb.pp("to_v"), dim, dim, config)?,
            to_out: LinearWithLoRA::from_vb(vb.pp("to_out.0"), dim, dim, config)?,
            num_heads,
            head_dim,
            scale,
            dropout: config.map(|c| c.dropout).unwrap_or(0.0),
        })
    }
    
    /// Get all trainable LoRA parameters
    pub fn trainable_parameters(&self) -> Vec<&Var> {
        let mut params = Vec::new();
        params.extend(self.to_q.trainable_parameters());
        params.extend(self.to_k.trainable_parameters());
        params.extend(self.to_v.trainable_parameters());
        params.extend(self.to_out.trainable_parameters());
        params
    }
    
    /// Reshape tensor for multi-head attention
    fn reshape_for_attention(&self, x: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, _) = x.dims3()?;
        x.reshape(&[batch_size, seq_len, self.num_heads, self.head_dim])?
            .transpose(1, 2) // [batch, heads, seq_len, head_dim]
    }
    
    /// Scaled dot-product attention
    fn attention(&self, q: &Tensor, k: &Tensor, v: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let scores = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)?
            .affine(self.scale, 0.0)?;
        
        let scores = if let Some(mask) = mask {
            scores.broadcast_add(mask)?
        } else {
            scores
        };
        
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        
        // Apply dropout if configured
        let probs = if self.dropout > 0.0 {
            // Create dropout mask for attention
            let keep_prob = 1.0 - self.dropout;
            let random = Tensor::rand(0f32, 1f32, probs.shape(), probs.device())?;
            let mask = random.ge(self.dropout)?;
            let scale = 1.0 / keep_prob;
            probs.broadcast_mul(&mask)?.affine(scale as f64, 0.0)?
        } else {
            probs
        };
        
        probs.matmul(v)
    }
}

impl Module for AttentionWithLoRA {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward_with_context(x, None, None)
    }
}

impl AttentionWithLoRA {
    /// Forward pass with optional context and mask
    pub fn forward_with_context(
        &self,
        x: &Tensor,
        context: Option<&Tensor>,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (batch_size, seq_len, dim) = x.dims3()?;
        
        // Compute Q from input
        let q = self.to_q.forward(x)?;
        
        // Compute K,V from context (or input if no context)
        let kv_input = context.unwrap_or(x);
        let k = self.to_k.forward(kv_input)?;
        let v = self.to_v.forward(kv_input)?;
        
        // Reshape for multi-head attention
        let q = self.reshape_for_attention(&q)?;
        let k = self.reshape_for_attention(&k)?;
        let v = self.reshape_for_attention(&v)?;
        
        // Compute attention
        let attn_out = self.attention(&q, &k, &v, mask)?;
        
        // Reshape back
        let attn_out = attn_out
            .transpose(1, 2)?
            .reshape(&[batch_size, seq_len, dim])?;
        
        // Output projection
        self.to_out.forward(&attn_out)
    }
}

/// Feed-forward network with LoRA support
pub struct FeedForwardWithLoRA {
    w1: LinearWithLoRA,
    w2: LinearWithLoRA,
    activation: Activation,
}

#[derive(Debug, Clone, Copy)]
pub enum Activation {
    Gelu,
    Silu,
    Relu,
}

impl FeedForwardWithLoRA {
    pub fn new(
        dim: usize,
        hidden_dim: usize,
        config: Option<&LoRALayerConfig>,
        activation: Activation,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            w1: LinearWithLoRA::from_vb(vb.pp("linear1"), dim, hidden_dim, config)?,
            w2: LinearWithLoRA::from_vb(vb.pp("linear2"), hidden_dim, dim, config)?,
            activation,
        })
    }
    
    pub fn trainable_parameters(&self) -> Vec<&Var> {
        let mut params = Vec::new();
        params.extend(self.w1.trainable_parameters());
        params.extend(self.w2.trainable_parameters());
        params
    }
}

impl Module for FeedForwardWithLoRA {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.w1.forward(x)?;
        let x = match self.activation {
            Activation::Gelu => x.gelu_erf()?,
            Activation::Silu => x.silu()?,
            Activation::Relu => x.relu()?,
        };
        self.w2.forward(&x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    
    #[test]
    fn test_linear_with_lora_forward() -> Result<()> {
        let device = Device::Cpu;
        let batch_size = 2;
        let seq_len = 10;
        let in_features = 512;
        let out_features = 512;
        let rank = 16;
        
        // Create base weight
        let weight = Tensor::randn(0.0, 1.0, &[out_features, in_features], &device)?;
        
        // Create layer with LoRA
        let layer = LinearWithLoRA::new(weight.clone(), None, Some(rank), 16.0, &device)?;
        
        // Test input
        let x = Tensor::randn(0.0, 1.0, &[batch_size, seq_len, in_features], &device)?;
        
        // Forward pass
        let output = layer.forward(&x)?;
        assert_eq!(output.shape().dims(), &[batch_size, seq_len, out_features]);
        
        // Verify LoRA parameters exist
        assert_eq!(layer.trainable_parameters().len(), 2);
        
        Ok(())
    }
    
    #[test]
    fn test_lora_gradient_flow() -> Result<()> {
        let device = Device::Cpu;
        let layer = LinearWithLoRA::new(
            Tensor::randn(0.0, 1.0, &[256, 256], &device)?,
            None,
            Some(8),
            8.0,
            &device,
        )?;
        
        let x = Tensor::randn(0.0, 1.0, &[1, 10, 256], &device)?;
        let output = layer.forward(&x)?;
        
        // Create a dummy loss
        let target = Tensor::randn(0.0, 1.0, &[1, 10, 256], &device)?;
        let loss = output.sub(&target)?.sqr()?.mean_all()?;
        
        // Check that we can compute gradients
        let grads = loss.backward()?;
        
        // Verify gradients exist for LoRA parameters
        for param in layer.trainable_parameters() {
            assert!(param.grad().is_ok(), "LoRA parameter missing gradient");
        }
        
        Ok(())
    }
    
    #[test]
    fn test_attention_with_lora() -> Result<()> {
        let device = Device::Cpu;
        let dim = 512;
        let num_heads = 8;
        let config = LoRALayerConfig {
            rank: 16,
            alpha: 16.0,
            dropout: 0.0,
            use_bias: false,
        };
        
        // Create dummy var builder
        let vs = candle_nn::VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&vs, DType::F32, &device);
        
        // Initialize required tensors
        let _ = vb.get((dim, dim), "to_q.weight")?;
        let _ = vb.get((dim, dim), "to_k.weight")?;
        let _ = vb.get((dim, dim), "to_v.weight")?;
        let _ = vb.get((dim, dim), "to_out.0.weight")?;
        
        let attn = AttentionWithLoRA::new(dim, num_heads, Some(&config), vb)?;
        
        // Test forward pass
        let x = Tensor::randn(0.0, 1.0, &[2, 10, dim], &device)?;
        let output = attn.forward(&x)?;
        
        assert_eq!(output.shape().dims(), &[2, 10, dim]);
        assert_eq!(attn.trainable_parameters().len(), 8); // 4 projections * 2 LoRA matrices
        
        Ok(())
    }
}