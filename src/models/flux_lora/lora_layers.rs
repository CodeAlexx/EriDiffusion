//! Temporary LoRA layers until networks crate is fixed

use candle_core::{Tensor, Module, Var, Result, Device, DType, D};
use candle_nn::VarBuilder;
use super::lora_config::LoRALayerConfig;

/// Linear layer with optional LoRA adaptation
pub struct LinearWithLoRA {
    // Base layer parameters (frozen during training)
    weight: Tensor,
    bias: Option<Tensor>,
    
    // LoRA parameters (trainable)
    lora_a: Option<Var>,
    lora_b: Option<Var>,
    
    // Configuration
    in_features: usize,
    out_features: usize,
    rank: Option<usize>,
    alpha: f32,
    dropout: f32,
    enabled: bool,
}

impl LinearWithLoRA {
    pub fn from_vb(
        vb: VarBuilder,
        in_features: usize,
        out_features: usize,
        config: Option<&LoRALayerConfig>,
    ) -> Result<Self> {
        let weight = vb.get((out_features, in_features), "weight")?;
        let bias = vb.get(out_features, "bias").ok();
        
        let (rank, alpha, dropout) = config
            .map(|c| (Some(c.rank), c.alpha, c.dropout))
            .unwrap_or((None, 1.0, 0.0));
        
        let device = weight.device();
        
        let (lora_a, lora_b) = if let Some(r) = rank {
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
            dropout,
            enabled: true,
        })
    }
    
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
}

impl Module for LinearWithLoRA {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Base forward pass
        let mut output = x.matmul(&self.weight.t()?)?;
        
        // Add LoRA adaptation if enabled and present
        if self.enabled {
            if let (Some(a), Some(b)) = (&self.lora_a, &self.lora_b) {
                let lora_out = x
                    .matmul(&a.as_tensor().t()?)?
                    .matmul(&b.as_tensor().t()?)?;
                
                let scale = self.alpha / (self.rank.unwrap_or(1) as f32);
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
    pub to_q: LinearWithLoRA,
    pub to_k: LinearWithLoRA,
    pub to_v: LinearWithLoRA,
    pub to_out: LinearWithLoRA,
    
    pub num_heads: usize,
    pub head_dim: usize,
    pub scale: f64,
    pub dropout: f32,
}

impl AttentionWithLoRA {
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
    
    pub fn trainable_parameters(&self) -> Vec<&Var> {
        let mut params = Vec::new();
        params.extend(self.to_q.trainable_parameters());
        params.extend(self.to_k.trainable_parameters());
        params.extend(self.to_v.trainable_parameters());
        params.extend(self.to_out.trainable_parameters());
        params
    }
}

impl Module for AttentionWithLoRA {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward_with_context(x, None, None)
    }
}

impl AttentionWithLoRA {
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
        let q = q.reshape(&[batch_size, seq_len, self.num_heads, self.head_dim])?
            .transpose(1, 2)?;
        let k = k.reshape(&[batch_size, seq_len, self.num_heads, self.head_dim])?
            .transpose(1, 2)?;
        let v = v.reshape(&[batch_size, seq_len, self.num_heads, self.head_dim])?
            .transpose(1, 2)?;
        
        // Compute attention
        let scores = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)?
            .affine(self.scale, 0.0)?;
        
        let scores = if let Some(mask) = mask {
            scores.broadcast_add(mask)?
        } else {
            scores
        };
        
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let attn_out = probs.matmul(&v)?;
        
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
            w1: LinearWithLoRA::from_vb(vb.pp("w1"), dim, hidden_dim, config)?,
            w2: LinearWithLoRA::from_vb(vb.pp("w2"), hidden_dim, dim, config)?,
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