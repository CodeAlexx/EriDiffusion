use crate::loaders::WeightLoader;
use flame_core::{DType, Shape, Tensor};
use std::sync::Arc;
use flame_core::device::Device;
use crate::ops::Linear;
use crate::ops::LayerNorm;
use flame_core::Module;
use std::collections::HashMap;
use flame_core::{Result};

// LoRA-enabled attention layers for SDXL
// This module provides attention mechanisms with integrated LoRA support

/// Module for managing SDXL LoRA layers
pub mod sdxl_lora_layer {
    use super::*;
    
    /// LoRA linear layer for SDXL
    pub struct LoRALinear {
        pub lora_a: Tensor,
        pub lora_b: Tensor,
        pub scale: f32,
        pub rank: usize,
        pub alpha: f32,
    }
    
    impl LoRALinear {
        pub fn new(
            in_features: usize,
            out_features: usize,
            rank: usize,
            alpha: f32,
            dropout: Option<f32>,
            device: &Device,
        ) -> Result<Self> {
            // Initialize LoRA matrices
            let lora_a = Tensor::randn(0.0, 0.02, (rank, in_features), DType::F32, device)?;
            let lora_b = Tensor::zeros((out_features, rank), DType::F32, device)?;
            let scale = alpha / (rank as f32);
            
            Ok(Self {
                lora_a,
                lora_b,
                scale,
                rank,
                alpha,
            })
        }
        
        pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
            // x @ A^T @ B^T * scale
            let h = x.matmul(&self.lora_a.transpose_dims(0, 1)?)?;
            let out = h.matmul(&self.lora_b.transpose_dims(0, 1)?)?;
            out.mul_scalar(self.scale)
        }
    }
}

use sdxl_lora_layer::LoRALinear;

/// Configuration for LoRA in attention layers
#[derive(Clone)]
pub struct LoRAAttentionConfig {
    pub rank: usize,
    pub alpha: f32,
    pub dropout: Option<f32>,
    pub target_modules: Vec<String>,
}

impl Default for LoRAAttentionConfig {
    fn default() -> Self {
        Self {
            rank: 16,
            alpha: 16.0,
            dropout: None,
            target_modules: vec![
                "to_q".to_string(),
                "to_k".to_string(),
                "to_v".to_string(),
                "to_out.0".to_string(),
            ],
        }
    }
}

/// Cross attention with LoRA support
pub struct CrossAttentionWithLoRA {
    pub to_q_base: Linear,
    pub to_k_base: Linear,
    pub to_v_base: Linear,
    pub to_out_base: Linear,
    pub to_q_lora: Option<LoRALinear>,
    pub to_k_lora: Option<LoRALinear>,
    pub to_v_lora: Option<LoRALinear>,
    pub to_out_lora: Option<LoRALinear>,
    pub heads: usize,
    pub scale: f32,
    pub use_flash_attn: bool,
}

impl CrossAttentionWithLoRA {
    pub fn new(
        vs: &WeightLoader,
        query_dim: usize,
        context_dim: Option<usize>,
        heads: usize,
        dim_head: usize,
        use_flash_attn: bool,
        lora_config: Option<&LoRAAttentionConfig>,
        device: Device,
    ) -> Result<Self> {
        let inner_dim = dim_head * heads;
        let context_dim = context_dim.unwrap_or(query_dim);
        let scale = 1.0 / (dim_head as f32).sqrt();

        // Load base layers (pre-trained weights)
        // TODO: Load weights from pre-trained model
        // For now, create with random initialization
        let device = Device::from(vs.device().clone());
        let to_q_base = Linear::new(query_dim, inner_dim, true, &device.cuda_device())?;
        let to_k_base = Linear::new(context_dim, inner_dim, true, &device.cuda_device())?;
        let to_v_base = Linear::new(context_dim, inner_dim, true, &device.cuda_device())?;
        let to_out_base = Linear::new(inner_dim, query_dim, true, &device.cuda_device())?;

        // Create LoRA layers if config provided
        let (to_q_lora, to_k_lora, to_v_lora, to_out_lora) = if let Some(config) = lora_config {
            let mut q_lora = None;
            let mut k_lora = None;
            let mut v_lora = None;
            let mut out_lora = None;

            for target in &config.target_modules {
                match target.as_str() {
                    "to_q" => {
                        q_lora = Some(LoRALinear::new(
                            query_dim,
                            inner_dim,
                            config.rank,
                            config.alpha,
                            config.dropout,
                            &device,
                        )?);
                    }
                    "to_k" => {
                        k_lora = Some(LoRALinear::new(
                            context_dim,
                            inner_dim,
                            config.rank,
                            config.alpha,
                            config.dropout,
                            &device,
                        )?);
                    }
                    "to_v" => {
                        v_lora = Some(LoRALinear::new(
                            context_dim,
                            inner_dim,
                            config.rank,
                            config.alpha,
                            config.dropout,
                            &device,
                        )?);
                    }
                    "to_out" | "to_out.0" => {
                        out_lora = Some(LoRALinear::new(
                            inner_dim,
                            query_dim,
                            config.rank,
                            config.alpha,
                            config.dropout,
                            &device,
                        )?);
                    }
                    _ => {
                        // Skip unrecognized projection layers
                    }
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
            use_flash_attn,
        })
    }

    pub fn forward(&self, hidden_states: &Tensor, context: Option<&Tensor>) -> Result<Tensor> {
        let context = context.unwrap_or(hidden_states);
        let batch_size = hidden_states.shape().dims()[0];
        let sequence_length = hidden_states.shape().dims()[1];

        // Apply base + LoRA for Q, K, V
        let mut q = self.to_q_base.forward(hidden_states)?;
        if let Some(lora) = &self.to_q_lora {
            q = q.add(&lora.forward(hidden_states)?)?;
        }

        let mut k = self.to_k_base.forward(context)?;
        if let Some(lora) = &self.to_k_lora {
            k = k.add(&lora.forward(context)?)?;
        }

        let mut v = self.to_v_base.forward(context)?;
        if let Some(lora) = &self.to_v_lora {
            v = v.add(&lora.forward(context)?)?;
        }

        // Reshape for multi-head attention
        let inner_dim = q.dim(-1)?;
        let head_dim = inner_dim / self.heads;
        
        let q = q.reshape((batch_size, sequence_length, self.heads, head_dim))?
            .transpose_dims(1, 2)?;
        let context_len = context.shape().dims()[1];
        let k = k.reshape((batch_size, context_len, self.heads, head_dim))?
            .transpose_dims(1, 2)?;
        let v = v.reshape((batch_size, context_len, self.heads, head_dim))?
            .transpose_dims(1, 2)?;

        // Compute attention
        let attn = if self.use_flash_attn {
            // Use flash attention if available
            self.flash_attention(&q, &k, &v)?
        } else {
            // Standard attention
            self.standard_attention(&q, &k, &v)?
        };

        // Reshape back
        let attn = attn.transpose_dims(1, 2)?
            .reshape((batch_size, sequence_length, inner_dim))?;

        // Apply output projection
        let mut out = self.to_out_base.forward(&attn)?;
        if let Some(lora) = &self.to_out_lora {
            out = out.add(&lora.forward(&attn)?)?;
        }

        Ok(out)
    }

    fn standard_attention(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let scores = q.matmul(&k.transpose_dims(2, 3)?)?.mul_scalar(self.scale)?;
        let probs = scores.softmax(-1)?;
        probs.matmul(v)
    }

    fn flash_attention(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        // For now, fall back to standard attention
        // TODO: Implement actual flash attention
        self.standard_attention(q, k, v)
    }

    /// Collect LoRA layers for saving/loading
    pub fn collect_lora_layers(&self) -> Vec<(String, &LoRALinear)> {
        let mut layers = Vec::new();
        
        if let Some(lora) = &self.to_q_lora {
            layers.push(("to_q".to_string(), lora));
        }
        if let Some(lora) = &self.to_k_lora {
            layers.push(("to_k".to_string(), lora));
        }
        if let Some(lora) = &self.to_v_lora {
            layers.push(("to_v".to_string(), lora));
        }
        if let Some(lora) = &self.to_out_lora {
            layers.push(("to_out.0".to_string(), lora));
        }
        
        layers
    }

    /// Get trainable parameters
    pub fn trainable_parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        
        if let Some(lora) = &self.to_q_lora {
            params.push(&lora.lora_a);
            params.push(&lora.lora_b);
        }
        if let Some(lora) = &self.to_k_lora {
            params.push(&lora.lora_a);
            params.push(&lora.lora_b);
        }
        if let Some(lora) = &self.to_v_lora {
            params.push(&lora.lora_a);
            params.push(&lora.lora_b);
        }
        if let Some(lora) = &self.to_out_lora {
            params.push(&lora.lora_a);
            params.push(&lora.lora_b);
        }
        
        params
    }
}

/// Feed forward network
struct FeedForward {
    net: Vec<Box<dyn Module>>,
}

impl FeedForward {
    pub fn new(
        vs: &WeightLoader,
        dim: usize,
        hidden_dim: Option<usize>,
        dropout: f32,
        device: Device,
    ) -> Result<Self> {
        let hidden_dim = hidden_dim.unwrap_or(dim * 4);
        
        // GEGLU activation
        let net = vec![
            // TODO: Load weights from pre-trained model
            Box::new(Linear::new(dim, hidden_dim * 2, true, &vs.device())?) as Box<dyn Module>,
            // Dropout and second linear layer would be added here
            // TODO: Load weights from pre-trained model
            Box::new(Linear::new(hidden_dim, dim, true, &vs.device())?) as Box<dyn Module>,
        ];
        
        Ok(Self { net })
    }
    
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = x.clone();
        
        // First layer with GEGLU
        h = self.net[0].forward(&h)?;
        // Chunk along the last dimension (dimension 2 for 3D tensors)
        let ndims = h.shape().dims().len();
        let (h1, h2) = crate::ops::chunk(&h, 2, ndims - 1)?;
        h = h1.gelu()?.mul(&h2)?;
        
        // Second layer
        h = self.net[1].forward(&h)?;
        
        Ok(h)
    }
}

/// Basic transformer block with LoRA
pub struct BasicTransformerBlockWithLoRA {
    attn1: CrossAttentionWithLoRA,
    attn2: CrossAttentionWithLoRA,
    ff: FeedForward,
    norm1: LayerNorm,
    norm2: LayerNorm,
    norm3: LayerNorm,
}

impl BasicTransformerBlockWithLoRA {
    pub fn new(
        vs: &WeightLoader,
        dim: usize,
        n_heads: usize,
        d_head: usize,
        context_dim: Option<usize>,
        use_flash_attn: bool,
        lora_config: Option<&LoRAAttentionConfig>,
        device: Device,
        dtype: DType,
    ) -> Result<Self> {
        // Layer norms
        let cuda_device = device.cuda_device();
        let norm1 = LayerNorm::new(vec![dim], 1e-5, cuda_device.clone())?;
        let norm2 = LayerNorm::new(vec![dim], 1e-5, cuda_device.clone())?;
        let norm3 = LayerNorm::new(vec![dim], 1e-5, cuda_device.clone())?;

        // Self attention
        let attn1 = CrossAttentionWithLoRA::new(
            &vs.pp("attn1"),
            dim,
            None,
            n_heads,
            d_head,
            use_flash_attn,
            lora_config,
            device,
        )?;

        // Cross attention
        let attn2 = CrossAttentionWithLoRA::new(
            &vs.pp("attn2"),
            dim,
            context_dim,
            n_heads,
            d_head,
            use_flash_attn,
            lora_config,
            device,
        )?;

        // Feed forward
        let ff = FeedForward::new(&vs.pp("ff"), dim, None, 0.0, device)?;

        Ok(Self {
            attn1,
            attn2,
            ff,
            norm1,
            norm2,
            norm3,
        })
    }

    pub fn forward(&self, hidden_states: &Tensor, context: Option<&Tensor>) -> Result<Tensor> {
        // Self attention
        let normed = self.norm1.forward(hidden_states)?;
        let attn_out = self.attn1.forward(&normed, None)?;
        let hidden_states = hidden_states.add(&attn_out)?;

        // Cross attention
        let normed = self.norm2.forward(&hidden_states)?;
        let attn_out = self.attn2.forward(&normed, context)?;
        let hidden_states = hidden_states.add(&attn_out)?;

        // Feed forward
        let normed = self.norm3.forward(&hidden_states)?;
        let ff_out = self.ff.forward(&normed)?;
        let hidden_states = hidden_states.add(&ff_out)?;

        Ok(hidden_states)
    }

    /// Collect all LoRA layers
    pub fn collect_lora_layers(&self) -> Vec<(String, &LoRALinear)> {
        let mut layers = Vec::new();
        
        // Self attention LoRA layers
        for (name, lora) in self.attn1.collect_lora_layers() {
            layers.push((format!("attn1.{}", name), lora));
        }
        
        // Cross attention LoRA layers
        for (name, lora) in self.attn2.collect_lora_layers() {
            layers.push((format!("attn2.{}", name), lora));
        }
        
        layers
    }

    /// Get trainable parameters
    pub fn trainable_parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        params.extend(self.attn1.trainable_parameters());
        params.extend(self.attn2.trainable_parameters());
        params
    }
}

// Extension trait for missing tensor methods
trait TensorExt {
    fn mul_scalar(&self, scalar: f32) -> Result<Tensor>;
    fn gelu(&self) -> Result<Tensor>;
    fn chunk(&self, chunks: usize, dim: isize) -> Result<(Tensor, Tensor)>;
}

impl TensorExt for Tensor {
    fn mul_scalar(&self, scalar: f32) -> Result<Tensor> {
        let scalar_tensor = Tensor::full(self.shape(), scalar, self.device())?;
        self.mul(&scalar_tensor)
    }
    
    fn gelu(&self) -> Result<Tensor> {
        // GELU activation: x * 0.5 * (1 + erf(x / sqrt(2)))
        let sqrt2 = std::f32::consts::SQRT_2;
        let x_scaled = self.div_scalar(sqrt2)?;
        let erf_x = x_scaled.erf()?;
        let one_plus_erf = erf_x.add_scalar(1.0)?;
        let half_one_plus_erf = one_plus_erf.mul_scalar(0.5)?;
        self.mul(&half_one_plus_erf)
    }
    
    fn chunk(&self, chunks: usize, dim: isize) -> Result<(Tensor, Tensor)> {
        let dim_size = self.dim(dim)?;
        let chunk_size = dim_size / chunks;
        
        let chunk1 = self.slice(dim, 0, 0 + chunk_size)?;
        let chunk2 = self.slice(dim, chunk_size, chunk_size + chunk_size)?;
        
        Ok((chunk1, chunk2))
    }
}

// Additional helper trait
trait TensorExtHelpers {
    fn add_scalar(&self, scalar: f32) -> Result<Tensor>;
    fn div_scalar(&self, scalar: f32) -> Result<Tensor>;
}

impl TensorExtHelpers for Tensor {
    fn add_scalar(&self, scalar: f32) -> Result<Tensor> {
        let scalar_tensor = Tensor::full(self.shape(), scalar, self.device())?;
        self.add(&scalar_tensor)
    }
    
    fn div_scalar(&self, scalar: f32) -> Result<Tensor> {
        let scalar_tensor = Tensor::full(self.shape(), scalar, self.device())?;
        self.div(&scalar_tensor)
    }
}