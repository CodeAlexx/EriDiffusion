use candle_core::{Device, Result as CandleResult, Tensor, DType, Shape, Module};
use candle_nn::{VarBuilder, Linear};
use std::collections::HashMap;

/// Production Flash Attention implementation for diffusion models
pub struct FlashAttention {
    pub head_dim: usize,
    pub num_heads: usize,
    pub scale: f32,
    pub block_size: usize,
    pub device: Device,
    pub use_causal: bool,
}

impl FlashAttention {
    pub fn new(
        head_dim: usize,
        num_heads: usize,
        device: Device,
        use_causal: bool,
        block_size: Option<usize>,
    ) -> Self {
        let scale = 1.0 / (head_dim as f32).sqrt();
        let block_size = block_size.unwrap_or(64.min(head_dim));
        
        Self {
            head_dim,
            num_heads,
            scale,
            block_size,
            device,
            use_causal,
        }
    }

    /// Forward pass optimized for diffusion transformer blocks
    /// q, k, v: [batch_size, seq_len, num_heads, head_dim]
    pub fn forward(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> CandleResult<Tensor> {
        let q_shape = q.shape();
        let k_shape = k.shape();
        let v_shape = v.shape();
        
        if q_shape.dims().len() != 4 || k_shape.dims().len() != 4 || v_shape.dims().len() != 4 {
            return Err(candle_core::Error::Msg("QKV tensors must be 4D".to_string()));
        }
        
        let (batch_size, seq_len, num_heads, head_dim) = (
            q_shape.dims()[0], q_shape.dims()[1], q_shape.dims()[2], q_shape.dims()[3]
        );
        
        // Validate dimensions
        if k_shape.dims() != [batch_size, seq_len, num_heads, head_dim] ||
           v_shape.dims() != [batch_size, seq_len, num_heads, head_dim] {
            return Err(candle_core::Error::Msg("Dimension mismatch in QKV tensors".to_string()));
        }

        // Reshape to [batch_size * num_heads, seq_len, head_dim]
        let q = q.transpose(1, 2)?.reshape(&[batch_size * num_heads, seq_len, head_dim])?;
        let k = k.transpose(1, 2)?.reshape(&[batch_size * num_heads, seq_len, head_dim])?;
        let v = v.transpose(1, 2)?.reshape(&[batch_size * num_heads, seq_len, head_dim])?;

        let output = if seq_len <= self.block_size * 2 {
            // Use standard attention for small sequences
            self.standard_attention(&q, &k, &v)?
        } else {
            // Use memory-efficient flash attention for large sequences
            self.flash_attention_impl(&q, &k, &v)?
        };

        // Reshape back to [batch_size, num_heads, seq_len, head_dim] then transpose
        output
            .reshape(&[batch_size, num_heads, seq_len, head_dim])?
            .transpose(1, 2)
    }

    fn standard_attention(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> CandleResult<Tensor> {
        // Compute attention scores: Q @ K^T
        let scores = q.matmul(&k.transpose(1, 2)?)?;
        let scores = scores.mul(&Tensor::new(self.scale, &self.device)?)?;
        
        // Apply causal mask if needed
        let scores = if self.use_causal {
            self.apply_causal_mask(&scores)?
        } else {
            scores
        };
        
        // Softmax and apply to values
        let attention_weights = candle_nn::ops::softmax(&scores, 2)?;
        attention_weights.matmul(v)
    }

    fn flash_attention_impl(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> CandleResult<Tensor> {
        let q_shape = q.shape();
        let (batch_heads, seq_len, head_dim) = (
            q_shape.dims()[0], q_shape.dims()[1], q_shape.dims()[2]
        );
        let block_size = self.block_size;
        let num_blocks = (seq_len + block_size - 1) / block_size;
        
        // Initialize output tensor
        let mut output = Tensor::zeros(&[batch_heads, seq_len, head_dim], q.dtype(), &self.device)?;
        
        // Process blocks sequentially
        for i in 0..num_blocks {
            let q_start = i * block_size;
            let q_end = (q_start + block_size).min(seq_len);
            let q_block_size = q_end - q_start;
            
            let q_block = q.narrow(1, q_start, q_block_size)?;
            
            // Initialize block statistics
            let mut block_output = Tensor::zeros(&[batch_heads, q_block_size, head_dim], q.dtype(), &self.device)?;
            let mut global_max = Tensor::new(f32::NEG_INFINITY, &self.device)?
                .broadcast_as(&[batch_heads, q_block_size, 1])?;
            let mut global_sum = Tensor::zeros(&[batch_heads, q_block_size, 1], q.dtype(), &self.device)?;
            
            for j in 0..num_blocks {
                let kv_start = j * block_size;
                let kv_end = (kv_start + block_size).min(seq_len);
                let kv_block_size = kv_end - kv_start;
                
                // Skip if causal mask would zero this block
                if self.use_causal && kv_start > q_end - 1 {
                    continue;
                }
                
                let k_block = k.narrow(1, kv_start, kv_block_size)?;
                let v_block = v.narrow(1, kv_start, kv_block_size)?;
                
                // Compute attention scores for this block
                let scores = q_block.matmul(&k_block.transpose(1, 2)?)?;
                let scores = scores.mul(&Tensor::new(self.scale, &self.device)?)?;
                
                // Apply causal mask to scores
                let scores = if self.use_causal {
                    self.apply_block_causal_mask(&scores, q_start, kv_start)?
                } else {
                    scores
                };
                
                // Online softmax computation
                let local_max = scores.max_keepdim(2)?;
                let new_max = global_max.maximum(&local_max)?;
                
                // Compute exponentials with numerical stability
                let exp_scores = scores.broadcast_sub(&new_max)?.exp()?;
                let local_sum = exp_scores.sum_keepdim(2)?;
                
                // Update global statistics
                let alpha = global_max.broadcast_sub(&new_max)?.exp()?;
                global_sum = global_sum.broadcast_mul(&alpha)?.add(&local_sum)?;
                
                // Update block output
                let weighted_values = exp_scores.matmul(&v_block)?;
                block_output = block_output.broadcast_mul(&alpha)?.add(&weighted_values)?;
                global_max = new_max;
            }
            
            // Normalize the block output
            block_output = block_output.broadcast_div(&global_sum)?;
            
            // Update the global output tensor
            let output_slice = output.narrow(1, q_start, q_block_size)?;
            let updated_slice = output_slice.add(&block_output)?;
            output = self.tensor_slice_assign(&output, 1, q_start, &updated_slice)?;
        }
        
        Ok(output)
    }

    fn tensor_slice_assign(&self, tensor: &Tensor, dim: usize, start: usize, values: &Tensor) -> CandleResult<Tensor> {
        // Manual slice assignment implementation
        let tensor_shape = tensor.shape();
        let mut slices = Vec::new();
        
        for (i, &dim_size) in tensor_shape.dims().iter().enumerate() {
            if i == dim {
                if start > 0 {
                    slices.push(tensor.narrow(dim, 0, start)?);
                }
                slices.push(values.clone());
                let remaining = dim_size - start - values.dim(dim)?;
                if remaining > 0 {
                    slices.push(tensor.narrow(dim, start + values.dim(dim)?, remaining)?);
                }
                break;
            }
        }
        
        if slices.is_empty() {
            return Ok(values.clone());
        }
        
        Tensor::cat(&slices, dim)
    }

    fn apply_causal_mask(&self, scores: &Tensor) -> CandleResult<Tensor> {
        let scores_shape = scores.shape();
        let seq_len = scores_shape.dims()[2];
        
        // Create causal mask
        let mut mask_data = vec![0.0f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in 0..seq_len {
                if i < j {
                    mask_data[i * seq_len + j] = f32::NEG_INFINITY;
                }
            }
        }
        
        let mask = Tensor::from_slice(&mask_data, &[seq_len, seq_len], &self.device)?;
        let mask = mask.broadcast_as(scores.shape())?;
        
        scores.add(&mask)
    }

    fn apply_block_causal_mask(
        &self, 
        scores: &Tensor, 
        q_offset: usize, 
        kv_offset: usize
    ) -> CandleResult<Tensor> {
        let scores_shape = scores.shape();
        let q_len = scores_shape.dims()[1];
        let kv_len = scores_shape.dims()[2];
        
        let mut mask_data = vec![0.0f32; q_len * kv_len];
        for i in 0..q_len {
            for j in 0..kv_len {
                if q_offset + i < kv_offset + j {
                    mask_data[i * kv_len + j] = f32::NEG_INFINITY;
                }
            }
        }
        
        let mask = Tensor::from_slice(&mask_data, &[q_len, kv_len], &self.device)?;
        let mask = mask.broadcast_as(scores.shape())?;
        
        scores.add(&mask)
    }
}

/// Multi-Head Attention layer for diffusion transformers
pub struct MultiHeadAttention {
    pub flash_attention: FlashAttention,
    pub q_proj: candle_nn::Linear,
    pub k_proj: candle_nn::Linear,
    pub v_proj: candle_nn::Linear,
    pub out_proj: candle_nn::Linear,
    pub dropout: Option<f32>,
    pub training: bool,
}

impl MultiHeadAttention {
    pub fn new(
        dim: usize,
        num_heads: usize,
        device: Device,
        vb: VarBuilder,
        use_causal: bool,
        dropout: Option<f32>,
    ) -> CandleResult<Self> {
        let head_dim = dim / num_heads;
        if dim % num_heads != 0 {
            return Err(candle_core::Error::Msg("Dimension must be divisible by number of heads".to_string()));
        }

        let q_proj = candle_nn::linear(dim, dim, vb.pp("q_proj"))?;
        let k_proj = candle_nn::linear(dim, dim, vb.pp("k_proj"))?;
        let v_proj = candle_nn::linear(dim, dim, vb.pp("v_proj"))?;
        let out_proj = candle_nn::linear(dim, dim, vb.pp("out_proj"))?;

        let flash_attention = FlashAttention::new(head_dim, num_heads, device, use_causal, None);

        Ok(Self {
            flash_attention,
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            dropout,
            training: false,
        })
    }

    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let x_shape = x.shape();
        if x_shape.dims().len() != 3 {
            return Err(candle_core::Error::Msg("Input tensor must be 3D".to_string()));
        }
        
        let (batch_size, seq_len, dim) = (
            x_shape.dims()[0], x_shape.dims()[1], x_shape.dims()[2]
        );
        let num_heads = self.flash_attention.num_heads;
        let head_dim = self.flash_attention.head_dim;

        // Project to Q, K, V
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape to [batch_size, seq_len, num_heads, head_dim]
        let q = q.reshape(&[batch_size, seq_len, num_heads, head_dim])?;
        let k = k.reshape(&[batch_size, seq_len, num_heads, head_dim])?;
        let v = v.reshape(&[batch_size, seq_len, num_heads, head_dim])?;

        // Apply flash attention
        let attention_output = self.flash_attention.forward(&q, &k, &v)?;

        // Reshape back and project
        let attention_output = attention_output.reshape(&[batch_size, seq_len, dim])?;
        let output = self.out_proj.forward(&attention_output)?;

        // Apply dropout if specified and in training mode
        if let Some(dropout_p) = self.dropout {
            if self.training {
                return candle_nn::ops::dropout(&output, dropout_p);
            }
        }

        Ok(output)
    }
}

/// Cross-attention layer for diffusion models (e.g., text conditioning)
pub struct CrossAttention {
    pub flash_attention: FlashAttention,
    pub q_proj: candle_nn::Linear,
    pub k_proj: candle_nn::Linear,
    pub v_proj: candle_nn::Linear,
    pub out_proj: candle_nn::Linear,
}

impl CrossAttention {
    pub fn new(
        query_dim: usize,
        context_dim: usize,
        num_heads: usize,
        device: Device,
        vb: VarBuilder,
    ) -> CandleResult<Self> {
        let head_dim = query_dim / num_heads;
        if query_dim % num_heads != 0 {
            return Err(candle_core::Error::Msg("Query dimension must be divisible by number of heads".to_string()));
        }

        let q_proj = candle_nn::linear(query_dim, query_dim, vb.pp("to_q"))?;
        let k_proj = candle_nn::linear(context_dim, query_dim, vb.pp("to_k"))?;
        let v_proj = candle_nn::linear(context_dim, query_dim, vb.pp("to_v"))?;
        let out_proj = candle_nn::linear(query_dim, query_dim, vb.pp("to_out"))?;

        let flash_attention = FlashAttention::new(head_dim, num_heads, device, false, None);

        Ok(Self {
            flash_attention,
            q_proj,
            k_proj,
            v_proj,
            out_proj,
        })
    }

    pub fn forward(&self, x: &Tensor, context: &Tensor) -> CandleResult<Tensor> {
        let x_shape = x.shape();
        let context_shape = context.shape();
        
        if x_shape.dims().len() != 3 || context_shape.dims().len() != 3 {
            return Err(candle_core::Error::Msg("Input tensors must be 3D".to_string()));
        }
        
        let (batch_size, seq_len, _) = (
            x_shape.dims()[0], x_shape.dims()[1], x_shape.dims()[2]
        );
        let context_len = context_shape.dims()[1];
        let num_heads = self.flash_attention.num_heads;
        let head_dim = self.flash_attention.head_dim;

        // Project queries from x, keys and values from context
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(context)?;
        let v = self.v_proj.forward(context)?;

        // Reshape for multi-head attention
        let q = q.reshape(&[batch_size, seq_len, num_heads, head_dim])?;
        let k = k.reshape(&[batch_size, context_len, num_heads, head_dim])?;
        let v = v.reshape(&[batch_size, context_len, num_heads, head_dim])?;

        // For cross-attention, we need to handle different sequence lengths
        let attention_output = self.cross_attention_forward(&q, &k, &v)?;

        // Reshape and project
        let attention_output = attention_output.reshape(&[batch_size, seq_len, num_heads * head_dim])?;
        self.out_proj.forward(&attention_output)
    }

    fn cross_attention_forward(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> CandleResult<Tensor> {
        let q_shape = q.shape();
        let k_shape = k.shape();
        
        let (batch_size, q_len, num_heads, head_dim) = (
            q_shape.dims()[0], q_shape.dims()[1], q_shape.dims()[2], q_shape.dims()[3]
        );
        let kv_len = k_shape.dims()[1];

        // Reshape for batch processing
        let q = q.transpose(1, 2)?.reshape(&[batch_size * num_heads, q_len, head_dim])?;
        let k = k.transpose(1, 2)?.reshape(&[batch_size * num_heads, kv_len, head_dim])?;
        let v = v.transpose(1, 2)?.reshape(&[batch_size * num_heads, kv_len, head_dim])?;

        // Compute attention scores
        let scores = q.matmul(&k.transpose(1, 2)?)?;
        let scores = scores.mul(&Tensor::new(self.flash_attention.scale, &self.flash_attention.device)?)?;

        // Apply softmax and compute output
        let attention_weights = candle_nn::ops::softmax(&scores, 2)?;
        let output = attention_weights.matmul(&v)?;

        // Reshape back
        output
            .reshape(&[batch_size, num_heads, q_len, head_dim])?
            .transpose(1, 2)
    }
}

/// Utility function for creating attention layers with proper weight initialization
pub fn create_attention_layer(
    dim: usize,
    num_heads: usize,
    device: &Device,
    vb: VarBuilder,
    attention_type: AttentionType,
) -> CandleResult<Box<dyn AttentionLayer>> {
    match attention_type {
        AttentionType::SelfAttention { use_causal, dropout } => {
            let layer = MultiHeadAttention::new(dim, num_heads, device.clone(), vb, use_causal, dropout)?;
            Ok(Box::new(layer))
        },
        AttentionType::CrossAttention { context_dim } => {
            let layer = CrossAttention::new(dim, context_dim, num_heads, device.clone(), vb)?;
            Ok(Box::new(layer))
        }
    }
}

pub enum AttentionType {
    SelfAttention { use_causal: bool, dropout: Option<f32> },
    CrossAttention { context_dim: usize },
}

pub trait AttentionLayer {
    fn forward_self(&self, x: &Tensor) -> CandleResult<Tensor>;
    fn forward_cross(&self, x: &Tensor, context: &Tensor) -> CandleResult<Tensor>;
}

impl AttentionLayer for MultiHeadAttention {
    fn forward_self(&self, x: &Tensor) -> CandleResult<Tensor> {
        self.forward(x)
    }
    
    fn forward_cross(&self, _x: &Tensor, _context: &Tensor) -> CandleResult<Tensor> {
        Err(candle_core::Error::Msg("Cross attention not supported for self-attention layer".to_string()))
    }
}

impl AttentionLayer for CrossAttention {
    fn forward_self(&self, _x: &Tensor) -> CandleResult<Tensor> {
        Err(candle_core::Error::Msg("Self attention not supported for cross-attention layer".to_string()))
    }
    
    fn forward_cross(&self, x: &Tensor, context: &Tensor) -> CandleResult<Tensor> {
        self.forward(x, context)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, DType};

    #[test]
    fn test_flash_attention_basic() -> CandleResult<()> {
        let device = Device::Cpu;
        let flash_attn = FlashAttention::new(64, 8, device.clone(), false, Some(32));
        
        let batch_size = 2;
        let seq_len = 128;
        let num_heads = 8;
        let head_dim = 64;
        
        let q = Tensor::randn(0f32, 1f32, &[batch_size, seq_len, num_heads, head_dim], &device)?;
        let k = Tensor::randn(0f32, 1f32, &[batch_size, seq_len, num_heads, head_dim], &device)?;
        let v = Tensor::randn(0f32, 1f32, &[batch_size, seq_len, num_heads, head_dim], &device)?;
        
        let output = flash_attn.forward(&q, &k, &v)?;
        let output_shape = output.shape();
        assert_eq!(output_shape.dims(), &[batch_size, seq_len, num_heads, head_dim]);
        
        Ok(())
    }

    #[test]
    fn test_multi_head_attention() -> CandleResult<()> {
        let device = Device::Cpu;
        let vs = candle_nn::VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&vs, DType::F32, &device);
        
        let dim = 512;
        let num_heads = 8;
        let seq_len = 64;
        let batch_size = 2;
        
        let mut mha = MultiHeadAttention::new(dim, num_heads, device.clone(), vb, false, None)?;
        mha.set_training(false);
        
        let input = Tensor::randn(0f32, 1f32, &[batch_size, seq_len, dim], &device)?;
        let output = mha.forward(&input)?;
        
        let output_shape = output.shape();
        assert_eq!(output_shape.dims(), &[batch_size, seq_len, dim]);
        
        Ok(())
    }

    #[test]
    fn test_cross_attention() -> CandleResult<()> {
        let device = Device::Cpu;
        let vs = candle_nn::VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&vs, DType::F32, &device);
        
        let query_dim = 512;
        let context_dim = 768;
        let num_heads = 8;
        let seq_len = 64;
        let context_len = 77;
        let batch_size = 2;
        
        let cross_attn = CrossAttention::new(query_dim, context_dim, num_heads, device.clone(), vb)?;
        
        let query = Tensor::randn(0f32, 1f32, &[batch_size, seq_len, query_dim], &device)?;
        let context = Tensor::randn(0f32, 1f32, &[batch_size, context_len, context_dim], &device)?;
        
        let output = cross_attn.forward(&query, &context)?;
        
        let output_shape = output.shape();
        assert_eq!(output_shape.dims(), &[batch_size, seq_len, query_dim]);
        
        Ok(())
    }
}
