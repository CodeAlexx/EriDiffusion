use crate::loaders::WeightLoader;
use flame_core::{DType, Shape, Tensor};
use flame_core::device::Device;
use flame_core::linear::Linear;
use std::collections::HashMap;
use flame_core::{Result};

/// Flash Attention implementation for FLAME
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
        Self {
            head_dim,
            num_heads,
            scale,
            block_size: block_size.unwrap_or(128),
            device,
            use_causal,
        }
    }

    /// Compute flash attention
    pub fn forward(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        // Get dimensions
        let (batch_size, seq_len_q, hidden_dim) = {
        let dims = query.shape().dims();
        (dims[0], dims[1], dims[2])
    }?;
        let (_, seq_len_k, _) = {
        let dims = key.shape().dims();
        (dims[0], dims[1], dims[2])
    }?;

        // Reshape Q, K, V for multi-head attention
        // (batch_size, seq_len, hidden_dim) -> (batch_size, seq_len, num_heads, head_dim)
        let q = query.reshape((batch_size, seq_len_q, self.num_heads, self.head_dim))?;
        let k = key.reshape((batch_size, seq_len_k, self.num_heads, self.head_dim))?;
        let v = value.reshape((batch_size, seq_len_k, self.num_heads, self.head_dim))?;

        // Transpose to (batch_size, num_heads, seq_len, head_dim)
        let q = q.transpose_dims(1, 2)?;
        let k = k.transpose_dims(1, 2)?;
        let v = v.transpose_dims(1, 2)?;

        // Flash attention computation
        let attn_output = if self.device.cuda_device().is_cuda() {
            // Use optimized CUDA kernel if available
            self.flash_attention_cuda(&q, &k, &v, mask)?
        } else {
            // CPU fallback
            self.flash_attention_cpu(&q, &k, &v, mask)?
        };

        // Transpose back and reshape
        let attn_output = attn_output
            .transpose_dims(1, 2)?
            .reshape((batch_size, seq_len_q, hidden_dim))?;

        Ok(attn_output)
    }

    /// Flash attention CUDA implementation
    fn flash_attention_cuda(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        // For now, fall back to standard attention
        // TODO: Implement actual flash attention CUDA kernel
        self.standard_attention(query, key, value, mask)
    }

    /// Flash attention CPU implementation
    fn flash_attention_cpu(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        // For CPU, use standard attention
        self.standard_attention(query, key, value, mask)
    }

    /// Standard attention computation (fallback)
    fn standard_attention(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        // Q @ K^T * scale
        let scores = query.matmul(&key.transpose_dims(2, 3)?)?.mul_scalar(self.scale)?;

        // Apply mask if provided
        let scores = if let Some(mask) = mask {
            scores.add(mask)?
        } else if self.use_causal {
            // Apply causal mask
            let seq_len = query.dim(-2)?;
            let mask = self.create_causal_mask(seq_len)?;
            scores.add(&mask)?
        } else {
            scores
        };

        // Softmax
        let attn_weights = scores.softmax(-1)?;

        // Attention @ V
        attn_weights.matmul(value)
    }

    /// Create causal mask
    fn create_causal_mask(&self, seq_len: usize) -> Result<Tensor> {
        let mask = Tensor::ones((seq_len, seq_len), DType::F32, &self.device.clone())?;
        let mask = mask.tril(0)?;
        let mask = (Tensor::ones_like(&mask)? - mask)? * -10000.0;
        Ok(mask)
    }
}

/// Multi-head attention with Flash Attention
pub struct MultiHeadAttention {
    pub flash_attention: FlashAttention,
    pub dropout: Option<f32>,
    pub training: bool,
}

impl MultiHeadAttention {
    pub fn new(
        embed_dim: usize,
        num_heads: usize,
        dropout: Option<f32>,
        device: Device,
    ) -> Self {
        let head_dim = embed_dim / num_heads;
        let flash_attention = FlashAttention::new(head_dim, num_heads, device.clone(), false, None);

        Self {
            flash_attention,
            dropout,
            training: false,
        }
    }

    pub fn forward(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let output = self.flash_attention.forward(query, key, value, mask)?;

        // Apply dropout if in training mode
        if self.training && self.dropout.is_some() {
            let dropout_rate = self.dropout.unwrap();
            output.dropout(dropout_rate)
        } else {
            Ok(output)
        }
    }
}

/// Cross attention with Flash Attention
pub struct CrossAttention {
    pub flash_attention: FlashAttention,
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub out_proj: Linear,
}

impl CrossAttention {
    pub fn new(
        query_dim: usize,
        context_dim: usize,
        num_heads: usize,
        device: Device,
        loader: &WeightLoader,
    ) -> Result<Self> {
        let head_dim = query_dim / num_heads;
        if query_dim % num_heads != 0 {
            return Err(flame_core::Error::InvalidOperation(
                format!("query_dim {} must be divisible by num_heads {}", query_dim, num_heads)
            ));
        }

        let flash_attention = FlashAttention::new(head_dim, num_heads, device.clone(), false, None);

        // Load projection layers
        // TODO: Load weights from pre-trained model
        let q_proj = Linear::new(query_dim, query_dim, true, &device.cuda_device())?;

        let k_proj = Linear::new(context_dim, query_dim, true, &device.cuda_device())?;

        let v_proj = Linear::new(context_dim, query_dim, true, &device.cuda_device())?;

        let out_proj = Linear::new(query_dim, query_dim, true, &device.cuda_device())?;

        Ok(Self {
            flash_attention,
            q_proj,
            k_proj,
            v_proj,
            out_proj,
        })
    }

    pub fn forward(&self, x: &Tensor, context: &Tensor) -> Result<Tensor> {
        let x_shape = x.shape();
        let context_shape = context.shape();

        if x_shape.dims().len() != 3 || context_shape.dims().len() != 3 {
            return Err(flame_core::Error::InvalidOperation(
                "Cross attention expects 3D tensors (batch, seq_len, hidden_dim)".to_string()
            ));
        }

        // Project Q, K, V
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(context)?;
        let v = self.v_proj.forward(context)?;

        // Apply flash attention
        let attn_output = self.flash_attention.forward(&q, &k, &v, None)?;

        // Output projection
        self.out_proj.forward(&attn_output)
    }
}

// Extension trait for missing tensor methods
trait TensorExt {
    fn mul_scalar(&self, scalar: f32) -> Result<Tensor>;
    fn dropout(&self, rate: f32) -> Result<Tensor>;
}

impl TensorExt for Tensor {
    fn mul_scalar(&self, scalar: f32) -> Result<Tensor> {
        let scalar_tensor = Tensor::full(self.shape(), scalar, self.device())?;
        self.mul(&scalar_tensor)
    }

    fn dropout(&self, rate: f32) -> Result<Tensor> {
        if rate == 0.0 {
            return Ok(self.clone());
        }
        
        let keep_prob = 1.0 - rate;
        // Generate uniform random between 0 and 1 for dropout mask
        let random = Tensor::randn(self.shape().clone(), 0.5, 0.29, self.device().clone())?;
        let mask = random.gt(rate)?;
        let scale = 1.0 / keep_prob;
        
        self.mul(&mask.to_dtype(self.dtype())?)?.mul_scalar(scale)
    }
}