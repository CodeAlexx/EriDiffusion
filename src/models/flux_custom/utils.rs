//! Utility modules for Flux custom implementation

use candle_core::{Device, DType, Module, Result, Tensor, D, IndexOp};
use candle_nn::{linear, Linear, VarBuilder, Activation};
use std::f32::consts::PI;

/// Compute RoPE embeddings for positional encoding
pub fn rope(pos: &Tensor, dim: usize, theta: usize) -> Result<Tensor> {
    if dim % 2 == 1 {
        candle_core::bail!("dim {dim} is odd")
    }
    let original_dev = pos.device();
    
    // Move to CPU to avoid CUDA kernel issues
    let pos_cpu = pos.to_device(&Device::Cpu)?;
    
    let (b_sz, n_pos) = match pos_cpu.dims() {
        &[b, n] => (b, n),
        _ => candle_core::bail!("unexpected pos shape {:?}", pos_cpu.shape()),
    };
    let theta = theta as f32;
    let inv_freq: Vec<_> = (0..dim / 2)
        .map(|i| 1f32 / theta.powf(i as f32 * 2.0 / dim as f32))
        .collect();
    let inv_freq_len = inv_freq.len();
    let inv_freq = Tensor::from_vec(inv_freq, (1, 1, inv_freq_len), &Device::Cpu)?;
    let inv_freq = inv_freq.to_dtype(pos_cpu.dtype())?;
    let freqs = pos_cpu.unsqueeze(2)?.broadcast_mul(&inv_freq)?;
    let cos = freqs.cos()?;
    let sin = freqs.sin()?;
    let out = Tensor::stack(&[&cos, &sin.neg()?, &sin, &cos], 3)?;
    let (b, n, d, _ij) = out.dims4()?;
    let out = out.reshape((b, n, d, 2, 2))?;
    
    // Move result back to original device
    out.to_device(original_dev)
}

/// Simple MLP implementation matching Flux MlpEmbedder structure
pub struct MLP {
    in_layer: Linear,
    out_layer: Linear,
    activation: Activation,
}

impl MLP {
    pub fn new(
        in_features: usize,
        hidden_features: usize,
        out_features: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        // Use Flux naming convention: mlp.0.fc1, mlp.0.fc2
        Ok(Self {
            in_layer: linear(in_features, hidden_features, vb.pp("mlp").pp("0").pp("fc1"))?,
            out_layer: linear(hidden_features, out_features, vb.pp("mlp").pp("0").pp("fc2"))?,
            activation: Activation::Gelu,
        })
    }
}

impl Module for MLP {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.in_layer.forward(x)?;
        let h = self.activation.forward(&h)?;
        self.out_layer.forward(&h)
    }
}

/// N-dimensional embedding for positional encoding
/// This is a compute-only module that doesn't have learnable parameters
pub struct EmbedND {
    dim: usize,
    theta: usize,
    axes_dim: Vec<usize>,
}

impl EmbedND {
    pub fn new(
        dim: usize,
        axes_dims: &[usize],
        _vb: VarBuilder, // Not used, but kept for API compatibility
    ) -> Result<Self> {
        Ok(Self {
            dim,
            theta: 10_000, // Default theta value for Flux
            axes_dim: axes_dims.to_vec(),
        })
    }
    
    pub fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        // Compute RoPE embeddings on the fly
        // ids shape should be [batch, ..., n_axes]
        let original_shape = ids.dims();
        let n_axes = ids.dim(D::Minus1)?;
        
        // Convert to float if needed (ids might be I64)
        let ids_float = if matches!(ids.dtype(), DType::I64) {
            ids.to_dtype(DType::F32)?
        } else {
            ids.clone()
        };
        
        // Flatten all but the last dimension
        let batch_size = original_shape[0];
        let flattened_len: usize = original_shape[1..original_shape.len()-1].iter().product();
        let ids_flat = ids_float.reshape((batch_size, flattened_len, n_axes))?;
        
        let mut emb = Vec::with_capacity(n_axes);
        
        for idx in 0..n_axes {
            // Extract the position values for this axis
            let pos = ids_flat.get_on_dim(D::Minus1, idx)?;
            // pos is now [batch_size, flattened_len]
            let r = rope(
                &pos,
                self.axes_dim[idx],
                self.theta,
            )?;
            // r should be [batch_size, flattened_len, axes_dim[idx], 2, 2]
            // But we need to flatten the last two dimensions
            let r_shape = r.dims();
            let r_flat = r.reshape((r_shape[0], r_shape[1], r_shape[2] * 4))?;
            emb.push(r_flat)
        }
        
        // Concatenate along the embedding dimension
        let emb = Tensor::cat(&emb, 2)?;
        
        // The embedding dimension is the sum of all axes embeddings
        let embed_dim = emb.dim(2)?;
        
        // Reshape back to match input shape (except last dim which is replaced by embedding dim)
        let mut out_shape = original_shape[..original_shape.len()-1].to_vec();
        out_shape.push(embed_dim);
        
        emb.reshape(out_shape)
    }
}