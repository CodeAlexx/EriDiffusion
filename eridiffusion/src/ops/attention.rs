use flame_core::device::Device;
use flame_core::optimizers::{Adam, SGD};
use flame_core::{DType, Result, Shape, Tensor};

// Attention operations including RoPE support

/// Apply rotary position embeddings and compute attention

// Extension trait for Tensor to add missing methods
pub trait TensorExt {
    fn sum_dim(&self, dim: usize) -> flame_core::Result<Tensor>;
    fn add_scalar(&self, scalar: f32) -> flame_core::Result<Tensor>;
    fn mul_scalar(&self, scalar: f32) -> flame_core::Result<Tensor>;
    fn square(&self) -> flame_core::Result<Tensor>;
}

impl TensorExt for Tensor {
    fn sum_dim(&self, dim: usize) -> flame_core::Result<Tensor> {
        // Sum along dimension
        self.sum_dim(dim)
    }

    fn add_scalar(&self, scalar: f32) -> flame_core::Result<Tensor> {
        // Add scalar to all elements
        let scalar_tensor = Tensor::full(self.shape().clone(), scalar, self.device().clone())?;
        self.add(&scalar_tensor)
    }

    fn mul_scalar(&self, scalar: f32) -> flame_core::Result<Tensor> {
        // Multiply all elements by scalar
        let scalar_tensor = Tensor::full(self.shape().clone(), scalar, self.device().clone())?;
        self.mul(&scalar_tensor)
    }

    fn square(&self) -> flame_core::Result<Tensor> {
        // Element-wise square
        self.mul(self)
    }
}

pub fn attention(q: &Tensor, k: &Tensor, v: &Tensor, pe: &Tensor) -> flame_core::Result<Tensor> {
    // Apply RoPE to queries and keys
    let q = apply_rope(q, pe)?;
    let k = apply_rope(k, pe)?;

    // Compute scaled dot-product attention
    scaled_dot_product_attention(&q, &k, v)
}

/// Apply RoPE (Rotary Position Embeddings) to tensor
fn apply_rope(x: &Tensor, freq_cis: &Tensor) -> flame_core::Result<Tensor> {
    let dims = x.shape();
    let shape_dims = x.shape().dims();
    let b_sz = shape_dims[0];
    let n_head = shape_dims[1];
    let seq_len = shape_dims[2];
    let n_embd = shape_dims[3];
    let x = x.reshape(&[b_sz, n_head, seq_len, n_embd / 2, 2])?;
    let x0 = x.slice(&[(0, 1)])?;
    let x1 = x.slice(&[(1, 1)])?;
    let fr0 = freq_cis.slice(&[(0, 1)])?;
    let fr1 = freq_cis.slice(&[(1, 1)])?;
    fr0.mul(&x0)?.add(&fr1.mul(&x1)?)?.reshape(dims.dims())
}

/// Scaled dot-product attention
fn scaled_dot_product_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> flame_core::Result<Tensor> {
    let dims = q.shape().dims();
    let dim = dims[dims.len() - 1];
    let scale_factor = 1.0 / (dim as f64).sqrt();
    let mut batch_dims = q.shape().dims().to_vec();
    batch_dims.pop();
    batch_dims.pop();
    let q = q.flatten_from(batch_dims.len())?;
    let k = k.flatten_from(batch_dims.len())?;
    let v = v.flatten_from(batch_dims.len())?;
    let scale_tensor =
        Tensor::full(Shape::from_dims(&[1]), scale_factor as f32, q.device().clone())?;
    let attn_weights = q.matmul(&k.transpose_dims(0, 1)?)?.mul(&scale_tensor)?;
    let weights_shape = attn_weights.shape().dims();
    batch_dims.push(weights_shape[weights_shape.len() - 2]);
    batch_dims.push(weights_shape[weights_shape.len() - 1]);
    attn_weights.reshape(&batch_dims)
}
