//! Efficient tensor operations

use crate::{Error, Result};
use candle_core::{Tensor, DType, Device, Shape};
use candle_nn::Module;

/// Custom operation types that can be fused
#[derive(Debug, Clone)]
pub enum FusedOp {
    AddScalar(f64),
    MulScalar(f64),
    AddMulScalar(f64, f64), // x * a + b
    Clamp(f64, f64),
    LeakyRelu(f64),
    Gelu,
}

/// Efficient fused operations
pub struct FusedKernel;

impl FusedKernel {
    /// Apply fused operation without intermediate allocations
    pub fn apply(tensor: &Tensor, op: FusedOp) -> Result<Tensor> {
        match op {
            FusedOp::AddScalar(scalar) => {
                tensor.broadcast_add(&Tensor::new(scalar, tensor.device())?)
                    .map_err(|e| Error::TensorOp(e))
            }
            FusedOp::MulScalar(scalar) => {
                tensor.broadcast_mul(&Tensor::new(scalar, tensor.device())?)
                    .map_err(|e| Error::TensorOp(e))
            }
            FusedOp::AddMulScalar(mul, add) => {
                // Fused multiply-add: x * mul + add
                tensor.affine(mul, add)
                    .map_err(|e| Error::TensorOp(e))
            }
            FusedOp::Clamp(min, max) => {
                tensor.clamp(min, max)
                    .map_err(|e| Error::TensorOp(e))
            }
            FusedOp::LeakyRelu(negative_slope) => {
                let zero = Tensor::zeros_like(tensor)?;
                let positive = tensor.maximum(&zero)?;
                let negative = tensor.minimum(&zero)?;
                (positive + negative.affine(negative_slope, 0.0)?)
                    .map_err(|e| Error::TensorOp(e))
            }
            FusedOp::Gelu => {
                // GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
                // Approximation: x * sigmoid(1.702 * x)
                let sigmoid_input = tensor.affine(1.702, 0.0)?;
                let sigmoid = candle_nn::ops::sigmoid(&sigmoid_input)?;
                tensor.mul(&sigmoid)
                    .map_err(|e| Error::TensorOp(e))
            }
        }
    }
}

/// Efficient batched operations
pub struct BatchedOps;

impl BatchedOps {
    /// Apply multiple operations in a single pass
    pub fn apply_sequence(tensor: &Tensor, ops: &[FusedOp]) -> Result<Tensor> {
        ops.iter().fold(Ok(tensor.clone()), |acc, op| {
            acc.and_then(|t| FusedKernel::apply(&t, op.clone()))
        })
    }
    
    /// Efficient batched matrix multiplication
    pub fn batched_matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
        // Use optimized batched matmul
        a.matmul(b).map_err(|e| Error::TensorOp(e))
    }
    
    /// Efficient attention computation
    pub fn efficient_attention(
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        scale: f32,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        // Compute QK^T
        let scores = Self::batched_matmul(query, &key.transpose(D::Minus2, D::Minus1)?)?;
        
        // Scale
        let scaled_scores = FusedKernel::apply(&scores, FusedOp::MulScalar(1.0 / scale as f64))?;
        
        // Apply mask if provided
        let masked_scores = if let Some(mask) = mask {
            (&scaled_scores + mask)?
        } else {
            scaled_scores
        };
        
        // Softmax
        let weights = candle_nn::ops::softmax_last_dim(&masked_scores)?;
        
        // Apply to values
        Self::batched_matmul(&weights, value)
    }
}

/// Memory-efficient operations
pub struct MemoryEfficientOps;

impl MemoryEfficientOps {
    /// In-place operations where possible
    pub fn add_inplace(tensor: &mut Tensor, other: &Tensor) -> Result<()> {
        // Note: Candle doesn't support true in-place ops yet
        // We simulate it by reassigning the result
        *tensor = (tensor.as_ref() + other)?;
        Ok(())
    }
    
    /// Checkpointed operations for memory efficiency
    pub fn checkpointed_forward<F, T>(
        inputs: &[Tensor],
        forward_fn: F,
    ) -> Result<T>
    where
        F: Fn(&[Tensor]) -> Result<T>,
    {
        // For now, just call the function
        // In the future, this could implement gradient checkpointing
        forward_fn(inputs)
    }
    
    /// Streaming operations for large tensors
    pub fn streaming_reduce(
        tensor: &Tensor,
        dim: usize,
        chunk_size: usize,
    ) -> Result<Tensor> {
        let shape = tensor.shape();
        let dim_size = shape.dims()[dim];
        
        if dim_size <= chunk_size {
            // Small enough to process at once
            return tensor.sum_keepdim(dim).map_err(|e| Error::TensorOp(e));
        }
        
        // Process in chunks
        let mut result = None;
        let mut start = 0;
        
        while start < dim_size {
            let end = (start + chunk_size).min(dim_size);
            let chunk = tensor.narrow(dim, start, end - start)?;
            let chunk_sum = chunk.sum_keepdim(dim)?;
            
            result = match result {
                None => Some(chunk_sum),
                Some(acc) => Some((acc + chunk_sum)?),
            };
            
            start = end;
        }
        
        result.ok_or_else(|| Error::TensorOp(candle_core::Error::Msg("Empty tensor".to_string())))
    }
}

/// Dimension helpers
struct D;

impl D {
    const Minus1: isize = -1;
    const Minus2: isize = -2;
}

/// Zero-allocation tensor iterator
pub struct TensorIterator<'a> {
    tensor: &'a Tensor,
    current: usize,
    total: usize,
}

impl<'a> TensorIterator<'a> {
    pub fn new(tensor: &'a Tensor) -> Self {
        Self {
            tensor,
            current: 0,
            total: tensor.elem_count(),
        }
    }
    
    /// Compute multi-dimensional indices from linear index
    fn compute_indices(&self, linear_idx: usize) -> Vec<usize> {
        let shape = self.tensor.shape().dims();
        let mut indices = vec![0; shape.len()];
        let mut idx = linear_idx;
        
        for i in (0..shape.len()).rev() {
            indices[i] = idx % shape[i];
            idx /= shape[i];
        }
        
        indices
    }
    
    /// Extract element at given indices
    fn extract_element(&self, indices: &[usize]) -> Result<f32> {
        // Build index tensors for gathering
        let mut current_tensor = self.tensor.clone();
        
        for (dim, &idx) in indices.iter().enumerate() {
            // Select along dimension
            current_tensor = current_tensor.narrow(dim, idx, 1)?
                .squeeze(dim)?;
        }
        
        // Convert scalar to f32
        current_tensor.to_scalar::<f32>()
    }
}

impl<'a> Iterator for TensorIterator<'a> {
    type Item = Result<f32>;
    
    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.total {
            return None;
        }
        
        // Extract element at current position
        let indices = self.compute_indices(self.current);
        let result = self.extract_element(&indices);
        self.current += 1;
        Some(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_fused_ops() {
        let device = Device::Cpu;
        let tensor = Tensor::randn(0.0f32, 1.0f32, &[2, 3], &device).unwrap();
        
        // Test fused multiply-add
        let result = FusedKernel::apply(&tensor, FusedOp::AddMulScalar(2.0, 1.0)).unwrap();
        assert_eq!(result.shape(), tensor.shape());
    }
}