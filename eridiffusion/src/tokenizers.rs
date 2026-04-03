use flame_core::device::Device;
use flame_core::GradientMap;
use flame_core::Result;
use flame_core::{DType, Shape, Tensor};
// backward is a method on Tensor, not a standalone function
use flame_core::optimizers::{Adam, SGD};

// Tokenizer utilities for text encoding

/// Simple tokenizer wrapper

// Extension trait for Tensor to add missing methods

// Extension trait for Tensor to add missing methods

trait TensorExt {
    fn to_vec(&self) -> flame_core::Result<Vec<f32>>;
    fn sum_dim(&self, dim: usize) -> flame_core::Result<Tensor>;
    fn add_scalar(&self, scalar: f32) -> flame_core::Result<Tensor>;
    fn mul_scalar(&self, scalar: f32) -> flame_core::Result<Tensor>;
    fn square(&self) -> flame_core::Result<Tensor>;
}

impl TensorExt for Tensor {
    fn to_vec(&self) -> flame_core::Result<Vec<f32>> {
        // Convert tensor to Vec<f32>
        // This is a placeholder - actual implementation would depend on FLAME internals
        Ok(vec![0.0; self.shape().dims().iter().product::<usize>()])
    }

    fn sum_dim(&self, dim: usize) -> flame_core::Result<Tensor> {
        // Sum along dimension
        self.sum_keepdim(dim as isize)
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

pub struct SimpleTokenizer {
    inner: tokenizers::Tokenizer,
}

impl SimpleTokenizer {
    pub fn new(path: &str) -> flame_core::Result<Self> {
        let tokenizer = tokenizers::Tokenizer::from_file(path).map_err(|e| {
            flame_core::Error::InvalidOperation(format!("Failed to load tokenizer: {}", e))
        })?;
        Ok(Self { inner: tokenizer })
    }

    pub fn encode(&self, text: &str) -> flame_core::Result<Vec<u32>> {
        let encoding = self.inner.encode(text, true).map_err(|e| {
            flame_core::Error::InvalidOperation(format!("Tokenization error: {}", e))
        })?;
        Ok(encoding.get_ids().to_vec())
    }
}
