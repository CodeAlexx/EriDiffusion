//! Tensor operations and views for zero-copy operations

use crate::{Result, Error};
use candle_core::{Tensor, Shape, Device, DType};
use std::ops::Range;

/// Zero-copy tensor view
#[derive(Debug, Clone)]
pub struct TensorView {
    /// Reference to the underlying tensor
    tensor: Tensor,
    /// Offset into the tensor
    offset: usize,
    /// Shape of the view
    shape: Shape,
    /// Strides for indexing
    strides: Vec<usize>,
}

impl TensorView {
    /// Create a new tensor view
    pub fn new(tensor: Tensor) -> Self {
        let shape = tensor.shape().clone();
        let strides = Self::compute_strides(&shape);
        
        Self {
            tensor,
            offset: 0,
            shape,
            strides,
        }
    }
    
    /// Create a slice view without copying data
    pub fn slice(&self, ranges: &[Range<usize>]) -> Result<Self> {
        if ranges.len() != self.shape.rank() {
            return Err(Error::TensorOp(candle_core::Error::ShapeMismatchBinaryOp {
                lhs: self.shape.clone(),
                rhs: Shape::from_dims(&vec![1; ranges.len()]),
                op: "slice",
            }));
        }
        
        let mut new_shape_dims = vec![];
        let mut new_offset = self.offset;
        
        for (i, range) in ranges.iter().enumerate() {
            let dim_size = self.shape.dims()[i];
            
            // Comprehensive bounds checking
            if range.start >= dim_size {
                return Err(Error::TensorOp(candle_core::Error::DimOutOfRange {
                    shape: self.shape.clone(),
                    dim: i as i32,
                    op: "slice_start",
                }));
            }
            
            if range.end > dim_size {
                return Err(Error::TensorOp(candle_core::Error::DimOutOfRange {
                    shape: self.shape.clone(),
                    dim: i as i32,
                    op: "slice_end",
                }));
            }
            
            if range.start > range.end {
                return Err(Error::TensorOp(candle_core::Error::Msg(format!(
                    "Invalid slice range at dimension {}: start {} > end {}",
                    i, range.start, range.end
                ))));
            }
            
            new_shape_dims.push(range.end - range.start);
            new_offset += range.start * self.strides[i];
        }
        
        // Validate that the new offset doesn't exceed tensor bounds
        let tensor_elements = self.tensor.elem_count();
        if new_offset >= tensor_elements {
            return Err(Error::TensorOp(candle_core::Error::Msg(
                "Slice offset exceeds tensor bounds".to_string()
            )));
        }
        
        Ok(Self {
            tensor: self.tensor.clone(),
            offset: new_offset,
            shape: Shape::from_dims(&new_shape_dims),
            strides: self.strides.clone(),
        })
    }
    
    /// Reshape the view without copying data
    pub fn reshape(&self, new_shape: &[usize]) -> Result<Self> {
        let total_elements: usize = self.shape.dims().iter().product();
        let new_total: usize = new_shape.iter().product();
        
        if total_elements != new_total {
            return Err(Error::TensorOp(candle_core::Error::ShapeMismatchBinaryOp {
                lhs: self.shape.clone(),
                rhs: Shape::from_dims(new_shape),
                op: "reshape",
            }));
        }
        
        Ok(Self {
            tensor: self.tensor.clone(),
            offset: self.offset,
            shape: Shape::from_dims(new_shape),
            strides: Self::compute_strides(&Shape::from_dims(new_shape)),
        })
    }
    
    /// Get the shape of the view
    pub fn shape(&self) -> &Shape {
        &self.shape
    }
    
    /// Get the underlying tensor
    pub fn as_tensor(&self) -> &Tensor {
        &self.tensor
    }
    
    /// Materialize the view into a new tensor
    pub fn materialize(&self) -> Result<Tensor> {
        if self.offset == 0 && self.shape == *self.tensor.shape() {
            Ok(self.tensor.clone())
        } else {
            // Create index tensor for gathering
            let indices = self.compute_indices()?;
            self.tensor.gather(&indices, 0)
                .map_err(|e| Error::TensorOp(e))
        }
    }
    
    /// Compute strides for a shape
    fn compute_strides(shape: &Shape) -> Vec<usize> {
        let dims = shape.dims();
        let mut strides = vec![1; dims.len()];
        
        for i in (0..dims.len() - 1).rev() {
            strides[i] = strides[i + 1] * dims[i + 1];
        }
        
        strides
    }
    
    /// Compute flat indices for gathering
    fn compute_indices(&self) -> Result<Tensor> {
        let num_elements: usize = self.shape.dims().iter().product();
        let tensor_elements = self.tensor.elem_count();
        
        // Pre-allocate with capacity for efficiency
        let mut indices = Vec::with_capacity(num_elements);
        
        for i in 0..num_elements {
            let mut flat_idx = self.offset;
            let mut remaining = i;
            
            // Compute multi-dimensional index more efficiently
            for (dim_idx, &dim_size) in self.shape.dims().iter().enumerate() {
                let coord = remaining % dim_size;
                remaining /= dim_size;
                flat_idx += coord * self.strides[dim_idx];
            }
            
            // Bounds check
            if flat_idx >= tensor_elements {
                return Err(Error::TensorOp(candle_core::Error::Msg(format!(
                    "Index {} out of bounds for tensor with {} elements",
                    flat_idx, tensor_elements
                ))));
            }
            
            indices.push(flat_idx as u32);
        }
        
        Tensor::from_vec(indices, num_elements, self.tensor.device())
            .map_err(|e| Error::TensorOp(e))
    }
}

/// Extended tensor operations
pub trait TensorOps {
    /// Apply a function element-wise
    fn apply_fn<F>(&self, f: F) -> Result<Tensor>
    where
        F: Fn(f32) -> f32;
    
    /// Compute cosine similarity
    fn cosine_similarity(&self, other: &Tensor) -> Result<Tensor>;
    
    /// Normalize along a dimension
    fn normalize(&self, dim: usize) -> Result<Tensor>;
    
    /// Compute mean along multiple dimensions
    fn mean_dims(&self, dims: &[usize]) -> Result<Tensor>;
    
    /// Efficient attention computation
    fn scaled_dot_product_attention(
        &self,
        key: &Tensor,
        value: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor>;
}

impl TensorOps for Tensor {
    fn apply_fn<F>(&self, f: F) -> Result<Tensor>
    where
        F: Fn(f32) -> f32,
    {
        // Use candle's built-in apply operation for efficiency
        use candle_core::Module;
        
        // Create a custom module that applies the function
        struct ApplyFn<F: Fn(f32) -> f32> {
            f: F,
        }
        
        impl<F: Fn(f32) -> f32> Module for ApplyFn<F> {
            fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
                // For now, we still need to convert to vec, but we can optimize this later
                let shape = x.shape().clone();
                let device = x.device();
                
                match x.dtype() {
                    candle_core::DType::F32 => {
                        let data = x.to_vec1::<f32>()?;
                        let result: Vec<f32> = data.into_iter().map(&self.f).collect();
                        Tensor::from_vec(result, shape, device)
                    }
                    _ => Err(candle_core::Error::Msg("apply_fn only supports f32".to_string())),
                }
            }
        }
        
        let module = ApplyFn { f };
        module.forward(self).map_err(|e| Error::TensorOp(e))
    }
    
    fn cosine_similarity(&self, other: &Tensor) -> Result<Tensor> {
        let dot_product = (self * other)?
            .sum_keepdim(self.rank() - 1)?;
        
        let self_norm = self.sqr()?
            .sum_keepdim(self.rank() - 1)?
            .sqrt()?;
        
        let other_norm = other.sqr()?
            .sum_keepdim(other.rank() - 1)?
            .sqrt()?;
        
        let denominator = (self_norm * other_norm)? + 1e-8;
        
        (dot_product / denominator)
            .map_err(|e| Error::TensorOp(e))
    }
    
    fn normalize(&self, dim: usize) -> Result<Tensor> {
        let norm = self.sqr()?
            .sum_keepdim(dim)?
            .sqrt()? + 1e-8;
        
        (self / norm)
            .map_err(|e| Error::TensorOp(e))
    }
    
    fn mean_dims(&self, dims: &[usize]) -> Result<Tensor> {
        let mut result = self.clone();
        
        // Sort dimensions in descending order to avoid index shifting
        let mut sorted_dims = dims.to_vec();
        sorted_dims.sort_by(|a, b| b.cmp(a));
        
        for &dim in &sorted_dims {
            result = result.mean_keepdim(dim)?;
        }
        
        Ok(result)
    }
    
    fn scaled_dot_product_attention(
        &self,
        key: &Tensor,
        value: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (batch_size, num_heads, seq_len, head_dim) = self.dims4()
            .map_err(|e| Error::TensorOp(e))?;
        
        // Compute attention scores
        let scores = self.matmul(&key.transpose(2, 3)?)?;
        
        // Scale by sqrt(head_dim)
        let scale = (head_dim as f32).sqrt();
        let scale_tensor = Tensor::new(&[scale], scores.device())?;
        let scaled_scores = scores.broadcast_div(&scale_tensor)?;
        
        // Apply mask if provided
        let masked_scores = if let Some(mask) = mask {
            (scaled_scores + mask)?
        } else {
            scaled_scores
        };
        
        // Softmax
        let attention_weights = candle_nn::ops::softmax(&masked_scores, 3)?;
        
        // Apply attention to values
        attention_weights.matmul(value)
            .map_err(|e| Error::TensorOp(e))
    }
}

/// Extension trait for missing Tensor methods
pub trait TensorExt {
    /// Add scalar to tensor
    fn add_scalar(&self, scalar: f64) -> Result<Tensor>;
    
    /// Get element at index (for 2D tensors)
    fn i(&self, coords: (usize, usize)) -> Result<Tensor>;
    
    /// Compute variance over all elements
    fn var_all(&self) -> Result<Tensor>;
}

impl TensorExt for Tensor {
    fn add_scalar(&self, scalar: f64) -> Result<Tensor> {
        let scalar_tensor = Tensor::new(&[scalar], self.device())?;
        self.broadcast_add(&scalar_tensor)
            .map_err(|e| Error::TensorOp(e))
    }
    
    fn i(&self, coords: (usize, usize)) -> Result<Tensor> {
        self.narrow(0, coords.0, 1)?
            .narrow(1, coords.1, 1)
            .map_err(|e| Error::TensorOp(e))
    }
    
    fn var_all(&self) -> Result<Tensor> {
        let mean = self.mean_all()?;
        let diff = self.sub(&mean)?;
        let squared = diff.sqr()?;
        squared.mean_all()
            .map_err(|e| Error::TensorOp(e))
    }
}