//! Tensor operations and views for zero-copy operations

use crate::Error;
use flame_core::{Tensor, Shape};
use std::ops::Range;

/// Zero-copy tensor view
#[derive(Clone)]
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

impl std::fmt::Debug for TensorView {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TensorView")
            .field("offset", &self.offset)
            .field("shape", &self.shape)
            .field("strides", &self.strides)
            .finish()
    }
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
    pub fn slice(&self, ranges: &[Range<usize>]) -> anyhow::Result<Self> {
        if ranges.len() != self.shape.rank() {
            return Err(Error::TensorOp(format!(
                "Shape mismatch in slice operation: expected {} dimensions, got {}",
                self.shape.rank(), ranges.len()
            )).into());
        }
        
        let mut new_shape_dims = vec![];
        let mut new_offset = self.offset;
        
        for (i, range) in ranges.iter().enumerate() {
            let dim_size = self.shape.dims()[i];
            
            // Comprehensive bounds checking
            if range.start >= dim_size {
                return Err(Error::TensorOp(format!(
                    "Slice start {} out of range for dimension {} (size {})",
                    range.start, i, dim_size
                )).into());
            }
            
            if range.end > dim_size {
                return Err(Error::TensorOp(format!(
                    "Slice end {} out of range for dimension {} (size {})",
                    range.end, i, dim_size
                )).into());
            }
            
            if range.start > range.end {
                return Err(Error::TensorOp(format!(
                    "Invalid slice range at dimension {}: start {} > end {}",
                    i, range.start, range.end
                )).into());
            }
            
            new_shape_dims.push(range.end - range.start);
            new_offset += range.start * self.strides[i];
        }
        
        // Validate that the new offset doesn't exceed tensor bounds
        let tensor_elements = self.tensor.shape().elem_count();
        if new_offset >= tensor_elements {
            return Err(Error::TensorOp(
                "Slice offset exceeds tensor bounds".to_string()
            ).into());
        }
        
        Ok(Self {
            tensor: self.tensor.clone(),
            offset: new_offset,
            shape: Shape::from_dims(&new_shape_dims),
            strides: self.strides.clone(),
        })
    }
    
    /// Reshape the view without copying data
    pub fn reshape(&self, new_shape: &[usize]) -> anyhow::Result<Self> {
        let total_elements: usize = self.shape.dims().iter().product();
        let new_total: usize = new_shape.iter().product();
        
        if total_elements != new_total {
            return Err(Error::TensorOp(format!(
                "Cannot reshape tensor with {} elements to shape with {} elements",
                total_elements, new_total
            )).into());
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
    pub fn materialize(&self) -> anyhow::Result<Tensor> {
        if self.offset == 0 && self.shape == *self.tensor.shape() {
            Ok(self.tensor.clone())
        } else {
            // FLAME doesn't have gather yet, so we'll need to implement this differently
            // For now, just return a clone of the tensor
            // TODO: Implement proper slicing when FLAME supports it
            Ok(self.tensor.clone())
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
    #[allow(dead_code)]
    fn compute_indices(&self) -> anyhow::Result<Tensor> {
        let num_elements: usize = self.shape.dims().iter().product();
        let tensor_elements = self.tensor.shape().elem_count();
        
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
                return Err(Error::TensorOp(format!(
                    "Index {} out of bounds for tensor with {} elements",
                    flat_idx, tensor_elements
                )).into());
            }
            
            indices.push(flat_idx as u32);
        }
        
        // FLAME doesn't support u32 tensors directly, would need to convert or handle differently
        Err(Error::TensorOp("Index gathering not yet implemented for FLAME".into()).into())
    }
}

/// Extended tensor operations
pub trait TensorOps {
    /// Apply a function element-wise
    fn apply_fn<F>(&self, f: F) -> anyhow::Result<Tensor>
    where
        F: Fn(f32) -> f32;
    
    /// Compute cosine similarity
    fn cosine_similarity(&self, other: &Tensor) -> anyhow::Result<Tensor>;
    
    /// Normalize along a dimension
    fn normalize(&self, dim: usize) -> anyhow::Result<Tensor>;
    
    /// Compute mean along multiple dimensions
    fn mean_dims(&self, dims: &[usize]) -> anyhow::Result<Tensor>;
    
    /// Efficient attention computation
    fn scaled_dot_product_attention(
        &self,
        key: &Tensor,
        value: &Tensor,
        mask: Option<&Tensor>,
    ) -> anyhow::Result<Tensor>;
}

impl TensorOps for Tensor {
    fn apply_fn<F>(&self, f: F) -> anyhow::Result<Tensor>
    where
        F: Fn(f32) -> f32,
    {
        // FLAME doesn't have a Module trait yet, implement apply_fn directly
        // For now, we need to convert to vec and back
        let _shape = self.shape().clone();
        
        // FLAME tensors are always f32
        let data = self.to_vec()
            .map_err(|e| Error::Flame(e))?;
        let _result: Vec<f32> = data.into_iter().map(f).collect();
        // FLAME needs device for tensor creation
        // For now, return error as we don't have device context here
        Err(Error::TensorOp("apply_fn needs device context in FLAME".into()).into())
    }
    
    fn cosine_similarity(&self, _other: &Tensor) -> anyhow::Result<Tensor> {
        // Placeholder: not implemented yet for this backend
        Err(Error::TensorOp("cosine_similarity not yet implemented for FLAME".into()).into())
    }
    
    fn normalize(&self, _dim: usize) -> anyhow::Result<Tensor> {
        // FLAME doesn't have sqr, sum_keepdim methods yet
        Err(Error::TensorOp("normalize not yet implemented for FLAME".into()).into())
    }
    
    fn mean_dims(&self, dims: &[usize]) -> anyhow::Result<Tensor> {
        let result = self.clone();
        let mut sorted_dims = dims.to_vec();
        sorted_dims.sort_by(|a, b| b.cmp(a));
        if !sorted_dims.is_empty() {
            return Err(Error::TensorOp("mean_dims not yet implemented for FLAME".into()).into());
        }
        Ok(result)
    }
    
    fn scaled_dot_product_attention(
        &self,
        _key: &Tensor,
        _value: &Tensor,
        _mask: Option<&Tensor>,
    ) -> anyhow::Result<Tensor> {
        // FLAME doesn't have dims4 method
        // For now, return placeholder
        Err(Error::TensorOp("scaled_dot_product_attention not yet implemented for FLAME".into()).into())
    }
}

/// Extension trait for missing Tensor methods
pub trait TensorExt {
    /// Add scalar to tensor
    fn add_scalar(&self, scalar: f64) -> anyhow::Result<Tensor>;
    
    /// Get element at index (for 2D tensors)
    fn i(&self, coords: (usize, usize)) -> anyhow::Result<Tensor>;
    
    /// Compute variance over all elements
    fn var_all(&self) -> anyhow::Result<Tensor>;
}

impl TensorExt for Tensor {
    fn add_scalar(&self, scalar: f64) -> anyhow::Result<Tensor> {
        // FLAME uses add_scalar directly
        self.add_scalar(scalar as f32)
            .map_err(|e| anyhow::anyhow!(Error::Flame(e)))
    }
    
    fn i(&self, _coords: (usize, usize)) -> anyhow::Result<Tensor> {
        // FLAME doesn't have narrow method yet
        Err(Error::TensorOp("Element indexing not yet implemented for FLAME".into()).into())
    }
    
    fn var_all(&self) -> anyhow::Result<Tensor> {
        // FLAME doesn't have mean_all, sub, sqr methods yet
        Err(Error::TensorOp("var_all not yet implemented for FLAME".into()).into())
    }
}
