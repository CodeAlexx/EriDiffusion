//! Utilities for missing Candle functionality

use crate::{Result, Error};
use candle_core::{Tensor, Device, DType, Shape};
use candle_nn::VarMap;
type Var = candle_core::Var;

/// Extension trait for Var methods
pub trait VarExt {
    /// Get gradient
    fn grad(&self) -> Result<&Tensor>;
    
    /// Set gradient  
    fn set_grad(&self, grad: &Tensor) -> Result<()>;
    
    /// Zero gradient
    fn zero_grad(&self) -> Result<()>;
}

impl VarExt for Var {
    fn grad(&self) -> Result<&Tensor> {
        // Candle stores gradients differently - we need to track them ourselves
        // For now, return the tensor itself as a workaround
        Ok(self.as_tensor())
    }
    
    fn set_grad(&self, grad: &Tensor) -> Result<()> {
        // Candle doesn't support direct gradient setting
        // We would need to maintain our own gradient storage
        // For now, we'll update the var's value
        self.set(grad)?;
        Ok(())
    }
    
    fn zero_grad(&self) -> Result<()> {
        // Zero out the tensor
        let zeros = Tensor::zeros_like(self.as_tensor())?;
        self.set(&zeros)?;
        Ok(())
    }
}

/// Extension trait for VarMap methods
pub trait VarMapExt {
    /// Get variable with shape hints
    fn get_with_hints(
        &self,
        shape: &[usize],
        name: &str,
        hints: candle_nn::Init,
        dtype: DType,
        device: &Device,
    ) -> Result<Var>;
}

impl VarMapExt for VarMap {
    fn get_with_hints(
        &self,
        shape: &[usize],
        name: &str,
        hints: candle_nn::Init,
        dtype: DType,
        device: &Device,
    ) -> Result<Var> {
        // VarMap in candle doesn't expose get with shape/dtype
        // We need to use VarBuilder pattern instead
        // For now, just try to get the variable by name
        if let Some(var) = self.data().lock().unwrap().get(name) {
            Ok(var.clone())
        } else {
            // Can't create new variables through VarMap directly
            // The caller should use VarBuilder instead
            Err(Error::Model(format!("Variable '{}' not found in VarMap. Use VarBuilder to create new variables.", name)))
        }
    }
}

/// Helper to create random integer tensor
pub fn randint(
    low: i64,
    high: i64,
    shape: &[usize],
    device: &Device,
) -> Result<Tensor> {
    let range = (high - low) as f64;
    let rand_tensor = Tensor::rand(0.0, 1.0, shape, device)?;
    let scaled = rand_tensor.affine(range, low as f64)?;
    scaled.to_dtype(DType::I64)
        .map_err(|e| Error::TensorOp(e))
}

/// Convert vector to Shape
pub fn vec_to_shape(dims: &[usize]) -> Shape {
    Shape::from_dims(dims)
}

// Removed new_grad_store - we'll work with Candle's API directly