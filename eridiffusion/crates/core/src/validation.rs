//! Input validation utilities

use crate::{Error, Result};
use candle_core::{Tensor, DType, Shape};

/// Validate tensor inputs
pub struct TensorValidator;

impl TensorValidator {
    /// Validate tensor is finite (no NaN or Inf)
    pub fn validate_finite(tensor: &Tensor, name: &str) -> Result<()> {
        match tensor.dtype() {
            DType::F32 => {
                let data = tensor.to_vec1::<f32>()
                    .map_err(|e| Error::Validation(format!("Failed to read {}: {}", name, e)))?;
                
                for (i, &val) in data.iter().enumerate() {
                    if !val.is_finite() {
                        return Err(Error::Validation(format!(
                            "{} contains non-finite value {} at index {}",
                            name, val, i
                        )));
                    }
                }
            }
            DType::F64 => {
                let data = tensor.to_vec1::<f64>()
                    .map_err(|e| Error::Validation(format!("Failed to read {}: {}", name, e)))?;
                
                for (i, &val) in data.iter().enumerate() {
                    if !val.is_finite() {
                        return Err(Error::Validation(format!(
                            "{} contains non-finite value {} at index {}",
                            name, val, i
                        )));
                    }
                }
            }
            _ => {
                // Other dtypes are always finite
            }
        }
        
        Ok(())
    }
    
    /// Validate tensor shape
    pub fn validate_shape(
        tensor: &Tensor,
        expected_rank: Option<usize>,
        expected_dims: Option<&[usize]>,
        name: &str,
    ) -> Result<()> {
        let shape = tensor.shape();
        
        // Check rank
        if let Some(rank) = expected_rank {
            if shape.rank() != rank {
                return Err(Error::Validation(format!(
                    "{} has rank {}, expected {}",
                    name,
                    shape.rank(),
                    rank
                )));
            }
        }
        
        // Check dimensions
        if let Some(dims) = expected_dims {
            if shape.dims() != dims {
                return Err(Error::Validation(format!(
                    "{} has shape {:?}, expected {:?}",
                    name,
                    shape.dims(),
                    dims
                )));
            }
        }
        
        Ok(())
    }
    
    /// Validate tensor range
    pub fn validate_range(
        tensor: &Tensor,
        min: Option<f64>,
        max: Option<f64>,
        name: &str,
    ) -> Result<()> {
        match tensor.dtype() {
            DType::F32 => {
                let data = tensor.to_vec1::<f32>()
                    .map_err(|e| Error::Validation(format!("Failed to read {}: {}", name, e)))?;
                
                for (i, &val) in data.iter().enumerate() {
                    if let Some(min_val) = min {
                        if (val as f64) < min_val {
                            return Err(Error::Validation(format!(
                                "{} value {} at index {} is below minimum {}",
                                name, val, i, min_val
                            )));
                        }
                    }
                    
                    if let Some(max_val) = max {
                        if (val as f64) > max_val {
                            return Err(Error::Validation(format!(
                                "{} value {} at index {} exceeds maximum {}",
                                name, val, i, max_val
                            )));
                        }
                    }
                }
            }
            DType::F64 => {
                let data = tensor.to_vec1::<f64>()
                    .map_err(|e| Error::Validation(format!("Failed to read {}: {}", name, e)))?;
                
                for (i, &val) in data.iter().enumerate() {
                    if let Some(min_val) = min {
                        if val < min_val {
                            return Err(Error::Validation(format!(
                                "{} value {} at index {} is below minimum {}",
                                name, val, i, min_val
                            )));
                        }
                    }
                    
                    if let Some(max_val) = max {
                        if val > max_val {
                            return Err(Error::Validation(format!(
                                "{} value {} at index {} exceeds maximum {}",
                                name, val, i, max_val
                            )));
                        }
                    }
                }
            }
            _ => {}
        }
        
        Ok(())
    }
    
    /// Validate tensor is not empty
    pub fn validate_not_empty(tensor: &Tensor, name: &str) -> Result<()> {
        if tensor.elem_count() == 0 {
            return Err(Error::Validation(format!("{} is empty", name)));
        }
        Ok(())
    }
}

/// Validate configuration values
pub struct ConfigValidator;

impl ConfigValidator {
    /// Validate numeric config value
    pub fn validate_range<T: PartialOrd + std::fmt::Display>(
        value: T,
        min: Option<T>,
        max: Option<T>,
        name: &str,
    ) -> Result<()> {
        if let Some(min_val) = min {
            if value < min_val {
                return Err(Error::Validation(format!(
                    "{} value {} is below minimum {}",
                    name, value, min_val
                )));
            }
        }
        
        if let Some(max_val) = max {
            if value > max_val {
                return Err(Error::Validation(format!(
                    "{} value {} exceeds maximum {}",
                    name, value, max_val
                )));
            }
        }
        
        Ok(())
    }
    
    /// Validate string is not empty
    pub fn validate_not_empty_string(value: &str, name: &str) -> Result<()> {
        if value.trim().is_empty() {
            return Err(Error::Validation(format!("{} cannot be empty", name)));
        }
        Ok(())
    }
    
    /// Validate path exists
    pub fn validate_path_exists(path: &std::path::Path, name: &str) -> Result<()> {
        if !path.exists() {
            return Err(Error::Validation(format!(
                "{} path does not exist: {:?}",
                name, path
            )));
        }
        Ok(())
    }
    
    /// Validate dimensions are compatible
    pub fn validate_dimensions_compatible(
        dims1: &[usize],
        dims2: &[usize],
        name: &str,
    ) -> Result<()> {
        if dims1.len() != dims2.len() {
            return Err(Error::Validation(format!(
                "{}: dimension mismatch - rank {} vs {}",
                name,
                dims1.len(),
                dims2.len()
            )));
        }
        
        for (i, (&d1, &d2)) in dims1.iter().zip(dims2.iter()).enumerate() {
            if d1 != d2 && d1 != 1 && d2 != 1 {
                return Err(Error::Validation(format!(
                    "{}: dimension {} incompatible - {} vs {}",
                    name, i, d1, d2
                )));
            }
        }
        
        Ok(())
    }
}

/// Validate model inputs
pub fn validate_model_inputs(inputs: &crate::ModelInputs) -> Result<()> {
    // Validate latents
    TensorValidator::validate_not_empty(&inputs.latents, "latents")?;
    TensorValidator::validate_finite(&inputs.latents, "latents")?;
    
    // Validate timestep
    TensorValidator::validate_not_empty(&inputs.timestep, "timestep")?;
    
    // Validate encoder hidden states if present
    if let Some(ref encoder_hidden_states) = inputs.encoder_hidden_states {
        TensorValidator::validate_not_empty(encoder_hidden_states, "encoder_hidden_states")?;
        TensorValidator::validate_finite(encoder_hidden_states, "encoder_hidden_states")?;
    }
    
    // Validate timestep
    if let Some(timestep) = inputs.additional.get("timestep") {
        TensorValidator::validate_finite(timestep, "timestep")?;
        TensorValidator::validate_range(timestep, Some(0.0), Some(1000.0), "timestep")?;
    }
    
    Ok(())
}

/// Path sanitization
pub fn sanitize_path(path: &std::path::Path) -> Result<std::path::PathBuf> {
    use std::path::Component;
    
    let mut sanitized = std::path::PathBuf::new();
    
    for component in path.components() {
        match component {
            Component::ParentDir => {
                return Err(Error::Validation(
                    "Path traversal detected: '..' not allowed".to_string()
                ));
            }
            Component::RootDir => {
                // Allow absolute paths but be careful
                sanitized.push("/");
            }
            Component::CurDir => {
                // Skip current directory markers
            }
            Component::Normal(c) => {
                let s = c.to_string_lossy();
                // Check for suspicious patterns
                if s.contains("..") || s.contains('\0') {
                    return Err(Error::Validation(format!(
                        "Invalid path component: {:?}",
                        c
                    )));
                }
                sanitized.push(c);
            }
            Component::Prefix(p) => {
                // Windows drive letters
                sanitized.push(p.as_os_str());
            }
        }
    }
    
    Ok(sanitized)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_tensor_validation() {
        // Test implementation would go here
    }
    
    #[test]
    fn test_path_sanitization() {
        // Test implementation would go here
    }
}