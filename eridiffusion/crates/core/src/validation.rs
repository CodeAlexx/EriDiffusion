//! Input validation utilities

use crate::Error;
use flame_core::Tensor;

/// Validate tensor inputs
pub struct TensorValidator;

impl TensorValidator {
    /// Validate tensor is finite (no NaN or Inf)
    pub fn validate_finite(tensor: &Tensor, name: &str) -> anyhow::Result<()> {
        // FLAME tensors are always f32
        let data = tensor
            .to_vec()
            .map_err(|e| anyhow::anyhow!(format!("Failed to read {}: {}", name, e)))?;
        
        for (i, &val) in data.iter().enumerate() {
            if !val.is_finite() {
                return Err(Error::Validation(format!(
                    "{} contains non-finite value {} at index {}",
                    name, val, i
                )).into());
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
    ) -> anyhow::Result<()> {
        let shape = tensor.shape();
        
        // Check rank
        if let Some(rank) = expected_rank {
            if shape.rank() != rank {
                return Err(Error::Validation(format!(
                    "{} has rank {}, expected {}",
                    name,
                    shape.rank(),
                    rank
                )).into());
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
                )).into());
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
    ) -> anyhow::Result<()> {
        // FLAME tensors are always f32
        let data = tensor
            .to_vec()
            .map_err(|e| anyhow::anyhow!(format!("Failed to read {}: {}", name, e)))?;
        
        for (i, &val) in data.iter().enumerate() {
            if let Some(min_val) = min {
                if (val as f64) < min_val {
                    return Err(Error::Validation(format!(
                        "{} value {} at index {} is below minimum {}",
                        name, val, i, min_val
                    )).into());
                }
            }
            
            if let Some(max_val) = max {
                if (val as f64) > max_val {
                    return Err(Error::Validation(format!(
                        "{} value {} at index {} exceeds maximum {}",
                        name, val, i, max_val
                    )).into());
                }
            }
        }
        
        Ok(())
    }
    
    /// Validate tensor is not empty
    pub fn validate_not_empty(tensor: &Tensor, name: &str) -> anyhow::Result<()> {
        if tensor.shape().elem_count() == 0 {
            return Err(Error::Validation(format!("{} is empty", name)).into());
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
    ) -> anyhow::Result<()> {
        if let Some(min_val) = min {
            if value < min_val {
                return Err(Error::Validation(format!(
                    "{} value {} is below minimum {}",
                    name, value, min_val
                )).into());
            }
        }
        
        if let Some(max_val) = max {
            if value > max_val {
                return Err(Error::Validation(format!(
                    "{} value {} exceeds maximum {}",
                    name, value, max_val
                )).into());
            }
        }
        
        Ok(())
    }
    
    /// Validate string is not empty
    pub fn validate_not_empty_string(value: &str, name: &str) -> anyhow::Result<()> {
        if value.trim().is_empty() {
            return Err(Error::Validation(format!("{} cannot be empty", name)).into());
        }
        Ok(())
    }
    
    /// Validate path exists
    pub fn validate_path_exists(path: &std::path::Path, name: &str) -> anyhow::Result<()> {
        if !path.exists() {
            return Err(Error::Validation(format!(
                "{} path does not exist: {:?}",
                name, path
            )).into());
        }
        Ok(())
    }
    
    /// Validate dimensions are compatible
    pub fn validate_dimensions_compatible(
        dims1: &[usize],
        dims2: &[usize],
        name: &str,
    ) -> anyhow::Result<()> {
        if dims1.len() != dims2.len() {
            return Err(Error::Validation(format!(
                "{}: dimension mismatch - rank {} vs {}",
                name,
                dims1.len(),
                dims2.len()
            )).into());
        }
        
        for (i, (&d1, &d2)) in dims1.iter().zip(dims2.iter()).enumerate() {
            if d1 != d2 && d1 != 1 && d2 != 1 {
                return Err(Error::Validation(format!(
                    "{}: dimension {} incompatible - {} vs {}",
                    name, i, d1, d2
                )).into());
            }
        }
        
        Ok(())
    }
}

/// Validate model inputs
pub fn validate_model_inputs(inputs: &crate::ModelInputs) -> anyhow::Result<()> {
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
pub fn sanitize_path(path: &std::path::Path) -> anyhow::Result<std::path::PathBuf> {
    use std::path::Component;
    
    let mut sanitized = std::path::PathBuf::new();
    
    for component in path.components() {
        match component {
            Component::ParentDir => {
                return Err(Error::Validation(
                    "Path traversal detected: '..' not allowed".to_string()
                ).into());
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
                    )).into());
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
