//! Tests for validation module

use eridiffusion_core::validation::*;
use eridiffusion_core::{Error, Result};
use candle_core::{Tensor, Device, DType};

#[test]
fn test_tensor_finite_validation() {
    let device = Device::Cpu;
    
    // Test valid tensor
    let valid = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], 3, &device).unwrap();
    assert!(TensorValidator::validate_finite(&valid, "test").is_ok());
    
    // Test tensor with NaN
    let with_nan = Tensor::from_vec(vec![1.0f32, f32::NAN, 3.0], 3, &device).unwrap();
    assert!(TensorValidator::validate_finite(&with_nan, "test").is_err());
    
    // Test tensor with Inf
    let with_inf = Tensor::from_vec(vec![1.0f32, f32::INFINITY, 3.0], 3, &device).unwrap();
    assert!(TensorValidator::validate_finite(&with_inf, "test").is_err());
}

#[test]
fn test_tensor_shape_validation() {
    let device = Device::Cpu;
    let tensor = Tensor::zeros(&[2, 3, 4], DType::F32, &device).unwrap();
    
    // Test correct rank
    assert!(TensorValidator::validate_shape(&tensor, Some(3), None, "test").is_ok());
    assert!(TensorValidator::validate_shape(&tensor, Some(2), None, "test").is_err());
    
    // Test correct dimensions
    assert!(TensorValidator::validate_shape(&tensor, None, Some(&[2, 3, 4]), "test").is_ok());
    assert!(TensorValidator::validate_shape(&tensor, None, Some(&[2, 3, 5]), "test").is_err());
}

#[test]
fn test_tensor_range_validation() {
    let device = Device::Cpu;
    let tensor = Tensor::from_vec(vec![0.0f32, 0.5, 1.0], 3, &device).unwrap();
    
    // Test valid range
    assert!(TensorValidator::validate_range(&tensor, Some(0.0), Some(1.0), "test").is_ok());
    
    // Test below minimum
    assert!(TensorValidator::validate_range(&tensor, Some(0.1), None, "test").is_err());
    
    // Test above maximum
    assert!(TensorValidator::validate_range(&tensor, None, Some(0.9), "test").is_err());
}

#[test]
fn test_config_validation() {
    // Test numeric range
    assert!(ConfigValidator::validate_range(5, Some(0), Some(10), "test").is_ok());
    assert!(ConfigValidator::validate_range(5, Some(6), None, "test").is_err());
    
    // Test string validation
    assert!(ConfigValidator::validate_not_empty_string("hello", "test").is_ok());
    assert!(ConfigValidator::validate_not_empty_string("", "test").is_err());
    assert!(ConfigValidator::validate_not_empty_string("  ", "test").is_err());
}

#[test]
fn test_path_sanitization() {
    use std::path::Path;
    
    // Test normal path
    let normal = Path::new("/home/user/model.safetensors");
    assert!(sanitize_path(normal).is_ok());
    
    // Test path with parent directory traversal
    let traversal = Path::new("/home/user/../../../etc/passwd");
    assert!(sanitize_path(traversal).is_err());
    
    // Test relative path
    let relative = Path::new("./models/test.bin");
    assert!(sanitize_path(relative).is_ok());
}

#[test]
fn test_dimensions_compatibility() {
    // Test compatible dimensions
    assert!(ConfigValidator::validate_dimensions_compatible(
        &[2, 3, 4],
        &[2, 3, 4],
        "test"
    ).is_ok());
    
    // Test broadcasting compatible
    assert!(ConfigValidator::validate_dimensions_compatible(
        &[2, 1, 4],
        &[2, 3, 4],
        "test"
    ).is_ok());
    
    // Test incompatible dimensions
    assert!(ConfigValidator::validate_dimensions_compatible(
        &[2, 3, 4],
        &[2, 5, 4],
        "test"
    ).is_err());
    
    // Test different ranks
    assert!(ConfigValidator::validate_dimensions_compatible(
        &[2, 3],
        &[2, 3, 4],
        "test"
    ).is_err());
}