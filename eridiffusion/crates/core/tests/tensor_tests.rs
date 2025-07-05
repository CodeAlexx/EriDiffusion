//! Tests for tensor operations

use eridiffusion_core::tensor::{TensorView, TensorOps};
use eridiffusion_core::{Error, Result};
use candle_core::{Tensor, Device, DType, Shape};

#[test]
fn test_tensor_view_creation() {
    let device = Device::Cpu;
    let tensor = Tensor::arange(0f32, 24f32, &device)
        .unwrap()
        .reshape(&[2, 3, 4])
        .unwrap();
    
    let view = TensorView::new(tensor.clone());
    assert_eq!(view.shape().dims(), &[2, 3, 4]);
}

#[test]
fn test_tensor_view_slice() {
    let device = Device::Cpu;
    let tensor = Tensor::arange(0f32, 24f32, &device)
        .unwrap()
        .reshape(&[2, 3, 4])
        .unwrap();
    
    let view = TensorView::new(tensor.clone());
    
    // Valid slice
    let sliced = view.slice(&[0..1, 1..3, 0..2]).unwrap();
    assert_eq!(sliced.shape().dims(), &[1, 2, 2]);
    
    // Invalid slice - out of bounds
    assert!(view.slice(&[0..3, 0..3, 0..4]).is_err());
    
    // Invalid slice - start > end
    assert!(view.slice(&[1..0, 0..3, 0..4]).is_err());
}

#[test]
fn test_tensor_view_reshape() {
    let device = Device::Cpu;
    let tensor = Tensor::arange(0f32, 24f32, &device).unwrap();
    let view = TensorView::new(tensor.clone());
    
    // Valid reshape
    let reshaped = view.reshape(&[2, 12]).unwrap();
    assert_eq!(reshaped.shape().dims(), &[2, 12]);
    
    // Invalid reshape - wrong total elements
    assert!(view.reshape(&[2, 10]).is_err());
}

#[test]
fn test_tensor_ops_normalize() {
    let device = Device::Cpu;
    let tensor = Tensor::from_vec(
        vec![1.0f32, 2.0, 3.0, 4.0],
        &[2, 2],
        &device
    ).unwrap();
    
    // Normalize along dimension 1
    let normalized = tensor.normalize(1).unwrap();
    
    // Check that norm is 1
    let norm = normalized.sqr()
        .unwrap()
        .sum_keepdim(1)
        .unwrap()
        .sqrt()
        .unwrap();
    
    let norm_values = norm.to_vec2::<f32>().unwrap();
    assert!((norm_values[0][0] - 1.0).abs() < 1e-5);
    assert!((norm_values[1][0] - 1.0).abs() < 1e-5);
}

#[test]
fn test_tensor_ops_cosine_similarity() {
    let device = Device::Cpu;
    
    // Test with identical vectors (cosine similarity = 1)
    let a = Tensor::from_vec(vec![1.0f32, 0.0, 0.0], &[1, 3], &device).unwrap();
    let b = Tensor::from_vec(vec![1.0f32, 0.0, 0.0], &[1, 3], &device).unwrap();
    
    let sim = a.cosine_similarity(&b).unwrap();
    let sim_value = sim.to_scalar::<f32>().unwrap();
    assert!((sim_value - 1.0).abs() < 1e-5);
    
    // Test with orthogonal vectors (cosine similarity = 0)
    let c = Tensor::from_vec(vec![1.0f32, 0.0, 0.0], &[1, 3], &device).unwrap();
    let d = Tensor::from_vec(vec![0.0f32, 1.0, 0.0], &[1, 3], &device).unwrap();
    
    let sim2 = c.cosine_similarity(&d).unwrap();
    let sim_value2 = sim2.to_scalar::<f32>().unwrap();
    assert!(sim_value2.abs() < 1e-5);
}

#[test]
fn test_tensor_ops_mean_dims() {
    let device = Device::Cpu;
    let tensor = Tensor::arange(0f32, 24f32, &device)
        .unwrap()
        .reshape(&[2, 3, 4])
        .unwrap();
    
    // Mean along dimensions 0 and 2
    let mean = tensor.mean_dims(&[0, 2]).unwrap();
    assert_eq!(mean.shape().dims(), &[1, 3, 1]);
    
    // Verify values
    let values = mean.to_vec3::<f32>().unwrap();
    // First row average: (0+1+2+3 + 12+13+14+15) / 8 = 7.5
    assert!((values[0][0][0] - 7.5).abs() < 1e-5);
}

#[test]
fn test_scaled_dot_product_attention() {
    let device = Device::Cpu;
    let batch_size = 2;
    let num_heads = 4;
    let seq_len = 8;
    let head_dim = 16;
    
    // Create Q, K, V tensors
    let query = Tensor::randn(0f32, 1f32, &[batch_size, num_heads, seq_len, head_dim], &device).unwrap();
    let key = Tensor::randn(0f32, 1f32, &[batch_size, num_heads, seq_len, head_dim], &device).unwrap();
    let value = Tensor::randn(0f32, 1f32, &[batch_size, num_heads, seq_len, head_dim], &device).unwrap();
    
    // Compute attention
    let output = query.scaled_dot_product_attention(&key, &value, None).unwrap();
    
    // Check output shape
    assert_eq!(output.shape().dims(), &[batch_size, num_heads, seq_len, head_dim]);
}

// TODO: Implement this test once scaled_dot_product_attention is available
#[ignore]
#[test]
fn test_attention_with_mask() {
    let device = Device::Cpu;
    let batch_size = 1;
    let num_heads = 1;
    let seq_len = 4;
    let head_dim = 2;
    
    // Simple tensors for testing
    let query = Tensor::ones(&[batch_size, num_heads, seq_len, head_dim], DType::F32, &device).unwrap();
    let key = Tensor::ones(&[batch_size, num_heads, seq_len, head_dim], DType::F32, &device).unwrap();
    let value = Tensor::eye(seq_len, DType::F32, &device)
        .unwrap()
        .reshape(&[1, 1, seq_len, seq_len])
        .unwrap()
        .narrow(3, 0, head_dim)
        .unwrap();
    
    // Create causal mask
    let mut mask_values = vec![0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            if j > i {
                mask_values[i * seq_len + j] = -1e9;
            }
        }
    }
    let mask = Tensor::from_vec(mask_values, &[1, 1, seq_len, seq_len], &device).unwrap();
    
    // Compute attention with mask
    let output = query.scaled_dot_product_attention(&key, &value, Some(&mask)).unwrap();
    
    // Check that future positions are masked
    let output_values = output.to_vec4::<f32>().unwrap();
    
    // First position should only attend to itself
    assert!(output_values[0][0][0][0] > 0.9);
    assert!(output_values[0][0][0][1] < 0.1);
}

#[test]
fn test_tensor_apply_fn() {
    let device = Device::Cpu;
    let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], 4, &device).unwrap();
    
    // Apply square function
    let squared = tensor.apply_fn(|x| x * x).unwrap();
    let values = squared.to_vec1::<f32>().unwrap();
    
    assert_eq!(values, vec![1.0, 4.0, 9.0, 16.0]);
}

#[test]
fn test_edge_cases() {
    let device = Device::Cpu;
    
    // Empty tensor
    let empty = Tensor::zeros(&[0], DType::F32, &device).unwrap();
    assert!(empty.normalize(0).is_ok());
    
    // Single element
    let single = Tensor::from_vec(vec![42.0f32], 1, &device).unwrap();
    let normalized = single.normalize(0).unwrap();
    assert_eq!(normalized.to_scalar::<f32>().unwrap(), 1.0);
}