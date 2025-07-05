//! Tests for loss functions

use eridiffusion_training::loss::*;
use candle_core::{Tensor, Device, DType};

#[test]
fn test_mse_loss() {
    let device = Device::Cpu;
    let pred = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[2, 2], &device).unwrap();
    let target = Tensor::from_vec(vec![1.5f32, 2.5, 3.5, 4.5], &[2, 2], &device).unwrap();
    
    let config = LossConfig {
        weight: 1.0,
        reduction: ReductionType::Mean,
    };
    
    let loss = MSELoss::new(config).unwrap();
    let result = loss.compute(&pred, &target).unwrap();
    
    // Expected: mean((0.5^2 + 0.5^2 + 0.5^2 + 0.5^2)) = 0.25
    let value = result.to_scalar::<f32>().unwrap();
    assert!((value - 0.25).abs() < 1e-5);
}

#[test]
fn test_mae_loss() {
    let device = Device::Cpu;
    let pred = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[2, 2], &device).unwrap();
    let target = Tensor::from_vec(vec![2.0f32, 1.0, 4.0, 3.0], &[2, 2], &device).unwrap();
    
    let config = LossConfig {
        weight: 1.0,
        reduction: ReductionType::Mean,
    };
    
    let loss = MAELoss::new(config).unwrap();
    let result = loss.compute(&pred, &target).unwrap();
    
    // Expected: mean(|1-2| + |2-1| + |3-4| + |4-3|) = 1.0
    let value = result.to_scalar::<f32>().unwrap();
    assert!((value - 1.0).abs() < 1e-5);
}

#[test]
fn test_huber_loss() {
    let device = Device::Cpu;
    let pred = Tensor::from_vec(vec![0.0f32, 0.5, 2.0, 5.0], 4, &device).unwrap();
    let target = Tensor::zeros(4, DType::F32, &device).unwrap();
    
    let config = LossConfig {
        weight: 1.0,
        reduction: ReductionType::Mean,
    };
    
    let loss = HuberLoss::new(config, 1.0).unwrap();
    let result = loss.compute(&pred, &target).unwrap();
    
    // Huber loss with delta=1.0:
    // |0-0| = 0 < 1: 0.5 * 0^2 = 0
    // |0.5-0| = 0.5 < 1: 0.5 * 0.5^2 = 0.125
    // |2-0| = 2 > 1: 1.0 * (2 - 0.5) = 1.5
    // |5-0| = 5 > 1: 1.0 * (5 - 0.5) = 4.5
    // Mean = (0 + 0.125 + 1.5 + 4.5) / 4 = 1.53125
    let value = result.to_scalar::<f32>().unwrap();
    assert!((value - 1.53125).abs() < 1e-4);
}

#[test]
fn test_loss_weight_validation() {
    // Test negative weight
    let config = LossConfig {
        weight: -1.0,
        reduction: ReductionType::Mean,
    };
    
    assert!(MSELoss::new(config).is_err());
    
    // Test NaN weight
    let config2 = LossConfig {
        weight: f32::NAN,
        reduction: ReductionType::Mean,
    };
    
    assert!(MAELoss::new(config2).is_err());
}

#[test]
fn test_loss_reduction_types() {
    let device = Device::Cpu;
    let pred = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], 4, &device).unwrap();
    let target = Tensor::from_vec(vec![2.0f32, 3.0, 4.0, 5.0], 4, &device).unwrap();
    
    // Test Sum reduction
    let config_sum = LossConfig {
        weight: 1.0,
        reduction: ReductionType::Sum,
    };
    
    let loss_sum = MSELoss::new(config_sum).unwrap();
    let result_sum = loss_sum.compute(&pred, &target).unwrap();
    
    // Expected: sum(1^2 + 1^2 + 1^2 + 1^2) = 4.0
    let value_sum = result_sum.to_scalar::<f32>().unwrap();
    assert!((value_sum - 4.0).abs() < 1e-5);
    
    // Test None reduction (should keep dimensions)
    let config_none = LossConfig {
        weight: 1.0,
        reduction: ReductionType::None,
    };
    
    let loss_none = MSELoss::new(config_none).unwrap();
    let result_none = loss_none.compute(&pred, &target).unwrap();
    
    assert_eq!(result_none.shape().dims(), &[4]);
    let values = result_none.to_vec1::<f32>().unwrap();
    assert!(values.iter().all(|&v| (v - 1.0).abs() < 1e-5));
}

#[test]
fn test_flow_matching_loss() {
    let device = Device::Cpu;
    let pred = Tensor::randn(0f32, 1f32, &[2, 4, 32, 32], &device).unwrap();
    let target = Tensor::randn(0f32, 1f32, &[2, 4, 32, 32], &device).unwrap();
    
    let config = LossConfig {
        weight: 1.0,
        reduction: ReductionType::Mean,
    };
    
    let loss = FlowMatchingLoss::new(config).unwrap();
    let result = loss.compute(&pred, &target).unwrap();
    
    // Should return a scalar
    assert_eq!(result.shape().rank(), 0);
}

#[test]
fn test_lpips_loss() {
    let device = Device::Cpu;
    let pred = Tensor::randn(0f32, 1f32, &[2, 3, 64, 64], &device).unwrap();
    let target = Tensor::randn(0f32, 1f32, &[2, 3, 64, 64], &device).unwrap();
    
    let config = LossConfig {
        weight: 1.0,
        reduction: ReductionType::Mean,
    };
    
    let loss = LPIPSLoss::new(config).unwrap();
    let result = loss.compute(&pred, &target).unwrap();
    
    // Should return a scalar
    assert_eq!(result.shape().rank(), 0);
    
    // Test with mismatched shapes
    let wrong_target = Tensor::randn(0f32, 1f32, &[2, 3, 32, 32], &device).unwrap();
    assert!(loss.compute(&pred, &wrong_target).is_err());
}

#[test]
fn test_loss_edge_cases() {
    let device = Device::Cpu;
    
    // Test with zero tensors
    let zeros = Tensor::zeros(&[2, 2], DType::F32, &device).unwrap();
    let config = LossConfig {
        weight: 1.0,
        reduction: ReductionType::Mean,
    };
    
    let mse = MSELoss::new(config).unwrap();
    let result = mse.compute(&zeros, &zeros).unwrap();
    assert_eq!(result.to_scalar::<f32>().unwrap(), 0.0);
    
    // Test with single element
    let single = Tensor::from_vec(vec![42.0f32], 1, &device).unwrap();
    let result2 = mse.compute(&single, &single).unwrap();
    assert_eq!(result2.to_scalar::<f32>().unwrap(), 0.0);
}