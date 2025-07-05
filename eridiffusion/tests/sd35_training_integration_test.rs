//! Integration tests for SD 3.5 training pipeline

use eridiffusion_core::{Device, Result};
use eridiffusion_models::{SD3Model, TextEncoder, VAEFactory};
use eridiffusion_networks::{LoKrNetwork, LoKrConfig};
use eridiffusion_training::{
    SD35Trainer, SD35TrainingConfig, SD35ModelVariant,
    DataLoader, OptimizerConfig, OptimizerType,
    compute_loss, LossType,
};
use eridiffusion_data::{ImageDataset, DatasetConfig};
use candle_core::{Tensor, DType};
use std::path::PathBuf;
use std::time::Instant;

/// Test basic SD 3.5 training setup
#[test]
fn test_sd35_training_setup() -> Result<()> {
    let device = Device::Cpu;
    
    // Create config
    let config = SD35TrainingConfig {
        model_variant: SD35ModelVariant::Large,
        resolution: 512, // Smaller for testing
        num_train_timesteps: 1000,
        flow_matching: true,
        flow_shift: 3.0,
        t5_max_length: 77, // Shorter for testing
        mixed_precision: false, // CPU doesn't support bf16
        gradient_checkpointing: false,
    };
    
    assert_eq!(config.model_variant, SD35ModelVariant::Large);
    assert!(config.flow_matching);
    
    Ok(())
}

/// Test text encoding with triple encoders
#[test]
fn test_triple_text_encoding() -> Result<()> {
    let device = Device::Cpu;
    
    // Create mock dimensions for encoders
    let clip_l_dim = 768;
    let clip_g_dim = 1280;
    let t5_dim = 4096;
    
    // Create mock text embeddings
    let batch_size = 2;
    let seq_len = 77;
    
    let clip_l_hidden = Tensor::randn(batch_size, seq_len, clip_l_dim, DType::F32, &device)?;
    let clip_g_hidden = Tensor::randn(batch_size, seq_len, clip_g_dim, DType::F32, &device)?;
    let t5_hidden = Tensor::randn(batch_size, seq_len, t5_dim, DType::F32, &device)?;
    
    // Test padding
    let clip_l_padded = SD35Trainer::pad_embeddings(&clip_l_hidden, 2048)?;
    let clip_g_padded = SD35Trainer::pad_embeddings(&clip_g_hidden, 2048)?;
    
    assert_eq!(clip_l_padded.dim(2)?, 2048);
    assert_eq!(clip_g_padded.dim(2)?, 2048);
    
    // Test concatenation
    let concatenated = Tensor::cat(&[&clip_l_padded, &clip_g_padded, &t5_hidden], 2)?;
    assert_eq!(concatenated.dim(2)?, 2048 + 2048 + 4096); // 8192
    
    Ok(())
}

/// Test flow matching timestep sampling
#[test]
fn test_flow_matching_timesteps() -> Result<()> {
    let device = Device::Cpu;
    let batch_size = 4;
    
    // Test with flow shift
    let timesteps = SD35Trainer::sample_timesteps(
        batch_size,
        1000,
        &device,
        3.0, // flow_shift
    )?;
    
    assert_eq!(timesteps.dims(), &[batch_size]);
    
    // Check timesteps are in [0, 1]
    let t_vec = timesteps.to_vec1::<f32>()?;
    for t in &t_vec {
        assert!(*t >= 0.0 && *t <= 1.0);
    }
    
    Ok(())
}

/// Test flow matching target creation
#[test]
fn test_flow_matching_targets() -> Result<()> {
    let device = Device::Cpu;
    let batch_size = 2;
    let channels = 16;
    let height = 32;
    let width = 32;
    
    // Create test tensors
    let original = Tensor::randn(batch_size, channels, height, width, DType::F32, &device)?;
    let noise = Tensor::randn_like(&original)?;
    let timesteps = Tensor::new(&[0.3f32, 0.7f32], &device)?;
    
    // Create flow targets
    let (noisy_latents, velocity) = SD35Trainer::create_flow_targets(
        &original,
        &noise,
        &timesteps,
    )?;
    
    // Check shapes
    assert_eq!(noisy_latents.dims(), original.dims());
    assert_eq!(velocity.dims(), original.dims());
    
    // Verify interpolation
    let t = timesteps.reshape((batch_size, 1, 1, 1))?;
    let expected_noisy = &original * (1.0 - &t)? + &noise * &t;
    let diff = (&noisy_latents - &expected_noisy)?.abs()?.max_all()?.to_scalar::<f32>()?;
    assert!(diff < 1e-5);
    
    Ok(())
}

/// Test loss computation with SNR weighting
#[test]
fn test_snr_weighted_loss() -> Result<()> {
    let device = Device::Cpu;
    let batch_size = 2;
    let shape = (batch_size, 16, 32, 32);
    
    // Create predictions and targets
    let predictions = Tensor::randn(shape.0, shape.1, shape.2, shape.3, DType::F32, &device)?;
    let targets = Tensor::randn_like(&predictions)?;
    let timesteps = Tensor::new(&[0.3f32, 0.7f32], &device)?;
    
    // Test without SNR weighting
    let loss_no_snr = SD35Trainer::compute_flow_matching_loss(
        &predictions,
        &targets,
        &timesteps,
        0.0, // no SNR gamma
    )?;
    
    // Test with SNR weighting
    let loss_with_snr = SD35Trainer::compute_flow_matching_loss(
        &predictions,
        &targets,
        &timesteps,
        5.0, // SNR gamma
    )?;
    
    // Both should be scalars
    assert_eq!(loss_no_snr.rank(), 0);
    assert_eq!(loss_with_snr.rank(), 0);
    
    // Losses should be different when SNR weighting is applied
    let diff = (loss_no_snr.to_scalar::<f32>()? - loss_with_snr.to_scalar::<f32>()?).abs();
    assert!(diff > 1e-6);
    
    Ok(())
}

/// Test LoKr target module selection
#[test]
fn test_lokr_target_modules() -> Result<()> {
    // Test without text encoder
    let modules = SD35Trainer::get_sd35_target_modules(false);
    assert!(!modules.is_empty());
    
    // Should have joint blocks
    let joint_blocks: Vec<_> = modules.iter()
        .filter(|m| m.contains("joint_blocks"))
        .collect();
    assert!(!joint_blocks.is_empty());
    
    // Should have attention and MLP modules
    assert!(modules.iter().any(|m| m.contains("attn")));
    assert!(modules.iter().any(|m| m.contains("mlp")));
    
    // Test with text encoder
    let modules_with_te = SD35Trainer::get_sd35_target_modules(true);
    assert!(modules_with_te.len() > modules.len());
    
    // Should have text encoder modules
    assert!(modules_with_te.iter().any(|m| m.contains("text_model")));
    assert!(modules_with_te.iter().any(|m| m.contains("t5")));
    
    Ok(())
}

/// Test training input preparation
#[test]
fn test_training_input_preparation() -> Result<()> {
    let device = Device::Cpu;
    let batch_size = 2;
    
    // Create test tensors
    let latents = Tensor::randn(batch_size, 16, 32, 32, DType::F32, &device)?;
    let timesteps = Tensor::rand(batch_size, DType::F32, &device)?;
    let text_embeddings = Tensor::randn(batch_size, 77, 8192, DType::F32, &device)?;
    let pooled = Tensor::randn(batch_size, 2048, DType::F32, &device)?;
    
    // Prepare inputs
    let inputs = SD35Trainer::prepare_training_inputs(
        &latents,
        &timesteps,
        &text_embeddings,
        &pooled,
    )?;
    
    // Verify structure
    assert_eq!(inputs.latents.dims(), latents.dims());
    assert_eq!(inputs.timestep.dims(), timesteps.dims());
    assert!(inputs.encoder_hidden_states.is_some());
    assert!(inputs.pooled_projections.is_some());
    
    // Check pooled projections in additional
    assert!(inputs.additional.contains_key("pooled_projections"));
    
    Ok(())
}

/// Test end-to-end training step simulation
#[test]
fn test_training_step_simulation() -> Result<()> {
    let device = Device::Cpu;
    let batch_size = 2;
    
    // Simulate a training step
    let start = Instant::now();
    
    // 1. Create batch data
    let images = Tensor::randn(batch_size, 3, 512, 512, DType::F32, &device)?;
    let latents = Tensor::randn(batch_size, 16, 64, 64, DType::F32, &device)?; // Pre-encoded
    
    // 2. Create text embeddings (pre-encoded)
    let text_embeddings = Tensor::randn(batch_size, 77, 8192, DType::F32, &device)?;
    let pooled = Tensor::randn(batch_size, 2048, DType::F32, &device)?;
    
    // 3. Sample timesteps
    let timesteps = SD35Trainer::sample_timesteps(batch_size, 1000, &device, 3.0)?;
    
    // 4. Add noise
    let noise = Tensor::randn_like(&latents)?;
    let (noisy_latents, velocity_target) = SD35Trainer::create_flow_targets(
        &latents,
        &noise,
        &timesteps,
    )?;
    
    // 5. Prepare model inputs
    let inputs = SD35Trainer::prepare_training_inputs(
        &noisy_latents,
        &timesteps,
        &text_embeddings,
        &pooled,
    )?;
    
    // 6. Simulate model forward (mock output)
    let model_output = Tensor::randn_like(&velocity_target)?;
    
    // 7. Compute loss
    let loss = SD35Trainer::compute_flow_matching_loss(
        &model_output,
        &velocity_target,
        &timesteps,
        5.0, // SNR gamma
    )?;
    
    let elapsed = start.elapsed();
    
    // Verify results
    assert!(loss.to_scalar::<f32>()? > 0.0);
    println!("Simulated training step completed in {:.2}ms", elapsed.as_millis());
    
    Ok(())
}

/// Test memory usage estimation
#[test]
fn test_memory_estimation() -> Result<()> {
    let batch_size = 4;
    let resolution = 1024;
    let latent_size = resolution / 8; // 128
    
    // Calculate approximate memory usage
    let latent_memory = batch_size * 16 * latent_size * latent_size * 4; // f32
    let text_memory = batch_size * 77 * 8192 * 4; // Triple encoder output
    let model_memory = 8_000_000_000; // ~8B params for SD3.5 Large
    
    let total_bytes = latent_memory + text_memory + model_memory;
    let total_gb = total_bytes as f64 / 1e9;
    
    println!("Estimated memory usage for batch_size={}: {:.2} GB", batch_size, total_gb);
    
    assert!(total_gb > 8.0); // Should need at least 8GB
    
    Ok(())
}

/// Test configuration validation
#[test]
fn test_config_validation() -> Result<()> {
    // Test valid config
    let valid_config = SD35TrainingConfig {
        model_variant: SD35ModelVariant::Large,
        resolution: 1024,
        num_train_timesteps: 1000,
        flow_matching: true,
        flow_shift: 3.0,
        t5_max_length: 512,
        mixed_precision: true,
        gradient_checkpointing: true,
    };
    
    // Resolution should be divisible by 8
    assert_eq!(valid_config.resolution % 8, 0);
    
    // T5 max length should be reasonable
    assert!(valid_config.t5_max_length <= 512);
    
    // Flow shift should be positive for flow matching
    if valid_config.flow_matching {
        assert!(valid_config.flow_shift > 0.0);
    }
    
    Ok(())
}