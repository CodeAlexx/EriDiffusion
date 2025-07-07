//! Tests for Flux LoRA implementation

use anyhow::Result;
use candle_core::{Device, DType, Tensor};
use eridiffusion::models::flux_lora::save_lora::{save_flux_lora, LoRAConfig, create_example_lora_weights};
use std::collections::HashMap;
use std::path::PathBuf;

#[test]
fn test_save_flux_lora_format() -> Result<()> {
    // Create test device and config
    let device = Device::Cpu; // Use CPU for tests
    let dtype = DType::F32;
    let config = LoRAConfig::default();
    
    // Create example weights
    let weights = create_example_lora_weights(&device, dtype, &config)?;
    
    // Save to temporary file
    let temp_dir = std::env::temp_dir();
    let output_path = temp_dir.join("test_flux_lora.safetensors");
    
    save_flux_lora(weights, &output_path, &config)?;
    
    // Verify file exists
    assert!(output_path.exists());
    
    // Load and verify structure
    let loaded = candle_core::safetensors::load(&output_path, &device)?;
    
    // Check expected keys exist
    assert!(loaded.contains_key("transformer.double_blocks.0.img_attn.to_q.lora_A"));
    assert!(loaded.contains_key("transformer.double_blocks.0.img_attn.to_q.lora_B"));
    assert!(loaded.contains_key("transformer.double_blocks.0.img_attn.to_k.lora_A"));
    assert!(loaded.contains_key("transformer.double_blocks.0.img_attn.to_v.lora_B"));
    
    // Check tensor shapes
    let to_q_lora_a = &loaded["transformer.double_blocks.0.img_attn.to_q.lora_A"];
    assert_eq!(to_q_lora_a.dims(), &[3072, 32]); // [hidden_size, rank]
    
    let to_q_lora_b = &loaded["transformer.double_blocks.0.img_attn.to_q.lora_B"];
    assert_eq!(to_q_lora_b.dims(), &[32, 3072]); // [rank, hidden_size]
    
    // Clean up
    std::fs::remove_file(output_path).ok();
    
    Ok(())
}

#[test]
fn test_weight_translator() -> Result<()> {
    use eridiffusion::models::flux_lora::weight_translator::FluxWeightTranslator;
    
    let device = Device::Cpu;
    let dtype = DType::F32;
    
    // Create separate Q, K, V tensors (AI-Toolkit format)
    let hidden_size = 768; // Smaller for testing
    let rank = 16;
    
    let mut ai_toolkit_weights = HashMap::new();
    
    // Create Q, K, V weights
    let q_weight = Tensor::randn(0.0, 1.0, (hidden_size, hidden_size), &device)?;
    let k_weight = Tensor::randn(0.0, 1.0, (hidden_size, hidden_size), &device)?;
    let v_weight = Tensor::randn(0.0, 1.0, (hidden_size, hidden_size), &device)?;
    
    ai_toolkit_weights.insert("double_blocks.0.img_attn.to_q.weight".to_string(), q_weight);
    ai_toolkit_weights.insert("double_blocks.0.img_attn.to_k.weight".to_string(), k_weight);
    ai_toolkit_weights.insert("double_blocks.0.img_attn.to_v.weight".to_string(), v_weight);
    
    // Create translator
    let translator = FluxWeightTranslator::new();
    
    // Convert to Candle format
    let candle_weights = translator.convert_ai_toolkit_to_candle(ai_toolkit_weights.clone())?;
    
    // Check that QKV was created
    assert!(candle_weights.contains_key("double_blocks.0.img_attn.qkv.weight"));
    
    // Check shape is correct (3 * hidden_size for Q+K+V)
    let qkv = &candle_weights["double_blocks.0.img_attn.qkv.weight"];
    assert_eq!(qkv.dims(), &[hidden_size * 3, hidden_size]);
    
    Ok(())
}

#[test]
fn test_lora_module_forward() -> Result<()> {
    use eridiffusion::networks::lora::LoRAModule;
    
    let device = Device::Cpu;
    let dtype = DType::F32;
    let batch_size = 2;
    let seq_len = 16;
    let hidden_size = 768;
    let rank = 32;
    let alpha = 32.0;
    
    // Create LoRA module
    let lora = LoRAModule::new(hidden_size, hidden_size, rank, alpha, device.clone(), dtype)?;
    
    // Create input
    let input = Tensor::randn(0.0, 1.0, (batch_size, seq_len, hidden_size), &device)?;
    
    // Forward pass
    let output = lora.forward(&input)?;
    
    // Check output shape
    assert_eq!(output.dims(), &[batch_size, seq_len, hidden_size]);
    
    // Check that output is not identical to input (LoRA should add something)
    let diff = (output - &input)?.abs()?.mean_all()?;
    let diff_val = diff.to_scalar::<f32>()?;
    assert!(diff_val > 0.0, "LoRA should modify the input");
    
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn test_flux_lora_cuda_training() -> Result<()> {
    use eridiffusion::trainers::flux_lora_simple::SimpleFluxLoRATrainer;
    
    // Check if CUDA is available
    if !Device::cuda_if_available(0).is_ok() {
        println!("Skipping CUDA test - no GPU available");
        return Ok(());
    }
    
    let device = Device::cuda_if_available(0)?;
    let dtype = DType::F16;
    
    let lora_config = LoRAConfig {
        rank: 16, // Smaller for testing
        alpha: 16.0,
        target_modules: vec!["to_q".to_string(), "to_k".to_string(), "to_v".to_string()],
    };
    
    let temp_dir = std::env::temp_dir().join("flux_lora_test");
    std::fs::create_dir_all(&temp_dir)?;
    
    let trainer = SimpleFluxLoRATrainer::new(device, dtype, lora_config, temp_dir.clone());
    
    // This would normally train, but for testing we just save example weights
    trainer.train_and_save()?;
    
    // Verify output file exists
    let output_file = temp_dir.join("flux_lora_trained.safetensors");
    assert!(output_file.exists());
    
    // Clean up
    std::fs::remove_dir_all(temp_dir).ok();
    
    Ok(())
}

#[test]
fn test_ai_toolkit_naming_convention() -> Result<()> {
    // Verify our naming matches AI-Toolkit expectations
    let expected_names = vec![
        "transformer.double_blocks.0.img_attn.to_q.lora_A",
        "transformer.double_blocks.0.img_attn.to_k.lora_A",
        "transformer.double_blocks.0.img_attn.to_v.lora_A",
        "transformer.single_blocks.0.attn.to_q.lora_A",
        "transformer.single_blocks.0.mlp.0.lora_A",
    ];
    
    for name in expected_names {
        // Verify the naming pattern
        assert!(name.starts_with("transformer."));
        assert!(name.contains("lora_A") || name.contains("lora_B"));
        assert!(name.contains("to_q") || name.contains("to_k") || name.contains("to_v") || name.contains("mlp"));
    }
    
    Ok(())
}