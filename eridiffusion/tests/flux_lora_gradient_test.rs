//! Test to verify gradient flow through Flux LoRA implementation

use anyhow::Result;
use candle_core::{Device, Tensor, DType, Module};
use candle_nn::VarBuilder;

// Include the necessary modules
include!("../crates/networks/src/lora_layers.rs");
include!("../crates/models/src/flux/modulation.rs"); 
include!("../crates/models/src/flux_lora/double_block.rs");
include!("../crates/models/src/flux_lora/model.rs");

#[test]
fn test_flux_lora_gradient_flow() -> Result<()> {
    println!("Testing Flux LoRA gradient flow...");
    
    let device = Device::Cpu;  // Use CPU for testing
    
    // Create a minimal Flux config
    let config = FluxConfig {
        in_channels: 16,
        out_channels: 16,
        hidden_size: 256,  // Smaller for testing
        num_heads: 8,
        patch_size: 2,
        image_size: 8,  // Small image for testing
        num_double_blocks: 1,  // Minimal blocks
        num_single_blocks: 1,
        mlp_ratio: 4.0,
        guidance_embed: true,
        text_hidden_size: 512,
        max_seq_length: 77,
    };
    
    // LoRA configuration
    let lora_config = LoRALayerConfig {
        rank: 4,  // Small rank for testing
        alpha: 4.0,
        dropout: 0.0,
        use_bias: false,
    };
    
    // Create var map and builder
    let vs = candle_nn::VarMap::new();
    let vb = VarBuilder::from_varmap(&vs, DType::F32, &device);
    
    // Initialize all required weights for the model
    initialize_test_weights(&config, &vb)?;
    
    // Create model
    let model = FluxModelWithLoRA::new(config.clone(), Some(lora_config), vb)?;
    
    // Get LoRA parameters
    let lora_params = model.trainable_parameters();
    assert!(lora_params.len() > 0, "Model should have LoRA parameters");
    
    println!("Number of LoRA parameters: {}", lora_params.len());
    
    // Store initial values
    let initial_values: Vec<f32> = lora_params.iter()
        .map(|p| p.as_tensor().mean_all().unwrap().to_scalar::<f32>().unwrap())
        .collect();
    
    // Create test inputs
    let batch_size = 2;
    let latents = Tensor::randn(
        0.0, 1.0,
        &[batch_size, config.in_channels, 8, 8],
        &device
    )?;
    let timestep = Tensor::new(&[0.5f32, 0.7f32], &device)?;
    let text_embeds = Tensor::randn(
        0.0, 1.0,
        &[batch_size, 77, config.text_hidden_size],
        &device
    )?;
    let pooled = Tensor::randn(
        0.0, 1.0,
        &[batch_size, config.hidden_size],
        &device
    )?;
    
    // Create model inputs
    let inputs = eridiffusion_core::ModelInputs {
        latents,
        timestep,
        encoder_hidden_states: Some(text_embeds),
        pooled_projections: Some(pooled),
        guidance_scale: Some(3.5),
        attention_mask: None,
        additional: std::collections::HashMap::new(),
    };
    
    // Forward pass
    let output = model.forward(&inputs)?;
    
    // Create a dummy target
    let target = Tensor::randn_like(&output.sample)?;
    
    // Compute loss
    let loss = (output.sample - target).sqr()?.mean_all()?;
    
    println!("Loss: {}", loss.to_scalar::<f32>()?);
    
    // Backward pass
    let grads = loss.backward()?;
    
    // Check gradients exist and are non-zero
    let mut num_grads = 0;
    let mut num_nonzero = 0;
    
    for (i, param) in lora_params.iter().enumerate() {
        if let Ok(grad) = param.grad() {
            num_grads += 1;
            let grad_norm = grad.abs()?.mean_all()?.to_scalar::<f32>()?;
            if grad_norm > 1e-8 {
                num_nonzero += 1;
            }
            println!("LoRA param {} gradient norm: {:.6}", i, grad_norm);
        }
    }
    
    assert_eq!(num_grads, lora_params.len(), "All LoRA parameters should have gradients");
    assert!(num_nonzero > 0, "At least some gradients should be non-zero");
    
    // Simulate optimizer step
    let lr = 0.01;
    for param in lora_params.iter() {
        if let Ok(grad) = param.grad() {
            let new_value = param.as_tensor() - (grad * lr)?;
            param.set(&new_value)?;
        }
    }
    
    // Check weights changed
    let updated_values: Vec<f32> = lora_params.iter()
        .map(|p| p.as_tensor().mean_all().unwrap().to_scalar::<f32>().unwrap())
        .collect();
    
    let mut num_changed = 0;
    for (initial, updated) in initial_values.iter().zip(updated_values.iter()) {
        if (initial - updated).abs() > 1e-6 {
            num_changed += 1;
        }
    }
    
    assert!(num_changed > 0, "Some LoRA weights should have changed after optimizer step");
    
    println!("✓ Gradient flow test passed!");
    println!("  - {} LoRA parameters", lora_params.len());
    println!("  - {} parameters with gradients", num_grads);
    println!("  - {} parameters with non-zero gradients", num_nonzero);
    println!("  - {} parameters changed after update", num_changed);
    
    Ok(())
}

/// Initialize test weights for all required model components
fn initialize_test_weights(config: &FluxConfig, vb: &VarBuilder) -> Result<()> {
    // Image input conv
    let _ = vb.get(
        (config.hidden_size, config.in_channels, config.patch_size, config.patch_size),
        "img_in.weight"
    )?;
    
    // Text input projection
    let _ = vb.get(
        (config.hidden_size, config.text_hidden_size),
        "txt_in.weight"
    )?;
    
    // Time embedding layers
    let _ = vb.get((config.hidden_size, 256), "time_in.0.weight")?;
    let _ = vb.get((config.hidden_size, config.hidden_size), "time_in.2.weight")?;
    
    // Guidance embedding
    let _ = vb.get((config.hidden_size, config.hidden_size), "guidance_in.weight")?;
    
    // Position embeddings
    let num_patches = (config.image_size / config.patch_size).pow(2);
    let _ = vb.get((num_patches, config.hidden_size), "pos_embed.weight")?;
    
    // Initialize double blocks
    for i in 0..config.num_double_blocks {
        let prefix = format!("double_blocks.{}", i);
        initialize_double_block_weights(config, &vb.pp(&prefix))?;
    }
    
    // Initialize single blocks
    for i in 0..config.num_single_blocks {
        let prefix = format!("single_blocks.{}", i);
        initialize_single_block_weights(config, &vb.pp(&prefix))?;
    }
    
    // Final norm and projection
    let _ = vb.get(config.hidden_size, "final_norm.weight")?;
    let _ = vb.get(config.hidden_size, "final_norm.bias")?;
    let _ = vb.get(
        (config.patch_size * config.patch_size * config.out_channels, config.hidden_size),
        "proj_out.weight"
    )?;
    
    Ok(())
}

fn initialize_double_block_weights(config: &FluxConfig, vb: &VarBuilder) -> Result<()> {
    let hidden_size = config.hidden_size;
    let mlp_hidden = (hidden_size as f32 * config.mlp_ratio) as usize;
    
    // Image stream
    initialize_attention_weights(hidden_size, &vb.pp("img_attn"))?;
    initialize_mlp_weights(hidden_size, mlp_hidden, &vb.pp("img_mlp"))?;
    initialize_norm_weights(hidden_size, &vb.pp("img_norm1"))?;
    initialize_norm_weights(hidden_size, &vb.pp("img_norm2"))?;
    initialize_modulation_weights(hidden_size, &vb.pp("img_mod1"))?;
    initialize_modulation_weights(hidden_size, &vb.pp("img_mod2"))?;
    
    // Text stream
    initialize_attention_weights(hidden_size, &vb.pp("txt_attn"))?;
    initialize_mlp_weights(hidden_size, mlp_hidden, &vb.pp("txt_mlp"))?;
    initialize_norm_weights(hidden_size, &vb.pp("txt_norm1"))?;
    initialize_norm_weights(hidden_size, &vb.pp("txt_norm2"))?;
    initialize_modulation_weights(hidden_size, &vb.pp("txt_mod1"))?;
    initialize_modulation_weights(hidden_size, &vb.pp("txt_mod2"))?;
    
    Ok(())
}

fn initialize_single_block_weights(config: &FluxConfig, vb: &VarBuilder) -> Result<()> {
    let hidden_size = config.hidden_size;
    let mlp_hidden = (hidden_size as f32 * config.mlp_ratio) as usize;
    
    initialize_attention_weights(hidden_size, &vb.pp("attn"))?;
    initialize_mlp_weights(hidden_size, mlp_hidden, &vb.pp("mlp"))?;
    initialize_norm_weights(hidden_size, &vb.pp("norm1"))?;
    initialize_norm_weights(hidden_size, &vb.pp("norm2"))?;
    initialize_modulation_weights(hidden_size, &vb.pp("mod1"))?;
    initialize_modulation_weights(hidden_size, &vb.pp("mod2"))?;
    
    Ok(())
}

fn initialize_attention_weights(hidden_size: usize, vb: &VarBuilder) -> Result<()> {
    let _ = vb.get((hidden_size, hidden_size), "to_q.weight")?;
    let _ = vb.get((hidden_size, hidden_size), "to_k.weight")?;
    let _ = vb.get((hidden_size, hidden_size), "to_v.weight")?;
    let _ = vb.get((hidden_size, hidden_size), "to_out.0.weight")?;
    Ok(())
}

fn initialize_mlp_weights(hidden_size: usize, mlp_hidden: usize, vb: &VarBuilder) -> Result<()> {
    let _ = vb.get((mlp_hidden, hidden_size), "w1.weight")?;
    let _ = vb.get((hidden_size, mlp_hidden), "w2.weight")?;
    Ok(())
}

fn initialize_norm_weights(hidden_size: usize, vb: &VarBuilder) -> Result<()> {
    let _ = vb.get(hidden_size, "weight")?;
    let _ = vb.get(hidden_size, "bias")?;
    Ok(())
}

fn initialize_modulation_weights(hidden_size: usize, vb: &VarBuilder) -> Result<()> {
    let _ = vb.get((hidden_size * 3, hidden_size), "linear.weight")?;
    let _ = vb.get(hidden_size * 3, "linear.bias")?;
    Ok(())
}

// Mock ModelInputs and ModelOutput since we're including files directly
mod eridiffusion_core {
    use candle_core::Tensor;
    use std::collections::HashMap;
    
    pub struct ModelInputs {
        pub latents: Tensor,
        pub timestep: Tensor,
        pub encoder_hidden_states: Option<Tensor>,
        pub pooled_projections: Option<Tensor>,
        pub guidance_scale: Option<f32>,
        pub attention_mask: Option<Tensor>,
        pub additional: HashMap<String, Tensor>,
    }
    
    #[derive(Default)]
    pub struct ModelOutput {
        pub sample: Tensor,
    }
    
    #[derive(Clone, Copy)]
    pub enum ModelArchitecture {
        Flux,
    }
}

// Mock Activation enum
mod eridiffusion_networks {
    pub use crate::{LinearWithLoRA, AttentionWithLoRA, FeedForwardWithLoRA, LoRALayerConfig, Activation};
}

// Mock DiffusionModel trait
trait DiffusionModel {
    fn forward(&self, inputs: &eridiffusion_core::ModelInputs) -> Result<eridiffusion_core::ModelOutput>;
    fn architecture(&self) -> eridiffusion_core::ModelArchitecture;
    fn in_channels(&self) -> usize;
    fn out_channels(&self) -> usize;
    fn train_mode(&mut self, mode: bool);
}