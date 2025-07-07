//! Test minimal Flux model creation and forward pass

use anyhow::Result;
use candle_core::{Device, Tensor, DType};
use candle_nn::{VarBuilder, VarMap};

use eridiffusion::models::flux_custom::{FluxConfig, FluxModelWithLoRA, create_flux_lora_model};
use eridiffusion::models::flux_custom::lora::LoRAConfig;

fn main() -> Result<()> {
    println!("=== Testing Minimal Flux Model ===\n");
    
    // Create device using cached pattern
    let device = Device::new_cuda(0)?;
    println!("Device: {:?}", device);
    
    // Create minimal config with smaller dimensions
    let config = FluxConfig {
        in_channels: 16,      // Reduced from 64
        vec_in_dim: 128,      // Reduced from 768
        context_in_dim: 512,  // Reduced from 4096
        hidden_size: 256,     // Reduced from 3072
        mlp_ratio: 2.0,       // Reduced from 4.0
        num_heads: 8,         // Reduced from 24
        depth: 1,             // Only 1 double block for testing
        depth_single_blocks: 1,  // Only 1 single block
        axes_dim: vec![4, 14, 14],  // Smaller dimensions
        theta: 10_000.0,
        qkv_bias: true,
        guidance_embed: true,
    };
    
    println!("Creating model with config:");
    println!("  Double blocks: {}", config.depth);
    println!("  Single blocks: {}", config.depth_single_blocks);
    println!("  Hidden size: {}", config.hidden_size);
    
    // Test 1: Create model with create_flux_lora_model
    println!("\nTest 1: Using create_flux_lora_model");
    match create_flux_lora_model(Some(config.clone()), &device, DType::F32, None) {
        Ok(mut model) => {
            println!("✅ Model created successfully!");
            
            // Add LoRA
            let lora_config = LoRAConfig {
                rank: 16,
                alpha: 16.0,
                dropout: Some(0.0),
                target_modules: vec![],
                module_filters: vec![],
                init_scale: 0.01,
            };
            
            model.add_lora_to_all(&lora_config, &device, DType::F32)?;
            println!("✅ LoRA added successfully!");
            
            // Test forward pass with dummy inputs
            println!("\nTesting forward pass...");
            let batch_size = 1;
            let h = 32;  // Smaller image size
            let w = 32;
            let patch_size = 2;
            let num_patches = (h / patch_size) * (w / patch_size);
            
            // Create dummy inputs with correct shapes
            // img should be [batch, num_patches, in_channels] not [batch, in_channels, num_patches]
            let img = Tensor::randn(0.0f32, 1.0, &[batch_size, num_patches, config.in_channels], &device)?;
            // img_ids should have shape [batch, height, width, 3] for position encoding
            let img_ids = Tensor::zeros(&[batch_size, h/patch_size, w/patch_size, 3], DType::F32, &device)?;
            let txt = Tensor::randn(0.0f32, 1.0, &[batch_size, 77, config.context_in_dim], &device)?;  // Smaller text sequence
            let txt_ids = Tensor::zeros(&[batch_size, 77, 3], DType::F32, &device)?;
            let timesteps = Tensor::ones(&[batch_size], DType::F32, &device)?;
            let y = Tensor::randn(0.0f32, 1.0, &[batch_size, config.vec_in_dim], &device)?;
            let guidance = Some(Tensor::ones(&[batch_size], DType::F32, &device)?);
            
            println!("Input shapes:");
            println!("  img: {:?}", img.shape());
            println!("  txt: {:?}", txt.shape());
            println!("  timesteps: {:?}", timesteps.shape());
            
            match model.forward(&img, &img_ids, &txt, &txt_ids, &timesteps, &y, guidance.as_ref()) {
                Ok(output) => println!("✅ Forward pass successful! Output shape: {:?}", output.shape()),
                Err(e) => println!("❌ Forward pass failed: {}", e),
            }
        }
        Err(e) => println!("❌ Model creation failed: {}", e),
    }
    
    println!("\n=== Test Complete ===");
    Ok(())
}