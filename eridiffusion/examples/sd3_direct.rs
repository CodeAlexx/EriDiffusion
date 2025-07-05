//! Direct SD3 generation using the working parts of our pipeline

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::mmdit::model::{Config as MMDiTConfig, MMDiT};
use std::time::Instant;

fn main() -> Result<()> {
    println!("🎨 AI-Toolkit-RS SD 3.5 Direct Generation\n");
    
    // Configuration
    let prompt = "a lady at the beach";
    let height = 768;
    let width = 768;
    let steps = 20;
    let cfg_scale = 4.0;
    let seed = 42;
    
    println!("Configuration:");
    println!("  Prompt: {}", prompt);
    println!("  Resolution: {}x{}", width, height);
    println!("  Steps: {}", steps);
    println!("  CFG: {}", cfg_scale);
    println!("  Seed: {}", seed);
    
    // Setup device
    let device = Device::cuda_if_available(0)?;
    println!("\nDevice: {:?}", device);
    
    // SD3.5 Large configuration
    let config = MMDiTConfig::sd3_5_large();
    println!("\nModel: SD 3.5 Large");
    println!("  Depth: {}", config.depth);
    println!("  Hidden size: {}", config.hidden_size);
    println!("  Heads: {}", config.num_heads);
    
    // Model path
    let model_path = "/home/alex/SwarmUI/Models/diffusion_models/sd3.5_large.safetensors";
    
    println!("\nLoading model from: {}", model_path);
    let start = Instant::now();
    
    // Load the model
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[model_path], DType::F32, &device)?
    };
    
    let mmdit = MMDiT::new(&config, &vb)?;
    println!("✓ Model loaded in {:.2}s", start.elapsed().as_secs_f32());
    
    // Create dummy inputs for testing
    let batch_size = 1;
    let latent_channels = 16;
    let latent_h = height / 8;
    let latent_w = width / 8;
    let seq_len = 77;
    let hidden_size = config.hidden_size;
    
    // Dummy latents
    let x = Tensor::randn(
        0f32, 
        1f32, 
        &[batch_size, latent_channels, latent_h, latent_w],
        &device
    )?;
    
    // Dummy embeddings (would come from CLIP + T5)
    let y = Tensor::randn(
        0f32,
        1f32, 
        &[batch_size, seq_len, hidden_size],
        &device
    )?;
    
    // Dummy timestep
    let t = Tensor::new(&[500.0f32], &device)?;
    
    println!("\nRunning forward pass...");
    let start = Instant::now();
    
    // Run the model
    let output = mmdit.forward(&x, &t, &y)?;
    
    let elapsed = start.elapsed();
    println!("✓ Forward pass complete in {:.2}s", elapsed.as_secs_f32());
    println!("  Output shape: {:?}", output.shape());
    
    // To complete the generation, we would need:
    // 1. Text encoders (CLIP-L, CLIP-G, T5-XXL)
    // 2. VAE decoder
    // 3. Sampling loop
    // 4. Image saving
    
    println!("\n✅ SD 3.5 model successfully loaded and tested!");
    println!("   (Full generation requires text encoders and VAE)");
    
    Ok(())
}