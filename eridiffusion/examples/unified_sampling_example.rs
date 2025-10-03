//! Example demonstrating unified sampling/inference for all models
//!
//! Shows how to generate images with SDXL, SD 3.5, and Flux models
//! using the unified sampling interface.

use anyhow::Result;
use eridiffusion::inference::flux_sampling::FluxSampler;
use eridiffusion::inference::sd35_sampling::SD35Sampler;
use eridiffusion::inference::sdxl_sampling::{SDXLSampler, TextEncoders};
use eridiffusion::inference::unified_sampling::{
    generate_validation_samples, get_optimal_settings, ModelComponents, ModelType, SamplingConfig,
    UnifiedSampler,
};
use flame_core::{DType, Device};
use std::path::PathBuf;

/// Example: Generate with SDXL
async fn sample_sdxl() -> Result<()> {
    println!("=== SDXL Sampling Example ===");

    let device = Device::cuda(0)?;
    let sampler = UnifiedSampler::new(device);

    // Configure for SDXL
    let mut config = get_optimal_settings(ModelType::SDXL);
    config.prompt = "A majestic mountain landscape at sunset, highly detailed, 8k".to_string();
    config.negative_prompt = "blurry, low quality, distorted".to_string();
    config.seed = Some(42);
    config.output_dir = PathBuf::from("outputs/sdxl");

    // Mock model components (in real usage, these would be loaded models)
    let components = ModelComponents::SDXL {
        unet: std::collections::HashMap::new(), // Loaded UNet weights
        vae: Box::new(|latents| {
            // Mock VAE decoder
            Ok(latents.clone())
        }),
        text_encoders: &mut TextEncoders {
            clip_l: Box::new(|_prompt| {
                // Mock CLIP-L encoder
                let embeds = flame_core::Tensor::randn(0.0, 1.0, &[1, 77, 768], device)?;
                let pooled = flame_core::Tensor::randn(0.0, 1.0, &[1, 768], device)?;
                Ok((embeds, pooled))
            }),
            clip_g: Box::new(|_prompt| {
                // Mock CLIP-G encoder
                let embeds = flame_core::Tensor::randn(0.0, 1.0, &[1, 77, 1280], device)?;
                let pooled = flame_core::Tensor::randn(0.0, 1.0, &[1, 1280], device)?;
                Ok((embeds, pooled))
            }),
        } as &mut dyn std::any::Any,
        lora: Box::new(()),
    };

    // Generate
    let paths = sampler.sample(&config, components)?;
    println!("Generated {} SDXL images: {:?}", paths.len(), paths);

    Ok(())
}

/// Example: Generate with SD 3.5
async fn sample_sd35() -> Result<()> {
    println!("\n=== SD 3.5 Sampling Example ===");

    let device = Device::cuda(0)?;
    let sampler = UnifiedSampler::new(device);

    // Configure for SD 3.5 Large
    let mut config = get_optimal_settings(ModelType::SD35Large);
    config.prompt = "A futuristic cityscape with flying cars, cyberpunk style".to_string();
    config.negative_prompt = "".to_string(); // SD 3.5 works well without negative prompts
    config.seed = Some(123);
    config.output_dir = PathBuf::from("outputs/sd35");

    // Mock model components
    let components = ModelComponents::SD35 {
        mmdit: Box::new(|_latents, _timestep, _text_embeds, _pooled| {
            // Mock MMDiT forward
            Ok(flame_core::Tensor::randn(0.0, 1.0, &[1, 16, 128, 128], device)?)
        }),
        vae: Box::new(|latents| {
            // Mock VAE decoder for 16-channel latents
            Ok(latents.clone())
        }),
        clip_l: Box::new(|_prompt| {
            // Mock CLIP-L encoder
            let embeds = flame_core::Tensor::randn(0.0, 1.0, &[1, 77, 768], device)?;
            let pooled = flame_core::Tensor::randn(0.0, 1.0, &[1, 768], device)?;
            Ok((embeds, pooled))
        }),
        clip_g: Box::new(|_prompt| {
            // Mock CLIP-G encoder
            let embeds = flame_core::Tensor::randn(0.0, 1.0, &[1, 77, 1280], device)?;
            let pooled = flame_core::Tensor::randn(0.0, 1.0, &[1, 1280], device)?;
            Ok((embeds, pooled))
        }),
        t5: Some(Box::new(|_prompt, _max_length| {
            // Mock T5-XXL encoder
            Ok(flame_core::Tensor::randn(0.0, 1.0, &[1, 256, 4096], device)?)
        })),
    };

    // Generate
    let paths = sampler.sample(&config, components)?;
    println!("Generated {} SD 3.5 images: {:?}", paths.len(), paths);

    Ok(())
}

/// Example: Generate with Flux
async fn sample_flux() -> Result<()> {
    println!("\n=== Flux Sampling Example ===");

    let device = Device::cuda(0)?;
    let sampler = UnifiedSampler::new(device);

    // Configure for Flux Dev
    let mut config = get_optimal_settings(ModelType::FluxDev);
    config.prompt = "A serene Japanese garden with cherry blossoms, photorealistic".to_string();
    config.seed = Some(456);
    config.output_dir = PathBuf::from("outputs/flux");

    // Mock model components
    let components = ModelComponents::Flux {
        model: Box::new(|_latents, _timestep, _text_embeds, _guidance| {
            // Mock Flux model forward
            Ok(flame_core::Tensor::randn(0.0, 1.0, &[1, 16384, 64], device)?)
        }),
        vae: Box::new(|latents| {
            // Mock VAE decoder
            Ok(latents.clone())
        }),
        clip: Box::new(|_prompt| {
            // Mock CLIP encoder
            Ok(flame_core::Tensor::randn(0.0, 1.0, &[1, 77, 768], device)?)
        }),
        t5: Box::new(|_prompt, _max_length| {
            // Mock T5 encoder
            Ok(flame_core::Tensor::randn(0.0, 1.0, &[1, 512, 4096], device)?)
        }),
    };

    // Generate
    let paths = sampler.sample(&config, components)?;
    println!("Generated {} Flux images: {:?}", paths.len(), paths);

    Ok(())
}

/// Example: Generate validation samples during training
async fn validation_example() -> Result<()> {
    println!("\n=== Validation Sampling Example ===");

    let validation_prompts =
        vec!["A cute cat wearing a hat", "A beautiful sunset over mountains", "A futuristic robot"];

    let device = Device::cuda(0)?;

    // Mock components for validation
    let components = ModelComponents::SDXL {
        unet: std::collections::HashMap::new(),
        vae: Box::new(|latents| Ok(latents.clone())),
        text_encoders: &mut TextEncoders {
            clip_l: Box::new(|_| {
                let embeds = flame_core::Tensor::randn(0.0, 1.0, &[1, 77, 768], device)?;
                let pooled = flame_core::Tensor::randn(0.0, 1.0, &[1, 768], device)?;
                Ok((embeds, pooled))
            }),
            clip_g: Box::new(|_| {
                let embeds = flame_core::Tensor::randn(0.0, 1.0, &[1, 77, 1280], device)?;
                let pooled = flame_core::Tensor::randn(0.0, 1.0, &[1, 1280], device)?;
                Ok((embeds, pooled))
            }),
        } as &mut dyn std::any::Any,
        lora: Box::new(()),
    };

    // Generate validation samples at training step 1000
    let step = 1000;
    let output_dir = PathBuf::from("outputs/validation");

    let paths = generate_validation_samples(
        ModelType::SDXL,
        components,
        &validation_prompts.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        step,
        &output_dir,
    )?;

    println!("Generated {} validation images at step {}", paths.len(), step);

    Ok(())
}

/// Demonstrate different scheduler configurations
fn scheduler_examples() {
    println!("\n=== Scheduler Configuration Examples ===");

    // SDXL with different schedulers
    let mut sdxl_config = get_optimal_settings(ModelType::SDXL);
    println!(
        "SDXL default: {} steps, guidance {}",
        sdxl_config.num_inference_steps, sdxl_config.guidance_scale
    );

    // SD 3.5 with turbo variant
    let mut sd35_turbo = get_optimal_settings(ModelType::SD35Medium);
    println!(
        "SD 3.5 Medium (turbo): {} steps, guidance {}",
        sd35_turbo.num_inference_steps, sd35_turbo.guidance_scale
    );

    // Flux Schnell (distilled)
    let flux_schnell = get_optimal_settings(ModelType::FluxSchnell);
    println!(
        "Flux Schnell: {} steps, no CFG (guidance={})",
        flux_schnell.num_inference_steps, flux_schnell.guidance_scale
    );
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    env_logger::init();

    println!("Unified Sampling Example for EriDiffusion");
    println!("========================================\n");

    // Run examples
    sample_sdxl().await?;
    sample_sd35().await?;
    sample_flux().await?;
    validation_example().await?;
    scheduler_examples();

    println!("\n✅ All examples completed successfully!");

    Ok(())
}
