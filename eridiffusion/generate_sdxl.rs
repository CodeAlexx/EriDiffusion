#!/usr/bin/env rust-script
//! Generate images with SDXL using eridiffusion-rs

use eridiffusion_models::{UNet, UNetConfig, VAE, VAEFactory, TextEncoder};
use eridiffusion_inference::{InferenceEngine, InferenceConfig, GenerationParams};
use candle_core::{Device, DType, Tensor};
use candle_nn::VarBuilder;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Generating SDXL image...");
    
    let device = Device::cuda_if_available(0)?;
    let dtype = DType::F16;
    
    // Load SDXL model weights
    let model_path = "/home/alex/models/sdxl/sdxl_base_1.0.safetensors";
    let vae_path = "/home/alex/models/sdxl/sdxl_vae.safetensors";
    
    // Load UNet
    println!("Loading SDXL UNet...");
    let unet = UNet::from_pretrained(model_path, &device, true)?;
    
    // Load VAE
    println!("Loading SDXL VAE...");
    let vae = VAEFactory::load_sdxl_vae(vae_path, &device)?;
    
    // Load text encoders
    println!("Loading CLIP text encoders...");
    let clip_l = TextEncoder::load_clip_l("/home/alex/models/sdxl/clip_l.safetensors", &device)?;
    let clip_g = TextEncoder::load_clip_g("/home/alex/models/sdxl/clip_g.safetensors", &device)?;
    
    // Create inference engine
    let engine = InferenceEngine::new(
        Box::new(unet),
        Box::new(vae),
        vec![Box::new(clip_l), Box::new(clip_g)],
        device,
    )?;
    
    // Generation parameters
    let params = GenerationParams {
        prompt: "a majestic lion with a flowing mane, highly detailed, 8k, photorealistic".to_string(),
        negative_prompt: Some("blurry, low quality, distorted".to_string()),
        width: 1024,
        height: 1024,
        steps: 30,
        cfg_scale: 7.5,
        seed: Some(42),
        scheduler: "euler_a".to_string(),
        ..Default::default()
    };
    
    // Generate image
    println!("Generating with prompt: {}", params.prompt);
    let image = engine.generate(&params)?;
    
    // Save image
    let output_path = "sdxl_output.png";
    image.save(output_path)?;
    println!("Image saved to: {}", output_path);
    
    Ok(())
}