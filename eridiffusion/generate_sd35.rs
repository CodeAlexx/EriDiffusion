#!/usr/bin/env rust-script
//! Generate images with SD3.5 using eridiffusion-rs

use eridiffusion_models::{MMDiT, MMDiTConfig, VAE, VAEFactory, TextEncoder};
use eridiffusion_inference::{InferenceEngine, InferenceConfig, GenerationParams};
use candle_core::{Device, DType, Tensor};
use candle_nn::VarBuilder;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Generating SD3.5 image...");
    
    let device = Device::cuda_if_available(0)?;
    let dtype = DType::F16;
    
    // Load SD3.5 model weights
    let model_path = "/home/alex/models/sd35/sd3.5_large.safetensors";
    let vae_path = "/home/alex/models/sd35/sd3_vae.safetensors";
    
    // Load MMDiT
    println!("Loading SD3.5 MMDiT...");
    let mmdit = MMDiT::from_pretrained(model_path, &device)?;
    
    // Load VAE
    println!("Loading SD3 VAE...");
    let vae = VAEFactory::load_sd3_vae(vae_path, &device)?;
    
    // Load text encoders (CLIP-L, CLIP-G, T5)
    println!("Loading text encoders...");
    let clip_l = TextEncoder::load_clip_l("/home/alex/models/sd35/clip_l.safetensors", &device)?;
    let clip_g = TextEncoder::load_clip_g("/home/alex/models/sd35/clip_g.safetensors", &device)?;
    let t5 = TextEncoder::load_t5_xxl("/home/alex/models/sd35/t5_xxl.safetensors", &device)?;
    
    // Create inference engine
    let engine = InferenceEngine::new(
        Box::new(mmdit),
        Box::new(vae),
        vec![Box::new(clip_l), Box::new(clip_g), Box::new(t5)],
        device,
    )?;
    
    // Generation parameters
    let params = GenerationParams {
        prompt: "a futuristic cyberpunk city at night, neon lights, rain, reflections, ultra detailed".to_string(),
        negative_prompt: Some("blurry, low quality, distorted, ugly".to_string()),
        width: 1024,
        height: 1024,
        steps: 40,
        cfg_scale: 7.0,
        seed: Some(1337),
        scheduler: "flow_matching".to_string(),
        ..Default::default()
    };
    
    // Generate image
    println!("Generating with prompt: {}", params.prompt);
    let image = engine.generate(&params)?;
    
    // Save image
    let output_path = "sd35_output.png";
    image.save(output_path)?;
    println!("Image saved to: {}", output_path);
    
    Ok(())
}