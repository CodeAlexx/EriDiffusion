#!/usr/bin/env rust-script
//! Generate images with Flux using eridiffusion-rs

use eridiffusion_models::{FluxModel, VAE, VAEFactory, TextEncoder};
use eridiffusion_inference::{InferenceEngine, InferenceConfig, GenerationParams};
use candle_core::{Device, DType, Tensor};
use candle_nn::VarBuilder;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Generating Flux image...");
    
    let device = Device::cuda_if_available(0)?;
    let dtype = DType::BF16;  // Flux prefers BF16
    
    // Load Flux model weights
    let model_path = "/home/alex/models/flux/flux_dev.safetensors";
    let vae_path = "/home/alex/models/flux/flux_vae.safetensors";
    
    // Load Flux transformer
    println!("Loading Flux transformer...");
    let flux = FluxModel::from_pretrained(model_path, &device)?;
    
    // Load VAE (Flux uses its own VAE)
    println!("Loading Flux VAE...");
    let vae = VAEFactory::load_flux_vae(vae_path, &device)?;
    
    // Load text encoders (CLIP-L and T5)
    println!("Loading text encoders...");
    let clip_l = TextEncoder::load_clip_l("/home/alex/models/flux/clip_l.safetensors", &device)?;
    let t5 = TextEncoder::load_t5_xxl("/home/alex/models/flux/t5_xxl.safetensors", &device)?;
    
    // Create inference engine
    let engine = InferenceEngine::new(
        Box::new(flux),
        Box::new(vae),
        vec![Box::new(clip_l), Box::new(t5)],
        device,
    )?;
    
    // Generation parameters
    let params = GenerationParams {
        prompt: "a mystical forest with glowing mushrooms, ethereal light beams, fantasy art, magical atmosphere".to_string(),
        negative_prompt: None,  // Flux doesn't use negative prompts
        width: 1024,
        height: 1024,
        steps: 20,  // Flux is efficient
        cfg_scale: 1.0,  // Flux uses lower CFG
        seed: Some(2024),
        scheduler: "flow_matching".to_string(),
        ..Default::default()
    };
    
    // Generate image
    println!("Generating with prompt: {}", params.prompt);
    let image = engine.generate(&params)?;
    
    // Save image
    let output_path = "flux_output.png";
    image.save(output_path)?;
    println!("Image saved to: {}", output_path);
    
    Ok(())
}