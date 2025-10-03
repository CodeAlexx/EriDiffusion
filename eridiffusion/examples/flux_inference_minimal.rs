//! Minimal Flux inference example

use anyhow::Result;
use eridiffusion::Device;
use flame_core::{DType, Tensor};
use std::path::Path;

fn main() -> Result<()> {
    println!("Flux Inference Example");

    // Model paths
    let flux_model = "/home/alex/SwarmUI/Models/diffusion_models/flux1-dev.safetensors";
    let vae_path = "/home/alex/SwarmUI/Models/vae/ae.safetensors";
    let clip_l = "/home/alex/SwarmUI/Models/clip/clip_l.safetensors";
    let t5_xxl = "/home/alex/SwarmUI/Models/clip/t5xxl_fp16.safetensors";

    // Check files exist
    for path in &[flux_model, vae_path, clip_l, t5_xxl] {
        if !Path::new(path).exists() {
            eprintln!("Model file not found: {}", path);
            return Ok(());
        }
    }

    println!("All model files found!");

    // Create device
    let device = eridiffusion::cuda_device(0)?;

    println!("CUDA device created");

    // Load models
    println!("Loading Flux model from: {}", flux_model);
    let flux_weights = eridiffusion::loaders::WeightLoader::from_safetensors(flux_model, device)?;
    println!("Loaded {} weights", flux_weights.weights.len());

    // Sample some weight names to understand structure
    println!("\nSample weight names:");
    for (i, key) in flux_weights.weights.keys().enumerate() {
        if i < 10 {
            println!("  {}", key);
        }
    }

    println!("\nFlux model structure loaded successfully!");

    Ok(())
}
