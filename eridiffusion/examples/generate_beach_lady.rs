//! Generate "lady at the beach" using Flux - Pure Rust implementation

use eridiffusion_core::{Device, Result};
use eridiffusion_models::{ModelManager, ModelArchitecture};
use eridiffusion_training::{
    flux_trainer::FluxTrainer,
    pipelines::sampling::{TrainingSampler, SamplingConfig},
};
use std::path::PathBuf;

// Since the full trainer has compilation issues, let's use the working parts directly
// This uses the sampling implementation we added to the trainer

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize device
    let device = Device::cuda_if_available(0)?;
    println!("Using device: {:?}", device);
    
    // Model paths - ALL LOCAL, NO DOWNLOADS
    let model_path = PathBuf::from("/home/alex/SwarmUI/Models/diffusion_models/flux1-dev.safetensors");
    let vae_path = PathBuf::from("/home/alex/SwarmUI/Models/vae/flux_vae.safetensors");
    let t5_path = PathBuf::from("/home/alex/SwarmUI/Models/text_encoders/t5-v1_1-xxl");
    let clip_path = PathBuf::from("/home/alex/SwarmUI/Models/text_encoders/clip-vit-large-patch14");
    
    // Load models
    println!("Loading Flux model...");
    let model_manager = ModelManager::new();
    let model = model_manager.load_model(&model_path, ModelArchitecture::Flux, &device).await?;
    
    println!("Loading VAE...");
    let vae = model_manager.load_vae(&vae_path, &device).await?;
    
    println!("Loading text encoders...");
    let (_clip_encoder, t5_encoder) = FluxTrainer::load_text_encoders(&clip_path, &t5_path, &device).await?;
    
    // Create sampling configuration specifically for "lady at the beach"
    let sampling_config = SamplingConfig {
        num_inference_steps: 28,
        guidance_scale: 3.5,
        eta: 0.0,
        generator_seed: Some(42),
        output_dir: PathBuf::from("beach_lady_output"),
        sample_prompts: vec!["lady at the beach".to_string()],
        negative_prompt: None, // Flux doesn't use negative prompts
        height: 1024,
        width: 1024,
    };
    
    // Create output directory
    std::fs::create_dir_all(&sampling_config.output_dir)?;
    
    // Create sampler
    let sampler = TrainingSampler::new(sampling_config, device.clone());
    
    // Generate the image
    println!("\nGenerating 'lady at the beach'...");
    let saved_paths = sampler.sample_flux(
        model.as_ref(),
        vae.as_ref(),
        t5_encoder.as_ref(),
        0, // step number
    ).await?;
    
    println!("\nGeneration complete!");
    println!("Image saved to: {}", saved_paths[0].display());
    
    Ok(())
}