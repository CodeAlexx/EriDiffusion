//! Example of Flux inference/sampling

use eridiffusion_core::{Device, Result};
use eridiffusion_models::{ModelManager, ModelArchitecture, DiffusionModel, VAE};
use eridiffusion_training::{
    flux_trainer::{FluxTrainer, FluxTrainingConfig},
    pipelines::sampling::{TrainingSampler, SamplingConfig},
};
use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize device
    let device = Device::cuda_if_available(0)?;
    println!("Using device: {:?}", device);
    
    // Model paths
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
    let (clip_encoder, t5_encoder) = FluxTrainer::load_text_encoders(&clip_path, &t5_path, &device).await?;
    
    // For Flux, we need to combine both encoders
    // In a real implementation, this would be handled by a FluxTextEncoder wrapper
    
    // Create sampling configuration
    let sampling_config = SamplingConfig {
        num_inference_steps: 28, // Flux default
        guidance_scale: 3.5, // Flux Dev default
        eta: 0.0,
        generator_seed: Some(42),
        output_dir: PathBuf::from("flux_samples"),
        sample_prompts: vec![
            "a beautiful landscape painting of mountains at sunset, highly detailed".to_string(),
            "a cyberpunk cat wearing sunglasses, neon lights, digital art".to_string(),
            "a medieval castle on a floating island in the clouds, fantasy art".to_string(),
        ],
        negative_prompt: None, // Flux doesn't use negative prompts
        height: 1024,
        width: 1024,
    };
    
    // Create output directory
    std::fs::create_dir_all(&sampling_config.output_dir)?;
    
    // Create sampler
    let sampler = TrainingSampler::new(sampling_config, device.clone());
    
    // Generate samples
    println!("Generating samples...");
    let saved_paths = sampler.sample_flux(
        model.as_ref(),
        vae.as_ref(),
        t5_encoder.as_ref(), // In real implementation, would use combined encoder
        0, // step number
    ).await?;
    
    println!("Saved {} samples:", saved_paths.len());
    for path in saved_paths {
        println!("  - {}", path.display());
    }
    
    // Alternative: Generate samples with custom prompts
    println!("\nGenerating custom samples...");
    let config = FluxTrainingConfig::default();
    let custom_paths = FluxTrainer::generate_samples(
        model.as_ref(),
        vae.as_ref(),
        t5_encoder.as_ref(),
        &config,
        &device,
        1, // step number
        &PathBuf::from("flux_samples"),
    ).await?;
    
    println!("Saved {} custom samples:", custom_paths.len());
    for path in custom_paths {
        println!("  - {}", path.display());
    }
    
    Ok(())
}