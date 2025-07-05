//! Example of Flux training with integrated sampling

use eridiffusion_core::{Device, Result};
use eridiffusion_models::{ModelManager, ModelArchitecture};
use eridiffusion_networks::{LoRAConfig, NetworkAdapter};
use eridiffusion_training::{
    Trainer, TrainerConfig,
    flux_trainer::{FluxTrainer, FluxTrainingConfig},
    pipelines::sampling::{TrainingSampler, SamplingConfig},
};
use eridiffusion_data::{DatasetConfig, DataLoader};
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
    
    // Training configuration
    let flux_config = FluxTrainingConfig {
        gradient_checkpointing: true,
        train_text_encoder: false,
        text_encoder_lr_multiplier: 0.1,
        guidance_scale: 3.5,
        text_drop_prob: 0.1,
        num_inference_steps: 28,
        mixed_precision: true,
    };
    
    // LoRA configuration
    let lora_config = LoRAConfig {
        rank: 32,
        alpha: 32.0,
        dropout: 0.0,
        target_modules: vec![
            "attn.to_q".to_string(),
            "attn.to_k".to_string(),
            "attn.to_v".to_string(),
            "attn.to_out".to_string(),
        ],
    };
    
    // Dataset configuration
    let dataset_config = DatasetConfig {
        dataset_path: PathBuf::from("/path/to/dataset"),
        batch_size: 1,
        num_workers: 4,
        resolution: 1024,
        center_crop: true,
        random_flip: true,
        caption_dropout: 0.1,
        shuffle: true,
    };
    
    // Trainer configuration
    let trainer_config = TrainerConfig {
        output_dir: PathBuf::from("flux_training_output"),
        num_epochs: 10,
        learning_rate: 1e-5,
        gradient_accumulation_steps: 4,
        save_steps: 500,
        sample_steps: 250,
        logging_steps: 10,
        checkpointing_steps: 1000,
        validation_steps: Some(500),
        warmup_steps: 100,
        max_grad_norm: 1.0,
        mixed_precision: true,
        ..Default::default()
    };
    
    // Load models
    println!("Loading models...");
    let model_manager = ModelManager::new();
    let mut model = model_manager.load_model(&model_path, ModelArchitecture::Flux, &device).await?;
    let vae = model_manager.load_vae(&vae_path, &device).await?;
    let (clip_encoder, t5_encoder) = FluxTrainer::load_text_encoders(&clip_path, &t5_path, &device).await?;
    
    // Apply LoRA
    println!("Applying LoRA adapter...");
    let lora_adapter = NetworkAdapter::new_lora(lora_config, &model)?;
    model.apply_adapter(&lora_adapter)?;
    
    // Create data loader
    let data_loader = DataLoader::new(dataset_config)?;
    
    // Create trainer
    let mut trainer = Trainer::new(trainer_config, device.clone())?;
    
    // Training loop with integrated sampling
    println!("Starting training...");
    for epoch in 0..trainer_config.num_epochs {
        println!("Epoch {}/{}", epoch + 1, trainer_config.num_epochs);
        
        for (step, batch) in data_loader.iter().enumerate() {
            // Training step
            let loss = trainer.training_step(
                &mut model,
                &batch,
                &flux_config,
            )?;
            
            // Log metrics
            if step % trainer_config.logging_steps == 0 {
                println!("Step {}: loss = {:.4}", step, loss);
            }
            
            // Generate samples
            if step % trainer_config.sample_steps == 0 && step > 0 {
                println!("Generating samples at step {}...", step);
                
                let sample_paths = FluxTrainer::generate_samples(
                    &model,
                    &vae,
                    &t5_encoder, // In real implementation, would use combined encoder
                    &flux_config,
                    &device,
                    step,
                    &trainer_config.output_dir.join("samples"),
                ).await?;
                
                println!("Generated {} samples", sample_paths.len());
            }
            
            // Save checkpoint
            if step % trainer_config.checkpointing_steps == 0 && step > 0 {
                println!("Saving checkpoint at step {}...", step);
                let checkpoint_path = trainer_config.output_dir
                    .join(format!("checkpoint-{}", step));
                trainer.save_checkpoint(&model, &checkpoint_path)?;
            }
        }
    }
    
    // Final sampling
    println!("Generating final samples...");
    let final_config = SamplingConfig {
        num_inference_steps: 50, // More steps for final quality
        guidance_scale: 3.5,
        eta: 0.0,
        generator_seed: Some(42),
        output_dir: trainer_config.output_dir.join("final_samples"),
        sample_prompts: vec![
            "a masterpiece digital painting of a fantasy landscape".to_string(),
            "a highly detailed portrait of a cyberpunk character".to_string(),
            "an epic space battle scene with spaceships and explosions".to_string(),
            "a serene zen garden with perfect composition".to_string(),
            "a steampunk city with intricate mechanical details".to_string(),
        ],
        negative_prompt: None,
        height: 1024,
        width: 1024,
    };
    
    let sampler = TrainingSampler::new(final_config, device);
    let final_paths = sampler.sample_flux(&model, &vae, &t5_encoder, 9999).await?;
    
    println!("Training complete! Generated {} final samples", final_paths.len());
    
    Ok(())
}