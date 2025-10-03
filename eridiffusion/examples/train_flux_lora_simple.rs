//! Simple Flux LoRA training example
//!
//! This uses the existing pipeline_flux_lora_optimized with your requested settings:
//! - Batch size 2 with gradient accumulation 2 (effective batch size 4)
//! - SNR gamma 5.0
//! - Flow schedule shift 3.0
//! - Dropout 0.0
//! - No fallback on batch size - if 2 is too much, tough!

use eridiffusion::trainers::flux_data_loader::{DatasetConfig, FluxDataLoader};
use eridiffusion::trainers::pipeline_flux_lora::{
    FluxTrainer, FluxTrainingConfig, TextEncoderPaths, TrainMode,
};
use eridiffusion::trainers::pipeline_flux_lora_optimized::create_flux_trainer_optimized;
use flame_core::device::Device;
use log::info;
use std::path::PathBuf;

fn main() -> flame_core::Result<()> {
    // Initialize logging
    env_logger::Builder::from_default_env().filter_level(log::LevelFilter::Info).init();

    info!("🚀 Starting Flux LoRA Training!");
    info!("32 H100s with 80GB each = 2.56TB total VRAM - that's insane compute power!");
    info!("Your 3090 with 24GB is perfect for batch size 2 development\n");

    // Create device (GPU 0)
    let device = Device::cuda(0)?;
    info!("Using device: {:?}", device);

    // Training configuration with your requested settings
    let config = FluxTrainingConfig {
        // Model paths
        model_path: PathBuf::from("/home/alex/SwarmUI/Models/unet/flux1-schnell.safetensors"),
        vae_path: PathBuf::from("/home/alex/SwarmUI/Models/VAE/ae.safetensors"),
        text_encoder_paths: TextEncoderPaths {
            clip_l: PathBuf::from("/home/alex/SwarmUI/Models/clip/clip_l.safetensors"),
            t5_xxl: PathBuf::from("/home/alex/SwarmUI/Models/clip/t5xxl_fp16.safetensors"),
        },

        // Training configuration - YOUR EXACT SETTINGS
        train_mode: TrainMode::LoRA,
        batch_size: 2,                  // No fallback as requested!
        gradient_accumulation_steps: 2, // Effective batch size 4
        learning_rate: 1e-4,
        warmup_steps: 100,
        max_train_steps: 1000,
        checkpointing_steps: 100,

        // Optimization
        mixed_precision: true, // Use FP16 for efficiency
        gradient_checkpointing: true,
        use_8bit_adam: true, // Save memory with 8-bit Adam
        max_grad_norm: 1.0,

        // LoRA configuration
        lora_rank: 16,
        lora_alpha: 16.0,
        lora_dropout: 0.0, // Dropout 0.0 as requested
        lora_target_modules: vec![
            "img_attn".to_string(),
            "txt_attn".to_string(),
            "img_mlp".to_string(),
            "txt_mlp".to_string(),
        ],

        // Data configuration - 1024x1024 as requested
        resolution: 1024,
        center_crop: false,
        random_flip: true,
        caption_dropout_rate: 0.0,

        // Flux-specific settings
        guidance_scale: 3.5,
        bypass_guidance_embedding: true,
        shift_schedule: 3.0, // Flow schedule shift as requested

        // Logging
        logging_dir: PathBuf::from("/home/alex/diffusers-rs/output/flux_lora_simple"),
        report_to: vec![],
        validation_prompts: vec![
            "a photo of a woman".to_string(),
            "a portrait of a woman smiling".to_string(),
        ],
        validation_steps: 250,

        // Caching configuration
        cache_latents_to_disk: true,
        cache_dir: None,
        force_reencode: false,
        dataset_name: "40_woman".to_string(),
    };

    // Create dataset configuration
    let dataset_config = DatasetConfig {
        folder_path: PathBuf::from("/home/alex/diffusers-rs/datasets/40_woman"),
        caption_ext: "txt".to_string(),
        caption_dropout_rate: 0.0,
        shuffle_tokens: false,
        cache_latents_to_disk: true,
        resolutions: vec![(1024, 1024)], // 1024x1024 only
        center_crop: false,
        random_flip: true,
        force_recache: Some(false),
    };

    info!("\n=== Configuration ===");
    info!("Dataset: {:?}", dataset_config.folder_path);
    info!("Resolution: 1024x1024");
    info!("Batch size: {} (no fallback!)", config.batch_size);
    info!("Gradient accumulation: {}", config.gradient_accumulation_steps);
    info!("Effective batch size: {}", config.batch_size * config.gradient_accumulation_steps);
    info!("SNR gamma: 5.0 (hardcoded in trainer)");
    info!("Flow schedule shift: {}", config.shift_schedule);
    info!("LoRA dropout: {}", config.lora_dropout);
    info!("Learning rate: {}", config.learning_rate);
    info!("Max steps: {}", config.max_train_steps);

    // Create optimized trainer with CPU offloading
    info!("\nCreating optimized Flux trainer...");
    let mut trainer = create_flux_trainer_optimized(config, device, dataset_config.clone())?;

    // Create data loader for training
    let mut dataloader = FluxDataLoader::new(dataset_config, trainer.device.clone())?;

    // Start training!
    info!("\n🎯 Starting training loop...");
    info!("If batch size 2 causes OOM, tough! No fallback as requested.\n");

    trainer.train(&mut dataloader)?;

    info!("\n✨ Training complete!");
    Ok(())
}
