//! Example of training Flux LoRA with sequential loading for 24GB GPUs
//!
//! This demonstrates the proper loading order:
//! 1. Encode latents with VAE (then free VAE)
//! 2. Encode text with CLIP/T5 (then free text encoders)
//! 3. Load Flux model and train

use eridiffusion::trainers::{
    flux_data_loader::{DatasetConfig, FluxDataLoader},
    pipeline_flux_lora::{FluxTrainingConfig, TextEncoderPaths, TrainMode},
    pipeline_flux_lora_sequential::FluxTrainerSequential,
};
use std::path::PathBuf;

fn main() -> flame_core::Result<()> {
    // Initialize logging
    env_logger::init();

    println!("=== Flux LoRA Training with Sequential Loading ===");
    println!("Optimized for 24GB GPUs (3090, 4090, etc.)");

    // Create device
    let device = flame_core::device::Device::cuda(0)?;
    println!("✅ GPU detected: {:?}", device);

    // Configuration for batch size 1
    let config = FluxTrainingConfig {
        // Model paths
        model_path: PathBuf::from(
            "/home/alex/SwarmUI/Models/diffusion_models/flux1-dev.safetensors",
        ),
        vae_path: PathBuf::from("/home/alex/SwarmUI/Models/VAE/ae.safetensors"),
        text_encoder_paths: TextEncoderPaths {
            clip_l: PathBuf::from("/home/alex/SwarmUI/Models/clip/clip_l.safetensors"),
            t5_xxl: PathBuf::from("/home/alex/SwarmUI/Models/clip/t5xxl_fp16.safetensors"),
        },

        // Training settings optimized for 24GB
        train_mode: TrainMode::LoRA,
        batch_size: 1,                  // Batch size 1 as requested
        gradient_accumulation_steps: 4, // Effective batch size 4
        learning_rate: 1e-4,
        warmup_steps: 100,
        max_train_steps: 1000,
        checkpointing_steps: 100,

        // Memory optimizations
        mixed_precision: true,        // Use BF16
        gradient_checkpointing: true, // CPU offload for gradients
        use_8bit_adam: true,          // 8-bit optimizer
        max_grad_norm: 1.0,

        // LoRA configuration
        lora_rank: 16,
        lora_alpha: 16.0,
        lora_dropout: 0.0, // No dropout as requested
        lora_target_modules: vec![
            "img_attn".to_string(),
            "txt_attn".to_string(),
            "img_mlp".to_string(),
            "txt_mlp".to_string(),
        ],

        // Data configuration
        resolution: 1024,
        center_crop: false,
        random_flip: false,
        caption_dropout_rate: 0.0, // No dropout as requested

        // Flux-specific settings
        guidance_scale: 3.5,
        bypass_guidance_embedding: true,
        shift_schedule: 3.0, // Flow schedule shift

        // Output configuration
        logging_dir: PathBuf::from("./output/flux_lora_sequential"),
        report_to: vec![],
        validation_prompts: vec![
            "a photo of a woman".to_string(),
            "a portrait of a woman smiling".to_string(),
            "a woman in professional attire".to_string(),
        ],
        validation_steps: 250,

        // IMPORTANT: Enable caching for sequential loading
        cache_latents_to_disk: true,
        cache_dir: None, // Will use default under logging_dir
        force_reencode: false,
        dataset_name: "40_woman".to_string(),

        // New required fields
        use_layer_streaming: Some(true),
        streaming_memory_limit_gb: Some(20.0),
        chroma_config: None,
    };

    // Dataset configuration
    let dataset_config = DatasetConfig {
        folder_path: PathBuf::from("/home/alex/diffusers-rs/datasets/40_woman"),
        caption_ext: "txt".to_string(),
        caption_dropout_rate: 0.0,
        shuffle_tokens: false,
        cache_latents_to_disk: true, // Enable caching
        resolutions: vec![(1024, 1024)],
        center_crop: false,
        random_flip: false,
        force_recache: Some(false),
    };

    // Create data loader for encoding phase
    let mut data_loader = FluxDataLoader::new(dataset_config.clone(), device.clone())?;
    println!("\n✅ Data loader created with {} samples", data_loader.total_samples());

    // Create trainer with sequential loading
    println!("\n[Sequential Loading]");
    println!("This will:");
    println!("1. Load VAE → encode latents → free VAE");
    println!("2. Load text encoders → encode embeddings → free encoders");
    println!("3. Load Flux model with all available memory");

    let mut trainer = FluxTrainerSequential::new_with_sequential_loading(
        config,
        device.clone(),
        &mut data_loader,
    )?;

    // Create fresh data loader for training (after encoding)
    let mut training_loader = FluxDataLoader::new(dataset_config, device)?;

    // Run training
    println!("\n=== Starting Training ===");
    println!("Memory efficient training with:");
    println!("  - Batch size: 1");
    println!("  - Gradient accumulation: 4 (effective batch size 4)");
    println!("  - No quantization (as requested)");
    println!("  - Sequential loading (SimpleTuner style)");
    println!("  - 8-bit Adam optimizer");
    println!("  - Gradient checkpointing with CPU offload");

    trainer.train(&mut training_loader)?;

    println!("\n=== Training Complete! ===");
    Ok(())
}
