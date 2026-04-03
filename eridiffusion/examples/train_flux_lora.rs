//! Train Flux LoRA with optimized 1024x1024 pipeline
//!
//! This example demonstrates training a Flux LoRA adapter using:
//! - FastVAE for quick encoding (<1 second per image)
//! - Memory-efficient T5-XXL loading
//! - Batch size 1 for 24GB VRAM (no automatic fallback)
//! - SNR weighting (gamma=5)
//! - Pre-cached latents and embeddings

use eridiffusion::trainers::flux_data_loader::{DatasetConfig, FluxDataLoader};
use eridiffusion::trainers::{FluxTrainerSequential, FluxTrainingConfig};
use flame_core::device::Device;
use log::info;
use std::path::PathBuf;

fn main() -> flame_core::Result<()> {
    // Initialize logging
    env_logger::Builder::from_default_env().filter_level(log::LevelFilter::Info).init();

    info!("🚀 Starting Flux LoRA 1024x1024 Training!");

    // Create device (GPU 0)
    let device = Device::cuda(0)?;
    info!("Using device: {:?}", device);

    // Create training configuration
    use eridiffusion::trainers::pipeline_flux_lora::{TextEncoderPaths, TrainMode};

    let config = FluxTrainingConfig {
        // Model paths
        model_path: PathBuf::from("/home/alex/SwarmUI/Models/unet/flux1-schnell.safetensors"),
        vae_path: PathBuf::from("/home/alex/SwarmUI/Models/VAE/ae.safetensors"),
        text_encoder_paths: TextEncoderPaths {
            clip_l: PathBuf::from("/home/alex/SwarmUI/Models/clip/clip_l.safetensors"),
            t5_xxl: PathBuf::from("/home/alex/SwarmUI/Models/clip/t5xxl_fp16.safetensors"),
        },

        // Training configuration
        train_mode: TrainMode::LoRA,
        batch_size: 1, // No automatic fallback
        gradient_accumulation_steps: 2,
        learning_rate: 1e-4,
        warmup_steps: 0,
        max_train_steps: 500,
        checkpointing_steps: 250,

        // Optimization
        mixed_precision: true,
        gradient_checkpointing: true,
        use_8bit_adam: true,
        max_grad_norm: 1.0,

        // LoRA configuration
        lora_rank: 16,
        lora_alpha: 16.0,
        lora_dropout: 0.0,
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
        caption_dropout_rate: 0.0,

        // Flux-specific
        guidance_scale: 3.5,
        bypass_guidance_embedding: true,
        shift_schedule: 3.0,

        // Logging
        logging_dir: PathBuf::from("/home/alex/diffusers-rs/output/flux_lora_1024"),
        report_to: vec![],
        validation_prompts: vec![], // No validation for this run
        validation_steps: 0,        // Disabled

        // Caching configuration
        cache_latents_to_disk: true,
        cache_dir: None,
        force_reencode: false,
        dataset_name: "flux_lora_training".to_string(),

        // Layer streaming configuration
        use_layer_streaming: Some(true),
        streaming_memory_limit_gb: Some(20.0),

        // ChromaXL configuration
        chroma_config: None,
    };

    // Create dataset configuration
    let dataset_config = DatasetConfig {
        folder_path: PathBuf::from("/home/alex/diffusers-rs/datasets/1stone"),
        caption_ext: "txt".to_string(),
        caption_dropout_rate: 0.0,
        shuffle_tokens: false,
        cache_latents_to_disk: config.cache_latents_to_disk,
        resolutions: vec![(1024, 1024)], // Will be aligned to 1088x1088 internally
        center_crop: false,
        random_flip: false,
        force_recache: Some(false),
    };

    // Create data loader
    let mut data_loader = FluxDataLoader::new(dataset_config, device.clone())?;

    // Create the training pipeline
    info!("Creating training pipeline with sequential loading...");
    let mut pipeline =
        FluxTrainerSequential::new_with_sequential_loading(config, device, &mut data_loader)?;

    // Pre-encode all data (VAE latents and text embeddings)
    info!("Pre-encoding dataset (VAE latents and text embeddings)...");
    pipeline.pre_encode_data(&mut data_loader)?;

    // Check for existing checkpoints and resume if available
    info!("Checking for existing checkpoints...");
    if pipeline.resume_from_latest()? {
        info!("Resumed from checkpoint at step {}", pipeline.global_step);
    } else {
        info!("No checkpoints found, starting fresh training");
    }

    // Run training
    info!("Starting training loop...");
    pipeline.train(&mut data_loader)?;

    info!("✨ Training complete!");
    Ok(())
}
