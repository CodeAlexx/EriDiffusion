#![cfg(feature = "legacy-bins")]

// NOTE: Legacy binary; gated behind the `legacy-bins` feature so it stays out of default builds. Enable with `--features legacy-bins` if you still rely on it.

use eridiffusion::config::trainer_config::TrainingConfig;
use eridiffusion::trainers::flux_data_loader::{
    DatasetConfig as FluxDatasetConfig, FluxDataLoader,
};
use eridiffusion::trainers::pipeline_flux_lora::{FluxTrainingConfig, TextEncoderPaths, TrainMode};
use eridiffusion::trainers::pipeline_flux_lora_sequential::FluxTrainerSequential;
use flame_core::{Device, Error};
use std::env;
use std::path::PathBuf;

fn main() -> flame_core::Result<()> {
    // Initialize logging
    if std::env::var("RUST_LOG").is_err() {
        std::env::set_var("RUST_LOG", "info");
    }
    env_logger::init();

    // Get config path from args
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <config.yaml>", args[0]);
        eprintln!("Example: {} /home/alex/diffusers-rs/config/train_lora_flux_24gb.yaml", args[0]);
        std::process::exit(1);
    }

    let config_path = PathBuf::from(&args[1]);

    // Check if file exists
    if !config_path.exists() {
        eprintln!("Error: Config file not found: {}", config_path.display());
        std::process::exit(1);
    }

    println!("Starting FLUX trainer with config: {}", config_path.display());

    // Load YAML configuration
    let config_str = std::fs::read_to_string(&config_path)?;
    let training_config: TrainingConfig = serde_yaml::from_str(&config_str)
        .map_err(|e| Error::InvalidOperation(format!("Failed to parse YAML config: {}", e)))?;
    let config = training_config.config;

    // Find the trainer process config
    let process_config = config
        .process
        .iter()
        .find(|p| p.process_type == "sd_trainer" || p.process_type == "flux_trainer")
        .ok_or_else(|| {
            flame_core::Error::InvalidOperation(
                "No trainer process found in config".to_string(),
            )
        })?;

    // Check if this is a FLUX model
    if !process_config.model.is_flux.unwrap_or(false) {
        eprintln!(
            "Error: This trainer is specifically for FLUX models. Set 'is_flux: true' in config."
        );
        std::process::exit(1);
    }

    println!("\n=== FLUX Sequential Trainer (SimpleTuner-style) ===");
    println!("This trainer properly implements memory-efficient loading:");
    println!("  1. Load VAE → Encode ALL images → FREE VAE completely");
    println!("  2. Load CLIP/T5 → Encode ALL text → FREE encoders completely");
    println!("  3. Only THEN load 23GB Flux model with all 24GB available!");
    println!("\nThis is EXACTLY how SimpleTuner fits 23GB models in 24GB VRAM!");

    // Create device
    let device = Device::cuda(0)?;

    // Convert YAML config to FluxTrainingConfig
    let flux_config = FluxTrainingConfig {
        // Model paths
        model_path: PathBuf::from(&process_config.model.name_or_path),
        vae_path: process_config
            .model
            .vae_path
            .as_ref()
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("/home/alex/SwarmUI/Models/VAE/ae.safetensors")),
        text_encoder_paths: TextEncoderPaths {
            clip_l: process_config
                .model
                .clip_l_path
                .as_ref()
                .or(process_config.model.text_encoder_path.as_ref())
                .cloned()
                .unwrap_or_else(|| {
                    PathBuf::from("/home/alex/SwarmUI/Models/clip/clip_l.safetensors")
                }),
            t5_xxl: process_config
                .model
                .t5_path
                .as_ref()
                .or(process_config.model.text_encoder_2_path.as_ref())
                .cloned(),
        },

        // Training configuration
        train_mode: TrainMode::LoRA,
        batch_size: process_config.train.batch_size,
        gradient_accumulation_steps: process_config.train.gradient_accumulation_steps,
        learning_rate: process_config.train.lr as f64,
        warmup_steps: 0,
        max_train_steps: process_config.train.steps,
        checkpointing_steps: process_config.save.save_every,

        // Optimization
        mixed_precision: process_config.train.dtype == "bf16"
            || process_config.train.dtype == "fp16",
        gradient_checkpointing: process_config.train.gradient_checkpointing,
        use_8bit_adam: process_config.train.optimizer == "adamw8bit",
        max_grad_norm: 1.0,

        // LoRA configuration
        lora_rank: process_config.network.linear,
        lora_alpha: process_config.network.linear_alpha,
        lora_dropout: 0.0,
        lora_target_modules: vec![
            "img_attn".to_string(),
            "txt_attn".to_string(),
            "img_mlp".to_string(),
            "txt_mlp".to_string(),
        ],

        // Data configuration
        resolution: process_config.datasets[0].resolution[0],
        center_crop: false,
        random_flip: false,
        caption_dropout_rate: process_config.datasets[0].caption_dropout_rate,

        // Flux-specific
        guidance_scale: process_config.sample.guidance_scale,
        bypass_guidance_embedding: process_config.train.bypass_guidance_embedding,
        shift_schedule: 3.0,

        // Logging
        logging_dir: PathBuf::from("output").join(&config.name),
        report_to: vec![],
        validation_prompts: process_config.sample.prompts.clone(),
        validation_steps: process_config.sample.sample_every,

        // Caching configuration
        cache_latents_to_disk: process_config.datasets[0].cache_latents_to_disk,
        cache_dir: None,
        force_reencode: false,
        dataset_name: config.name.clone(),

        // Use layer streaming for memory efficiency
        use_layer_streaming: Some(true), // Enable streaming for 24GB GPUs
        streaming_memory_limit_gb: Some(18.0), // Leave 6GB for gradients/optimizer

        // INT8 quantization - not needed with proper streaming
        use_int8_base_model: Some(false), // We use streaming instead

        // ChromaXL config - not needed for standard FLUX
        chroma_config: None,
    };

    // Create dataset config
    let dataset_config = FluxDatasetConfig {
        folder_path: PathBuf::from(&process_config.datasets[0].folder_path),
        caption_ext: process_config.datasets[0].caption_ext.clone(),
        caption_dropout_rate: process_config.datasets[0].caption_dropout_rate,
        shuffle_tokens: process_config.datasets[0].shuffle_tokens,
        cache_latents_to_disk: process_config.datasets[0].cache_latents_to_disk,
        resolutions: process_config.datasets[0].resolution.iter().map(|&r| (r, r)).collect(),
        center_crop: false,
        random_flip: false,
        // force_recache is managed at higher level; not part of DatasetConfig here
    };

    // Create data loader
    println!("\nCreating data loader...");
    let mut dataloader = FluxDataLoader::new(dataset_config, device.clone())?;

    // Create the sequential trainer with proper memory management
    println!("\n=== Initializing Sequential Trainer ===");
    println!("This will:");
    println!("  1. Pre-encode all images with VAE then FREE it");
    println!("  2. Pre-encode all text with CLIP/T5 then FREE them");
    println!("  3. Load FLUX with maximum available memory");

    let mut trainer = FluxTrainerSequential::new_with_sequential_loading(
        flux_config,
        device.clone(),
        &mut dataloader,
    )?;

    // Now run training with the Flux model loaded
    println!("\n✅ Starting training with REAL 23GB Flux Dev model!");
    println!("All pre-encoding is done, maximum memory available for training!");
    trainer.train(&mut dataloader)?;

    println!("\nTraining completed successfully!");
    Ok(())
}
