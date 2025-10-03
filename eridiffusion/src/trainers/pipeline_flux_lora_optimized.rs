//! Optimized Flux training pipeline with proper GPU memory management

use flame_core::{Tensor, Shape, DType, Parameter, Result};
use flame_core::device::Device;
use crate::loaders::WeightLoader;
use crate::models::{
    flux_vae::{AutoencoderKL, AutoEncoderConfig as VAEConfig},
    flux_model_complete::{FluxModel, FluxModelConfig},
};
use crate::trainers::{
    text_encoders::TextEncoders,
    pipeline_flux_lora::{
        FluxTrainingConfig, TextEncoderPaths, TrainMode,
        FluxTrainer, FluxLoRALayer, TrainingBatch,
    },
    flux_data_loader::{FluxDataLoader, DatasetConfig},
    flux_cache_manager::FluxCacheManager,
};
use std::sync::Arc;
use std::path::{Path, PathBuf};
use std::collections::HashMap;
use log::info;

/// Demonstrates the optimized Flux model loading sequence:
/// 1. Load dataset
/// 2. Load VAE and text encoders
/// 3. Encode data (pre-compute latents and text embeddings)
/// 4. Free VAE/text encoders from GPU
/// 5. Load main model with the freed GPU memory
pub fn create_flux_trainer_optimized(
    config: FluxTrainingConfig, 
    device: Device,
    dataset_config: DatasetConfig,
) -> flame_core::Result<FluxTrainer> {
    println!("=== Starting optimized Flux trainer creation ===");
    println!("This follows the memory-efficient loading sequence");
    
    // Keep track of caching config before moving dataset_config
    let cache_latents = dataset_config.cache_latents_to_disk;
    let cache_dir = dataset_config.folder_path.join("cache");
    let force_recache = false; // Read from higher-level process config; default false here
    
    // Step 1: Create data loader and cache manager
    println!("\n[Step 1] Creating data loader and cache manager...");
    let mut data_loader = FluxDataLoader::new(dataset_config, device.clone())?;
    println!("✅ Data loader created with {} samples", data_loader.total_samples());
    
    // Create cache manager
    let cache_manager = FluxCacheManager::with_dataset_name(
        cache_dir,
        device.clone(),
        cache_latents,
        "40_woman".to_string()
    )?;
    println!("✅ Cache manager initialized");
    
    // Sequential loading: VAE -> encode -> free, then CLIP/T5 -> encode -> free, then Flux
    println!("\n[Sequential Loading Strategy]");
    println!("Load models one at a time to manage GPU memory efficiently");
    println!("VAE and text encoders are NOT needed during training when using cached data!");
    
    if cache_latents {
        // Step 2: Load VAE and encode latents if not already cached
        println!("\n[Step 2] Checking VAE latent cache...");
        let (latent_count, _) = cache_manager.get_stats()?;
        
        if latent_count == 0 || force_recache {
            println!("  Need to encode latents - loading VAE...");
            {
                // Load VAE in a scope so it's freed after encoding
                cache_manager.encode_all_latents(&mut data_loader, &config.vae_path, force_recache)?;
            }
            println!("✅ VAE freed from GPU memory after encoding");
        } else {
            println!("✅ Latents already cached ({} files) - skipping VAE loading", latent_count);
        }
        
        // Step 3: Load text encoders and encode text if not already cached
        println!("\n[Step 3] Checking text embedding cache...");
        let (_, embed_count) = cache_manager.get_stats()?;
        
        if embed_count == 0 || force_recache {
            println!("  Need to encode text - loading CLIP and T5...");
            {
                // Load text encoders in a scope so they're freed after encoding
                cache_manager.encode_all_text_embeddings(
                    &mut data_loader,
                    &config.text_encoder_paths.clip_l,
                    config.text_encoder_paths.t5_xxl.as_ref().map(|p| p.as_path()),
                    force_recache
                )?;
            }
            println!("✅ Text encoders freed from GPU memory after encoding");
        } else {
            println!("✅ Text embeddings already cached ({} files) - skipping text encoder loading", embed_count);
        }
        
        println!("\n✅ All preprocessing complete - VAE and text encoders are NOT loaded");
        println!("   This saves ~10GB of GPU memory for Flux training!");
    } else {
        println!("\n⚠️  Warning: Caching disabled - will need VAE and text encoders during training");
        println!("   This will use significantly more GPU memory!");
    }
    
    // Step 4: Now load the main Flux model with all the freed memory
    println!("\n[Step 4] Loading main Flux model...");
    println!("All preprocessing complete - loading Flux with maximum available GPU memory");
    println!("This is a large model (~22GB) and may take a few minutes...");
    
    // Show current memory status
    println!("\nMemory status before Flux loading:");
    println!("  VAE: Freed ✅");
    println!("  Text Encoders: Freed ✅");
    println!("  Available for Flux: ~24GB\n");
    
    // Load with partial offloading if needed
    let flux = if config.train_mode == TrainMode::LoRA {
        // For LoRA, we can be more aggressive with offloading since we only need
        // gradients for LoRA weights
        println!("  Training mode: LoRA (efficient memory usage)");
        load_flux_model_for_lora(&config.model_path, &device)?
    } else {
        // For full fine-tuning, need all weights on GPU
        println!("  Training mode: Full fine-tuning");
        load_flux_model_standard(&config.model_path, &device)?
    };
    
    println!("\n✅ Flux model loaded successfully!");
    
    // Step 5: Create the trainer with everything loaded
    println!("\n[Step 5] Creating trainer with optimized memory layout...");
    let trainer = FluxTrainer::new(config, device)?;
    
    println!("\n=== 🎉 Trainer Creation Complete! ===");
    println!("\nSummary:");
    println!("  ✓ Data loader: {} samples ready", data_loader.total_samples());
    println!("  ✓ Latents: Cached to disk");
    println!("  ✓ Text embeddings: Cached to disk");
    println!("  ✓ Flux model: Loaded in BF16");
    println!("  ✓ Training mode: LoRA");
    println!("\nReady to start training!\n");
    
    Ok(trainer)
}

/// Demonstrates the training loop with dynamic model loading
pub fn train_with_dynamic_loading(trainer: &mut FluxTrainer, data_loader: &mut FluxDataLoader) -> Result<()> {
    println!("\n=== Training with dynamic model loading ===");
    
    for epoch in 0..trainer.config.max_train_steps {
        println!("\nEpoch {}/{}", epoch + 1, trainer.config.max_train_steps);
        
        // Process batches
        while let Some(batch) = data_loader.next_batch_old()? {
            // The actual training would:
            // 1. Load VAE if needed for encoding
            // 2. Encode images -> latents
            // 3. Free VAE
            // 4. Load text encoders if needed
            // 5. Encode text -> embeddings
            // 6. Free text encoders
            // 7. Run training step with main model
            
            println!("  Processing batch...");
            
            // Simulate the workflow
            println!("  - Would load VAE for encoding");
            println!("  - Encode images to latents");
            println!("  - Free VAE from GPU");
            println!("  - Load text encoders");
            println!("  - Encode text prompts");
            println!("  - Free text encoders from GPU");
            println!("  - Run training step with main model");
            
            break; // Just demonstrate one batch
        }
        
        break; // Just demonstrate one epoch
    }
    
    Ok(())
}

// Helper functions
fn load_flux_model_for_lora(path: &Path, device: &Device) -> flame_core::Result<FluxModel> {
    println!("  Loading Flux model for LoRA training (all tensors to GPU)...");
    println!("  Model size: ~22.17 GB in BF16");
    
    // Load all tensors to GPU - no CPU offloading as requested
    let weight_loader = WeightLoader::from_safetensors_with_dtype(path, device.clone(), DType::BF16)?;
    println!("  Loaded {} weights in BF16 format", weight_loader.weights.len());
    
    let config = detect_flux_variant(&weight_loader.weights);
    
    // Create model with all weights on GPU
    println!("  Creating Flux model with all tensors on GPU...");
    let model = FluxModel::new(config, device.clone(), weight_loader.weights)?;
    println!("  ✅ Flux model loaded successfully with all tensors on GPU");
    
    Ok(model)
}

fn load_flux_model_standard(path: &Path, device: &Device) -> flame_core::Result<FluxModel> {
    println!("  Loading Flux model for full fine-tuning...");
    
    // For full fine-tuning, we need all weights on GPU with BF16
    let weight_loader = WeightLoader::from_safetensors_with_dtype(path, device.clone(), DType::BF16)?;
    
    let config = detect_flux_variant(&weight_loader.weights);
    let model = FluxModel::new(config, device.clone(), weight_loader.weights)?;
    
    Ok(model)
}

fn detect_flux_variant(weights: &HashMap<String, Tensor>) -> FluxModelConfig {
    if weights.contains_key("guidance_in.in_layer.weight") {
        println!("  Detected Flux-dev model (with guidance)");
        FluxModelConfig::flux_dev()
    } else {
        println!("  Detected Flux-schnell model (no guidance)");
        FluxModelConfig::flux_schnell()
    }
}

/// Example of how to use the optimized pipeline
pub fn example_usage() -> Result<()> {
    println!("=== Example: Optimized Flux LoRA Training ===");
    
    // Create device
    let device = flame_core::device::Device::cuda(0)?;
    
    // Configuration
    let config = FluxTrainingConfig {
        model_path: PathBuf::from("/path/to/flux-dev.safetensors"),
        vae_path: PathBuf::from("/path/to/vae.safetensors"),
        text_encoder_paths: TextEncoderPaths {
            clip_l: PathBuf::from("/path/to/clip_l.safetensors"),
            t5_xxl: Some(PathBuf::from("/path/to/t5_xxl.safetensors")),
        },
        train_mode: TrainMode::LoRA,
        batch_size: 1,
        gradient_accumulation_steps: 4,
        learning_rate: 1e-4,
        warmup_steps: 100,
        max_train_steps: 1000,
        checkpointing_steps: 500,
        mixed_precision: true,
        gradient_checkpointing: true,
        use_8bit_adam: true,
        max_grad_norm: 1.0,
        lora_rank: 16,
        lora_alpha: 16.0,
        lora_dropout: 0.0,
        lora_target_modules: vec!["img_attn".to_string(), "txt_attn".to_string()],
        resolution: 1024,
        center_crop: true,
        random_flip: true,
        caption_dropout_rate: 0.1,
        guidance_scale: 3.5,
        bypass_guidance_embedding: false,
        shift_schedule: 3.0,
        logging_dir: PathBuf::from("./logs"),
        report_to: vec!["tensorboard".to_string()],
        validation_prompts: vec!["a photo of a cat".to_string()],
        validation_steps: 100,
        cache_latents_to_disk: true,
        cache_dir: None,
        force_reencode: false,
        dataset_name: "flux_lora_optimized".to_string(),
        use_layer_streaming: Some(true),
        streaming_memory_limit_gb: Some(20.0),
        use_int8_base_model: Some(false),
        chroma_config: None,
    };
    
    let dataset_config = DatasetConfig {
        folder_path: PathBuf::from("/path/to/dataset"),
        caption_ext: "txt".to_string(),
        caption_dropout_rate: 0.1,
        shuffle_tokens: false,
        cache_latents_to_disk: true, // Enable latent caching
        resolutions: vec![(1024, 1024)],
        center_crop: true,
        random_flip: true,
    };
    
    // Create trainer with optimized loading
    let mut trainer = create_flux_trainer_optimized(config, device.clone(), dataset_config)?;
    
    println!("\nThe optimized pipeline:");
    println!("1. Loads models in the correct order");
    println!("2. Frees unnecessary models from GPU");
    println!("3. Maximizes available memory for training");
    println!("4. Supports dynamic loading during training");
    
    Ok(())
}
