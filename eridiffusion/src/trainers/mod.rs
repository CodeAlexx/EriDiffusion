use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::anyhow;
use eridiffusion_core::FluxVariant;
use eridiffusion_training::flux_trainer::{
    self as phase4_flux, FluxTrainingConfig as Phase4FluxTrainingConfig,
};
use eridiffusion_training::init::init_global_device;
use flame_core::device::Device;
use flame_core::{DType, Error, Result, Tensor};
use serde::{Deserialize, Serialize};
use tokio::runtime::Builder as TokioBuilder;

use crate::config::trainer_config::{Config, ProcessConfig};

// Removed duplicate Config - using the one from config::trainer_config

// Removed duplicate ProcessConfig - using the one from config::trainer_config

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    pub dataset_path: String,
    pub batch_size: Option<usize>,
    pub every_n_steps: Option<usize>,
}

// NetworkConfig is now imported from trainer_config
// struct definitions removed to avoid conflicts

// All config structs are now imported from crate::config::trainer_config
// Old duplicate definitions removed to avoid conflicts

// SD 3.5 modules
pub mod cuda_rms_norm;
pub mod mmdit_loader;
pub mod mmdit_patch;
pub mod rms_norm_fix;
pub mod rms_norm_patch;

// Production pipelines
pub mod pipeline_flux_lora; // Flux LoRA trainer
pub mod pipeline_flux_lora_optimized;
pub mod pipeline_sd35_lora; // SD3.5 LoRA trainer
pub mod pipeline_sdxl_lora; // SDXL LoRA trainer // Flux LoRA trainer with CPU offloading
                            // pub mod pipeline_flux_lora_1024;  // Optimized Flux LoRA trainer for 1024x1024 - has compilation errors
pub mod pipeline_flux_lora_sequential; // Flux LoRA trainer with proper sequential loading
// pub mod pipeline_flux_lora_sequential_fixed; // Disabled: depends on streaming_t5 (deleted)
// pub mod streaming_text_encoders; // Disabled: depends on streaming_t5 (deleted)

// Re-export important types
pub use pipeline_flux_lora::{
    FluxTrainer as LegacyFluxTrainer, FluxTrainingConfig as LegacyFluxTrainingConfig,
};
pub use pipeline_flux_lora_sequential::FluxTrainerSequential;

// Modular SDXL pipeline components
pub mod lora;
pub mod models;
pub mod sampling;
pub mod trainer_sanity_checks;
pub mod training;

// Memory optimization
pub mod cpu_offload_manager;
pub mod model_offloader;

// SDXL modules
pub mod sdxl_forward_sd_format_flash;
pub mod sdxl_memory_efficient;
pub mod sdxl_sampling_complete;
pub mod sdxl_transformer_block_flash;
pub mod sdxl_utils;

// Text encoding modules
pub mod embedded_tokenizers;
pub mod optimized_text_encoders;
pub mod real_tokenizers;
pub mod text_embedding_cache;
pub mod text_encoder_cached;
pub mod text_encoders;

// VAE modules
pub mod sdxl_vae_native;
pub mod sdxl_vae_wrapper;
pub mod vae_encoder;
pub mod vae_tiling;
// pub mod vae_tiling_advanced; // Disabled: depends on unified_vae (deleted)

// Training utilities
pub mod adam8bit;
pub mod adam8bit_enhanced;
pub mod adapters_util;
pub mod ddpm_scheduler;
pub mod efficient_attention;
pub mod enhanced_data_loader;
pub mod flash_attention_wrapper; // Flash Attention for SD 3.5 and Flux
pub mod generic_linear;
pub mod gpu_gradient_checkpoint;
pub mod gradient_accumulator;
pub mod memory_utils;

// Advanced training features
pub mod checkpoint_manager;
pub mod ema;
pub mod ema_enhanced;
pub mod lr_scheduler;
pub mod snr_weighting;
pub mod validation;
pub mod validation_advanced;
pub mod validation_formatter;

// Memory-efficient training modules
pub mod flux_lora_optimized;
pub mod gradient_accumulation;

// Sampling and inference
pub mod sdxl_forward_sampling;
pub mod unified_sampling;

// Device management
pub mod cached_device;
pub mod device_debug;
pub mod device_fix;
pub mod force_device_zero;
pub mod single_device_enforcer;

// Tensor conversion utilities
pub mod hybrid_tensor_ops;
pub mod tensor_conversion;

// Flux modules
pub mod checkpoint_saver;
pub mod cpu_offload_flux_loader;
pub mod flux_backward_optimizer;
pub mod flux_cache_manager;
pub mod flux_cache_manager_optimized; // Optimized cache with cuDNN and persistent text cache
pub mod flux_data_loader;
pub mod flux_int8_loader;
pub mod flux_int8_wrapper;
pub mod flux_layer_streaming;
pub mod gpu_lora_example;
pub mod memory_optimizer;
pub mod minimal_flux_loader; // Minimal loader for LoRA training
pub mod pipeline_progress_tracker;
pub mod quanto_var_builder; // CPU offloading for Flux model

// Removed duplicate TrainingConfig - using the one from config::trainer_config

// Removed duplicate JobConfig - using the one from config::trainer_config

#[derive(Debug)]
pub enum ModelType {
    SD35,
    SDXL,
    Flux,
    SD15,
    SD21,
}

pub fn detect_model_type(config: &ProcessConfig) -> flame_core::Result<ModelType> {
    // First check explicit arch field
    if let Some(arch) = &config.model.arch {
        match arch.to_lowercase().as_str() {
            "sd35" | "sd3.5" | "sd_3.5" => return Ok(ModelType::SD35),
            "sdxl" | "sd_xl" => return Ok(ModelType::SDXL),
            "flux" => return Ok(ModelType::Flux),
            "sd15" | "sd1.5" | "sd_1.5" => return Ok(ModelType::SD15),
            "sd21" | "sd2.1" | "sd_2.1" => return Ok(ModelType::SD21),
            _ => {} // Continue to other checks
        }
    }

    // Then check explicit flags
    if config.model.is_v3.unwrap_or(false) {
        return Ok(ModelType::SD35);
    }

    if config.model.is_flux.unwrap_or(false) {
        return Ok(ModelType::Flux);
    }

    // Check by model path/name
    let model_path = config.model.name_or_path.to_string_lossy().to_lowercase();

    if model_path.contains("sd3.5") || model_path.contains("sd35") || model_path.contains("sd_3.5")
    {
        return Ok(ModelType::SD35);
    }

    if model_path.contains("flux") {
        return Ok(ModelType::Flux);
    }

    if model_path.contains("sdxl") || model_path.contains("sd_xl") {
        return Ok(ModelType::SDXL);
    }

    if model_path.contains("sd2") || model_path.contains("v2") {
        return Ok(ModelType::SD21);
    }

    if model_path.contains("sd1") || model_path.contains("v1-5") || model_path.contains("v1.5") {
        return Ok(ModelType::SD15);
    }

    // Check by training config hints
    if config.train.linear_timesteps.unwrap_or(false) {
        // SD3.5 uses linear timesteps
        return Ok(ModelType::SD35);
    }

    if config.train.bypass_guidance_embedding {
        // Flux uses bypass_guidance_embedding
        return Ok(ModelType::Flux);
    }

    // Default to SDXL as it's the most common
    println!("Warning: Could not determine model type from config, defaulting to SDXL");
    Ok(ModelType::SDXL)
}

pub fn train_from_config(config_path: PathBuf) -> flame_core::Result<()> {
    // Check for GPU requirement using FLAME
    let device = flame_core::device::Device::cuda(0)
    .map_err(|_| flame_core::Error::InvalidOperation("GPU is required for training. No CUDA device found.\nThis trainer requires a CUDA-capable GPU.".into()))?;
    println!("GPU detected and verified for training using FLAME");

    // Load YAML configuration (robust parsing with fallbacks)
    let config_str = std::fs::read_to_string(&config_path).map_err(|_| {
        flame_core::Error::InvalidOperation(format!(
            "Failed to read config file: {}",
            config_path.display()
        ))
    })?;

    let training_config: crate::config::trainer_config::TrainingConfig = match serde_yaml::from_str(
        &config_str,
    ) {
        Ok(cfg) => cfg,
        Err(e) => {
            // Fallback: tolerate files that either omit `job` or wrap content directly under `config:`
            use crate::config::trainer_config::{Config as C, TrainingConfig as TC};
            #[derive(serde::Deserialize)]
            struct RootWrap {
                config: serde_yaml::Value,
                #[allow(dead_code)]
                job: Option<String>,
            }
            if let Ok(root) = serde_yaml::from_str::<RootWrap>(&config_str) {
                let cfg: C = serde_yaml::from_value(root.config).map_err(|e3| {
                    flame_core::Error::InvalidOperation(format!(
                        "Failed to parse YAML 'config' section: {}",
                        e3
                    ))
                })?;
                TC {
                    job: Some("extension".to_string()),
                    config: cfg,
                    meta: std::collections::HashMap::new(),
                }
            } else {
                // Last resorts:
                // 1) Extract the indented block under a line exactly equal to 'config:' and parse that
                let mut extracted = String::new();
                let mut in_config = false;
                for line in config_str.lines() {
                    let trimmed = line.trim_end();
                    if !in_config {
                        if trimmed.trim() == "config:" {
                            in_config = true;
                            continue;
                        }
                    } else {
                        if trimmed.starts_with("  ") {
                            // remove two leading spaces
                            extracted.push_str(&trimmed[2..]);
                            extracted.push('\n');
                        } else if trimmed.is_empty() {
                            // allow blank lines inside block
                            extracted.push('\n');
                        } else if trimmed.trim_start().starts_with('#') {
                            // skip top-level comment lines and continue scanning
                            continue;
                        } else {
                            // encountered another top-level key, stop
                            break;
                        }
                    }
                }
                if !extracted.is_empty() {
                    if let Ok(cfg) = serde_yaml::from_str::<C>(&extracted) {
                        TC {
                            job: Some("extension".to_string()),
                            config: cfg,
                            meta: std::collections::HashMap::new(),
                        }
                    } else {
                        // 2) As a final attempt, parse whole document as Config directly
                        let cfg: C = serde_yaml::from_str(&config_str).map_err(|e3| {
                        flame_core::Error::InvalidOperation(format!(
                            "Failed to parse YAML as TrainingConfig or Config: {} (original error: {})", e3, e
                        ))
                    })?;
                        TC {
                            job: Some("extension".to_string()),
                            config: cfg,
                            meta: std::collections::HashMap::new(),
                        }
                    }
                } else {
                    // 2) Parse whole doc as Config
                    let cfg: C = serde_yaml::from_str(&config_str).map_err(|e3| {
                    flame_core::Error::InvalidOperation(format!(
                        "Failed to parse YAML as TrainingConfig or Config: {} (original error: {})", e3, e
                    ))
                })?;
                    TC {
                        job: Some("extension".to_string()),
                        config: cfg,
                        meta: std::collections::HashMap::new(),
                    }
                }
            }
        }
    };

    let training_meta = training_config.meta.clone();
    let config = training_config.config;

    // Find the trainer process config (supports both sd_trainer and flux_trainer)
    let process_config = config
        .process
        .iter()
        .find(|p| p.process_type == "sd_trainer" || p.process_type == "flux_trainer")
        .ok_or_else(|| {
            flame_core::Error::InvalidOperation(
                "No 'sd_trainer' or 'flux_trainer' process found in config".into(),
            )
        })?;

    // Debug: Print config to see what we're working with
    println!("\n=== Model Detection Debug ===");
    println!("is_flux: {:?}", process_config.model.is_flux);
    println!("arch: {:?}", process_config.model.arch);
    println!("model path: {}", process_config.model.name_or_path.display());

    // Detect model type
    let model_type = detect_model_type(process_config)?;

    println!("\nDetected model type: {:?}", model_type);
    println!("Model path: {}", process_config.model.name_or_path.display());
    println!("Network type: {}", process_config.network.network_type);

    // Route to appropriate trainer
    match model_type {
        ModelType::SD35 => {
            println!("\nStarting SD 3.5 training...");
            match process_config.network.network_type.as_str() {
                "lokr" | "lokr_full_rank" => {
                    println!("Note: LoKr not yet implemented for SD 3.5");
                    return Err(flame_core::Error::Unsupported(
                        "SD 3.5 LoKr training temporarily disabled".into(),
                    ));
                    // sd35_lora_gpu::train_sd35_lora_gpu(&config, process_config)?;
                }
                "lora" => {
                    // Use GPU-only LoRA implementation
                    println!("SD 3.5 LoRA training temporarily disabled - use SDXL instead");
                    return Err(flame_core::Error::Unsupported(
                        "SD 3.5 LoRA training temporarily disabled".into(),
                    ));
                    // sd35_lora_gpu::train_sd35_lora_gpu(&config, process_config)?;
                }
                _ => {
                    return Err(flame_core::Error::InvalidOperation(format!(
                        "Unsupported network type '{}' for SD 3.5",
                        process_config.network.network_type
                    )));
                }
            }
        } // End of SD35 case
        ModelType::SDXL => {
            println!("\nStarting SDXL training...");
            match process_config.network.network_type.as_str() {
                "lora" => {
                    println!("Using production SDXL LoRA pipeline");
                    // TODO: Call pipeline_sdxl_lora when ready
                    return Err(flame_core::Error::Unsupported(
                        "SDXL LoRA training temporarily disabled while fixing imports".into(),
                    ));
                }
                "lokr" | "lokr_full_rank" => {
                    println!("Note: LoKr not yet implemented for SDXL, using LoRA instead");
                    return Err(flame_core::Error::Unsupported(
                        "LoKr training not yet implemented. Please use LoRA for now.".into(),
                    ));
                }
                _ => {
                    return Err(flame_core::Error::InvalidOperation(format!(
                        "Unsupported network type '{}' for SDXL",
                        process_config.network.network_type
                    )));
                }
            }
        } // End of SDXL case
        ModelType::Flux => {
            println!("\nStarting Flux training...");
            match process_config.network.network_type.as_str() {
                "lora" => {
                    println!("Using production Flux LoRA pipeline");
                    // flux_lora_gpu::train_flux_lora_gpu(&config, process_config)?;
                    return Err(flame_core::Error::Unsupported(
                        "Flux LoRA training temporarily disabled while fixing imports".into(),
                    ));
                }
                _ => {
                    return Err(flame_core::Error::InvalidOperation(format!(
                        "Unsupported network type '{}' for Flux",
                        process_config.network.network_type
                    )));
                }
            }
        } // End of Flux case
        ModelType::SD15 | ModelType::SD21 => {
            return Err(flame_core::Error::Unsupported(
            "SD 1.5/2.1 training not yet implemented in unified trainer. Use legacy diffusers-rs.".into()
        ));
        }
    } // End of match model_type

    println!("\n=== Training Complete ===");
    Ok(())
} // End of train_from_config

// Re-export GPU gradient checkpoint as the default
pub mod gradient_checkpoint {
    pub use super::gpu_gradient_checkpoint::SDXLGradientCheckpoint;
}

// Function to run Flux LoRA training (legacy pipeline kept for reference)
#[allow(dead_code)]
fn run_flux_lora_training_legacy(
    config: &Config,
    process_config: &ProcessConfig,
) -> flame_core::Result<()> {
    // Use the SEQUENTIAL trainer with GPU-only streaming and cuDNN optimization!
    use crate::trainers::flux_data_loader::{DatasetConfig as FluxDatasetConfig, FluxDataLoader};
    use crate::trainers::pipeline_flux_lora::{
        ChromaXLConfig, FluxTrainingConfig, TextEncoderPaths, TrainMode,
    };
    use crate::trainers::pipeline_flux_lora_sequential::FluxTrainerSequential;

    // Create device
    let device = crate::cuda_device(0)?;

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
                .cloned(), // Don't use default if null - respects config
        },

        // Training configuration
        train_mode: TrainMode::LoRA,
        batch_size: process_config.train.batch_size,
        gradient_accumulation_steps: process_config.train.gradient_accumulation_steps,
        learning_rate: process_config.train.lr as f64,
        warmup_steps: 0, // No warmup steps field in config
        max_train_steps: process_config.train.steps,
        checkpointing_steps: process_config.save.save_every,

        // Optimization
        mixed_precision: process_config.train.dtype == "bf16"
            || process_config.train.dtype == "fp16",
        gradient_checkpointing: process_config.train.gradient_checkpointing,
        // Prefer memory-efficient 8-bit Adam automatically on constrained VRAM
        use_8bit_adam: process_config.train.optimizer == "adamw8bit"
            || process_config.train.streaming_memory_limit_gb.unwrap_or(24.0) <= 24.0,
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
        resolution: process_config.datasets[0].resolution[0], // Take first resolution
        center_crop: false,
        random_flip: false,
        caption_dropout_rate: process_config.datasets[0].caption_dropout_rate,

        // Flux-specific
        guidance_scale: process_config.sample.guidance_scale,
        bypass_guidance_embedding: process_config.train.bypass_guidance_embedding,
        shift_schedule: 3.0, // Default shift for Flux

        // Logging
        logging_dir: PathBuf::from("output").join(&config.name),
        report_to: vec![],
        validation_prompts: process_config.sample.prompts.clone(),
        validation_steps: process_config.sample.sample_every,

        // Caching configuration
        cache_latents_to_disk: process_config.datasets[0].cache_latents_to_disk,
        cache_dir: None, // Will use default under logging_dir
        force_reencode: false,
        dataset_name: config.name.clone(),

        // Layer streaming configuration
        use_layer_streaming: process_config.train.use_layer_streaming.or(Some(false)), // Read from config, default to false (was causing endless loading)
        streaming_memory_limit_gb: process_config.train.streaming_memory_limit_gb.or(Some(16.0)), // Read from config or default to 16GB

        // INT8 quantization (not configured via YAML - use simple_trainer for INT8)
        use_int8_base_model: Some(false),

        // ChromaXL configuration - check if this is a ChromaXL config
        chroma_config: if process_config.model.arch.as_deref() == Some("chroma")
            || process_config.network.ramp_double_blocks.unwrap_or(false)
        {
            // Parse layer-specific learning rates if provided
            let layer_lr_multipliers = process_config
                .network
                .network_kwargs
                .as_ref()
                .and_then(|kwargs| kwargs.lr_if_contains.clone());

            Some(ChromaXLConfig {
                layer_pattern: "chroma_default".to_string(),
                layer_lr_multipliers,
                ramp_double_blocks: process_config.network.ramp_double_blocks.unwrap_or(false),
                ramp_target_lr: process_config.network.ramp_target_lr.unwrap_or(1.5e-6) as f32,
                ramp_warmup_steps: process_config.network.ramp_warmup_steps.unwrap_or(1000),
                ramp_type: process_config
                    .network
                    .ramp_type
                    .as_ref()
                    .map(|s| s.clone())
                    .unwrap_or_else(|| "linear".to_string()),
            })
        } else {
            None
        },
    };

    // Create data loader
    let dataset_config = FluxDatasetConfig {
        folder_path: PathBuf::from(&process_config.datasets[0].folder_path),
        caption_ext: process_config.datasets[0].caption_ext.clone(),
        caption_dropout_rate: process_config.datasets[0].caption_dropout_rate,
        shuffle_tokens: process_config.datasets[0].shuffle_tokens,
        cache_latents_to_disk: process_config.datasets[0].cache_latents_to_disk,
        resolutions: process_config.datasets[0].resolution.iter().map(|&r| (r, r)).collect(),
        center_crop: false,
        random_flip: false,
    };

    // Create data loader FIRST
    let mut dataloader = FluxDataLoader::new(dataset_config, device.clone())?;

    // Use SEQUENTIAL trainer that implements SimpleTuner's memory-efficient approach!
    println!("\n=== Using FluxTrainerSequential (SimpleTuner-style) ===");
    println!("This properly implements:");
    println!("  1. Load VAE → Encode ALL images → FREE VAE completely");
    println!("  2. Load CLIP/T5 → Encode ALL text → FREE encoders completely");
    println!("  3. Only THEN load 23GB Flux model with all 24GB available!");
    println!("\nThis is how SimpleTuner fits 23GB models in 24GB VRAM!");

    // Create the sequential trainer with proper memory management
    let mut trainer = FluxTrainerSequential::new_with_sequential_loading(
        flux_config,
        device.clone(),
        &mut dataloader,
    )?;

    // The sequential trainer already did pre-encoding in its constructor
    // Now run actual training with the Flux model loaded
    println!("\n✅ Starting training with REAL 23GB Flux Dev model!");
    trainer.train(&mut dataloader)?;

    Ok(())
}

pub use adam8bit_enhanced::{Adam8bit as Adam8bitEnhanced, Adam8bitConfig};
pub use unified_sampling::SamplingConfig;
pub use validation::create_sample_directory;
pub mod tensor_ops;

fn run_flux_lora_training_phase4(
    config: &Config,
    process_config: &ProcessConfig,
    meta: &HashMap<String, String>,
) -> flame_core::Result<()> {
    run_flux_lora_training_phase4_impl(config, process_config, meta)
        .map_err(|e| Error::InvalidOperation(format!("Flux Phase-4 trainer failed: {e}")))
}

fn run_flux_lora_training_phase4_impl(
    config: &Config,
    process_config: &ProcessConfig,
    meta: &HashMap<String, String>,
) -> anyhow::Result<()> {
    let device = init_global_device(&process_config.device)?;

    let clip_tokenizer = meta
        .get("flux_clip_tokenizer")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/home/alex/diffusers-rs/tokenizers/clip_tokenizer.json"));
    let t5_tokenizer = meta
        .get("flux_t5_tokenizer")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/home/alex/diffusers-rs/tokenizers/t5_tokenizer.json"));

    let model_path = process_config.model.name_or_path.clone();
    let vae_path =
        process_config.model.vae_path.clone().ok_or_else(|| anyhow!("model.vae_path missing"))?;
    let clip_path = process_config
        .model
        .clip_l_path
        .clone()
        .or_else(|| process_config.model.text_encoder_path.clone())
        .ok_or_else(|| anyhow!("model.clip_l_path missing"))?;
    let t5_path = process_config
        .model
        .t5_path
        .clone()
        .or_else(|| process_config.model.text_encoder_2_path.clone())
        .ok_or_else(|| anyhow!("model.t5_path missing"))?;

    let flux_cfg = Phase4FluxTrainingConfig {
        learning_rate: process_config.train.lr,
        warmup_steps: meta
            .get("flux_warmup_steps")
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0),
        max_steps: process_config.train.steps,
        gradient_accumulation_steps: process_config.train.gradient_accumulation_steps.max(1),
        gradient_checkpointing: process_config.train.gradient_checkpointing,
        guidance_scale: process_config.sample.guidance_scale,
        text_drop_prob: process_config
            .datasets
            .first()
            .map(|d| d.caption_dropout_rate)
            .unwrap_or(0.0),
        ema_decay: process_config
            .train
            .ema_config
            .as_ref()
            .filter(|ema| ema.use_ema)
            .map(|ema| ema.ema_decay)
            .unwrap_or(0.0),
        mixed_precision: matches!(
            process_config.train.dtype.to_ascii_lowercase().as_str(),
            "bf16" | "bfloat16" | "fp16" | "float16"
        ),
        max_grad_norm: meta
            .get("flux_max_grad_norm")
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(1.0),
        seed: meta
            .get("flux_seed")
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(process_config.sample.seed),
        lora_rank: process_config.network.linear.max(1),
        lora_alpha: process_config.network.linear_alpha,
        lora_zero_init: meta
            .get("flux_lora_zero_init")
            .and_then(|s| s.parse::<bool>().ok())
            .unwrap_or(true),
        sigma_min: meta.get("flux_sigma_min").and_then(|s| s.parse::<f32>().ok()).unwrap_or(0.01),
        sigma_max: meta.get("flux_sigma_max").and_then(|s| s.parse::<f32>().ok()).unwrap_or(50.0),
    };

    let flame_device = match device {
        eridiffusion_core::Device::Cuda(ix) => flame_core::device::Device::cuda(ix)?,
        _ => return Err(anyhow!("Flux training requires CUDA device")),
    };

    let dataset = process_config
        .datasets
        .first()
        .ok_or_else(|| anyhow!("Flux trainer requires at least one dataset entry"))?;

    let dataset_config = crate::trainers::flux_data_loader::DatasetConfig {
        folder_path: dataset.folder_path.clone(),
        caption_ext: dataset.caption_ext.clone(),
        caption_dropout_rate: dataset.caption_dropout_rate,
        shuffle_tokens: dataset.shuffle_tokens,
        cache_latents_to_disk: dataset.cache_latents_to_disk,
        resolutions: dataset.resolution.iter().map(|&r| (r, r)).collect(),
        center_crop: false,
        random_flip: false,
    };

    let mut loader = crate::trainers::flux_data_loader::FluxDataLoader::new(
        dataset_config,
        flame_device.clone(),
    )?;

    let rt = TokioBuilder::new_current_thread().enable_all().build()?;
    let variant = if let Some(arch) = process_config.model.arch.as_ref() {
        match arch.to_ascii_lowercase().as_str() {
            "flux-dev" | "flux_dev" | "fluxdev" | "dev" => FluxVariant::Dev,
            "flux-schnell" | "flux_schnell" | "fluxschnell" | "schnell" => FluxVariant::Schnell,
            _ => FluxVariant::Base,
        }
    } else {
        let name = process_config.model.name_or_path.to_string_lossy().to_ascii_lowercase();
        if name.contains("schnell") {
            FluxVariant::Schnell
        } else if name.contains("dev") {
            FluxVariant::Dev
        } else {
            FluxVariant::Base
        }
    };

    let mut trainer = rt.block_on(phase4_flux::create_flux_trainer(
        model_path.as_path(),
        vae_path.as_path(),
        t5_path.as_path(),
        clip_path.as_path(),
        t5_tokenizer.as_path(),
        clip_tokenizer.as_path(),
        variant,
        flux_cfg.clone(),
        device.clone(),
    ))?;

    if let Some(dir) = meta.get("flux_ckpt_dir").map(PathBuf::from) {
        if let Some((step, _)) = trainer.load_latest_checkpoint(dir.to_string_lossy().as_ref())? {
            println!("[flux] resumed at step {step}");
        }
    }

    let log_every = meta
        .get("flux_log_every")
        .and_then(|s| s.parse::<u64>().ok())
        .or_else(|| process_config.performance_log_every.map(|v| v as u64))
        .unwrap_or(10)
        .max(1);
    let ckpt_every = meta
        .get("flux_checkpoint_every")
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(process_config.save.save_every as u64)
        .max(1);
    let ckpt_dir = meta
        .get("flux_ckpt_dir")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(&process_config.training_folder).join(&config.name));
    std::fs::create_dir_all(&ckpt_dir)?;
    let ckpt_dir = ckpt_dir.to_string_lossy().to_string();

    let mut step: u64 = 0;
    let max_steps = flux_cfg.max_steps as u64;
    let batch_size = process_config.train.batch_size.max(1);
    let negative_prompts: Vec<String> = Vec::new();
    loop {
        let mut images_batches = Vec::new();
        let mut prompts = Vec::new();

        while images_batches.len() < batch_size {
            match loader.next_batch_old()? {
                Some(mut sample) => {
                    images_batches.push(sample.images);
                    prompts.append(&mut sample.prompts);
                }
                None => {
                    loader.shuffle_dataset()?;
                    continue;
                }
            }
        }

        let image_refs: Vec<&Tensor> = images_batches.iter().collect();
        let images_cat = Tensor::cat(&image_refs, 0)?;
        let loss = rt.block_on(trainer.train_step(&images_cat, &prompts, &negative_prompts))?;

        if step % log_every == 0 {
            println!("[flux][step {step}] loss={loss:.4}");
        }

        drop(images_cat);
        step += 1;
        if step % ckpt_every == 0 {
            let _ = trainer.save_checkpoint(&ckpt_dir, step, flux_cfg.seed)?;
        }
        if max_steps > 0 && step >= max_steps {
            break;
        }
    }

    Ok(())
}
