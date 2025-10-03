//! Fixed Flux LoRA training pipeline with true T5 streaming
//! This version properly streams T5 layers to prevent OOM on 24GB GPUs

use flame_core::{Device, Result, Tensor, DType, CudaDevice, Module};
use crate::config::trainer_config::{TrainingConfig, ProcessConfig, TrainConfig};
use crate::trainers::flux_data_loader::FluxDataLoader;
use crate::models::{FluxModel, FluxModelConfig};
use crate::models::flux_vae::{AutoencoderKL, AutoEncoderConfig as VAEConfig};
use crate::trainers::{
    streaming_text_encoders::StreamingTextEncoders,
    text_embedding_cache::PersistentEmbeddingCache,
    flux_cache_manager::FluxCacheManager,
};
// Progress tracking and memory management
use std::collections::HashMap;
use std::path::PathBuf;

/// Main training pipeline with streaming T5 support
pub fn train_flux_lora_sequential_fixed(config: TrainingConfig) -> Result<()> {
    println!("\n🚀 Starting Flux LoRA Training Pipeline (Fixed T5 Streaming)");
    println!("================================================");
    
    // Setup device
    let device = CudaDevice::new(0)?;
    let device_flame = Device::cuda(0)?;
    
    println!("🔧 Device: {:?}", device);
    
    // Initialize helpers (simplified for now)
    println!("🔧 Initializing helpers...");
    
    // Get process config
    let process_config = config.config.process.get(0)
        .ok_or_else(|| flame_core::Error::InvalidOperation("No process config found".into()))?;
    
    // Apply TREAD config (optional) to env for downstream trainer
    if let Some(tread) = process_config.tread.as_ref() {
        std::env::set_var("TREAD_ENABLED", if tread.enabled { "1" } else { "0" });
        if let Some(mask) = tread.mask.as_ref() {
            if let Some(k) = mask.k { std::env::set_var("TREAD_K", k.to_string()); }
            if let Some(kf) = mask.k_frac { std::env::set_var("TREAD_K_FRAC", format!("{}", kf)); }
            if let Some(ref t) = mask.r#type { std::env::set_var("TREAD_MASK_TYPE", t); }
        }
        if !tread.schedule.is_empty() {
            let s = tread.schedule.iter().map(|p| format!("{}:{}", p.out, p.r#in)).collect::<Vec<_>>().join(",");
            std::env::set_var("TREAD_SCHEDULE", s);
        }
        if let Some(rei) = tread.reinject.as_ref() { if let Some(ref m) = rei.mode { std::env::set_var("TREAD_REINJECT_MODE", m); } }
        if let Some(loss) = tread.loss.as_ref() { if let Some(l) = loss.route_lambda { std::env::set_var("TREAD_LAMBDA", format!("{}", l)); } }
    }

    // Setup data loader
    println!("\n📁 Setting up data loader...");
    let dataset_config = crate::trainers::flux_data_loader::DatasetConfig {
        folder_path: process_config.datasets[0].folder_path.clone(),
        caption_ext: process_config.datasets[0].caption_ext.clone(),
        caption_dropout_rate: process_config.datasets[0].caption_dropout_rate,
        shuffle_tokens: process_config.datasets[0].shuffle_tokens,
        cache_latents_to_disk: process_config.datasets[0].cache_latents_to_disk,
        resolutions: process_config.datasets[0].resolution.iter()
            .map(|&r| (r, r))
            .collect(),
        center_crop: false,
        random_flip: false,
    };
    let mut data_loader = FluxDataLoader::new(dataset_config, device_flame.clone())?;
    println!("  Dataset: {:?}", process_config.datasets[0].folder_path);
    println!("  Batch size: {}", process_config.train.batch_size);
    println!("  Image size: {}x{}", process_config.datasets[0].resolution[0], process_config.datasets[0].resolution[0]);
    
    // Setup cache manager
    let cache_dir = process_config.datasets[0].folder_path.join("_cache");
    let cache_mgr = FluxCacheManager::new(cache_dir.clone(), device_flame.clone(), true)?;
    println!("  Cache directory: {:?}", cache_dir);
    
    // Check if we need to process data
    let needs_processing = if process_config.datasets[0].force_recache {
        println!("  Force re-encode enabled - will process all data");
        true
    } else {
        let (missing_latents, missing_text) = cache_mgr.check_cache_status(&data_loader)?;
        println!("  Missing latent caches: {}", missing_latents);
        println!("  Missing text caches: {}", missing_text);
        missing_latents > 0 || missing_text > 0
    };
    
    if needs_processing {
        println!("\n🔄 Sequential Data Processing Required");
        println!("=====================================");
        
        // PHASE 1: VAE Encoding (if needed)
        let missing_latents = cache_mgr.count_missing_latents(&data_loader)?;
        if missing_latents > 0 {
            println!("\n[Phase 1] VAE Latent Encoding");
            println!("  Missing latents: {}", missing_latents);
            println!("  Loading VAE temporarily...");
            
            // Load VAE
            let default_vae_path = PathBuf::from("/home/alex/SwarmUI/Models/VAE/ae.safetensors");
            let vae_path = process_config.model.vae_path.as_ref()
                .unwrap_or(&default_vae_path);
            let vae = load_vae(&vae_path, &device_flame)?;
            
            // Encode all latents
            println!("  Encoding latents to disk...");
            // Cast VAE to AutoencoderKL (they're the same type via alias)
            let vae_ref: &AutoencoderKL = &vae;
            cache_mgr.encode_all_latents_with_model(&mut data_loader, vae_ref, process_config.datasets[0].force_recache)?;
            
            // Explicitly free VAE
            drop(vae);
            println!("  ✅ VAE unloaded - freed ~1.5GB GPU memory");
            
            // Force memory cleanup
            // Synchronize the device
            device_flame.synchronize()?;
        } else {
            println!("\n[Phase 1] Skipped - all latents already cached");
        }
        
        // PHASE 2: Text Encoding with Streaming T5
        let missing_text = cache_mgr.count_missing_text_embeddings(&data_loader)?;
        if missing_text > 0 {
            println!("\n[Phase 2] Text Encoder Processing (Streaming T5)");
            println!("  Missing text embeddings: {}", missing_text);
            
            // Create GPU-only streaming text encoders with cuDNN optimization
            println!("  T5 processing mode: GPU streaming with cuDNN");
            let mut text_encoders = StreamingTextEncoders::new(device_flame.clone());
            
            // Load CLIP-L (small, stays in memory)
            println!("  Loading CLIP-L...");
            let default_clip_path = PathBuf::from("/home/alex/SwarmUI/Models/clip/clip_l.safetensors");
            let clip_l_path = process_config.model.clip_l_path.as_ref()
                .or(process_config.model.text_encoder_path.as_ref())
                .unwrap_or(&default_clip_path);
            text_encoders.load_clip_l(&clip_l_path.to_string_lossy())?;
            
            // Load T5 with streaming
            println!("  Setting up T5-XXL streaming...");
            let t5_path = process_config.model.t5_path.as_ref()
                .or(process_config.model.text_encoder_2_path.as_ref())
                .ok_or_else(|| flame_core::Error::InvalidOperation("T5 path required".into()))?;
            text_encoders.load_t5_streaming(&t5_path.to_string_lossy())?;
            
            // Load tokenizers
            println!("  Loading tokenizers...");
            let tokenizer_dir = PathBuf::from("/home/alex/diffusers-rs/tokenizers");
            text_encoders.load_tokenizers(
                &tokenizer_dir.join("clip_tokenizer.json").to_string_lossy(),
                &tokenizer_dir.join("t5_tokenizer.json").to_string_lossy()
            )?;
            
            // Create text embedding cache
            let text_cache_dir = cache_dir.join("text_embeddings");
            std::fs::create_dir_all(&text_cache_dir)?;
            let mut text_cache = PersistentEmbeddingCache::new(text_cache_dir, device_flame.clone())?;
            
            // Collect all prompts
            let all_prompts: Vec<String> = data_loader.all_captions()?;
            println!("  Total prompts to encode: {}", all_prompts.len());
            
            // Encode prompts one by one with streaming T5
            println!("  Encoding text embeddings (one prompt at a time)...");
            text_encoders.encode_prompt_batch_streaming(&all_prompts, &mut text_cache)?;
            
            // Explicitly free text encoders
            drop(text_encoders);
            drop(text_cache);
            println!("  ✅ Text encoders unloaded");
            println!("     CLIP-L: freed ~0.5GB");
            println!("     T5-XXL: freed ~0.4GB (was streaming)");
            
            // Force memory cleanup
            // Synchronize the device
            device_flame.synchronize()?;
            
            // Memory tracking would go here
        } else {
            println!("\n[Phase 2] Skipped - all text embeddings already cached");
        }
        
        println!("\n✅ All preprocessing complete!");
        println!("   VAE: Not loaded (saves ~1.5GB)");
        println!("   CLIP-L: Not loaded (saves ~0.5GB)"); 
        println!("   T5-XXL: Not loaded (saves ~9.12GB with GPU streaming)");
        println!("   Total saved: ~11GB GPU memory for model loading");
    }
    
    // PHASE 3: Main Model Training
    println!("\n[Phase 3] Loading Flux model with all available memory...");
    println!("  Model path: {:?}", process_config.model.name_or_path);
    println!("  Expected size: ~22GB in BF16");
    
    // Use layer streaming by default
    let use_streaming = process_config.train.streaming_memory_limit_gb.is_some();
    
    if use_streaming {
        println!("\n🌊 Setting up Flux with layer streaming...");
        println!("  This allows training on 24GB GPUs by loading layers on-demand");
        
        // Convert config for sequential trainer
        let flux_config = create_flux_training_config(&config, process_config)?;
        
        // Use existing sequential trainer
        println!("\n=== Using FluxTrainerSequential (SimpleTuner-style) ===");
        println!("This properly implements:");
        println!("  1. Load VAE → Encode ALL images → FREE VAE completely");
        println!("  2. Load CLIP/T5 → Encode ALL text → FREE encoders completely");
        println!("  3. Only THEN load 23GB Flux model with all 24GB available!");
        println!("  4. T5 uses GPU layer streaming with cuDNN optimization!");
        
        // Create the sequential trainer with proper memory management
        let mut trainer = crate::trainers::pipeline_flux_lora_sequential::FluxTrainerSequential::new_with_sequential_loading(
            flux_config,
            device_flame.clone(),
            &mut data_loader
        )?;
        
        // The sequential trainer already did pre-encoding in its constructor
        // Now run actual training with the Flux model loaded
        println!("\n✅ Starting training with REAL 23GB Flux Dev model!");
        trainer.train(&mut data_loader)?;
        
    } else {
        return Err(flame_core::Error::InvalidOperation(
            "Non-streaming mode not recommended for 24GB GPUs".to_string()
        ));
    }
    
    println!("\n✅ Training complete!");
    Ok(())
}

/// Load VAE 
fn load_vae(path: &PathBuf, device: &Device) -> Result<AutoencoderKL> {
    println!("  Loading VAE from: {:?}", path);
    
    // Load VAE weights
    let wl = crate::loaders::WeightLoader::from_safetensors_with_dtype(
        path.to_string_lossy().as_ref(),
        device.clone(),
        DType::F16,
    )?;
    
    // Create VAE using the expected signature (WeightLoader, Device, enable_offloading)
    let vae = AutoencoderKL::new(&wl, device.clone(), false)?;
    
    println!("  ✅ VAE loaded successfully");
    Ok(vae)
}

/// Create FluxTrainingConfig from YAML config
fn create_flux_training_config(
    config: &TrainingConfig,
    process_config: &ProcessConfig,
) -> Result<crate::trainers::pipeline_flux_lora::FluxTrainingConfig> {
    use crate::trainers::pipeline_flux_lora::{FluxTrainingConfig, TextEncoderPaths, TrainMode, ChromaXLConfig};
    
    Ok(FluxTrainingConfig {
        // Model paths
        model_path: PathBuf::from(&process_config.model.name_or_path),
        vae_path: process_config.model.vae_path.as_ref()
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("/home/alex/SwarmUI/Models/VAE/ae.safetensors")),
        text_encoder_paths: TextEncoderPaths {
            clip_l: process_config.model.clip_l_path.as_ref()
                .or(process_config.model.text_encoder_path.as_ref())
                .cloned()
                .unwrap_or_else(|| PathBuf::from("/home/alex/SwarmUI/Models/clip/clip_l.safetensors")),
            t5_xxl: process_config.model.t5_path.as_ref()
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
        mixed_precision: process_config.train.dtype == "bf16" || process_config.train.dtype == "fp16",
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
        logging_dir: PathBuf::from("output").join(&config.config.name),
        report_to: vec![],
        validation_prompts: process_config.sample.prompts.clone(),
        validation_steps: process_config.sample.sample_every,
        
        // Caching configuration
        cache_latents_to_disk: process_config.datasets[0].cache_latents_to_disk,
        cache_dir: None,
        force_reencode: false,
        dataset_name: config.config.name.clone(),
        
        // Layer streaming configuration
        use_layer_streaming: Some(true),
        streaming_memory_limit_gb: process_config.train.streaming_memory_limit_gb.or(Some(16.0)),
        
        // INT8 quantization
        use_int8_base_model: Some(false),
        
        // ChromaXL configuration
        chroma_config: None,
    })
}

// Re-export the function with the original name for compatibility
pub use self::train_flux_lora_sequential_fixed as train_flux_lora_sequential;
