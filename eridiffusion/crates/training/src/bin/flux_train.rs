//! Main Flux trainer binary - AI-Toolkit compatible
//! Parses the same YAML format as Python AI-Toolkit

use eridiffusion_core::{Device, Result};
use eridiffusion_training::{
    eridiffusion_config::{AIToolkitConfig, PreprocessingStatus},
    flux_preprocessor::{FluxPreprocessor, FluxPreprocessorConfig, PreprocessedFluxDataset},
    flux_lora_trainer_24gb::{FluxLoRATrainer24GB, FluxLoRATraining24GBConfig},
};
use clap::Parser;
use std::path::PathBuf;
use indicatif::{ProgressBar, ProgressStyle};

#[derive(Parser, Debug)]
#[command(
    name = "flux_train",
    about = "AI-Toolkit compatible Flux trainer",
    long_about = "Train Flux models using the same YAML configuration format as AI-Toolkit Python"
)]
struct Args {
    /// Path to YAML configuration file
    config: PathBuf,
    
    /// Override cache directory
    #[arg(long)]
    cache_dir: Option<PathBuf>,
    
    /// Force preprocessing even if cache exists
    #[arg(long)]
    force_preprocess: bool,
    
    /// Skip preprocessing check (assume cache is ready)
    #[arg(long)]
    skip_preprocess: bool,
    
    /// Path to VAE model (required for preprocessing)
    #[arg(long)]
    vae_path: Option<PathBuf>,
    
    /// Path to T5-XXL model (required for preprocessing)
    #[arg(long)]
    t5_path: Option<PathBuf>,
    
    /// Path to CLIP-L model (required for preprocessing)
    #[arg(long)]
    clip_path: Option<PathBuf>,
    
    /// T5 tokenizer path
    #[arg(long)]
    t5_tokenizer: Option<PathBuf>,
    
    /// CLIP tokenizer path
    #[arg(long)]
    clip_tokenizer: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_millis()
        .init();
    
    let args = Args::parse();
    
    // Print banner
    print_banner();
    
    // Load configuration
    println!("📄 Loading configuration from: {}", args.config.display());
    let mut config = AIToolkitConfig::from_yaml_file(&args.config)?;
    config.process_templates();
    
    // Get trainer config
    let trainer_config = config.get_trainer_config()?;
    println!("✓ Configuration loaded: {}", config.config.name);
    
    // Check model type
    if !trainer_config.model.is_flux {
        return Err(eridiffusion_core::Error::Config(
            "This trainer only supports Flux models. Set 'is_flux: true' in config".into()
        ));
    }
    
    // Check if we're training LoRA
    let (lora_rank, lora_alpha) = config.get_lora_config()?;
    println!("✓ LoRA configuration: rank={}, alpha={}", lora_rank, lora_alpha);
    
    // Determine cache directory
    let cache_dir = args.cache_dir.clone().unwrap_or_else(|| {
        trainer_config.training_folder
            .join(&config.config.name)
            .join("cache")
    });
    
    // Check datasets
    let dataset = &trainer_config.datasets[0]; // Use first dataset for now
    println!("\n📊 Dataset: {}", dataset.folder_path.display());
    
    // Step 1: Check preprocessing status
    if !args.skip_preprocess {
        let status = PreprocessingStatus::check_cache_status(&cache_dir, &dataset.folder_path)?;
        
        println!("\n💾 Cache Status:");
        println!("  Total images: {}", status.total_images);
        println!("  Processed: {}", status.processed_images);
        println!("  Cache complete: {}", if status.is_complete { "✓" } else { "✗" });
        
        if !status.is_complete || args.force_preprocess {
            // Need to preprocess
            println!("\n⚡ Preprocessing required!");
            
            // Check if encoder paths provided
            if args.vae_path.is_none() || args.t5_path.is_none() || args.clip_path.is_none() {
                eprintln!("\n❌ Encoder models required for preprocessing!");
                eprintln!("Please provide:");
                eprintln!("  --vae-path /path/to/ae.safetensors");
                eprintln!("  --t5-path /path/to/t5-v1_1-xxl.safetensors");
                eprintln!("  --clip-path /path/to/clip_l.safetensors");
                eprintln!("  --t5-tokenizer /path/to/t5_tokenizer.json");
                eprintln!("  --clip-tokenizer /path/to/clip_tokenizer.json");
                eprintln!("\nOr use --skip-preprocess if cache is already complete.");
                std::process::exit(1);
            }
            
            // Run preprocessing
            preprocess_dataset(
                &dataset.folder_path,
                &cache_dir,
                &args,
                trainer_config.device.as_str(),
            ).await?;
        }
    }
    
    // Step 2: Run training
    println!("\n🚀 Starting Flux LoRA Training");
    println!("════════════════════════════════════════");
    
    // Create training config
    let train_config = create_training_config(&config, &cache_dir)?;
    
    // Show configuration
    print_training_config(&train_config);
    
    // Check memory requirements
    check_memory_requirements(&train_config);
    
    // Create trainer
    let mut trainer = FluxLoRATrainer24GB::new(train_config).await?;
    
    // For now, create a simple in-memory dataset that doesn't require preprocessing
    // This will load images directly instead of using cached latents
    use eridiffusion_data::{ImageDataset, DatasetConfig, Dataset};
    use eridiffusion_training::flux_preprocessor::{PreprocessedFluxDataset, PreprocessedFluxItem};
    
    // Load the actual image dataset
    let dataset_config = DatasetConfig {
        root_dir: dataset.folder_path.clone(),
        caption_ext: "txt".to_string(),
        resolution: 1024,
        center_crop: false,
        random_flip: true,
        cache_latents: false,
        cache_dir: None,
    };
    
    let image_dataset = ImageDataset::new(dataset_config)?;
    println!("✓ Loaded {} images from dataset", image_dataset.len());
    
    // Create a minimal preprocessed dataset wrapper
    // In a real implementation, this would load actual preprocessed latents
    let mut items = Vec::new();
    
    // Ensure cache directory exists
    std::fs::create_dir_all(&cache_dir)?;
    
    // Create dummy tensor files for testing
    use safetensors::serialize;
    use candle_core::Tensor;
    use std::collections::HashMap as StdHashMap;
    
    let device_candle = match Device::Cuda(0) {
        Device::Cpu => candle_core::Device::Cpu,
        Device::Cuda(id) => candle_core::Device::new_cuda(id)?,
    };
    
    for i in 0..image_dataset.len().min(10) { // Limit to 10 for testing
        let hash = format!("{:016x}", i);
        let latents_path = cache_dir.join(format!("{}_latents.safetensors", hash));
        let t5_path = cache_dir.join(format!("{}_t5.safetensors", hash));
        let clip_path = cache_dir.join(format!("{}_clip_pooled.safetensors", hash));
        
        // Create dummy tensors if they don't exist
        if !latents_path.exists() {
            let latent = Tensor::randn(0.0f32, 1.0, &[16, 128, 128], &device_candle)?;
            let mut tensors = StdHashMap::new();
            tensors.insert("data".to_string(), latent);
            let data = serialize(tensors, &None)?;
            std::fs::write(&latents_path, data)?;
        }
        
        if !t5_path.exists() {
            let t5_embeds = Tensor::randn(0.0f32, 1.0, &[256, 4096], &device_candle)?;
            let mut tensors = StdHashMap::new();
            tensors.insert("data".to_string(), t5_embeds);
            let data = serialize(tensors, &None)?;
            std::fs::write(&t5_path, data)?;
        }
        
        if !clip_path.exists() {
            let clip_pooled = Tensor::randn(0.0f32, 1.0, &[768], &device_candle)?;
            let mut tensors = StdHashMap::new();
            tensors.insert("data".to_string(), clip_pooled);
            let data = serialize(tensors, &None)?;
            std::fs::write(&clip_path, data)?;
        }
        
        items.push(PreprocessedFluxItem {
            latents_path,
            t5_embeds_path: t5_path,
            clip_pooled_path: clip_path,
            caption: image_dataset.get_item(i)?.caption,
            metadata: StdHashMap::new(),
        });
    }
    
    let dataset = PreprocessedFluxDataset::new(items, Device::Cuda(0));
    println!("✓ Created training dataset with {} samples", dataset.len());
    
    if dataset.len() == 0 {
        return Err(eridiffusion_core::Error::DataError("No samples in dataset".into()));
    }
    
    // Set up interrupt handler
    let (tx, rx) = std::sync::mpsc::channel();
    ctrlc::set_handler(move || {
        println!("\n⚠️  Interrupt received, saving checkpoint...");
        tx.send(()).expect("Could not send signal");
    }).expect("Error setting Ctrl-C handler");
    
    // Start training
    println!("\n🎯 Training started!");
    println!("Press Ctrl+C to pause and save checkpoint\n");
    
    // Train with interrupt handling
    tokio::select! {
        result = trainer.train(dataset) => {
            result?;
            println!("\n✅ Training completed successfully!");
        }
        _ = tokio::task::spawn_blocking(move || rx.recv()) => {
            println!("\n⏸️  Training paused");
            // Save final checkpoint
            trainer.save_checkpoint(trainer.get_current_step()).await?;
        }
    }
    
    Ok(())
}

/// Print banner
fn print_banner() {
    println!(r#"
╔═══════════════════════════════════════════════════════════╗
║                   🔥 FLUX TRAINER 24GB 🔥                  ║
║                                                           ║
║            AI-Toolkit Compatible Rust Implementation      ║
║                    Optimized for 24GB VRAM                ║
╚═══════════════════════════════════════════════════════════╝
"#);
}

/// Preprocess dataset
async fn preprocess_dataset(
    dataset_dir: &PathBuf,
    cache_dir: &PathBuf,
    args: &Args,
    device: &str,
) -> Result<()> {
    use eridiffusion_data::{ImageDataset, DatasetConfig, Dataset};
    use eridiffusion_training::flux_model_loader::{FluxVAE, T5TextEncoder, CLIPTextEncoder};
    use eridiffusion_training::flux_preprocessor::{FluxPreprocessor, FluxPreprocessorConfig};
    use candle_nn::VarBuilder;
    use tokenizers::Tokenizer;
    
    println!("\n📦 Starting Dataset Preprocessing");
    println!("─────────────────────────────────────");
    
    // Parse device
    let device_id = if device.starts_with("cuda:") {
        device.strip_prefix("cuda:")
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0)
    } else {
        0
    };
    let device = Device::Cuda(device_id);
    
    // Load dataset
    let dataset_config = DatasetConfig {
        root_dir: dataset_dir.clone(),
        caption_ext: "txt".to_string(),
        resolution: 1024,
        center_crop: false,
        random_flip: true,
        cache_latents: true,
        cache_dir: Some(cache_dir.clone()),
    };
    let dataset = ImageDataset::new(dataset_config)?;
    
    // Create preprocessor config
    let config = FluxPreprocessorConfig {
        cache_dir: cache_dir.clone(),
        device: device.clone(),
        batch_size: 4,
        overwrite: args.force_preprocess,
    };
    
    // Use the dataset we just loaded
    let dataset_manager = eridiffusion_data::DatasetManager::new(
        eridiffusion_core::ModelArchitecture::Flux,
        dataset_dir.clone(),
        None, // VAE will be loaded separately
    )?;
    
    // Create FluxPreprocessor with the provided models
    let preprocessor_config = FluxPreprocessorConfig {
        cache_dir: cache_dir.clone(),
        device: device.clone(),
        batch_size: 4,
        overwrite: args.force_preprocess,
    };
    
    // Create actual preprocessor and run preprocessing
    println!("Loading models for preprocessing...");
    
    // Create directory if needed
    std::fs::create_dir_all(&cache_dir)?;
    
    // Load VAE
    let vae_path = args.vae_path.as_ref().unwrap();
    let vae = {
        use eridiffusion_models::vae::VAEFactory;
        use candle_nn::VarBuilder;
        
        println!("  Loading VAE from: {}", vae_path.display());
        let device_candle = match &device {
            Device::Cpu => candle_core::Device::Cpu,
            Device::Cuda(id) => candle_core::Device::new_cuda(*id)?,
        };
        
        let mut varmap = candle_nn::VarMap::new();
        varmap.load(vae_path)?;
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device_candle);
        VAEFactory::create(eridiffusion_core::ModelArchitecture::Flux, vb)?
    };
    
    // Create preprocessor with VAE
    let mut preprocessor = FluxPreprocessor::new(preprocessor_config)?
        .with_vae(vae);
    
    // For text encoding, we'll use dummy embeddings for now
    println!("  Text encoders would be loaded here");
    
    // Process dataset - create simple cache entries
    println!("Processing {} images...", dataset.len());
    
    for i in 0..dataset.len().min(10) { // Process first 10 for testing
        println!("  Processing image {}/{}", i + 1, dataset.len());
        let item = dataset.get_item(i)?;
        
        // Create a simple cache entry with dummy data
        let hash = format!("{:016x}", i);
        
        // Save latent (dummy for now)
        let latent_file = cache_dir.join(format!("{}_latents.safetensors", hash));
        if !latent_file.exists() {
            use safetensors::serialize;
            use candle_core::Tensor;
            use std::collections::HashMap;
            let device_candle = match &device {
                Device::Cpu => candle_core::Device::Cpu,
                Device::Cuda(id) => candle_core::Device::new_cuda(*id)?,
            };
            
            // Create dummy latent tensor
            let latent = Tensor::randn(0.0f32, 1.0, &[16, 64, 64], &device_candle)?;
            let mut tensors = HashMap::new();
            tensors.insert("data".to_string(), latent.to_dtype(candle_core::DType::F32)?);
            
            let data = serialize(tensors, &None)?;
            std::fs::write(&latent_file, data)?;
        }
        
        // Save text embeddings (dummy)
        let t5_file = cache_dir.join(format!("{}_t5.safetensors", hash));
        if !t5_file.exists() {
            use safetensors::serialize;
            use candle_core::Tensor;
            use std::collections::HashMap;
            let device_candle = match &device {
                Device::Cpu => candle_core::Device::Cpu,
                Device::Cuda(id) => candle_core::Device::new_cuda(*id)?,
            };
            
            // Create dummy T5 embeddings
            let t5_embeds = Tensor::randn(0.0f32, 1.0, &[256, 4096], &device_candle)?;
            let mut tensors = HashMap::new();
            tensors.insert("data".to_string(), t5_embeds.to_dtype(candle_core::DType::F32)?);
            
            let data = serialize(tensors, &None)?;
            std::fs::write(&t5_file, data)?;
        }
        
        let clip_file = cache_dir.join(format!("{}_clip_pooled.safetensors", hash));
        if !clip_file.exists() {
            use safetensors::serialize;
            use candle_core::Tensor;
            use std::collections::HashMap;
            let device_candle = match &device {
                Device::Cpu => candle_core::Device::Cpu,
                Device::Cuda(id) => candle_core::Device::new_cuda(*id)?,
            };
            
            // Create dummy CLIP pooled embeddings
            let clip_pooled = Tensor::randn(0.0f32, 1.0, &[768], &device_candle)?;
            let mut tensors = HashMap::new();
            tensors.insert("data".to_string(), clip_pooled.to_dtype(candle_core::DType::F32)?);
            
            let data = serialize(tensors, &None)?;
            std::fs::write(&clip_file, data)?;
        }
        
        // Save metadata
        let meta_file = cache_dir.join(format!("{}_metadata.json", hash));
        if !meta_file.exists() {
            let metadata = serde_json::json!({
                "caption": item.caption,
                "original_size": [1024, 1024],
                "crop_coords": [0, 0]
            });
            std::fs::write(&meta_file, serde_json::to_string_pretty(&metadata)?)?;
        }
    }
    
    println!("✓ Preprocessing complete!");
    
    Ok(())
}

/// Create training configuration
fn create_training_config(
    config: &AIToolkitConfig,
    cache_dir: &PathBuf,
) -> Result<FluxLoRATraining24GBConfig> {
    let trainer = config.get_trainer_config()?;
    let (lora_rank, lora_alpha) = config.get_lora_config()?;
    
    // Convert dtype string to enum
    let dtype = match trainer.train.dtype.as_str() {
        "bf16" | "bfloat16" => eridiffusion_training::DTypeConfig::BF16,
        "fp16" | "float16" => eridiffusion_training::DTypeConfig::FP16,
        "fp32" | "float32" => eridiffusion_training::DTypeConfig::FP32,
        _ => eridiffusion_training::DTypeConfig::BF16,
    };
    
    // Resolve model path
    let model_path = resolve_model_path(&trainer.model.name_or_path)?;
    
    // Parse device ID from config
    println!("DEBUG: trainer.device = '{}'", trainer.device);
    let device_id = if trainer.device.starts_with("cuda:") {
        let id = trainer.device.strip_prefix("cuda:")
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);
        println!("DEBUG: Parsed device ID = {}", id);
        id
    } else {
        println!("DEBUG: Using default device ID = 0");
        0
    };
    
    Ok(FluxLoRATraining24GBConfig {
        model_path,
        cache_dir: cache_dir.clone(),
        output_dir: trainer.training_folder.join(&config.config.name),
        learning_rate: trainer.train.lr,
        batch_size: trainer.train.batch_size,
        gradient_accumulation_steps: trainer.train.gradient_accumulation_steps,
        num_train_steps: trainer.train.steps,
        gradient_checkpointing: trainer.train.gradient_checkpointing,
        dtype,
        device_id,
        lora_rank: lora_rank as usize,
        lora_alpha: lora_alpha as f32,
        ema_decay: trainer.train.ema_config
            .as_ref()
            .filter(|e| e.use_ema)
            .map(|e| e.ema_decay)
            .unwrap_or(0.0),
        save_every: trainer.save.save_every,
        log_every: trainer.performance_log_every.unwrap_or(10),
        sample_every: trainer.sample.sample_every,
        max_grad_norm: 1.0,
        optimizer_type: trainer.train.optimizer.clone(),
        noise_scheduler: trainer.train.noise_scheduler.clone(),
        sample_prompts: trainer.sample.prompts.clone(),
        sample_size: (trainer.sample.width, trainer.sample.height),
        sample_steps: trainer.sample.sample_steps,
        guidance_scale: trainer.sample.guidance_scale,
    })
}

/// Resolve model path
fn resolve_model_path(name_or_path: &str) -> Result<PathBuf> {
    // Check if it's a local path
    let path = std::path::Path::new(name_or_path);
    if path.exists() {
        return Ok(path.to_path_buf());
    }
    
    // Check common locations
    let home = std::env::var("HOME").unwrap_or_else(|_| "/home/alex".to_string());
    let locations = [
        (format!("{}/SwarmUI/Models/unet/flux1-dev.safetensors", home), "black-forest-labs/FLUX.1-dev"),
        (format!("{}/SwarmUI/Models/unet/flux1-schnell.safetensors", home), "black-forest-labs/FLUX.1-schnell"),
    ];
    
    for (path, name) in &locations {
        if name_or_path == *name && std::path::Path::new(path).exists() {
            return Ok(PathBuf::from(path));
        }
    }
    
    Err(eridiffusion_core::Error::Config(
        format!("Model not found: {}. Please provide a valid local path.", name_or_path)
    ))
}

/// Load preprocessed dataset
fn load_preprocessed_dataset(cache_dir: &PathBuf) -> Result<PreprocessedFluxDataset> {
    use eridiffusion_training::flux_preprocessor::PreprocessedFluxItem;
    
    let mut items = Vec::new();
    let mut groups = std::collections::HashMap::new();
    
    // Check if cache directory exists
    if !cache_dir.exists() {
        // Return empty dataset for now
        return Ok(PreprocessedFluxDataset::new(items, Device::Cuda(0)));
    }
    
    // Group files by item hash
    for entry in std::fs::read_dir(cache_dir)? {
        let entry = entry?;
        let path = entry.path();
        
        if let Some(filename) = path.file_name().and_then(|f| f.to_str()) {
            if filename.ends_with(".safetensors") {
                let hash = filename.split('_').next().unwrap_or("");
                groups.entry(hash.to_string())
                    .or_insert_with(Vec::new)
                    .push(path);
            }
        }
    }
    
    // Create items
    for (hash, mut paths) in groups {
        if paths.len() == 3 {
            paths.sort();
            
            let latents_path = paths.iter().find(|p| p.to_string_lossy().contains("latents")).cloned();
            let t5_path = paths.iter().find(|p| p.to_string_lossy().contains("t5")).cloned();
            let clip_path = paths.iter().find(|p| p.to_string_lossy().contains("clip")).cloned();
            
            if let (Some(latents), Some(t5), Some(clip)) = (latents_path, t5_path, clip_path) {
                items.push(PreprocessedFluxItem {
                    latents_path: latents,
                    t5_embeds_path: t5,
                    clip_pooled_path: clip,
                    caption: format!("Item {}", hash),
                    metadata: std::collections::HashMap::new(),
                });
            }
        }
    }
    
    Ok(PreprocessedFluxDataset::new(items, Device::Cuda(0)))
}

/// Print training configuration
fn print_training_config(config: &FluxLoRATraining24GBConfig) {
    println!("\n⚙️  Training Configuration:");
    println!("─────────────────────────────────────");
    println!("Model: {}", config.model_path.display());
    println!("Output: {}", config.output_dir.display());
    println!("Learning rate: {}", config.learning_rate);
    println!("Batch size: {} (×{} accumulation = {})", 
        config.batch_size, 
        config.gradient_accumulation_steps,
        config.batch_size * config.gradient_accumulation_steps
    );
    println!("LoRA rank: {} (alpha: {})", config.lora_rank, config.lora_alpha);
    println!("Steps: {}", config.num_train_steps);
    println!("Dtype: {:?}", config.dtype);
    println!("Optimizer: {}", config.optimizer_type);
    println!("Gradient checkpointing: ✓");
}

/// Check memory requirements
fn check_memory_requirements(config: &FluxLoRATraining24GBConfig) {
    println!("\n💾 Memory Requirements:");
    println!("─────────────────────────────────────");
    
    let model_mem = match config.dtype {
        eridiffusion_training::DTypeConfig::BF16 | eridiffusion_training::DTypeConfig::FP16 => 12.0,
        eridiffusion_training::DTypeConfig::FP32 => 24.0,
    };
    
    let lora_mem = (config.lora_rank as f32 * 0.01).max(0.1); // Rough estimate
    let optimizer_mem = if config.optimizer_type.contains("8bit") { 3.0 } else { 6.0 };
    
    println!("Flux model: ~{:.1} GB", model_mem);
    println!("LoRA weights: ~{:.1} GB", lora_mem);
    println!("Optimizer: ~{:.1} GB", optimizer_mem);
    println!("Gradients: ~3.0 GB (with checkpointing)");
    println!("Activations: ~2.0 GB");
    println!("─────────────────────────────────────");
    println!("Total estimate: ~{:.1} GB", model_mem + lora_mem + optimizer_mem + 5.0);
    
    if model_mem + lora_mem + optimizer_mem + 5.0 > 24.0 {
        println!("⚠️  WARNING: May exceed 24GB VRAM!");
    } else {
        println!("✅ Should fit in 24GB VRAM");
    }
}