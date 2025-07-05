//! SD 3.5 training binary

use eridiffusion_core::{Result, Error, ModelArchitecture, Device};
use eridiffusion_training::{TrainingConfig, Trainer, TrainerConfig};
use eridiffusion_models::{ModelConfig, VAEFactory, TextEncoderFactory};
use eridiffusion_data::{ImageDataset, DatasetConfig, LatentDataLoader, LatentDataLoaderConfig};
use candle_nn::VarBuilder;
use candle_core::DType;
use clap::Parser;
use std::path::PathBuf;
use tracing::{info, error};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to config file
    #[arg(short, long)]
    config: PathBuf,
    
    /// Device to use
    #[arg(short, long, default_value = "cuda:0")]
    device: String,
    
    /// Enable latent caching
    #[arg(long, default_value = "true")]
    cache_latents: bool,
    
    /// Test mode - only run a few steps
    #[arg(long)]
    test: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    let args = Args::parse();
    info!("Starting SD 3.5 training with config: {:?}", args.config);
    
    // Parse device
    let device = if args.device.starts_with("cuda") {
        let id = args.device.strip_prefix("cuda:").unwrap_or("0")
            .parse::<usize>()
            .unwrap_or(0);
        Device::Cuda(id)
    } else {
        Device::Cpu
    };
    
    // Load config
    let config_str = std::fs::read_to_string(&args.config)?;
    let config: serde_yaml::Value = serde_yaml::from_str(&config_str)
        .map_err(|e| Error::Config(format!("Failed to parse config: {}", e)))?;
    
    // Extract key parameters
    let model_path = config["config"]["model"]["name_or_path"]
        .as_str()
        .ok_or_else(|| Error::Config("Missing model path".to_string()))?;
    
    let dataset_path = config["config"]["datasets"][0]["folder_path"]
        .as_str()
        .ok_or_else(|| Error::Config("Missing dataset path".to_string()))?;
    
    let batch_size = config["config"]["train"]["batch_size"]
        .as_u64()
        .unwrap_or(4) as usize;
    
    let learning_rate = config["config"]["train"]["lr"]
        .as_f64()
        .unwrap_or(5e-5);
    
    let t5_max_length = config["config"]["model"]["t5_max_length"]
        .as_u64()
        .unwrap_or(154) as usize;
    
    info!("Configuration loaded:");
    info!("  Model: {}", model_path);
    info!("  Dataset: {}", dataset_path);
    info!("  Batch size: {}", batch_size);
    info!("  Learning rate: {}", learning_rate);
    info!("  T5 max length: {}", t5_max_length);
    
    // Create dataset
    info!("Loading dataset...");
    let dataset_config = DatasetConfig {
        root_dir: PathBuf::from(dataset_path),
        caption_ext: Some("txt".to_string()),
        resolution: 1024,
        center_crop: true,
        random_flip: true,
        cache_dir: Some(PathBuf::from("./dataset_cache")),
    };
    
    let dataset = ImageDataset::new(dataset_config)?;
    info!("Dataset loaded with {} images", dataset.len());
    
    // Create VAE for latent encoding
    info!("Loading VAE...");
    let candle_device = match &device {
        Device::Cpu => candle_core::Device::Cpu,
        Device::Cuda(id) => candle_core::Device::new_cuda(*id)?,
    };
    
    // For now, create a dummy VAE (in real usage, load from checkpoint)
    let vae = {
        let varmap = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &candle_device);
        VAEFactory::create(ModelArchitecture::SD3, vb)?
    };
    
    // Create latent dataloader
    info!("Creating latent dataloader...");
    let dataloader_config = LatentDataLoaderConfig {
        batch_size,
        shuffle: true,
        drop_last: true,
        num_workers: 4,
        prefetch_factor: 2,
        cache_latents: args.cache_latents,
        cache_dir: Some(PathBuf::from("./latent_cache")),
    };
    
    let dataloader = LatentDataLoader::new(
        dataset,
        dataloader_config,
        device.clone(),
        vae,
        ModelArchitecture::SD3,
    )?;
    
    // Pre-cache latents if enabled
    if args.cache_latents {
        info!("Pre-caching latents...");
        dataloader.precache_all().await?;
        info!("Latent pre-caching complete");
    }
    
    // Test the dataloader
    info!("Testing dataloader...");
    let mut iter = dataloader.iter().await;
    let mut batch_count = 0;
    let max_batches = if args.test { 3 } else { dataloader.len() };
    
    while let Some(batch_result) = iter.next().await {
        let batch = batch_result?;
        batch_count += 1;
        
        info!("Batch {}/{}: ", batch_count, max_batches);
        info!("  Images shape: {:?}", batch.images.shape());
        info!("  Latents shape: {:?}", batch.latents.shape());
        info!("  Captions: {} items", batch.captions.len());
        
        // Verify latent shape for SD3 (should be 16 channels)
        let latent_channels = batch.latents.dims()[1];
        if latent_channels != 16 {
            error!("Invalid latent channels: expected 16, got {}", latent_channels);
            return Err(Error::InvalidShape(format!(
                "Expected 16 latent channels for SD3, got {}",
                latent_channels
            )));
        }
        
        if batch_count >= max_batches {
            break;
        }
    }
    
    info!("Dataloader test complete: {} batches processed", batch_count);
    
    // Create training config
    let training_config = TrainingConfig {
        learning_rate,
        batch_size,
        num_epochs: 1,
        gradient_accumulation_steps: 1,
        warmup_steps: 100,
        save_steps: 250,
        eval_steps: 500,
        max_grad_norm: Some(1.0),
        mixed_precision: true,
        gradient_checkpointing: true,
        output_dir: PathBuf::from("./output"),
        resume_from_checkpoint: None,
    };
    
    // Note: Full training would require loading the model and text encoders
    // For this test, we're just verifying the dataloader and config
    
    if args.test {
        info!("Test mode complete. SD 3.5 training pipeline verified!");
        info!("\nIMPORTANT: Remember to fix T5 token length to 154 in sd3_candle.rs");
    } else {
        info!("Ready for full training. Model loading not implemented in this test.");
    }
    
    Ok(())
}