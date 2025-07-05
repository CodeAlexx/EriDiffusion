//! Flux training binary

use eridiffusion_core::{Device, Result};
use eridiffusion_training::{
    flux_trainer::{FluxTrainer, FluxTrainingConfig, create_flux_trainer},
    flux_model_loader::FluxVariant,
    metrics::MetricsLogger,
};
use eridiffusion_data::{DataLoader, DataLoaderConfig, ImageDataset, DatasetConfig};
use clap::Parser;
use std::path::PathBuf;
use tokio;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to Flux model
    #[arg(long)]
    model_path: PathBuf,
    
    /// Path to VAE model
    #[arg(long)]
    vae_path: PathBuf,
    
    /// Path to T5 text encoder
    #[arg(long)]
    t5_path: PathBuf,
    
    /// Path to CLIP text encoder
    #[arg(long)]
    clip_path: PathBuf,
    
    /// Path to T5 tokenizer
    #[arg(long, default_value = "t5-v1_1-xxl.tokenizer.json")]
    t5_tokenizer_path: PathBuf,
    
    /// Path to CLIP tokenizer
    #[arg(long, default_value = "clip-vit-large-patch14-tokenizer.json")]
    clip_tokenizer_path: PathBuf,
    
    /// Flux variant (dev or schnell)
    #[arg(long, default_value = "schnell")]
    variant: String,
    
    /// Path to training data directory
    #[arg(long)]
    data_path: PathBuf,
    
    /// Output directory for checkpoints
    #[arg(long)]
    output_dir: PathBuf,
    
    /// Configuration file (optional)
    #[arg(long)]
    config: Option<PathBuf>,
    
    /// Learning rate
    #[arg(long, default_value = "1e-4")]
    learning_rate: f64,
    
    /// Batch size
    #[arg(long, default_value = "1")]
    batch_size: usize,
    
    /// Gradient accumulation steps
    #[arg(long, default_value = "4")]
    gradient_accumulation_steps: usize,
    
    /// Max training steps
    #[arg(long, default_value = "100000")]
    max_steps: usize,
    
    /// Save checkpoint every N steps
    #[arg(long, default_value = "1000")]
    save_steps: usize,
    
    /// Log metrics every N steps
    #[arg(long, default_value = "10")]
    log_steps: usize,
    
    /// Resume from checkpoint
    #[arg(long)]
    resume_from: Option<PathBuf>,
    
    /// Use mixed precision
    #[arg(long, action)]
    mixed_precision: bool,
    
    /// Use gradient checkpointing
    #[arg(long, action)]
    gradient_checkpointing: bool,
    
    /// GPU device ID
    #[arg(long, default_value = "0")]
    gpu_id: usize,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Parse arguments
    let args = Args::parse();
    
    // Set up logging
    env_logger::init();
    
    // Create device
    let device = if args.gpu_id == usize::MAX {
        Device::Cpu
    } else {
        Device::Cuda(args.gpu_id)
    };
    
    println!("Using device: {:?}", device);
    
    // Create configuration
    let mut config = if let Some(config_path) = args.config {
        // Load from file
        let config_str = std::fs::read_to_string(&config_path)?;
        serde_json::from_str(&config_str)?
    } else {
        FluxTrainingConfig::default()
    };
    
    // Override with command line arguments
    config.learning_rate = args.learning_rate;
    config.gradient_accumulation_steps = args.gradient_accumulation_steps;
    config.max_steps = args.max_steps;
    config.mixed_precision = args.mixed_precision;
    config.gradient_checkpointing = args.gradient_checkpointing;
    
    // Create output directory
    std::fs::create_dir_all(&args.output_dir)?;
    
    // Parse variant
    let variant = match args.variant.to_lowercase().as_str() {
        "dev" => FluxVariant::Dev,
        "schnell" => FluxVariant::Schnell,
        _ => {
            eprintln!("Invalid variant '{}', using 'schnell'", args.variant);
            FluxVariant::Schnell
        }
    };
    
    // Create trainer
    let mut trainer = create_flux_trainer(
        &args.model_path,
        &args.vae_path,
        &args.t5_path,
        &args.clip_path,
        &args.t5_tokenizer_path,
        &args.clip_tokenizer_path,
        variant,
        config.clone(),
        device.clone(),
    ).await?;
    
    // Resume from checkpoint if specified
    if let Some(checkpoint_path) = args.resume_from {
        trainer.load_checkpoint(&checkpoint_path)?;
        println!("Resumed from checkpoint at step {}", trainer.global_step);
    }
    
    // Create dataset
    let dataset_config = DatasetConfig {
        root_dir: args.data_path.clone(),
        resolution: 1024,
        center_crop: false,
        random_flip: true,
        caption_dropout: 0.1,
        cache_latents: true,
        cache_dir: Some(args.output_dir.join("cache")),
        validation_split: 0.05,
    };
    
    let dataset = ImageDataset::new(dataset_config)?;
    
    // Create dataloader
    let dataloader_config = DataLoaderConfig {
        batch_size: args.batch_size,
        shuffle: true,
        num_workers: 4,
        pin_memory: true,
        drop_last: true,
        prefetch_factor: 2,
    };
    
    let mut dataloader = DataLoader::new(dataset, dataloader_config)?;
    
    // Create metrics logger
    let mut metrics_logger = MetricsLogger::new(&args.output_dir)?;
    
    // Training loop
    println!("Starting Flux training...");
    println!("Total steps: {}", config.max_steps);
    println!("Batch size: {} (x{} accumulation = {} effective)",
        args.batch_size,
        config.gradient_accumulation_steps,
        args.batch_size * config.gradient_accumulation_steps
    );
    
    let mut step_timer = std::time::Instant::now();
    
    while trainer.global_step < config.max_steps {
        // Get batch
        let batch = dataloader.next_batch().await?;
        
        // Prepare negative prompts (empty strings for Flux)
        let negative_prompts = vec!["".to_string(); batch.captions.len()];
        
        // Training step
        let loss = trainer.train_step(
            &batch.images,
            &batch.captions,
            &negative_prompts,
        ).await?;
        
        // Log metrics
        if trainer.global_step % args.log_steps == 0 {
            let step_time = step_timer.elapsed().as_secs_f32();
            let samples_per_sec = args.batch_size as f32 / step_time;
            
            println!(
                "Step {}/{} | Loss: {:.4} | Time: {:.2}s | {:.2} samples/sec",
                trainer.global_step,
                config.max_steps,
                loss,
                step_time,
                samples_per_sec,
            );
            
            metrics_logger.log_scalar("train/loss", loss, trainer.global_step)?;
            metrics_logger.log_scalar("train/samples_per_second", samples_per_sec, trainer.global_step)?;
            
            step_timer = std::time::Instant::now();
        }
        
        // Save checkpoint
        if trainer.global_step % args.save_steps == 0 && trainer.global_step > 0 {
            let checkpoint_dir = args.output_dir.join(format!("checkpoint-{}", trainer.global_step));
            trainer.save_checkpoint(&checkpoint_dir)?;
            println!("Saved checkpoint at step {}", trainer.global_step);
        }
    }
    
    // Final checkpoint
    let final_checkpoint = args.output_dir.join("checkpoint-final");
    trainer.save_checkpoint(&final_checkpoint)?;
    println!("Training completed! Final checkpoint saved.");
    
    Ok(())
}