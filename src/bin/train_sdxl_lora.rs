//! Complete SDXL LoRA training script with gradient tracking
//! 
//! Usage: cargo run --bin train_sdxl_lora -- /path/to/config.yaml

use anyhow::{Result, Context};
use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about = "Train SDXL LoRA with EriDiffusion")]
struct Args {
    /// Path to the training configuration YAML file
    config: PathBuf,
    
    /// Resume from checkpoint
    #[arg(long)]
    resume: Option<PathBuf>,
    
    /// Override learning rate
    #[arg(long)]
    lr: Option<f32>,
    
    /// Override batch size
    #[arg(long)]
    batch_size: Option<usize>,
    
    /// Device to use (cuda:0, cuda:1, cpu)
    #[arg(long, default_value = "cuda:0")]
    device: String,
}

fn main() -> Result<()> {
    // Parse command line arguments
    let args = Args::parse();
    
    // Initialize logging
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_secs()
        .init();
    
    println!("\n=== EriDiffusion SDXL LoRA Trainer ===\n");
    println!("Config: {}", args.config.display());
    println!("Device: {}", args.device);
    
    if let Some(resume) = &args.resume {
        println!("Resuming from: {}", resume.display());
    }
    
    // Load configuration
    let config = eridiffusion::trainers::load_config(&args.config)?;
    
    // Override config with command line args if provided
    let mut config = config;
    if let Some(lr) = args.lr {
        println!("Overriding learning rate: {}", lr);
        config.learning_rate = lr;
    }
    if let Some(batch_size) = args.batch_size {
        println!("Overriding batch size: {}", batch_size);
        config.batch_size = batch_size;
    }
    
    // Create and run trainer
    let mut trainer = eridiffusion::trainers::sdxl_lora_trainer_fixed::SDXLLoRATrainerFixed::new(
        &config,
        &config.config.process[0],
    )?;
    
    // Load models
    trainer.load_models()?;
    
    // Start training
    trainer.train()?;
    
    println!("\n=== Training Complete! ===\n");
    
    Ok(())
}