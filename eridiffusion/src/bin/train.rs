//! Main training binary for EriDiffusion
//! 
//! Supports training various diffusion models with different network adapters

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use eridiffusion::trainers;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
#[command(propagate_version = true)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
    
    /// Configuration file path (for backwards compatibility)
    #[arg(long, short)]
    config: Option<PathBuf>,
}

#[derive(Subcommand)]
enum Commands {
    /// Train a model using a configuration file
    Train {
        /// Path to the configuration file
        #[arg(value_name = "CONFIG")]
        config: PathBuf,
    },
    /// List available models
    List,
    /// Show version information
    Version,
}

fn main() -> Result<()> {
    // Initialize logging
    if std::env::var("RUST_LOG").is_err() {
        std::env::set_var("RUST_LOG", "info");
    }
    env_logger::init();
    
    let cli = Cli::parse();
    
    // Handle commands or fallback to config arg
    match cli.command {
        Some(Commands::Train { config }) => {
            train_model(config)?;
        }
        Some(Commands::List) => {
            list_models();
        }
        Some(Commands::Version) => {
            println!("EriDiffusion v{}", env!("CARGO_PKG_VERSION"));
            println!("Pure Rust diffusion model trainer");
        }
        None => {
            // Check for config argument (backwards compatibility)
            if let Some(config) = cli.config {
                train_model(config)?;
            } else {
                eprintln!("Usage: train --config <CONFIG>");
                eprintln!("       train train <CONFIG>");
                eprintln!("\nExample: train --config flux_lora_example.yaml");
                std::process::exit(1);
            }
        }
    }
    
    Ok(())
}

fn train_model(config_path: PathBuf) -> Result<()> {
    // Check if file exists
    if !config_path.exists() {
        anyhow::bail!("Config file not found: {}", config_path.display());
    }
    
    println!("Loading configuration from: {}", config_path.display());
    
    // Run the trainer with automatic model detection and routing
    trainers::train_from_config(config_path)?;
    
    Ok(())
}

fn list_models() {
    println!("Supported models:");
    println!("  - SD 3.5 (Medium/Large/Large-Turbo)");
    println!("  - SDXL");
    println!("  - Flux (Dev/Schnell)");
    println!("\nSupported network types:");
    println!("  - LoRA (Low-Rank Adaptation)");
    println!("  - LoKr (Low-Rank Kronecker)");
    println!("  - DoRA (coming soon)");
    println!("  - LoCoN (coming soon)");
}