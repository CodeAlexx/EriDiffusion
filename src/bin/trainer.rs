use anyhow::Result;
use std::env;
use std::path::PathBuf;
use log::info;

fn main() -> Result<()> {
    // Initialize logging
    eridiffusion::logging::init_logger();
    
    // Get config path from args
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <config.yaml>", args[0]);
        eprintln!("Example: {} /home/alex/diffusers-rs/config/eri1024.yaml", args[0]);
        std::process::exit(1);
    }
    
    let config_path = PathBuf::from(&args[1]);
    
    // Check if file exists
    if !config_path.exists() {
        eprintln!("Error: Config file not found: {}", config_path.display());
        std::process::exit(1);
    }
    
    info!("Starting trainer with config: {}", config_path.display());
    
    // Run the trainer with automatic model detection and routing
    eridiffusion::trainers::train_from_config(config_path)?;
    
    info!("Training completed successfully");
    Ok(())
}