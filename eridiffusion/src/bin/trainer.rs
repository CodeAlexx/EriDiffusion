use flame_core::Error;
use log::info;
use std::{env, path::PathBuf};

fn main() -> flame_core::Result<()> {
    // Initialize logging
    let _ = eridiffusion::logging::init_logger();

    // Initialize core device manager + plugins to avoid runtime panics
    eridiffusion_core::initialize().map_err(|e| Error::InvalidOperation(e.to_string()))?;

    // Get config path from args
    let args: Vec<String> = env::args().collect();
    eprintln!("DEBUG: args.len() = {}", args.len());
    eprintln!("DEBUG: args = {:?}", args);
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
