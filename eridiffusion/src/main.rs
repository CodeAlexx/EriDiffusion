use std::env;
use std::path::PathBuf;

use eridiffusion::trainers::train_from_config;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Simple unified trainer - just takes a config file as argument
    let args: Vec<String> = env::args().collect();

    let config_path = if args.len() > 1 {
        PathBuf::from(&args[1])
    } else {
        eprintln!("Usage: trainer <config.yaml>\nExample: trainer config/eri1024.yaml");
        return Err("Invalid arguments".into());
    };

    println!("\n=== Unified Diffusion Model Trainer ===");
    println!("Loading config: {}", config_path.display());

    // Dispatch to the appropriate trainer based on config
    train_from_config(config_path)?;

    Ok(())
}
