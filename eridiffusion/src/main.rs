use anyhow::Result;
use std::env;
use std::path::PathBuf;

mod trainers;

fn main() -> Result<()> {
    // Simple unified trainer - just takes a config file as argument
    let args: Vec<String> = env::args().collect();
    
    let config_path = if args.len() > 1 {
        PathBuf::from(&args[1])
    } else {
        return Err(anyhow::anyhow!(
            "Usage: trainer <config.yaml>\nExample: trainer config/eri1024.yaml"
        ));
    };
    
    println!("\n=== Unified Diffusion Model Trainer ===");
    println!("Loading config: {}", config_path.display());
    
    // Dispatch to the appropriate trainer based on config
    trainers::train_from_config(config_path)?;
    
    Ok(())
}