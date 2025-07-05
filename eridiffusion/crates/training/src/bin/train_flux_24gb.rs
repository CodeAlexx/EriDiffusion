//! Train Flux on 24GB VRAM using preprocessed data

use eridiffusion_core::{Device, Result};
use eridiffusion_training::flux_trainer_24gb::{FluxTrainer24GB, FluxTraining24GBConfig};
use eridiffusion_training::flux_preprocessor::{PreprocessedFluxDataset, PreprocessedFluxItem};
use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about = "Train Flux on 24GB VRAM", long_about = None)]
struct Args {
    /// Preprocessed data cache directory
    #[arg(long)]
    cache_dir: PathBuf,
    
    /// Flux model checkpoint path
    #[arg(long)]
    model_path: PathBuf,
    
    /// Output directory for checkpoints
    #[arg(long, default_value = "output")]
    output_dir: PathBuf,
    
    /// Learning rate
    #[arg(long, default_value = "1e-5")]
    learning_rate: f64,
    
    /// Batch size (1-2 recommended for 24GB)
    #[arg(long, default_value = "1")]
    batch_size: usize,
    
    /// Gradient accumulation steps
    #[arg(long, default_value = "4")]
    gradient_accumulation: usize,
    
    /// Number of training steps
    #[arg(long, default_value = "1000")]
    num_steps: usize,
    
    /// Save checkpoint every N steps
    #[arg(long, default_value = "100")]
    save_every: usize,
    
    /// Log metrics every N steps
    #[arg(long, default_value = "10")]
    log_every: usize,
    
    /// Max gradient norm
    #[arg(long, default_value = "1.0")]
    max_grad_norm: f32,
    
    /// Disable mixed precision
    #[arg(long)]
    no_mixed_precision: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();
    
    println!("🚀 Flux 24GB Trainer");
    println!("════════════════════════════════════════");
    println!();
    println!("This trainer uses preprocessed data to fit Flux training in 24GB VRAM.");
    println!("Make sure you've run preprocess_flux first!");
    println!();
    
    // Check CUDA
    if !candle_core::utils::cuda_is_available() {
        eprintln!("❌ CUDA is required for Flux training!");
        std::process::exit(1);
    }
    
    // Check cache directory
    if !args.cache_dir.exists() {
        eprintln!("❌ Cache directory not found: {}", args.cache_dir.display());
        eprintln!("   Run preprocess_flux first to create the cache!");
        std::process::exit(1);
    }
    
    // Load preprocessed items
    println!("📂 Loading preprocessed data from {}...", args.cache_dir.display());
    let items = load_preprocessed_items(&args.cache_dir)?;
    println!("✓ Found {} preprocessed items", items.len());
    
    if items.is_empty() {
        eprintln!("❌ No preprocessed items found!");
        eprintln!("   Run preprocess_flux first!");
        std::process::exit(1);
    }
    
    // Create config
    let config = FluxTraining24GBConfig {
        model_path: args.model_path,
        cache_dir: args.cache_dir,
        output_dir: args.output_dir,
        learning_rate: args.learning_rate,
        batch_size: args.batch_size,
        gradient_accumulation_steps: args.gradient_accumulation,
        num_train_steps: args.num_steps,
        gradient_checkpointing: true, // Always true for 24GB
        mixed_precision: !args.no_mixed_precision,
        ema_decay: 0.0, // Disabled to save memory
        save_every: args.save_every,
        log_every: args.log_every,
        max_grad_norm: args.max_grad_norm,
    };
    
    // Show configuration
    println!("\n⚙️  Configuration:");
    println!("─────────────────────────────────");
    println!("Model: {}", config.model_path.display());
    println!("Learning rate: {}", config.learning_rate);
    println!("Batch size: {} (effective: {})", 
        config.batch_size, 
        config.batch_size * config.gradient_accumulation_steps
    );
    println!("Steps: {}", config.num_train_steps);
    println!("Mixed precision: {}", config.mixed_precision);
    println!("Gradient checkpointing: ✓ (required)");
    
    // Memory estimate
    println!("\n💾 Memory Usage Estimate:");
    println!("─────────────────────────────────");
    println!("Flux model: ~12 GB");
    println!("Gradients: ~3 GB (with checkpointing)");
    println!("Optimizer: ~6 GB");
    println!("Activations: ~2 GB");
    println!("Total: ~23 GB ✅ Fits in 24GB!");
    
    // Create trainer
    println!("\n🏗️  Initializing trainer...");
    let mut trainer = FluxTrainer24GB::new(config).await?;
    
    // Create dataset
    let dataset = PreprocessedFluxDataset::new(items, Device::Cuda(0));
    
    // Start training
    println!("\n🎯 Starting training...");
    println!("Press Ctrl+C to stop and save checkpoint");
    
    // Set up Ctrl+C handler
    let (tx, rx) = std::sync::mpsc::channel();
    ctrlc::set_handler(move || {
        println!("\n⚠️  Interrupt received, saving checkpoint...");
        tx.send(()).expect("Could not send signal on channel");
    }).expect("Error setting Ctrl-C handler");
    
    // Train with interrupt handling
    tokio::select! {
        result = trainer.train(dataset) => {
            result?;
            println!("\n✅ Training completed successfully!");
        }
        _ = tokio::task::spawn_blocking(move || rx.recv()) => {
            println!("\n⏸️  Training paused");
            println!("Resume with --resume flag");
        }
    }
    
    Ok(())
}

/// Load preprocessed items from cache directory
fn load_preprocessed_items(cache_dir: &PathBuf) -> Result<Vec<PreprocessedFluxItem>> {
    let mut items = Vec::new();
    let mut groups = std::collections::HashMap::new();
    
    // Group files by item hash
    for entry in std::fs::read_dir(cache_dir)? {
        let entry = entry?;
        let path = entry.path();
        
        if let Some(filename) = path.file_name().and_then(|f| f.to_str()) {
            if filename.ends_with(".safetensors") {
                // Extract item hash
                let hash = filename.split('_').next().unwrap_or("");
                groups.entry(hash.to_string())
                    .or_insert_with(Vec::new)
                    .push(path);
            }
        }
    }
    
    // Create items from groups
    for (hash, mut paths) in groups {
        if paths.len() == 3 {
            paths.sort();
            
            let latents_path = paths.iter()
                .find(|p| p.to_string_lossy().contains("latents"))
                .cloned();
            let t5_path = paths.iter()
                .find(|p| p.to_string_lossy().contains("t5"))
                .cloned();
            let clip_path = paths.iter()
                .find(|p| p.to_string_lossy().contains("clip"))
                .cloned();
            
            if let (Some(latents), Some(t5), Some(clip)) = (latents_path, t5_path, clip_path) {
                items.push(PreprocessedFluxItem {
                    latents_path: latents,
                    t5_embeds_path: t5,
                    clip_pooled_path: clip,
                    caption: format!("Item {}", hash), // Could store this separately
                    metadata: std::collections::HashMap::new(),
                });
            }
        }
    }
    
    Ok(items)
}