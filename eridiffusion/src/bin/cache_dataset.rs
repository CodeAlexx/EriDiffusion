#![cfg(feature = "legacy-bins")]

// NOTE: Legacy binary; gated behind the `legacy-bins` feature so it stays out of default builds. Enable with `--features legacy-bins` if you still rely on it.

//! Cache dataset for Flux training
//!
//! This follows SimpleTuner's approach:
//! 1. Load VAE, encode images to latents, save to disk, free VAE
//! 2. Load text encoders, encode prompts, save to disk, free encoders
//! 3. Now training can proceed with just the main model loaded

use clap::Parser;
use eridiffusion::loaders::WeightLoader;
use eridiffusion::models::flux_vae::AutoencoderKL;
use eridiffusion::trainers::{
    flux_cache_manager::FluxCacheManager,
    flux_data_loader::{DatasetConfig, FluxDataLoader},
    text_encoders::TextEncoders,
};
use flame_core::DType;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Path to the YAML configuration file
    #[arg(short, long)]
    config: PathBuf,
}

#[derive(Debug, Deserialize)]
struct CacheConfig {
    device: String,
    weights_dir: String,
    dataset: DatasetConfigYaml,
}

#[derive(Debug, Deserialize)]
struct DatasetConfigYaml {
    folder: String,
    caption_ext: String,
    resolutions: Vec<(usize, usize)>,
    center_crop: bool,
    random_flip: bool,
    cache_dir: Option<String>,
    force_recache: Option<bool>,
}

fn main() -> flame_core::Result<()> {
    eridiffusion::logging::init_logger()?;

    // Parse command line arguments
    let cli = Cli::parse();

    // Load configuration from YAML
    let config_str = std::fs::read_to_string(&cli.config).map_err(|e| {
        flame_core::Error::InvalidOperation(format!("Failed to read config: {}", e))
    })?;
    let config: CacheConfig = serde_yaml::from_str(&config_str).map_err(|e| {
        flame_core::Error::InvalidOperation(format!("Failed to parse YAML: {}", e))
    })?;

    // Parse device
    let device = if config.device.starts_with("cuda:") {
        let ordinal =
            config.device.strip_prefix("cuda:").and_then(|s| s.parse::<usize>().ok()).unwrap_or(0);
        flame_core::device::Device::cuda(ordinal)?
    } else {
        flame_core::device::Device::cpu()?
    };

    // Try to find the weight files in various locations
    let vae_path = if PathBuf::from(&config.weights_dir).join("ae.safetensors").exists() {
        PathBuf::from(&config.weights_dir).join("ae.safetensors")
    } else {
        PathBuf::from(&config.weights_dir).join("VAE/ae.safetensors")
    };

    let clip_l_path = if PathBuf::from("/home/alex/kohya_ss/clip_l.safetensors").exists() {
        PathBuf::from("/home/alex/kohya_ss/clip_l.safetensors")
    } else {
        PathBuf::from(&config.weights_dir).join("clip/clip_l.safetensors")
    };

    let t5_path = if PathBuf::from("/home/alex/kohya_ss/t5xxl_fp16.safetensors").exists() {
        PathBuf::from("/home/alex/kohya_ss/t5xxl_fp16.safetensors")
    } else {
        PathBuf::from(&config.weights_dir).join("clip/t5xxl_fp16.safetensors")
    };

    // Extract dataset name from folder path
    let dataset_path = PathBuf::from(&config.dataset.folder);
    let dataset_name =
        dataset_path.file_name().and_then(|n| n.to_str()).unwrap_or("dataset").to_string();

    // Create cache manager
    let cache_dir =
        config.dataset.cache_dir.map(PathBuf::from).unwrap_or_else(|| dataset_path.join("cache"));

    let cache_manager = FluxCacheManager::with_dataset_name(
        cache_dir.clone(),
        device.clone(),
        true,
        dataset_name.clone(),
    )?;

    // Create data loader
    let dataset_config = DatasetConfig {
        folder_path: dataset_path,
        caption_ext: config.dataset.caption_ext.clone(),
        caption_dropout_rate: 0.0,
        shuffle_tokens: false,
        cache_latents_to_disk: true,
        resolutions: config.dataset.resolutions.clone(),
        center_crop: config.dataset.center_crop,
        random_flip: config.dataset.random_flip,
    };

    let mut data_loader = FluxDataLoader::new(dataset_config, device.clone())?;
    println!("Dataset loaded with {} samples", data_loader.total_samples());

    // Get force_recache setting
    let force_recache = config.dataset.force_recache.unwrap_or(false);
    if force_recache {
        println!("⚠️  Force recache enabled - will re-encode all data even if cached");
    }

    // Step 1: Cache VAE latents
    println!("\n=== Step 1: Caching VAE latents ===");
    {
        // Use the cache manager's encode_all_latents method instead
        cache_manager.encode_all_latents(&mut data_loader, &vae_path, force_recache)?;
    }

    // Step 2: Cache text embeddings
    println!("\n=== Step 2: Caching text embeddings ===");
    {
        // Use the cache manager's encode_all_text_embeddings method
        cache_manager.encode_all_text_embeddings(
            &mut data_loader,
            &clip_l_path,
            Some(&t5_path),
            force_recache,
        )?;
    }

    println!("\n=== Caching Complete ===");
    println!("All latents and embeddings have been cached to: {}", cache_dir.display());
    println!("\nNow you can run training with:");
    println!("  - cache_latents_to_disk: true");
    println!("  - The trainer will load cached data instead of encoding");
    println!("  - This saves ~10GB of VRAM (VAE + T5)");

    Ok(())
}
