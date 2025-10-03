#![cfg(feature = "legacy-bins")]

// NOTE: Legacy binary; gated behind the `legacy-bins` feature so it stays out of default builds. Enable with `--features legacy-bins` if you still rely on it.

use eridiffusion::trainers::{
    flux_cache_manager::FluxCacheManager,
    flux_data_loader::{DatasetConfig, FluxDataLoader},
};
use flame_core::device::Device;
use std::path::PathBuf;

fn main() -> flame_core::Result<()> {
    println!("=== Pre-encoding Text Embeddings for Flux ===");

    // Initialize device
    let device = Device::cuda(0)?;

    // Dataset configuration
    let dataset_config = DatasetConfig {
        folder_path: PathBuf::from("/home/alex/diffusers-rs/datasets/40_woman"),
        caption_ext: "txt".to_string(),
        caption_dropout_rate: 0.0,
        shuffle_tokens: false,
        cache_latents_to_disk: true,
        resolutions: vec![(1024, 1024)],
        center_crop: true,
        random_flip: true,
        // force_recache handled elsewhere
    };

    // Create data loader
    let mut data_loader = FluxDataLoader::new(dataset_config.clone(), device.clone())?;
    println!("Created data loader with {} samples", data_loader.total_samples());

    // Create cache manager
    let cache_dir = dataset_config.folder_path.join("cache");
    let cache_manager = FluxCacheManager::with_dataset_name(
        cache_dir,
        device.clone(),
        true, // enabled
        "40_woman".to_string(),
    )?;

    // Text encoder paths
    let clip_path = PathBuf::from("/home/alex/SwarmUI/Models/clip/clip_l.safetensors");
    let t5_path = PathBuf::from("/home/alex/SwarmUI/Models/clip/t5xxl_fp16.safetensors");

    // Check current status
    let (latent_count, embed_count) = cache_manager.get_stats()?;
    println!("\nCurrent cache status:");
    println!("  Latents cached: {}", latent_count);
    println!("  Text embeddings cached: {}", embed_count);

    // Encode text embeddings
    println!("\n=== Encoding Text Embeddings ===");

    // Check GPU memory before encoding
    println!("\n=== GPU Memory Before Encoding ===");
    std::process::Command::new("nvidia-smi")
        .arg("--query-gpu=memory.used,memory.free,memory.total")
        .arg("--format=csv,noheader,nounits")
        .status()
        .expect("Failed to run nvidia-smi");

    cache_manager.encode_all_text_embeddings(
        &mut data_loader,
        &clip_path,
        Some(&t5_path),
        false, // Don't force re-encode
    )?;

    // Check final status
    let (_, final_embed_count) = cache_manager.get_stats()?;
    println!("\n✅ Text encoding complete!");
    println!("  Total embeddings cached: {}", final_embed_count);

    Ok(())
}
