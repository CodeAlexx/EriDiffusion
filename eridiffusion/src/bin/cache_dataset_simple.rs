#![cfg(feature = "legacy-bins")]

// NOTE: Legacy binary; gated behind the `legacy-bins` feature so it stays out of default builds. Enable with `--features legacy-bins` if you still rely on it.

//! Simplified cache dataset for Flux training
//!
//! This version avoids the narrow operation that's causing CUDA issues

use eridiffusion::loaders::WeightLoader;
use eridiffusion::models::flux_vae::AutoencoderKL;
use eridiffusion::trainers::{
    flux_cache_manager::FluxCacheManager,
    flux_data_loader::{DatasetConfig, FluxDataLoader},
    text_encoders::TextEncoders,
};
use flame_core::Device;
use std::path::PathBuf;
use std::sync::Arc;

fn main() -> flame_core::Result<()> {
    eridiffusion::logging::init_logger()?;

    // Configuration
    let cache_dir = PathBuf::from("/home/alex/diffusers-rs/cache");
    let dataset_path = PathBuf::from("/home/alex/diffusers-rs/datasets/40_woman");
    let vae_path = PathBuf::from("/home/alex/SwarmUI/Models/VAE/ae.safetensors");
    let clip_l_path = PathBuf::from("/home/alex/SwarmUI/Models/clip/clip_l.safetensors");
    let t5_path = PathBuf::from("/home/alex/SwarmUI/Models/clip/t5xxl_fp16.safetensors");

    let device = flame_core::device::Device::cuda(0)?;
    let dataset_name = "40_woman";

    // Create cache manager
    let cache_manager = FluxCacheManager::with_dataset_name(
        cache_dir.clone(),
        device.clone(),
        true,
        dataset_name.to_string(),
    )?;

    // Create data loader
    let dataset_config = DatasetConfig {
        folder_path: dataset_path,
        caption_ext: "txt".to_string(),
        caption_dropout_rate: 0.0,
        shuffle_tokens: false,
        cache_latents_to_disk: true,
        resolutions: vec![(1024, 1024)],
        center_crop: true,
        random_flip: false,
    };

    let mut data_loader = FluxDataLoader::new(dataset_config, device.clone())?;
    println!("Dataset loaded with {} samples", data_loader.total_samples());

    // Step 1: Cache VAE latents
    println!("\n=== Step 1: Caching VAE latents ===");
    {
        println!("Loading VAE...");
        let vae_weights = WeightLoader::from_safetensors(&vae_path, device.clone())?;
        let vae = Arc::new(AutoencoderKL::new(&vae_weights, device.clone(), false)?);
        println!("✅ VAE loaded successfully");

        // Process all images one by one
        let mut processed = 0;
        let total = data_loader.total_samples();

        for _ in 0..total {
            let batch = data_loader.next_batch()?;
            if let Some(batch) = batch {
                processed += 1;
                println!("  [{}/{}] Processing sample", processed, total);

                // Process each image in the batch (should be just one)
                if let Some(path) = batch.image_paths.first() {
                    println!("    Encoding: {:?}", path);

                    // Check if already cached
                    if cache_manager.is_latent_cached(&path) {
                        println!("    ✓ Already cached, skipping");
                        continue;
                    }

                    // The batch contains all images, but we process the whole tensor
                    let latent = vae.encode(&batch.images)?;

                    // Save to cache
                    let cache_path = cache_manager.get_latent_cache_path(&path);
                    cache_manager.save_tensor(&latent, &cache_path, "latent")?;
                    println!("    ✓ Cached latent (shape: {:?})", latent.shape());
                }
            }
        }

        println!("\n✅ All {} latents cached", processed);
        println!("🗑️  Freeing VAE from memory...");
        // VAE will be dropped when going out of scope
    }

    // Step 2: Cache text embeddings
    println!("\n=== Step 2: Caching text embeddings ===");
    {
        println!("Loading text encoders...");
        let text_encoders = Arc::new(TextEncoders::from_safetensors(
            Some(&clip_l_path),
            None, // Flux doesn't use CLIP-G
            Some(&t5_path),
            device.clone(),
        )?);
        println!("✅ Text encoders loaded successfully");

        // Reset data loader
        data_loader = FluxDataLoader::new(data_loader.config.clone(), device.clone())?;

        // Process all captions
        let mut processed = 0;
        let total = data_loader.total_samples();

        for _ in 0..total {
            let batch = data_loader.next_batch()?;
            if let Some(batch) = batch {
                processed += 1;
                println!("  [{}/{}] Processing sample", processed, total);

                // Process the first prompt/path pair
                if let (Some(prompt), Some(path)) =
                    (batch.prompts.first(), batch.image_paths.first())
                {
                    println!("    Encoding caption for: {:?}", path);

                    // Check if already cached
                    if cache_manager.is_embed_cached(&path) {
                        println!("    ✓ Already cached, skipping");
                        continue;
                    }

                    // Encode caption
                    println!("    Caption: \"{}\"", prompt);
                    let (pooled_embeds, embeds) = text_encoders.encode_flux(prompt)?;

                    // Save embeddings to cache
                    let cache_path = cache_manager.get_embed_cache_path(&path);

                    // Save main embeddings
                    cache_manager.save_tensor(&embeds, &cache_path, "text_embeds")?;
                    // Save pooled embeddings
                    cache_manager.save_tensor(&pooled_embeds, &cache_path, "pooled_embeds")?;

                    println!("    ✓ Cached embeddings (shape: {:?})", embeds.shape());
                }
            }
        }

        println!("\n✅ All {} text embeddings cached", processed);
        println!("🗑️  Freeing text encoders from memory...");
        // Text encoders will be dropped when going out of scope
    }

    println!("\n=== Caching Complete ===");
    println!("All latents and embeddings have been cached to: {}", cache_dir.display());
    println!("\nNow you can run training with:");
    println!("  - cache_latents_to_disk: true");
    println!("  - The trainer will load cached data instead of encoding");
    println!("  - This saves ~10GB of VRAM (VAE + T5)");

    Ok(())
}
