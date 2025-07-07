//! Test dataset loading for Flux LoRA training

use anyhow::Result;
use eridiffusion::trainers::flux_data_loader::{FluxDataLoader, DatasetConfig};
use std::path::PathBuf;
use candle_core::Device;

fn main() -> Result<()> {
    env_logger::init();
    
    println!("Testing Flux dataset loading...");
    
    let device = Device::cuda_if_available(0)?;
    println!("Using device: {:?}", device);
    
    // Create dataset configuration
    let config = DatasetConfig {
        folder_path: PathBuf::from("/home/alex/diffusers-rs/datasets/40_woman"),
        caption_ext: "txt".to_string(),
        caption_dropout_rate: 0.05,
        shuffle_tokens: false,
        cache_latents_to_disk: false, // Don't cache for this test
        resolutions: vec![(512, 512), (768, 768), (1024, 1024)],
        center_crop: false,
        random_flip: true,
    };
    
    println!("\nDataset configuration:");
    println!("  Folder: {}", config.folder_path.display());
    println!("  Caption extension: {}", config.caption_ext);
    println!("  Resolutions: {:?}", config.resolutions);
    
    // Create data loader
    let mut data_loader = FluxDataLoader::new(config, device)?;
    
    // Test loading some batches
    println!("\nLoading batches...");
    for i in 0..3 {
        println!("\nBatch {}:", i + 1);
        let batch = data_loader.get_batch(4)?;
        
        if batch.is_empty() {
            println!("  No more data!");
            break;
        }
        
        println!("  Loaded {} samples", batch.len());
        for (j, (img, caption)) in batch.iter().enumerate() {
            println!("    Sample {}: {}x{} - {}", 
                j + 1, 
                img.width(), 
                img.height(),
                if caption.len() > 50 { 
                    format!("{}...", &caption[..50])
                } else {
                    caption.clone()
                }
            );
        }
    }
    
    // Test dataset statistics
    println!("\nDataset statistics:");
    let total_images = std::fs::read_dir("/home/alex/diffusers-rs/datasets/40_woman")?
        .filter_map(Result::ok)
        .filter(|entry| {
            entry.path().extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| matches!(ext, "jpg" | "jpeg" | "png"))
                .unwrap_or(false)
        })
        .count();
    
    println!("  Total images: {}", total_images);
    println!("  Total captions: {}", total_images); // Assuming each image has a caption
    
    println!("\nDataset test completed successfully!");
    Ok(())
}