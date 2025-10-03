#![cfg(feature = "legacy-bins")]

// NOTE: Legacy binary; gated behind the `legacy-bins` feature so it stays out of default builds. Enable with `--features legacy-bins` if you still rely on it.

use eridiffusion::trainers::{
    flux_cache_manager::FluxCacheManager,
    flux_data_loader::{DatasetConfig, FluxDataLoader},
    text_encoders::TextEncoders,
};
use flame_core::device::Device;
use flame_core::{DType, Tensor};
use safetensors::{serialize, tensor::TensorView, Dtype as SafeDtype};
use std::collections::HashMap;
use std::path::PathBuf;

fn main() -> flame_core::Result<()> {
    println!("=== Batch Text Embedding Encoder for Flux ===");

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
        cache_dir.clone(),
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

    // Get all samples and their captions
    let mut all_samples = Vec::new();
    for bucket_idx in 0..data_loader.buckets.len() {
        for sample in &data_loader.buckets[bucket_idx].samples {
            let caption = if sample.caption_path.exists() {
                std::fs::read_to_string(&sample.caption_path)
                    .unwrap_or_else(|_| "".to_string())
                    .trim()
                    .to_string()
            } else {
                "".to_string()
            };
            all_samples.push((sample.image_path.clone(), caption));
        }
    }

    println!("\n=== Loading Text Encoders ===");

    // Create text encoders (both will be loaded)
    let mut text_encoders = TextEncoders::new(device.clone());

    // Load models - load both at once since we have a simplified approach
    println!("Loading CLIP-L encoder...");
    text_encoders.load_clip_l(&clip_path.to_string_lossy())?;
    println!("✅ CLIP-L loaded successfully");

    // Check memory
    println!("\nGPU Memory after loading CLIP:");
    std::process::Command::new("nvidia-smi")
        .arg("--query-gpu=memory.used,memory.free,memory.total")
        .arg("--format=csv,noheader,nounits")
        .status()
        .expect("Failed to run nvidia-smi");

    println!("\nLoading T5-XXL encoder...");
    text_encoders.load_t5(&t5_path.to_string_lossy())?;
    println!("✅ T5-XXL loaded successfully");

    // Check memory again
    println!("\nGPU Memory after loading T5:");
    std::process::Command::new("nvidia-smi")
        .arg("--query-gpu=memory.used,memory.free,memory.total")
        .arg("--format=csv,noheader,nounits")
        .status()
        .expect("Failed to run nvidia-smi");

    // Process samples
    println!("\n=== Encoding Text Embeddings ===");
    for (idx, (image_path, caption)) in all_samples.iter().enumerate() {
        println!(
            "\n[{}/{}] Processing: {}",
            idx + 1,
            all_samples.len(),
            image_path.file_name().unwrap().to_str().unwrap()
        );

        let prompt_preview =
            if caption.len() > 80 { format!("{}...", &caption[..80]) } else { caption.clone() };
        println!("  Caption: \"{}\"", prompt_preview);

        // Encode using the flux method
        let (clip_embed, t5_embed) = text_encoders.encode_flux(caption)?;

        // Save embeddings to cache
        let cache_path = cache_manager.get_embed_cache_path(&image_path);

        // Prepare tensors for saving
        let mut tensors = HashMap::new();

        // CLIP embedding
        let clip_data = clip_embed.to_vec1::<f32>()?;
        let clip_shape = clip_embed.shape().dims().to_vec();
        let clip_bytes = unsafe {
            std::slice::from_raw_parts(clip_data.as_ptr() as *const u8, clip_data.len() * 4)
        };
        tensors.insert(
            "clip_embeds".to_string(),
            TensorView::new(SafeDtype::F32, clip_shape, clip_bytes)
                .map_err(|e| flame_core::Error::InvalidOperation(e.to_string()))?,
        );

        // T5 embedding
        let t5_data = t5_embed.to_vec1::<f32>()?;
        let t5_shape = t5_embed.shape().dims().to_vec();
        let t5_bytes =
            unsafe { std::slice::from_raw_parts(t5_data.as_ptr() as *const u8, t5_data.len() * 4) };
        tensors.insert(
            "t5_embeds".to_string(),
            TensorView::new(SafeDtype::F32, t5_shape, t5_bytes)
                .map_err(|e| flame_core::Error::InvalidOperation(e.to_string()))?,
        );

        // Add metadata
        let mut metadata = HashMap::new();
        metadata.insert("format".to_string(), "flux_text_cache".to_string());
        metadata.insert("version".to_string(), "1.0".to_string());
        metadata.insert("prompt".to_string(), caption.clone());

        // Serialize and save
        let serialized = serialize(tensors, &Some(metadata))
            .map_err(|e| flame_core::Error::InvalidOperation(e.to_string()))?;
        std::fs::write(&cache_path, serialized)
            .map_err(|e| flame_core::Error::Io(e.to_string()))?;

        println!("  ✅ Embeddings saved to cache");
    }

    // Check final status
    let (_, final_embed_count) = cache_manager.get_stats()?;
    println!("\n✅ Text encoding complete!");
    println!("  Total embeddings cached: {}", final_embed_count);

    // Final memory check
    println!("\nFinal GPU Memory:");
    std::process::Command::new("nvidia-smi")
        .arg("--query-gpu=memory.used,memory.free,memory.total")
        .arg("--format=csv,noheader,nounits")
        .status()
        .expect("Failed to run nvidia-smi");

    Ok(())
}
