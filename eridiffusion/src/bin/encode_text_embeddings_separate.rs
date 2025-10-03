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
    println!("=== Pre-encoding Text Embeddings for Flux (Separate Models) ===");

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

    // Tokenizer paths
    let clip_tokenizer_path = PathBuf::from("/home/alex/.cache/huggingface/hub/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41/tokenizer.json");
    let t5_tokenizer_path = PathBuf::from("/home/alex/FLUX.1-dev/tokenizer_2/tokenizer.json");

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

    println!("\n=== Phase 1: Encoding with CLIP-L ===");
    let mut clip_embeddings = HashMap::new();

    // Load only CLIP-L
    {
        println!("Loading CLIP-L encoder...");
        let mut text_encoders = TextEncoders::new(device.clone());
        text_encoders.load_clip_l(&clip_path.to_string_lossy())?;
        text_encoders.load_clip_tokenizer(&clip_tokenizer_path.to_string_lossy())?;
        println!("✅ CLIP-L loaded successfully");

        // Check memory
        println!("\nGPU Memory after loading CLIP:");
        std::process::Command::new("nvidia-smi")
            .arg("--query-gpu=memory.used,memory.free,memory.total")
            .arg("--format=csv,noheader,nounits")
            .status()
            .expect("Failed to run nvidia-smi");

        // Encode with CLIP
        for (idx, (image_path, caption)) in all_samples.iter().enumerate() {
            println!(
                "\n[CLIP {}/{}] Processing: {}",
                idx + 1,
                all_samples.len(),
                image_path.file_name().unwrap().to_str().unwrap()
            );

            let prompt_preview =
                if caption.len() > 80 { format!("{}...", &caption[..80]) } else { caption.clone() };
            println!("  Caption: \"{}\"", prompt_preview);

            // Tokenize and encode
            let clip_tokens = text_encoders.tokenize_clip(caption, 77)?;
            let clip_output = text_encoders.clip_l.as_ref().unwrap().forward(&clip_tokens, None)?;
            let clip_embed = clip_output.last_hidden_state;

            // Store in memory
            clip_embeddings.insert(image_path.clone(), clip_embed);
            println!("  ✅ CLIP encoding complete");
        }

        // Explicitly drop text encoders
        drop(text_encoders);
        println!("\n✅ CLIP-L encoder freed from GPU memory");
    }

    // Check memory after freeing CLIP
    println!("\nGPU Memory after freeing CLIP:");
    std::process::Command::new("nvidia-smi")
        .arg("--query-gpu=memory.used,memory.free,memory.total")
        .arg("--format=csv,noheader,nounits")
        .status()
        .expect("Failed to run nvidia-smi");

    println!("\n=== Phase 2: Encoding with T5-XXL ===");

    // Load only T5
    {
        println!("Loading T5-XXL encoder...");
        let mut text_encoders = TextEncoders::new(device.clone());
        text_encoders.load_t5(&t5_path.to_string_lossy())?;
        text_encoders.load_tokenizers(
            &clip_tokenizer_path.to_string_lossy(),
            &t5_tokenizer_path.to_string_lossy(),
        )?;
        println!("✅ T5-XXL loaded successfully");

        // Check memory
        println!("\nGPU Memory after loading T5:");
        std::process::Command::new("nvidia-smi")
            .arg("--query-gpu=memory.used,memory.free,memory.total")
            .arg("--format=csv,noheader,nounits")
            .status()
            .expect("Failed to run nvidia-smi");

        // Encode with T5 and save combined embeddings
        for (idx, (image_path, caption)) in all_samples.iter().enumerate() {
            println!(
                "\n[T5 {}/{}] Processing: {}",
                idx + 1,
                all_samples.len(),
                image_path.file_name().unwrap().to_str().unwrap()
            );

            // Tokenize and encode with T5
            let t5_tokens = text_encoders.tokenize_t5(caption, 256)?;
            let t5_output = text_encoders.t5.as_ref().unwrap().forward(&t5_tokens)?;
            let t5_embed = t5_output.last_hidden_state;

            // Get CLIP embedding from memory
            let clip_embed = clip_embeddings.get(image_path).unwrap();

            // Save both embeddings to cache
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
            let t5_bytes = unsafe {
                std::slice::from_raw_parts(t5_data.as_ptr() as *const u8, t5_data.len() * 4)
            };
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

            println!("  ✅ Saved combined embeddings to cache");
        }

        drop(text_encoders);
        println!("\n✅ T5-XXL encoder freed from GPU memory");
    }

    // Check final status
    let (_, final_embed_count) = cache_manager.get_stats()?;
    println!("\n✅ Text encoding complete!");
    println!("  Total embeddings cached: {}", final_embed_count);

    Ok(())
}
