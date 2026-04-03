#![cfg(feature = "legacy-bins")]

// NOTE: Legacy binary; gated behind the `legacy-bins` feature so it stays out of default builds. Enable with `--features legacy-bins` if you still rely on it.

use eridiffusion::trainers::optimized_text_encoders::OptimizedTextEncoders;
use flame_core::{DType, Device, Result};
use std::fs;
use std::path::{Path, PathBuf};

fn main() -> Result<()> {
    println!("=== FLUX TEXT EMBEDDING CACHING WITH BF16 ===");
    println!("Cache text embeddings ONCE, never load T5/CLIP during training!");
    println!();

    let device = Device::cuda(0)?;

    // Setup paths
    let dataset_path = PathBuf::from("/home/alex/diffusers-rs/datasets/1stone");
    let cache_dir = dataset_path.join("cached_embeddings");
    fs::create_dir_all(&cache_dir).map_err(|e| flame_core::Error::Io(e.to_string()))?;

    // Get list of text files
    let text_files: Vec<_> = fs::read_dir(&dataset_path)
        .map_err(|e| flame_core::Error::Io(e.to_string()))?
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            let path = entry.path();
            path.extension().and_then(|ext| ext.to_str()).map(|ext| ext == "txt").unwrap_or(false)
        })
        .collect();

    println!("Found {} text caption files", text_files.len());

    // Create text encoders
    println!("\n🧠 Loading text encoders with BF16/FP16...");
    let mut encoders = OptimizedTextEncoders::new(device.clone());

    // Load tokenizers
    println!("  Loading tokenizers...");
    encoders.load_tokenizers(
        "/home/alex/.cache/huggingface/hub/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41/tokenizer.json",
        "/home/alex/SwarmUI_/dlbackend/ComfyUI/comfy/text_encoders/t5_tokenizer/tokenizer.json"
    )?;

    // Load CLIP (small, always in memory)
    println!("  Loading CLIP-L with FP16...");
    encoders.load_clip_l("/home/alex/SwarmUI/Models/clip/clip_l.safetensors")?;

    // T5 will be loaded lazily when first needed
    let t5_path = "/home/alex/SwarmUI/Models/clip/t5xxl_fp16.safetensors";
    println!("  T5 XXL will be loaded on first use with FP16");

    println!("✅ Encoders ready!");

    let mut processed = 0;
    let mut skipped = 0;

    for (idx, entry) in text_files.iter().enumerate() {
        let text_path = entry.path();
        let base_name = text_path.file_stem().unwrap().to_str().unwrap();

        // Check if embeddings already cached
        let clip_cache_path = cache_dir.join(format!("{}_clip_embed.bin", base_name));
        let t5_cache_path = cache_dir.join(format!("{}_t5_embed.bin", base_name));

        if clip_cache_path.exists() && t5_cache_path.exists() {
            skipped += 1;
            if skipped % 100 == 0 || skipped <= 10 {
                println!("[{}/{}] Already cached: {}", idx + 1, text_files.len(), base_name);
            }
            continue;
        }

        // Read caption
        let caption = fs::read_to_string(&text_path)
            .map_err(|e| flame_core::Error::Io(e.to_string()))?;
        let caption = caption.trim();

        println!("[{}/{}] Processing: {}", idx + 1, text_files.len(), base_name);
        println!(
            "  Caption: \"{}\"",
            if caption.len() > 60 { format!("{}...", &caption[..60]) } else { caption.to_string() }
        );

        // Encode with both CLIP and T5 using encode_flux
        println!("  Encoding with CLIP+T5 (FP16)...");
        let (clip_embed, t5_embed) = encoders.encode_flux(caption, t5_path)?;

        println!("  CLIP embed shape: {:?}", clip_embed.shape());
        println!("  T5 embed shape: {:?}", t5_embed.shape());

        // Convert to F32 for saving (training expects F32)
        let clip_embed_f32 = clip_embed.to_dtype(DType::F32)?;
        let t5_embed_f32 = t5_embed.to_dtype(DType::F32)?;

        // Save CLIP embeddings
        let clip_data = clip_embed_f32.to_vec()?;
        let clip_shape_dims = clip_embed_f32.shape().dims().to_vec();
        save_tensor_binary(&clip_cache_path, &clip_data, &clip_shape_dims)?;

        // Save T5 embeddings
        let t5_data = t5_embed_f32.to_vec()?;
        let t5_shape_dims = t5_embed_f32.shape().dims().to_vec();
        save_tensor_binary(&t5_cache_path, &t5_data, &t5_shape_dims)?;

        processed += 1;
        println!("  ✅ Saved embeddings to cache");

        // Free memory periodically
        if processed % 10 == 0 {
            println!("\n🔄 Processed {} files, continuing...", processed);
        }

        // For initial test, process only first 20 files
        if processed >= 20 {
            println!("\n⚠️  Stopping after {} files for initial test", processed);
            break;
        }
    }

    // Free encoders
    println!("\n🧹 Freeing text encoders from GPU memory...");
    drop(encoders);

    println!("\n✅ TEXT EMBEDDING CACHING COMPLETE!");
    println!("Total processed: {}", processed);
    println!("Total skipped (already cached): {}", skipped);
    println!("Cache directory: {}", cache_dir.display());
    println!("\n🎯 NOW YOU CAN TRAIN WITHOUT TEXT ENCODERS!");
    println!("Training will only need:");
    println!("  - Flux model (~12GB with FP16)");
    println!("  - LoRA weights (~200MB)");
    println!("  - Optimizer states (~1GB)");
    println!("  - Cached latents + embeddings (loaded from disk)");
    println!("Total VRAM: <15GB!");

    Ok(())
}

fn save_tensor_binary(path: &Path, data: &[f32], shape: &[usize]) -> Result<()> {
    let mut binary = Vec::new();

    // Write shape
    binary.extend_from_slice(&(shape.len() as u32).to_le_bytes());
    for dim in shape {
        binary.extend_from_slice(&(*dim as u32).to_le_bytes());
    }

    // Write data
    for val in data {
        binary.extend_from_slice(&val.to_le_bytes());
    }

    fs::write(path, binary).map_err(|e| flame_core::Error::Io(e.to_string()))?;
    Ok(())
}
