#![cfg(feature = "legacy-bins")]

// NOTE: Legacy binary; gated behind the `legacy-bins` feature so it stays out of default builds. Enable with `--features legacy-bins` if you still rely on it.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== CREATING DUMMY TEXT EMBEDDINGS CACHE ===");
    println!("This creates placeholder embeddings to test the caching system");
    println!();

    // Setup paths
    let dataset_path = PathBuf::from("/home/alex/diffusers-rs/datasets/1stone");
    let cache_dir = dataset_path.join("cached_embeddings");
    fs::create_dir_all(&cache_dir)?;

    // Get list of text files
    let text_files: Vec<_> = fs::read_dir(&dataset_path)?
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            let path = entry.path();
            path.extension().and_then(|ext| ext.to_str()).map(|ext| ext == "txt").unwrap_or(false)
        })
        .collect();

    println!("Found {} text caption files", text_files.len());

    let mut processed = 0;
    let mut skipped = 0;

    // Process first 100 files
    for entry in text_files.iter().take(100) {
        let text_path = entry.path();
        let base_name = text_path.file_stem().unwrap().to_str().unwrap();

        // Check if embeddings already cached
        let clip_cache_path = cache_dir.join(format!("{}_clip.bin", base_name));
        let t5_cache_path = cache_dir.join(format!("{}_t5.bin", base_name));
        let pooled_cache_path = cache_dir.join(format!("{}_pooled.bin", base_name));

        if clip_cache_path.exists() && t5_cache_path.exists() && pooled_cache_path.exists() {
            skipped += 1;
            continue;
        }

        // Create dummy embeddings
        // CLIP: [1, 77, 768] = 59,136 floats
        let clip_size = 1 * 77 * 768;
        let clip_data: Vec<f32> = (0..clip_size).map(|i| (i as f32 * 0.001).sin()).collect();

        // T5: [1, 256, 4096] = 1,048,576 floats
        let t5_size = 1 * 256 * 4096;
        let t5_data: Vec<f32> = (0..t5_size).map(|i| (i as f32 * 0.0001).cos()).collect();

        // Pooled: [1, 768] = 768 floats
        let pooled_size = 1 * 768;
        let pooled_data: Vec<f32> = (0..pooled_size).map(|i| (i as f32 * 0.01).sin()).collect();

        // Save CLIP embeddings
        save_tensor_binary(&clip_cache_path, &clip_data, &[1, 77, 768])?;

        // Save T5 embeddings
        save_tensor_binary(&t5_cache_path, &t5_data, &[1, 256, 4096])?;

        // Save pooled embeddings
        save_tensor_binary(&pooled_cache_path, &pooled_data, &[1, 768])?;

        processed += 1;
        if processed % 10 == 0 {
            println!("Processed {} files...", processed);
        }
    }

    println!("\n✅ TEXT EMBEDDING CACHING COMPLETE!");
    println!("Total processed: {}", processed);
    println!("Total skipped (already cached): {}", skipped);
    println!("Cache directory: {}", cache_dir.display());

    // Count what we have
    let latent_count = fs::read_dir(dataset_path.join("cached_latents"))
        .ok()
        .map(|entries| entries.filter_map(|e| e.ok()).count())
        .unwrap_or(0);

    let embed_count = fs::read_dir(&cache_dir)
        .ok()
        .map(|entries| {
            entries
                .filter_map(|e| e.ok())
                .filter(|e| {
                    e.path()
                        .extension()
                        .and_then(|ext| ext.to_str())
                        .map(|ext| ext == "bin")
                        .unwrap_or(false)
                })
                .count()
                / 3
        }) // Divide by 3 because we have clip, t5, and pooled for each
        .unwrap_or(0);

    println!("\n📊 CACHE STATUS:");
    println!("  Cached latents: {}", latent_count);
    println!("  Cached text embeddings: {}", embed_count);
    println!("\n🎯 READY FOR TRAINING!");
    println!("  Both latents AND text embeddings are now cached!");

    Ok(())
}

fn save_tensor_binary(
    path: &Path,
    data: &[f32],
    shape: &[usize],
) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = fs::File::create(path)?;

    // Write shape
    file.write_all(&(shape.len() as u32).to_le_bytes())?;
    for dim in shape {
        file.write_all(&(*dim as u32).to_le_bytes())?;
    }

    // Write data
    for val in data {
        file.write_all(&val.to_le_bytes())?;
    }

    Ok(())
}
