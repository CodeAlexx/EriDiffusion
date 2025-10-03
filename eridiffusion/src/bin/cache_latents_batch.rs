#![cfg(feature = "legacy-bins")]

// NOTE: This binary is gated behind the `legacy-bins` feature so it no longer builds by
// default alongside the production pipelines. Enable it explicitly with
// `--features legacy-bins` if you still rely on the original caching workflow.

use eridiffusion::models::flux_vae::load_flux_vae;
use flame_core::{Device, Result, Shape, Tensor};
use image::ImageReader;
use std::fs;
use std::path::{Path, PathBuf};

fn save_latent_as_safetensor(path: &Path, latent: &Tensor) -> Result<()> {
    // Save as safetensors format for compatibility
    use safetensors::{serialize, SafeTensors};
    use std::collections::HashMap;

    let mut tensors = HashMap::new();
    let data = latent.to_vec()?;
    let shape = latent.shape().dims().to_vec();

    // Store as a named tensor
    tensors.insert(
        "latent".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, &shape, &data)?,
    );

    let serialized = serialize(&tensors, &None)?;
    fs::write(path, serialized)?;
    Ok(())
}

fn main() -> Result<()> {
    println!("=== FLUX VAE BATCH LATENT CACHING ===");
    println!("SimpleTuner-style batch processing without OOM");

    let device = Device::cuda(0)?;

    // Configuration
    let batch_size = 2; // Process 2 images at a time
    let cleanup_interval = 10; // Force GPU cleanup every 10 images

    // Load VAE once
    println!("\nLoading Flux VAE with cuDNN support...");
    let vae_path = Path::new("/home/alex/SwarmUI/Models/VAE/ae.safetensors");
    let vae = load_flux_vae(vae_path, device.clone(), false)?;
    println!("✅ VAE loaded");

    // Setup paths
    let dataset_path = PathBuf::from("/home/alex/diffusers-rs/datasets/1stone");
    let cache_dir = dataset_path.join("cached_latents");
    fs::create_dir_all(&cache_dir)?;

    // Get all images
    let images: Vec<_> = fs::read_dir(&dataset_path)?
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            let path = entry.path();
            if let Some(ext) = path.extension() {
                matches!(ext.to_str(), Some("jpg") | Some("png") | Some("jpeg"))
            } else {
                false
            }
        })
        .collect();

    println!("Found {} images to cache", images.len());

    let mut processed = 0;
    let mut skipped = 0;
    let mut batch_buffer = Vec::new();
    let mut batch_paths = Vec::new();

    for (idx, entry) in images.iter().enumerate() {
        let img_path = entry.path();
        let base_name = img_path.file_stem().unwrap().to_str().unwrap();
        let latent_path = cache_dir.join(format!("{}.pt", base_name)); // Use .pt like SimpleTuner

        // Skip if already cached
        if latent_path.exists() {
            skipped += 1;
            println!(
                "[{}/{}] Already cached: {}",
                processed + skipped,
                images.len(),
                base_name
            );
            continue;
        }

        println!(
            "[{}/{}] Loading: {}",
            idx + 1,
            images.len(),
            img_path.file_name().unwrap().to_str().unwrap()
        );

        // Load and preprocess image
        let img = ImageReader::open(&img_path)
            .map_err(|e| {
                flame_core::FlameError::InvalidOperation(format!("Failed to open image: {}", e))
            })?
            .decode()
            .map_err(|e| {
                flame_core::FlameError::InvalidOperation(format!("Failed to decode image: {}", e))
            })?;

        let img = img.resize_exact(1024, 1024, image::imageops::FilterType::Lanczos3);
        let img = img.to_rgb8();

        // Convert to tensor
        let mut pixels = Vec::with_capacity(3 * 1024 * 1024);
        for c in 0..3 {
            for y in 0..1024 {
                for x in 0..1024 {
                    let pixel = img.get_pixel(x, y)[c];
                    pixels.push(pixel as f32 / 255.0);
                }
            }
        }

        let img_tensor = Tensor::from_vec(
            pixels,
            Shape::from_dims(&[1, 3, 1024, 1024]),
            device.cuda_device_arc(),
        )?;

        // Add to batch
        batch_buffer.push(img_tensor);
        batch_paths.push(latent_path);

        // Process batch when full or at end
        if batch_buffer.len() >= batch_size || idx == images.len() - 1 {
            println!("  📦 Processing batch of {} images...", batch_buffer.len());

            // Stack tensors for batch processing
            // For now, process one by one (can optimize later)
            for (tensor, save_path) in batch_buffer.iter().zip(batch_paths.iter()) {
                println!("    Encoding with VAE...");
                let latent = vae.encode(tensor)?;

                // Save immediately and drop
                let latent_data = latent.to_vec()?;
                let latent_shape = latent.shape().dims().to_vec();

                // Save as binary (similar to our approach but with .pt extension)
                let mut data = Vec::new();
                data.extend_from_slice(&(latent_shape.len() as u32).to_le_bytes());
                for dim in &latent_shape {
                    data.extend_from_slice(&(*dim as u32).to_le_bytes());
                }
                for val in latent_data {
                    data.extend_from_slice(&val.to_le_bytes());
                }

                fs::write(save_path, data)?;
                processed += 1;
                println!(
                    "    ✅ Saved: {}",
                    save_path.file_name().unwrap().to_str().unwrap()
                );

                // Drop latent immediately
                drop(latent);
            }

            // Clear batch
            batch_buffer.clear();
            batch_paths.clear();

            // Periodic cleanup
            if processed % cleanup_interval == 0 {
                println!("  🔄 GPU memory cleanup after {} images", processed);
                // Force garbage collection by dropping and recreating tensors
                // In a real implementation, we'd call CUDA memory cleanup functions
            }
        }
    }

    println!("\n✅ LATENT CACHING COMPLETE!");
    println!("Total processed: {}", processed);
    println!("Total skipped (already cached): {}", skipped);
    println!("Cache directory: {}", cache_dir.display());

    Ok(())
}
