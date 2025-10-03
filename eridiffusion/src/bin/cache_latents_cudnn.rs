#![cfg(feature = "legacy-bins")]

// NOTE: This legacy caching binary remains available only when the `legacy-bins` feature
// is enabled. Gate keeps it out of default builds while we focus on the production
// training pipelines.

use eridiffusion::models::flux_vae::load_flux_vae;
use flame_core::{Device, Result, Shape, Tensor};
use image::ImageReader;
use std::fs;
use std::path::{Path, PathBuf};

fn main() -> Result<()> {
    println!("=== FLUX VAE LATENT CACHING WITH CUDNN ===");
    println!("Processing /home/alex/diffusers-rs/datasets/1stone");
    println!("Memory-optimized batch processing with cuDNN acceleration");

    // Set CUDA_MODULE_LOADING=EAGER to prevent lazy loading issues
    std::env::set_var("CUDA_MODULE_LOADING", "EAGER");

    let device = Device::cuda(0)?;

    // Load VAE with cuDNN support
    println!("\nLoading Flux VAE with cuDNN support...");
    let vae_path = Path::new("/home/alex/SwarmUI/Models/VAE/ae.safetensors");
    let vae = load_flux_vae(vae_path, device.clone(), false)?;
    println!("✅ VAE loaded with cuDNN acceleration");

    // Setup paths
    let dataset_path = PathBuf::from("/home/alex/diffusers-rs/datasets/1stone");
    let cache_dir = dataset_path.join("cached_latents");
    fs::create_dir_all(&cache_dir)?;

    // Get list of images
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

    println!("\nFound {} images to cache", images.len());

    let mut processed = 0;
    let mut skipped = 0;
    let batch_size = 1; // Process only 1 image then exit to free memory completely
    let mut batch_count = 0;

    for entry in images.iter() {
        let img_path = entry.path();
        let base_name = img_path.file_stem().unwrap().to_str().unwrap();
        let latent_path = cache_dir.join(format!("{}.bin", base_name));

        // Skip if already cached
        if latent_path.exists() {
            skipped += 1;
            println!("[{}/{}] Already cached: {}", processed + skipped, images.len(), base_name);
            continue;
        }

        println!(
            "[{}/{}] Processing: {}",
            processed + skipped + 1,
            images.len(),
            img_path.display()
        );

        // Load and preprocess image
        let img = ImageReader::open(&img_path)
            .map_err(|e| {
                flame_core::Error::InvalidOperation(format!("Failed to open image: {}", e))
            })?
            .decode()
            .map_err(|e| {
                flame_core::Error::InvalidOperation(format!("Failed to decode image: {}", e))
            })?;

        // Resize to 1024x1024
        let img = img.resize_exact(1024, 1024, image::imageops::FilterType::Lanczos3);
        let img = img.to_rgb8();

        // Convert to tensor [1, 3, 1024, 1024] and normalize to [0, 1]
        let mut pixels = Vec::with_capacity(3 * 1024 * 1024);
        for c in 0..3 {
            for y in 0..1024 {
                for x in 0..1024 {
                    let pixel = img.get_pixel(x, y)[c];
                    pixels.push(pixel as f32 / 255.0);
                }
            }
        }

        // Create tensor
        let img_tensor = Tensor::from_vec(
            pixels,
            Shape::from_dims(&[1, 3, 1024, 1024]),
            device.cuda_device_arc(),
        )?;

        // Encode to latent
        println!("  Encoding with VAE (cuDNN accelerated)...");
        let latent = vae.encode(&img_tensor)?;

        // Get data and shape before dropping
        let latent_data = latent.to_vec()?;
        let latent_shape = latent.shape().dims().to_vec();

        // Immediately drop tensors to free GPU memory
        drop(latent);
        drop(img_tensor);

        // Save as simple binary file with shape info
        let mut data = Vec::new();
        // Write shape
        data.extend_from_slice(&(latent_shape.len() as u32).to_le_bytes());
        for dim in &latent_shape {
            data.extend_from_slice(&(*dim as u32).to_le_bytes());
        }
        // Write data
        for val in latent_data {
            data.extend_from_slice(&val.to_le_bytes());
        }

        fs::write(&latent_path, data)?;
        processed += 1;
        batch_count += 1;

        println!("  ✅ Saved: {} (shape: {:?})", latent_path.display(), latent_shape);

        // Exit after batch_size images to free all GPU memory
        if batch_count >= batch_size {
            println!("\n🔄 Batch of {} images complete. Exiting to free GPU memory.", batch_size);
            println!(
                "📊 Total progress: {} processed, {} skipped, {} remaining",
                processed,
                skipped,
                images.len() - processed - skipped
            );
            println!("\n✅ Run this program again to continue caching!");
            return Ok(());
        }
    }

    println!("\n✅ LATENT CACHING COMPLETE!");
    println!("Total processed: {}", processed);
    println!("Total skipped (already cached): {}", skipped);
    println!("Cache directory: {}", cache_dir.display());
    println!("\n🎯 NOW YOU CAN TRAIN WITHOUT LOADING THE VAE!");
    println!("The latents are ready for Flux LoRA training!");

    Ok(())
}
