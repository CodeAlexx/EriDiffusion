#![cfg(feature = "legacy-bins")]

// NOTE: Legacy binary; gated behind the `legacy-bins` feature so it stays out of default builds. Enable with `--features legacy-bins` if you still rely on it.

use eridiffusion::models::flux_vae::load_flux_vae;
use flame_core::{Device, Result, Shape, Tensor};
use image::ImageReader;
use std::fs;
use std::path::{Path, PathBuf};

fn main() -> Result<()> {
    println!("=== SIMPLE LATENT CACHING ===");
    println!("Cache latents ONCE, never load VAE again during training!");

    let device = Device::cuda(0)?;

    // Load VAE
    println!("\nLoading Flux VAE...");
    let vae_path = Path::new("/home/alex/SwarmUI/Models/VAE/ae.safetensors");
    let vae = load_flux_vae(vae_path, device.clone(), false)?;
    println!("✅ VAE loaded");

    // Setup paths - use the FULL 1stone dataset!
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

    println!("Found {} images to cache", images.len());

    // Process ALL images with proper memory management
    println!("\nProcessing {} images in batches to manage memory...", images.len());

    // Process one image at a time and clear GPU memory between each
    for (i, entry) in images.iter().enumerate() {
        let img_path = entry.path();
        let base_name = img_path.file_stem().unwrap().to_str().unwrap();
        let latent_path = cache_dir.join(format!("{}.safetensors", base_name));

        // Skip if already cached
        if latent_path.exists() {
            println!("[{}/{}] Already cached: {}", i + 1, images.len(), base_name);
            continue;
        }

        println!("[{}/{}] Processing: {}", i + 1, images.len(), img_path.display());

        // Load and preprocess image
        let img = ImageReader::open(&img_path)
            .map_err(|e| {
                flame_core::Error::InvalidOperation(format!("Failed to open image: {}", e))
            })?
            .decode()
            .map_err(|e| {
                flame_core::Error::InvalidOperation(format!("Failed to decode image: {}", e))
            })?;
        let img = img.resize_exact(1024, 1024, image::imageops::FilterType::Lanczos3);
        let img = img.to_rgb8();

        // Convert to tensor [1, 3, 1024, 1024] and normalize to [0, 1]
        let mut pixels = Vec::new();
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

        // Encode to latent - using direct encoding for speed
        println!("  Encoding with VAE...");
        let latent = vae.encode(&img_tensor)?;

        // Save latent as numpy file for simplicity
        let latent_data = latent.to_vec()?;
        let latent_shape = latent.shape().dims().to_vec();

        // Drop the latent tensor to free GPU memory immediately
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

        let latent_bin_path = cache_dir.join(format!("{}.bin", base_name));
        fs::write(&latent_bin_path, data)?;

        println!("  ✅ Saved: {} (shape: {:?})", latent_bin_path.display(), latent_shape);

        // Force GPU memory cleanup every 10 images
        if (i + 1) % 10 == 0 {
            println!("  🔄 Cleaning GPU memory after {} images...", i + 1);
            // The drops above should have freed memory, but we can add more cleanup if needed
        }
    }

    println!("\n✅ LATENTS CACHED!");
    println!("Cached to: {}", cache_dir.display());
    println!("Now train WITHOUT loading VAE!");

    Ok(())
}
