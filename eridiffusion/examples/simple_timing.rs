use eridiffusion::loaders::WeightLoader;
use eridiffusion::models::flux_vae::AutoencoderKL;
use flame_core::{Device, Shape, Tensor};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Suppress debug output
    std::env::set_var("RUST_LOG", "error");

    // Initialize device
    let device = Device::cuda(0)?;

    // Load VAE
    let weights_file = "/home/alex/SwarmUI/Models/VAE/ae.safetensors";
    println!("Loading VAE...");
    let loader = WeightLoader::from_safetensors(weights_file, device.clone())?;
    let vae = AutoencoderKL::new(&loader, device.clone(), false)?;
    println!("VAE loaded");

    // Test 1024x1024
    println!("\nTesting 1024x1024 encoding speed:");

    // Create test image
    let test_image =
        Tensor::randn(Shape::from_dims(&[1, 3, 1024, 1024]), 0.0, 1.0, device.cuda_device_arc())?;

    // Warm-up
    let _ = vae.encode(&test_image)?;

    // Time single run
    let start = Instant::now();
    let latent = vae.encode(&test_image)?;
    let elapsed = start.elapsed();

    let shape = latent.shape();
    println!("  Encoded to shape: {:?}", shape.dims());
    println!("  Time: {:.2} seconds", elapsed.as_secs_f32());
    println!("  Speed: {:.2} images/second", 1.0 / elapsed.as_secs_f32());

    Ok(())
}
