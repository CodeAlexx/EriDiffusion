use eridiffusion::loaders::WeightLoader;
use eridiffusion::models::flux_vae::AutoencoderKL;
use flame_core::{Device, Shape, Tensor};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize device
    let device = Device::cuda(0)?;

    // Load VAE
    let weights_file = "/home/alex/SwarmUI/Models/VAE/ae.safetensors";
    println!("Loading VAE...");
    let loader = WeightLoader::from_safetensors(weights_file, device.clone())?;
    let vae = AutoencoderKL::new(&loader, device.clone(), false)?;
    println!("VAE loaded successfully");

    // Test 1024x1024 (the size that was causing issues)
    println!("\n=== Testing 1024x1024 (tiled) ===");

    // Create test image
    let test_image = Tensor::randn(
        Shape::from_dims(&[1, 3, 1024, 1024]),
        0.0, // mean
        1.0, // std
        device.cuda_device_arc(),
    )?;

    // Single warm-up run
    println!("Warming up...");
    let _ = vae.encode(&test_image)?;

    // Time 3 runs
    println!("Timing 3 runs...");
    let mut times = Vec::new();

    for i in 0..3 {
        let start = Instant::now();
        let latent = vae.encode(&test_image)?;
        let elapsed = start.elapsed().as_secs_f32();
        times.push(elapsed);

        // Verify output shape (should be 1x16x128x128 for Flux VAE)
        let shape = latent.shape();
        println!("  Run {}: {:.3}s -> latent shape: {:?}", i + 1, elapsed, shape.dims());
    }

    // Calculate average
    let avg_time: f32 = times.iter().sum::<f32>() / times.len() as f32;
    println!("\n=== Results ===");
    println!("Average time: {:.3}s per image", avg_time);
    println!("Throughput: {:.2} images/second", 1.0 / avg_time);

    // Compare with small image (no tiling)
    println!("\n=== Testing 512x512 (no tiling) for comparison ===");

    let small_image =
        Tensor::randn(Shape::from_dims(&[1, 3, 512, 512]), 0.0, 1.0, device.cuda_device_arc())?;

    // Warm up
    let _ = vae.encode(&small_image)?;

    // Time 3 runs
    let mut small_times = Vec::new();
    for i in 0..3 {
        let start = Instant::now();
        let _ = vae.encode(&small_image)?;
        let elapsed = start.elapsed().as_secs_f32();
        small_times.push(elapsed);
        println!("  512x512 run {}: {:.3}s", i + 1, elapsed);
    }

    let small_avg: f32 = small_times.iter().sum::<f32>() / small_times.len() as f32;
    println!("\n512x512 average: {:.3}s per image", small_avg);
    println!("1024x1024 average: {:.3}s per image (tiled)", avg_time);
    println!("Tiling overhead: {:.1}x slower", avg_time / small_avg);

    Ok(())
}
