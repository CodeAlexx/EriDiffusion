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
    let start = Instant::now();
    let loader = WeightLoader::from_safetensors(weights_file, device.clone())?;
    let vae = AutoencoderKL::new(&loader, device.clone(), false)?;
    println!("VAE loaded in {:.2}s", start.elapsed().as_secs_f32());

    // Test different image sizes
    let test_sizes = vec![
        (512, 512, "512x512 (no tiling)"),
        (768, 768, "768x768 (tiled)"),
        (1024, 1024, "1024x1024 (tiled)"),
    ];

    for (height, width, desc) in test_sizes {
        println!("\n=== Testing {} ===", desc);

        // Create test image
        let test_image = Tensor::randn(
            Shape::from_dims(&[1, 3, height, width]),
            0.0, // mean
            1.0, // std
            device.cuda_device_arc(),
        )?;

        // Warm up (first run is always slower)
        println!("Warming up...");
        let _ = vae.encode(&test_image)?;

        // Benchmark multiple runs
        let num_runs = 5;
        let mut total_time = 0.0;

        println!("Running {} iterations...", num_runs);
        for i in 0..num_runs {
            let start = Instant::now();
            let latent = vae.encode(&test_image)?;
            let elapsed = start.elapsed().as_secs_f32();
            total_time += elapsed;

            // Verify output shape
            let shape = latent.shape();
            let expected_h = height / 8;
            let expected_w = width / 8;

            println!("  Run {}: {:.3}s -> latent shape: {:?}", i + 1, elapsed, shape.dims());

            if shape.dims()[2] != expected_h || shape.dims()[3] != expected_w {
                println!("  WARNING: Unexpected shape!");
            }
        }

        let avg_time = total_time / num_runs as f32;
        let throughput = 1.0 / avg_time;

        println!("Average time: {:.3}s per image", avg_time);
        println!("Throughput: {:.2} images/second", throughput);

        // Calculate memory bandwidth estimate
        let pixels = (height * width) as f32;
        let megapixels = pixels / 1_000_000.0;
        let megapixels_per_sec = megapixels * throughput;
        println!("Processing speed: {:.2} megapixels/second", megapixels_per_sec);
    }

    Ok(())
}
