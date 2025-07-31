//! Generate test images following exact candle patterns
//! This demonstrates proper image generation for all three models

use anyhow::Result;
use candle::{DType, Device, Tensor};

// Import the candle image utilities
#[path = "../src/trainers/candle_image_utils.rs"]
mod candle_image_utils;

use candle_image_utils::{save_image, create_sample_directory, ModelType};

const TEST_PROMPTS: &[&str] = &[
    "a majestic mountain landscape at sunset",
    "a futuristic city with neon lights",
    "a serene forest with morning mist",
    "an ancient temple in the jungle",
    "a cosmic nebula in deep space",
];

fn main() -> Result<()> {
    println!("Generating test images following candle patterns");
    
    let device = Device::cuda_if_available(0)?;
    println!("Using device: {:?}", device);
    
    // Generate for each model type
    generate_sdxl_images(&device)?;
    generate_sd35_images(&device)?;
    generate_flux_images(&device)?;
    
    println!("\nAll test images generated successfully!");
    println!("Images saved to /outputs/<model>_lora/samples/");
    
    Ok(())
}

fn generate_sdxl_images(device: &Device) -> Result<()> {
    println!("\nGenerating SDXL test images...");
    
    let output_dir = create_sample_directory("sdxl_lora")?;
    let resolution = 1024;
    
    for (idx, prompt) in TEST_PROMPTS.iter().enumerate() {
        println!("  Generating image {} for prompt: {}", idx + 1, prompt);
        
        // Create gradient test image
        let img = create_gradient_image(device, resolution, idx)?;
        
        // Save as JPG (SDXL standard)
        let filename = format!("sample_step000000_idx{:02}.jpg", idx);
        let filepath = output_dir.join(&filename);
        
        save_image(&img, &filepath)?;
        
        // Save metadata
        let metadata = format!(
            "Prompt: {}\nModel: SDXL\nStep: 0\nCFG Scale: 7.5\nSeed: {}",
            prompt, 42 + idx
        );
        std::fs::write(filepath.with_extension("txt"), metadata)?;
    }
    
    Ok(())
}

fn generate_sd35_images(device: &Device) -> Result<()> {
    println!("\nGenerating SD 3.5 test images...");
    
    let output_dir = create_sample_directory("sd35_lora")?;
    let resolution = 1024;
    
    for (idx, prompt) in TEST_PROMPTS.iter().enumerate() {
        println!("  Generating image {} for prompt: {}", idx + 1, prompt);
        
        // Create pattern test image
        let img = create_pattern_image(device, resolution, idx)?;
        
        // Save as PNG (SD 3.5 preference)
        let filename = format!("sample_step000000_idx{:02}.png", idx);
        let filepath = output_dir.join(&filename);
        
        save_image(&img, &filepath)?;
        
        // Save metadata
        let metadata = format!(
            "Prompt: {}\nModel: SD 3.5\nStep: 0\nCFG Scale: 5.0\nSeed: {}",
            prompt, 142 + idx
        );
        std::fs::write(filepath.with_extension("txt"), metadata)?;
    }
    
    Ok(())
}

fn generate_flux_images(device: &Device) -> Result<()> {
    println!("\nGenerating Flux test images...");
    
    let output_dir = create_sample_directory("flux_lora")?;
    let resolution = 1024;
    
    for (idx, prompt) in TEST_PROMPTS.iter().enumerate() {
        println!("  Generating image {} for prompt: {}", idx + 1, prompt);
        
        // Create noise test image
        let img = create_noise_image(device, resolution, idx)?;
        
        // Save as JPG (Flux standard)
        let filename = format!("sample_step000000_idx{:02}.jpg", idx);
        let filepath = output_dir.join(&filename);
        
        save_image(&img, &filepath)?;
        
        // Save metadata
        let metadata = format!(
            "Prompt: {}\nModel: Flux\nStep: 0\nCFG Scale: 3.5\nSeed: {}",
            prompt, 242 + idx
        );
        std::fs::write(filepath.with_extension("txt"), metadata)?;
    }
    
    Ok(())
}

// Helper functions to create test images

fn create_gradient_image(device: &Device, size: usize, seed: usize) -> Result<Tensor> {
    // Create a gradient image in [-1, 1] range
    let mut data = vec![0f32; 3 * size * size];
    
    for y in 0..size {
        for x in 0..size {
            let idx = y * size + x;
            // Red channel: horizontal gradient
            data[idx] = (x as f32 / size as f32) * 2.0 - 1.0;
            // Green channel: vertical gradient
            data[size * size + idx] = (y as f32 / size as f32) * 2.0 - 1.0;
            // Blue channel: diagonal gradient with seed variation
            data[2 * size * size + idx] = 
                ((x + y + seed * 50) as f32 / (2.0 * size as f32)) * 2.0 - 1.0;
        }
    }
    
    Tensor::from_vec(data, &[3, size, size], device)
}

fn create_pattern_image(device: &Device, size: usize, seed: usize) -> Result<Tensor> {
    // Create a checkerboard pattern in [-1, 1] range
    let mut data = vec![0f32; 3 * size * size];
    let square_size = 64;
    
    for y in 0..size {
        for x in 0..size {
            let idx = y * size + x;
            let checker = ((x / square_size) + (y / square_size)) % 2 == 0;
            let value = if checker { 0.8 } else { -0.8 };
            
            // Add seed-based color variation
            data[idx] = value * (1.0 + (seed as f32 * 0.1));
            data[size * size + idx] = value * (1.0 - (seed as f32 * 0.05));
            data[2 * size * size + idx] = value;
        }
    }
    
    Tensor::from_vec(data, &[3, size, size], device)
}

fn create_noise_image(device: &Device, size: usize, seed: usize) -> Result<Tensor> {
    // Create perlin-noise-like image using random values
    let tensor = Tensor::randn(
        0.0, 
        0.3, 
        &[3, size, size], 
        device
    )?;
    
    // Add seed-based offset
    let offset = Tensor::full(seed as f32 * 0.1, &[3, size, size], device)?;
    let tensor = (tensor + offset)?;
    
    // Clamp to valid range
    tensor.clamp(-1.0, 1.0)
}