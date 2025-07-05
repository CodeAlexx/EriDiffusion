// Example showing how to use Candle models with eridiffusion-rs
use candle_core::{Device, DType, Tensor};
use candle_nn::VarBuilder;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🦀 Using Candle Models with Real Weights\n");
    
    let device = Device::cuda_if_available(0)?;
    let dtype = if device.is_cuda() { DType::F16 } else { DType::F32 };
    
    // Check available model files
    check_model_files();
    
    // For SDXL
    if let Some(vae_path) = find_vae_file() {
        println!("\n=== SDXL VAE Test ===");
        test_vae_decode(&vae_path, &device, dtype)?;
    }
    
    Ok(())
}

fn check_model_files() {
    println!("📂 Checking for model files...");
    
    let paths = vec![
        ("/home/alex/SwarmUI/Models/VAE/OfficialStableDiffusion/sdxl_vae.safetensors", "SDXL VAE"),
        ("/home/alex/SwarmUI/Models/VAE/OfficialStableDiffusion/sd35_vae.safetensors", "SD3.5 VAE"),
        ("/home/alex/SwarmUI/Models/clip/clip_l.safetensors", "CLIP L"),
        ("/home/alex/SwarmUI/Models/clip/clip_g.safetensors", "CLIP G"),
    ];
    
    for (path, name) in paths {
        if Path::new(path).exists() {
            let size = std::fs::metadata(path).unwrap().len() / 1024 / 1024;
            println!("  ✓ {}: {} MB", name, size);
        }
    }
}

fn find_vae_file() -> Option<String> {
    let vae_path = "/home/alex/SwarmUI/Models/VAE/OfficialStableDiffusion/sdxl_vae.safetensors";
    if Path::new(vae_path).exists() {
        Some(vae_path.to_string())
    } else {
        None
    }
}

fn test_vae_decode(vae_path: &str, device: &Device, dtype: DType) -> Result<(), Box<dyn std::error::Error>> {
    println!("Loading VAE from: {}", vae_path);
    
    // Load just the VAE decoder weights we need
    let vb = VarBuilder::from_safetensors(&[vae_path], dtype, device)?;
    
    // Create test latents
    println!("Creating test latents...");
    let latents = create_test_latents(device)?;
    
    // For now, let's just show what we would do
    println!("Would decode latents of shape: {:?}", latents.shape());
    println!("VAE decoder components needed:");
    println!("  - decoder.conv_in");
    println!("  - decoder.up_blocks (multiple)");
    println!("  - decoder.mid_block");
    println!("  - decoder.conv_norm_out");
    println!("  - decoder.conv_out");
    
    // Save a simple visualization of the latents
    visualize_latents(&latents)?;
    
    Ok(())
}

fn create_test_latents(device: &Device) -> Result<Tensor, Box<dyn std::error::Error>> {
    // Create structured latents that should produce visible patterns
    let mut latent_data = vec![0.0f32; 4 * 64 * 64];
    
    for c in 0..4 {
        for y in 0..64 {
            for x in 0..64 {
                let idx = c * 64 * 64 + y * 64 + x;
                
                // Different pattern per channel
                latent_data[idx] = match c {
                    0 => (x as f32 / 64.0 - 0.5) * 2.0, // Horizontal gradient
                    1 => (y as f32 / 64.0 - 0.5) * 2.0, // Vertical gradient
                    2 => ((x as f32 - 32.0).powi(2) + (y as f32 - 32.0).powi(2)).sqrt() / 45.0 - 0.5, // Radial
                    3 => ((x + y) as f32 / 128.0 - 0.5) * 2.0, // Diagonal
                    _ => 0.0,
                };
            }
        }
    }
    
    Tensor::from_vec(latent_data, (1, 4, 64, 64), device)
}

fn visualize_latents(latents: &Tensor) -> Result<(), Box<dyn std::error::Error>> {
    // Move to CPU for visualization
    let latents = latents.to_device(&Device::Cpu)?;
    let data = latents.to_vec4::<f32>()?;
    
    // Create a simple visualization by upscaling and mixing channels
    let mut image = vec![0u8; 512 * 512 * 3];
    
    for y in 0..512 {
        for x in 0..512 {
            let idx = (y * 512 + x) * 3;
            
            // Sample from 64x64 latents
            let lx = x / 8;
            let ly = y / 8;
            
            // Mix 4 channels to RGB
            let r = data[0][0][ly][lx] * 0.5 + data[3][ly][lx] * 0.3;
            let g = data[0][1][ly][lx] * 0.5 + data[3][ly][lx] * 0.3;
            let b = data[0][2][ly][lx] * 0.5 + data[3][ly][lx] * 0.3;
            
            image[idx] = ((r * 0.18215 + 0.5) * 255.0).clamp(0.0, 255.0) as u8;
            image[idx + 1] = ((g * 0.18215 + 0.5) * 255.0).clamp(0.0, 255.0) as u8;
            image[idx + 2] = ((b * 0.18215 + 0.5) * 255.0).clamp(0.0, 255.0) as u8;
        }
    }
    
    // Save visualization
    let header = format!("P6\n512 512\n255\n");
    let mut ppm_data = header.into_bytes();
    ppm_data.extend_from_slice(&image);
    
    std::fs::create_dir_all("generated_images")?;
    std::fs::write("generated_images/latent_visualization.ppm", ppm_data)?;
    println!("Saved latent visualization to: generated_images/latent_visualization.ppm");
    
    Ok(())
}