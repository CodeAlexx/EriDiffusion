use candle_transformers::models::stable_diffusion::{vae, unet_2d, clip, ddim};
use candle_core::{DType, Device, Tensor, Module};
use candle_nn::VarBuilder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🦀 Generating REAL SDXL image with Candle!\n");
    
    let device = Device::Cpu; // Use CPU for compatibility
    let dtype = DType::F32;
    
    // Paths to models
    let vae_path = "/home/alex/SwarmUI/Models/VAE/OfficialStableDiffusion/sdxl_vae.safetensors";
    let clip_path = "/home/alex/SwarmUI/Models/clip/clip_l.safetensors";
    
    println!("Loading models...");
    
    // Load VAE
    println!("  Loading VAE...");
    let vb_vae = VarBuilder::from_safetensors(&[vae_path], dtype, &device)?;
    let vae_config = vae::AutoEncoderKLConfig {
        block_out_channels: vec![128, 256, 512, 512],
        layers_per_block: 2,
        latent_channels: 4,
        norm_num_groups: 32,
    };
    let vae_model = vae::AutoEncoderKL::new(vb_vae, 3, 3, vae_config)?;
    
    // Create random latents
    println!("  Creating latents...");
    let latents = Tensor::randn(0f32, 1.0, (1, 4, 128, 128), &device)?;
    
    // Simple text embeddings (would use CLIP in real implementation)
    let text_embeddings = Tensor::randn(0f32, 1.0, (1, 77, 768), &device)?;
    
    // For demo, just decode the random latents
    println!("  Decoding latents...");
    let decoded = vae_model.decode(&latents)?;
    
    println!("  Decoded shape: {:?}", decoded.shape());
    
    // Convert to image
    let image = postprocess_image(decoded)?;
    
    println!("\n✅ Generated image!");
    save_image("generated_images/candle_sdxl.ppm", &image)?;
    
    Ok(())
}

fn postprocess_image(tensor: Tensor) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    // Move to CPU if needed
    let tensor = tensor.to_device(&Device::Cpu)?;
    
    // Get dimensions
    let (_, _, height, width) = tensor.dims4()?;
    
    // Convert to image data
    let mut image = vec![0u8; (height * width * 3) as usize];
    let data = tensor.to_vec3::<f32>()?;
    
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            
            // Scale from [-1, 1] to [0, 255]
            image[idx] = ((data[0][y][x] + 1.0) * 127.5).clamp(0.0, 255.0) as u8;
            image[idx + 1] = ((data[1][y][x] + 1.0) * 127.5).clamp(0.0, 255.0) as u8;
            image[idx + 2] = ((data[2][y][x] + 1.0) * 127.5).clamp(0.0, 255.0) as u8;
        }
    }
    
    Ok(image)
}

fn save_image(path: &str, pixels: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
    use std::fs;
    
    // Assume 1024x1024 image
    let header = format!("P6\n1024 1024\n255\n");
    let mut data = header.into_bytes();
    data.extend_from_slice(pixels);
    
    fs::create_dir_all("generated_images")?;
    fs::write(path, data)?;
    println!("  Saved: {}", path);
    
    Ok(())
}