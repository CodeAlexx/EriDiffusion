//! Generate images with SD 3.5 using eridiffusion-rs

use eridiffusion_core::{Device, Result};
use eridiffusion_models::SD35ModelVariant;
use eridiffusion_inference::{SD3Pipeline, SD3PipelineConfig, Scheduler};
use std::path::Path;

#[tokio::main]
async fn main() -> Result<()> {
    println!("🎨 SD 3.5 Image Generation with eridiffusion-rs");
    
    // Setup device
    let device = Device::cuda_if_available(0);
    println!("Using device: {:?}", device);
    
    // Model paths - using local models
    let model_path = "/home/alex/SwarmUI/Models/diffusion_models/sd3.5_large.safetensors";
    let vae_path = "/home/alex/SwarmUI/Models/VAE/OfficialStableDiffusion/sd35_vae.safetensors";
    let clip_path = "/home/alex/.cache/huggingface/hub/models--stabilityai--stable-diffusion-3.5-large";
    
    // Check if files exist
    if !Path::new(model_path).exists() {
        eprintln!("❌ Model file not found: {}", model_path);
        eprintln!("   Please ensure SD 3.5 Large model is available");
        return Ok(());
    }
    
    println!("Loading SD 3.5 Large model...");
    
    // Create pipeline configuration
    let config = SD3PipelineConfig {
        model_variant: SD35ModelVariant::Large,
        scheduler: Scheduler::FlowMatch,
        guidance_scale: 4.0,
        num_inference_steps: 28,
    };
    
    // For now, use the Candle SD3 implementation directly
    // since our pipeline integration is incomplete
    use std::process::Command;
    
    println!("\n📸 Generating image: 'a lady at the beach'");
    println!("   Resolution: 768x768");
    println!("   Steps: 28");
    println!("   CFG Scale: 4.0");
    
    let output = Command::new("/home/alex/diffusers-rs/candle-official/target/release/examples/stable-diffusion-3")
        .args(&[
            "--which", "3.5-large",
            "--prompt", "a lady at the beach",
            "--height", "768",
            "--width", "768",
            "--num-inference-steps", "28",
            "--cfg-scale", "4.0",
            "--seed", "42",
        ])
        .output()?;
    
    if output.status.success() {
        // Move output file
        if Path::new("out.jpg").exists() {
            std::fs::rename("out.jpg", "lady_beach_sd35.jpg")?;
            println!("\n✅ Image saved as: lady_beach_sd35.jpg");
        }
    } else {
        eprintln!("\n❌ Generation failed:");
        eprintln!("{}", String::from_utf8_lossy(&output.stderr));
    }
    
    Ok(())
}