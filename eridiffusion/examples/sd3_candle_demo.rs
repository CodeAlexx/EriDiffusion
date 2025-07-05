//! Minimal SD3 demo using Candle implementation directly

use anyhow::Result;
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use candle_transformers::models::mmdit::model::{Config as MMDiTConfig, MMDiT};
use std::path::PathBuf;

// Import from our sd3_candle module
use eridiffusion_models::sd3_candle::{
    StableDiffusion3TripleClipWithTokenizer, Which, build_sd3_vae_autoencoder,
    sd3_vae_vb_rename, euler_sample,
};

fn main() -> Result<()> {
    println!("SD3 Candle Demo - Direct Implementation");
    
    // Configuration
    let prompt = "A cute rusty robot holding a candle torch";
    let uncond_prompt = "";
    let height = 1024;
    let width = 1024;
    let num_inference_steps = 28;
    let cfg_scale = 7.0;
    let time_shift = 3.0;
    let seed = 42;
    
    // Set up device
    let device = if candle_core::utils::cuda_is_available() {
        println!("Using CUDA device");
        Device::new_cuda(0)?
    } else {
        println!("Using CPU device");
        Device::Cpu
    };
    
    // Set random seed
    device.set_seed(seed)?;
    
    // Model paths - adjust these to your actual model locations
    let model_base = PathBuf::from("/home/alex/SwarmUI/Models/Stable-Diffusion");
    let model_file = model_base.join("sd3_medium_incl_clips_t5xxlfp16.safetensors");
    
    println!("Loading model from: {:?}", model_file);
    
    // Load model weights
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[&model_file], DType::F16, &device)?
    };
    
    // Initialize text encoder
    println!("Initializing text encoders...");
    let mut text_encoder = StableDiffusion3TripleClipWithTokenizer::new(vb.pp("text_encoders"))?;
    
    // Encode prompts
    println!("Encoding prompts...");
    let (context, y) = text_encoder.encode_text_to_embedding(prompt, &device)?;
    let (context_uncond, y_uncond) = text_encoder.encode_text_to_embedding(uncond_prompt, &device)?;
    
    // Concatenate for classifier-free guidance
    let context = candle_core::Tensor::cat(&[context, context_uncond], 0)?;
    let y = candle_core::Tensor::cat(&[y, y_uncond], 0)?;
    
    // Drop text encoder to free memory
    drop(text_encoder);
    println!("Text encoding complete, freeing text encoder memory");
    
    // Initialize MMDiT
    println!("Initializing MMDiT model...");
    let mmdit_config = MMDiTConfig::sd3_medium();
    let mmdit = MMDiT::new(&mmdit_config, false, vb.pp("model.diffusion_model"))?;
    
    // Run sampling
    println!("Starting image generation...");
    let start_time = std::time::Instant::now();
    
    let x = euler_sample(
        &mmdit,
        &y,
        &context,
        num_inference_steps,
        cfg_scale,
        time_shift,
        height,
        width,
        None, // No skip layer guidance
    )?;
    
    let dt = start_time.elapsed().as_secs_f32();
    println!(
        "Sampling complete in {:.2}s ({:.2} steps/s)",
        dt,
        num_inference_steps as f32 / dt
    );
    
    // Initialize VAE for decoding
    println!("Initializing VAE decoder...");
    let vb_vae = vb.rename_f(sd3_vae_vb_rename).pp("first_stage_model");
    let vae = build_sd3_vae_autoencoder(vb_vae)?;
    
    // Decode latents
    println!("Decoding latents to image...");
    let scaled = ((x / 1.5305)? + 0.0609)?;
    let img = vae.decode(&scaled)?;
    
    // Convert to RGB
    let img = ((img.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?.to_dtype(DType::U8)?;
    
    // Save image
    println!("Saving image to sd3_output.jpg");
    candle_examples::save_image(&img.i(0)?, "sd3_output.jpg")?;
    
    println!("Done! Total time: {:.2}s", start_time.elapsed().as_secs_f32());
    
    Ok(())
}