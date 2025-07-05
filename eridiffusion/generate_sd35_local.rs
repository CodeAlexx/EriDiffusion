#!/usr/bin/env rust-script
//! Generate images with SD3.5 using local model weights
//!
//! ```cargo
//! [dependencies]
//! candle-core = { version = "0.8", features = ["cuda"] }
//! candle-nn = "0.8"
//! candle-transformers = "0.8"
//! anyhow = "1.0"
//! clap = { version = "4.0", features = ["derive"] }
//! image = "0.25"
//! tokenizers = "0.20"
//! hf-hub = "0.3"
//! ```

use anyhow::{Result, Context};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::stable_diffusion3::{
    mmdit::{Config as MMDiTConfig, MMDiT},
    text_encoder::{T5Config, T5TextEncoder, CLIPTextEncoder, CLIPConfig},
    vae::{AutoEncoderConfig, AutoEncoder},
    sampling::{FlowMatchingEulerSampler, SamplingMethod},
};
use clap::Parser;
use std::path::Path;

#[derive(Parser)]
struct Args {
    /// Text prompt for generation
    #[arg(long, default_value = "A majestic mountain landscape at sunset, photorealistic, high quality")]
    prompt: String,
    
    /// Negative prompt
    #[arg(long, default_value = "")]
    negative_prompt: String,
    
    /// Output image path
    #[arg(long, default_value = "sd35_output.png")]
    output: String,
    
    /// Image width
    #[arg(long, default_value = "1024")]
    width: usize,
    
    /// Image height
    #[arg(long, default_value = "1024")]
    height: usize,
    
    /// Number of denoising steps
    #[arg(long, default_value = "28")]
    steps: usize,
    
    /// Guidance scale
    #[arg(long, default_value = "7.5")]
    cfg_scale: f32,
    
    /// Random seed
    #[arg(long)]
    seed: Option<u64>,
    
    /// Use CPU instead of GPU
    #[arg(long)]
    cpu: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    
    // Setup device
    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::new_cuda(0)?
    };
    
    println!("SD 3.5 Image Generation (Local Models)");
    println!("=====================================");
    println!("Prompt: {}", args.prompt);
    println!("Resolution: {}x{}", args.width, args.height);
    println!("Steps: {}, CFG Scale: {}", args.steps, args.cfg_scale);
    println!("Device: {:?}", device);
    
    // Model paths
    let model_base = "/home/alex/SwarmUI/Models";
    let hf_cache = "/home/alex/.cache/huggingface/hub";
    
    // SD 3.5 Large model paths
    let mmdit_path = format!("{}/diffusion_models/sd3.5_large.safetensors", model_base);
    let vae_path = format!("{}/VAE/OfficialStableDiffusion/sd35_vae.safetensors", model_base);
    
    // Text encoder paths from HF cache
    let text_encoder_dir = format!("{}/models--stabilityai--stable-diffusion-3.5-large/snapshots/764a7c2c5b58de7099a102985f91ca87b656c279/text_encoders", hf_cache);
    let clip_l_path = format!("{}/clip_l.safetensors", text_encoder_dir);
    let clip_g_path = format!("{}/clip_g.safetensors", text_encoder_dir);
    let t5_path = format!("{}/t5xxl_fp16.safetensors", text_encoder_dir);
    
    // Check if all files exist
    for (name, path) in [
        ("MMDiT", &mmdit_path),
        ("VAE", &vae_path),
        ("CLIP-L", &clip_l_path),
        ("CLIP-G", &clip_g_path),
        ("T5", &t5_path),
    ] {
        if !Path::new(path).exists() {
            anyhow::bail!("{} not found at: {}", name, path);
        }
        println!("✓ Found {}: {}", name, path);
    }
    
    // Load models
    println!("\nLoading models...");
    
    // Load MMDiT
    println!("Loading MMDiT...");
    let mmdit_vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[mmdit_path], DType::F16, &device)?
    };
    let mmdit_config = MMDiTConfig::sd3_5_large();
    let mmdit = MMDiT::new(&mmdit_config, false, mmdit_vb)?;
    
    // Load VAE
    println!("Loading VAE...");
    let vae_vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[vae_path], DType::F16, &device)?
    };
    let vae_config = AutoEncoderConfig::sd3();
    let vae = AutoEncoder::new(&vae_config, vae_vb)?;
    
    // Load text encoders
    println!("Loading CLIP-L...");
    let clip_l_vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[clip_l_path], DType::F16, &device)?
    };
    let clip_l_config = CLIPConfig::clip_l();
    let clip_l = CLIPTextEncoder::new(&clip_l_config, clip_l_vb)?;
    
    println!("Loading CLIP-G...");
    let clip_g_vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[clip_g_path], DType::F16, &device)?
    };
    let clip_g_config = CLIPConfig::clip_g();
    let clip_g = CLIPTextEncoder::new(&clip_g_config, clip_g_vb)?;
    
    println!("Loading T5...");
    let t5_vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[t5_path], DType::F16, &device)?
    };
    let t5_config = T5Config::t5_xxl();
    let t5 = T5TextEncoder::new(&t5_config, t5_vb)?;
    
    // Encode prompts
    println!("\nEncoding prompts...");
    let (prompt_embeds, pooled_embeds) = encode_prompt(
        &args.prompt,
        &clip_l,
        &clip_g,
        &t5,
        &device,
    )?;
    
    let (neg_prompt_embeds, neg_pooled_embeds) = if args.cfg_scale > 1.0 {
        encode_prompt(
            &args.negative_prompt,
            &clip_l,
            &clip_g,
            &t5,
            &device,
        )?
    } else {
        (prompt_embeds.clone(), pooled_embeds.clone())
    };
    
    // Initialize sampler
    let sampler = FlowMatchingEulerSampler::new(
        1000, // num_train_timesteps
        3.0,  // shift for SD3.5
    );
    
    // Generate latents
    println!("\nGenerating image...");
    let latents = generate_latents(
        &mmdit,
        &sampler,
        &prompt_embeds,
        &pooled_embeds,
        &neg_prompt_embeds,
        &neg_pooled_embeds,
        args.width,
        args.height,
        args.steps,
        args.cfg_scale,
        args.seed,
        &device,
    )?;
    
    // Decode latents
    println!("Decoding latents...");
    let images = vae.decode(&latents)?;
    
    // Convert to image and save
    let image = tensor_to_image(&images)?;
    image.save(&args.output)?;
    
    println!("\n✅ Image saved to: {}", args.output);
    
    Ok(())
}

fn encode_prompt(
    prompt: &str,
    clip_l: &CLIPTextEncoder,
    clip_g: &CLIPTextEncoder,
    t5: &T5TextEncoder,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    // Tokenize and encode with each model
    let clip_l_embeds = clip_l.encode(prompt, device)?;
    let (clip_g_embeds, pooled) = clip_g.encode_with_pooled(prompt, device)?;
    let t5_embeds = t5.encode(prompt, device)?;
    
    // Pad CLIP embeddings to 2048 tokens each
    let clip_l_padded = pad_to_length(&clip_l_embeds, 2048, 1)?;
    let clip_g_padded = pad_to_length(&clip_g_embeds, 2048, 1)?;
    
    // Concatenate all embeddings
    let prompt_embeds = Tensor::cat(&[&clip_l_padded, &clip_g_padded, &t5_embeds], 1)?;
    
    Ok((prompt_embeds, pooled))
}

fn pad_to_length(tensor: &Tensor, target_len: usize, dim: usize) -> Result<Tensor> {
    let current_len = tensor.dim(dim)?;
    if current_len >= target_len {
        return Ok(tensor.narrow(dim, 0, target_len)?);
    }
    
    let padding_shape = {
        let mut shape = tensor.dims().to_vec();
        shape[dim] = target_len - current_len;
        shape
    };
    
    let padding = Tensor::zeros(&padding_shape, tensor.dtype(), tensor.device())?;
    Tensor::cat(&[tensor, &padding], dim)
}

fn generate_latents(
    mmdit: &MMDiT,
    sampler: &FlowMatchingEulerSampler,
    prompt_embeds: &Tensor,
    pooled_embeds: &Tensor,
    neg_prompt_embeds: &Tensor,
    neg_pooled_embeds: &Tensor,
    width: usize,
    height: usize,
    steps: usize,
    cfg_scale: f32,
    seed: Option<u64>,
    device: &Device,
) -> Result<Tensor> {
    // Set seed if provided
    if let Some(s) = seed {
        device.set_seed(s)?;
    }
    
    // Initialize latents
    let latent_height = height / 8;
    let latent_width = width / 8;
    let mut latents = Tensor::randn(
        0f32,
        1f32,
        &[1, 16, latent_height, latent_width], // SD3 uses 16-channel VAE
        device,
    )?;
    
    // Setup timesteps
    sampler.set_timesteps(steps);
    let timesteps = sampler.timesteps.clone();
    
    // Denoising loop
    for (i, &t) in timesteps.iter().enumerate() {
        // Prepare inputs for CFG
        let latent_input = if cfg_scale > 1.0 {
            Tensor::cat(&[&latents, &latents], 0)?
        } else {
            latents.clone()
        };
        
        let encoder_hidden_states = if cfg_scale > 1.0 {
            Tensor::cat(&[neg_prompt_embeds, prompt_embeds], 0)?
        } else {
            prompt_embeds.clone()
        };
        
        let pooled_projections = if cfg_scale > 1.0 {
            Tensor::cat(&[neg_pooled_embeds, pooled_embeds], 0)?
        } else {
            pooled_embeds.clone()
        };
        
        let timestep = Tensor::new(&[t as f32], device)?
            .unsqueeze(0)?
            .repeat(&[latent_input.dims()[0], 1])?;
        
        // Model prediction
        let noise_pred = mmdit.forward(
            &latent_input,
            &timestep,
            &encoder_hidden_states,
            &pooled_projections,
        )?;
        
        // Apply guidance
        let noise_pred = if cfg_scale > 1.0 {
            let (neg_pred, pos_pred) = noise_pred.chunk(2, 0)?;
            (neg_pred + ((pos_pred - neg_pred)? * cfg_scale)?)?
        } else {
            noise_pred
        };
        
        // Scheduler step
        latents = sampler.step(&noise_pred, t, &latents)?;
        
        if i % 5 == 0 {
            println!("  Step {}/{}", i + 1, steps);
        }
    }
    
    // Scale latents for VAE
    let scaled_latents = (latents / 1.5305)? + 0.0609)?;
    
    Ok(scaled_latents)
}

fn tensor_to_image(tensor: &Tensor) -> Result<image::DynamicImage> {
    // Ensure tensor is on CPU and convert to f32
    let tensor = tensor.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
    
    // Get dimensions [B, C, H, W]
    let (_, _, height, width) = tensor.dims4()?;
    
    // Convert from [-1, 1] to [0, 255]
    let pixel_data = ((tensor.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?;
    let pixels = pixel_data.flatten_all()?.to_vec1::<f32>()?;
    
    // Create RGB image
    let mut imgbuf = image::ImageBuffer::new(width as u32, height as u32);
    
    for (x, y, pixel) in imgbuf.enumerate_pixels_mut() {
        let idx = (y * width as u32 + x) as usize;
        let r = pixels[idx].clamp(0.0, 255.0) as u8;
        let g = pixels[idx + (height * width)].clamp(0.0, 255.0) as u8;
        let b = pixels[idx + 2 * (height * width)].clamp(0.0, 255.0) as u8;
        *pixel = image::Rgb([r, g, b]);
    }
    
    Ok(image::DynamicImage::ImageRgb8(imgbuf))
}