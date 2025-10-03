#![cfg(feature = "legacy-bins")]

// NOTE: Legacy binary; gated behind the `legacy-bins` feature so it stays out of default builds. Enable with `--features legacy-bins` if you still rely on it.

//! Simple Flux image generation binary that bypasses complex training code

use clap::Parser;
use eridiffusion::loaders::WeightLoader;
use eridiffusion::models::flux_vae::AutoencoderKL;
use eridiffusion::models::{FluxModel, FluxModelConfig};
use eridiffusion::inference::FluxScheduler;
use flame_core::{DType, Device, Result, Shape, Tensor};

#[derive(Parser, Debug)]
#[command(author, version, about = "Generate images with Flux", long_about = None)]
struct Args {
    /// Prompt for the image
    #[arg(short, long, default_value = "flamingo on mars")]
    prompt: String,

    /// Number of diffusion steps
    #[arg(short = 's', long, default_value_t = 20)]
    steps: usize,

    /// Guidance scale
    #[arg(short = 'g', long, default_value_t = 1.0)]
    cfg: f32,

    /// Output path
    #[arg(short, long, default_value = "output.png")]
    output: String,

    /// Image width
    #[arg(long, default_value_t = 1024)]
    width: usize,

    /// Image height  
    #[arg(long, default_value_t = 1024)]
    height: usize,

    /// Model variant (dev or schnell)
    #[arg(long, default_value = "schnell")]
    variant: String,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("🚀 Flux Simple Image Generation");
    println!("================================");
    println!("Prompt: {}", args.prompt);
    println!("Steps: {}", args.steps);
    println!("CFG: {}", args.cfg);
    println!("Size: {}x{}", args.width, args.height);
    println!("Variant: {}", args.variant);

    // Setup device
    let device = Device::cuda(0)?;
    println!("✅ Using CUDA device 0");

    // Load models with hardcoded paths
    let model_path = if args.variant == "schnell" {
        "/home/alex/SwarmUI/Models/diffusion_models/flux1-schnell.safetensors"
    } else {
        "/home/alex/SwarmUI/Models/diffusion_models/flux1-dev.safetensors"
    };

    let vae_path = "/home/alex/SwarmUI/Models/VAE/ae.safetensors";
    let _clip_path = "/home/alex/SwarmUI/Models/clip/clip_l.safetensors";
    let _t5_path = "/home/alex/SwarmUI/Models/clip/t5xxl_fp16.safetensors";

    println!("\n📦 Loading models...");

    // Load VAE
    println!("  Loading VAE...");
    let vae_weights = WeightLoader::from_safetensors(vae_path, device.clone())?;
    let vae = AutoencoderKL::new(&vae_weights, device.clone(), false)?;
    println!("  ✅ VAE loaded");

    // Load Flux model
    println!("  Loading Flux model...");
    let flux_config = match args.variant.as_str() {
        "schnell" => FluxModelConfig::flux_schnell(),
        _ => FluxModelConfig::flux_dev(),
    };

    let model_weights = WeightLoader::from_safetensors(model_path, device.clone())?;
    let _flux_model =
        FluxModel::new(flux_config.clone(), device.clone(), model_weights.weights.clone())?;
    println!("  ✅ Flux model loaded");

    // Simple text encoding (placeholder - just create random embeddings)
    println!("\n📝 Encoding text...");
    let batch_size = 1;
    let seq_len = 512; // T5 sequence length
    let clip_seq_len = 77; // CLIP sequence length

    // Create placeholder embeddings (in real implementation, use proper text encoders)
    let t5_embeds = Tensor::randn(
        Shape::from_dims(&[batch_size, seq_len, 4096]),
        0.0,
        1.0,
        device.cuda_device_arc(),
    )?;
    let clip_embeds = Tensor::randn(
        Shape::from_dims(&[batch_size, clip_seq_len, 768]),
        0.0,
        1.0,
        device.cuda_device_arc(),
    )?;
    let pooled_embeds = Tensor::randn(
        Shape::from_dims(&[batch_size, 768]),
        0.0,
        1.0,
        device.cuda_device_arc(),
    )?;

    // Concatenate embeddings for Flux
    let _encoder_hidden_states = Tensor::cat(&[&clip_embeds, &t5_embeds], 1)?;
    let _ = pooled_embeds;
    println!("  ✅ Text encoded (placeholder)");

    // Initialize scheduler
    println!("\n🎯 Starting diffusion process...");
    let scheduler = FluxScheduler::new(device.clone());

    // Create initial noise
    let latent_height = args.height / 8;
    let latent_width = args.width / 8;
    let mut latents = Tensor::randn(
        Shape::from_dims(&[batch_size, 16, latent_height, latent_width]),
        0.0,
        1.0,
        device.cuda_device_arc(),
    )?;

    // Get timesteps
    let timesteps = scheduler.get_timesteps(args.steps);

    // Denoising loop
    for (i, &t) in timesteps.iter().enumerate() {
        if i % 5 == 0 {
            println!("  Step {}/{}", i + 1, args.steps);
        }

        // Predict noise (simplified - in real implementation, properly call the model)
        // For now, just create random noise as placeholder
        let noise_pred =
            Tensor::randn(latents.shape().clone(), 0.0, 1.0, device.cuda_device_arc())?;

        latents = scheduler.step(&noise_pred, &latents, t)?;
    }

    println!("  ✅ Denoising complete");

    // Decode latents with VAE
    println!("\n🎨 Decoding image...");
    let images = vae.decode(&latents)?;

    // Convert to image and save
    println!("💾 Saving image to {}...", args.output);
    save_tensor_as_image(&images, &args.output)?;

    println!("✅ Done! Image saved to {}", args.output);

    Ok(())
}

fn save_tensor_as_image(tensor: &Tensor, path: &str) -> Result<()> {
    // Get tensor data
    let shape = tensor.shape();
    let dims = shape.dims();

    // Assume tensor is [1, 3, H, W]
    let height = dims[2];
    let width = dims[3];

    // Convert to CPU and get data
    let tensor_f32 = tensor.to_dtype(DType::F32)?;
    let data: Vec<f32> = tensor_f32.to_vec()?;

    // Create image buffer
    let mut img = image::RgbImage::new(width as u32, height as u32);

    // Fill image (convert from CHW to HWC)
    for y in 0..height {
        for x in 0..width {
            let r_idx = y * width + x;
            let g_idx = height * width + y * width + x;
            let b_idx = 2 * height * width + y * width + x;

            // Clamp and scale from [-1, 1] to [0, 255]
            let r = ((data[r_idx] * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0) as u8;
            let g = ((data[g_idx] * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0) as u8;
            let b = ((data[b_idx] * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0) as u8;

            img.put_pixel(x as u32, y as u32, image::Rgb([r, g, b]));
        }
    }

    // Save image
    img.save(path).map_err(|e| {
        flame_core::Error::InvalidOperation(format!("Failed to save image: {}", e))
    })?;

    Ok(())
}
