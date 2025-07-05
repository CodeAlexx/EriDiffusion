//! Flux image generation example with eridiffusion-rs

use eridiffusion_core::Device;
use candle_core::{DType, Tensor, D};
use candle_nn::VarBuilder;
use candle_transformers::models::flux;
use clap::Parser;
use std::path::PathBuf;
use image;
use tokenizers::Tokenizer;

#[derive(Parser)]
struct Args {
    /// Text prompt for image generation
    #[arg(long, default_value = "A cyberpunk city at night with neon lights")]
    prompt: String,
    
    /// Image width (must be multiple of 16)
    #[arg(long, default_value = "1024")]
    width: usize,
    
    /// Image height (must be multiple of 16)
    #[arg(long, default_value = "1024")]
    height: usize,
    
    /// Number of inference steps
    #[arg(long, default_value = "4")]
    steps: usize,
    
    /// Guidance scale
    #[arg(long, default_value = "3.5")]
    guidance: f32,
    
    /// Random seed for reproducibility
    #[arg(long)]
    seed: Option<u64>,
    
    /// Output image path
    #[arg(long, default_value = "flux_output.png")]
    output: PathBuf,
    
    /// Use CPU instead of GPU
    #[arg(long)]
    cpu: bool,
    
    /// Path to Flux model directory
    #[arg(long, default_value = "/home/alex/SwarmUI/Models/diffusion_models/flux1-schnell.safetensors")]
    model_path: PathBuf,
    
    /// Path to T5 encoder
    #[arg(long, default_value = "/home/alex/SwarmUI/Models/text_encoder/t5-v1_1-xxl")]
    t5_path: PathBuf,
    
    /// Path to CLIP encoder
    #[arg(long, default_value = "/home/alex/SwarmUI/Models/CLIP/clip_l.safetensors")]
    clip_path: PathBuf,
    
    /// Path to VAE decoder
    #[arg(long, default_value = "/home/alex/SwarmUI/Models/VAE/ae.safetensors")]
    vae_path: PathBuf,
    
    /// Use Flux-Schnell variant
    #[arg(long)]
    schnell: bool,
}

fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    let args = Args::parse();
    
    println!("🎨 Flux Image Generation with eridiffusion-rs\n");
    println!("Configuration:");
    println!("  Model: Flux {}", if args.schnell { "Schnell" } else { "Dev" });
    println!("  Prompt: {}", args.prompt);
    println!("  Resolution: {}x{}", args.width, args.height);
    println!("  Steps: {}", args.steps);
    println!("  Guidance: {}", args.guidance);
    println!("  Seed: {:?}", args.seed);
    println!("  Device: {}", if args.cpu { "CPU" } else { "CUDA" });
    
    // Validate dimensions
    if args.width % 16 != 0 || args.height % 16 != 0 {
        anyhow::bail!("Width and height must be multiples of 16");
    }
    
    // Set up device
    let device = if args.cpu {
        candle_core::Device::Cpu
    } else {
        candle_core::Device::cuda(0)?
    };
    
    // Load Flux model
    println!("\n⏳ Loading Flux model components...");
    
    // Load the main Flux transformer
    let flux_config = if args.schnell {
        flux::Config::schnell()
    } else {
        flux::Config::dev()
    };
    
    let tensors = unsafe { candle_core::safetensors::load(&args.model_path, &device)? };
    let vb = VarBuilder::from_tensors(tensors, DType::BF16, &device);
    let flux_model = flux::Flux::new(&flux_config, vb)?;
    
    // Load T5 encoder
    println!("Loading T5 text encoder...");
    let t5_tensors = unsafe { candle_core::safetensors::load(&args.t5_path.join("model.safetensors"), &device)? };
    let t5_vb = VarBuilder::from_tensors(t5_tensors, DType::F32, &device);
    let t5_config = candle_transformers::models::t5::Config::t5_xxl();
    let mut t5_model = candle_transformers::models::t5::T5EncoderModel::load(t5_vb, &t5_config)?;
    
    // Load CLIP encoder
    println!("Loading CLIP encoder...");
    let clip_tensors = unsafe { candle_core::safetensors::load(&args.clip_path, &device)? };
    let clip_vb = VarBuilder::from_tensors(clip_tensors, DType::F32, &device);
    let clip_config = candle_transformers::models::clip::text_model::Config::clip_large();
    let clip_model = candle_transformers::models::clip::text_model::ClipTextTransformer::new(&clip_config, clip_vb)?;
    
    // Load VAE decoder
    println!("Loading VAE decoder...");
    let vae_tensors = unsafe { candle_core::safetensors::load(&args.vae_path, &device)? };
    let vae_vb = VarBuilder::from_tensors(vae_tensors, DType::BF16, &device);
    let vae = flux::autoencoder::AutoEncoder::new(&flux::autoencoder::Config::dev(), vae_vb)?;
    
    println!("✅ Model components loaded successfully");
    
    // Encode prompt with T5
    println!("\n📝 Encoding prompt with T5...");
    let t5_tokenizer = Tokenizer::from_file(args.t5_path.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("Failed to load T5 tokenizer: {}", e))?;
    
    let t5_tokens = t5_tokenizer.encode(&args.prompt, false)
        .map_err(|e| anyhow::anyhow!("T5 tokenization failed: {}", e))?;
    let t5_token_ids = Tensor::new(t5_tokens.get_ids(), &device)?.unsqueeze(0)?;
    
    let t5_output = t5_model.forward(&t5_token_ids)?;
    
    // Encode prompt with CLIP
    println!("📝 Encoding prompt with CLIP...");
    let clip_tokenizer = candle_transformers::models::clip::Tokenizer::from_file(
        "/home/alex/SwarmUI/Models/CLIP/tokenizer.json"
    )?;
    let clip_tokens = clip_tokenizer.encode(&args.prompt)?;
    let clip_token_ids = Tensor::new(clip_tokens.as_slice(), &device)?.unsqueeze(0)?;
    
    let clip_output = clip_model.forward(&clip_token_ids)?;
    let pooled_output = clip_output.i((.., 0, ..))?; // Use CLS token
    
    println!("✅ Prompts encoded");
    
    // Prepare latents
    println!("\n🎲 Generating initial latents...");
    let latent_height = args.height / 8;
    let latent_width = args.width / 8;
    let latent_channels = 16; // Flux uses 16 latent channels
    
    let mut rng = if let Some(seed) = args.seed {
        rand::rngs::StdRng::seed_from_u64(seed)
    } else {
        rand::rngs::StdRng::from_entropy()
    };
    
    use rand::Rng;
    let latents_vec: Vec<f32> = (0..1 * latent_channels * latent_height * latent_width)
        .map(|_| rng.gen::<f32>() * 2.0 - 1.0)
        .collect();
    
    let mut latents = Tensor::from_vec(
        latents_vec,
        (1, latent_channels, latent_height, latent_width),
        &device
    )?;
    
    // Prepare image IDs for positional encoding
    let img_ids = {
        let mut ids = Vec::new();
        for y in 0..latent_height {
            for x in 0..latent_width {
                ids.push(vec![0u32, y as u32, x as u32]);
            }
        }
        let ids_flat: Vec<u32> = ids.into_iter().flatten().collect();
        Tensor::from_vec(ids_flat, (1, latent_height * latent_width, 3), &device)?
    };
    
    // Prepare text IDs
    let txt_ids = {
        let seq_len = t5_output.dim(1)?;
        let mut ids = Vec::new();
        for pos in 0..seq_len {
            ids.push(vec![0u32, 0u32, pos as u32]);
        }
        let ids_flat: Vec<u32> = ids.into_iter().flatten().collect();
        Tensor::from_vec(ids_flat, (1, seq_len, 3), &device)?
    };
    
    // Flux flow matching generation
    println!("\n🔄 Running flow matching generation...");
    
    // Schnell uses a different schedule
    let timesteps = if args.schnell {
        vec![1.0, 0.75, 0.5, 0.25]
    } else {
        // Linear schedule for dev
        (0..args.steps).map(|i| 1.0 - (i as f32 / args.steps as f32)).collect()
    };
    
    for (i, &t) in timesteps.iter().enumerate() {
        print!("\r  Step {}/{}", i + 1, timesteps.len());
        use std::io::Write;
        std::io::stdout().flush()?;
        
        // Prepare timestep tensor
        let timestep = Tensor::new(&[t], &device)?;
        
        // Prepare guidance tensor
        let guidance_vec = if args.schnell {
            vec![args.guidance] // Schnell uses guidance differently
        } else {
            vec![args.guidance]
        };
        let guidance = Tensor::new(&guidance_vec, &device)?;
        
        // Patchify latents for Flux (2x2 patches)
        let img = patchify(&latents)?;
        
        // Run Flux forward pass
        let pred = flux_model.forward(
            &img,
            &img_ids,
            &t5_output,
            &txt_ids,
            &timestep,
            &pooled_output,
            Some(&guidance),
        )?;
        
        // Unpatchify back to latent shape
        let pred = unpatchify(&pred, latent_height, latent_width)?;
        
        // Flow matching update (simplified for Schnell)
        if args.schnell {
            // Schnell uses a more direct update
            latents = ((latents * (1.0 - t))? + (pred * t)?)?;
        } else {
            // Standard flow matching update
            let dt = if i < timesteps.len() - 1 {
                timesteps[i] - timesteps[i + 1]
            } else {
                timesteps[i]
            };
            latents = (latents - (pred * dt)?)?;
        }
    }
    
    println!("\n✅ Generation complete");
    
    // Decode latents to image
    println!("\n🖼️  Decoding latents to image...");
    let image = vae.decode(&latents)?;
    
    // Convert to RGB image
    let image = ((image + 1.0)? * 127.5)?
        .to_dtype(DType::U8)?
        .clamp(0.0, 255.0)?;
    
    // Save image
    let (_, c, h, w) = image.dims4()?;
    let image_data = image.permute((0, 2, 3, 1))?.flatten_all()?.to_vec1::<u8>()?;
    
    let img = image::RgbImage::from_raw(w as u32, h as u32, image_data)
        .ok_or_else(|| anyhow::anyhow!("Failed to create image from tensor data"))?;
    
    img.save(&args.output)?;
    println!("✅ Image saved to: {}", args.output.display());
    
    println!("\n🎉 Generation complete!");
    
    Ok(())
}

/// Patchify latents for Flux (2x2 patches)
fn patchify(latents: &Tensor) -> anyhow::Result<Tensor> {
    let (b, c, h, w) = latents.dims4()?;
    
    // Ensure dimensions are divisible by 2
    if h % 2 != 0 || w % 2 != 0 {
        anyhow::bail!("Latent dimensions must be divisible by 2");
    }
    
    // Reshape to patches
    let latents = latents.reshape((b, c, h / 2, 2, w / 2, 2))?;
    let latents = latents.permute((0, 2, 4, 3, 5, 1))?; // (b, h/2, w/2, 2, 2, c)
    let latents = latents.reshape((b, (h / 2) * (w / 2), 4 * c))?;
    
    Ok(latents)
}

/// Unpatchify output back to latent shape
fn unpatchify(output: &Tensor, h: usize, w: usize) -> anyhow::Result<Tensor> {
    let (b, seq_len, channels) = output.dims3()?;
    let c = channels / 4;
    
    // Reshape from patches
    let output = output.reshape((b, h / 2, w / 2, 2, 2, c))?;
    let output = output.permute((0, 5, 1, 3, 2, 4))?; // (b, c, h/2, 2, w/2, 2)
    let output = output.reshape((b, c, h, w))?;
    
    Ok(output)
}