//! SD 1.5 image generation example with eridiffusion-rs

use eridiffusion_core::{Device, ModelInputs};
use eridiffusion_models::{SD15Model, ModelFactory, DiffusionModel};
use candle_core::{DType, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::stable_diffusion::{self, clip};
use clap::Parser;
use std::path::PathBuf;
use image;

#[derive(Parser)]
struct Args {
    /// Text prompt for image generation
    #[arg(long, default_value = "A fantasy landscape with mountains and a river")]
    prompt: String,
    
    /// Negative prompt (what to avoid)
    #[arg(long, default_value = "")]
    negative_prompt: String,
    
    /// Image width
    #[arg(long, default_value = "512")]
    width: usize,
    
    /// Image height  
    #[arg(long, default_value = "512")]
    height: usize,
    
    /// Number of inference steps
    #[arg(long, default_value = "50")]
    steps: usize,
    
    /// Guidance scale (higher = more prompt adherence)
    #[arg(long, default_value = "7.5")]
    cfg_scale: f32,
    
    /// Random seed for reproducibility
    #[arg(long)]
    seed: Option<u64>,
    
    /// Output image path
    #[arg(long, default_value = "sd15_output.png")]
    output: PathBuf,
    
    /// Use CPU instead of GPU
    #[arg(long)]
    cpu: bool,
    
    /// Path to SD 1.5 model
    #[arg(long, default_value = "/home/alex/SwarmUI/Models/Stable-Diffusion/v1-5-pruned-emaonly.safetensors")]
    model_path: PathBuf,
}

fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    let args = Args::parse();
    
    println!("🎨 SD 1.5 Image Generation with eridiffusion-rs\n");
    println!("Configuration:");
    println!("  Model: SD 1.5");
    println!("  Prompt: {}", args.prompt);
    println!("  Negative: {}", args.negative_prompt);
    println!("  Resolution: {}x{}", args.width, args.height);
    println!("  Steps: {}", args.steps);
    println!("  CFG Scale: {}", args.cfg_scale);
    println!("  Seed: {:?}", args.seed);
    println!("  Device: {}", if args.cpu { "CPU" } else { "CUDA" });
    
    // Set up device
    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::Cuda(0)
    };
    
    let candle_device = match &device {
        Device::Cpu => candle_core::Device::Cpu,
        Device::Cuda(id) => candle_core::Device::cuda(*id)?,
    };
    
    // Load SD 1.5 components from safetensors
    println!("\n⏳ Loading SD 1.5 model components...");
    
    // Load the safetensors file
    let tensors = unsafe { candle_core::safetensors::load(&args.model_path, &candle_device)? };
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &candle_device);
    
    // Create CLIP text encoder
    let text_encoder = {
        let config = clip::Config::v1_5();
        let text_model = clip::ClipTextTransformer::new(vb.pp("cond_stage_model.transformer"), &config)?;
        text_model
    };
    
    // Create tokenizer
    let tokenizer = clip::Tokenizer::from_file("/home/alex/SwarmUI/Models/CLIP/tokenizer.json")?;
    
    // Create VAE
    let vae = {
        let config = stable_diffusion::vae::AutoEncoderKLConfig {
            block_out_channels: vec![128, 256, 512, 512],
            layers_per_block: 2,
            latent_channels: 4,
            norm_num_groups: 32,
            use_quant_conv: true,
            use_post_quant_conv: true,
        };
        stable_diffusion::vae::AutoEncoderKL::new(
            vb.pp("first_stage_model"),
            3, 3, config
        )?
    };
    
    // Create UNet
    let unet = {
        let config = stable_diffusion::unet_2d::UNet2DConditionModelConfig {
            blocks: vec![
                stable_diffusion::unet_2d::BlockConfig {
                    out_channels: 320,
                    use_cross_attn: Some(1),
                    attention_head_dim: 8,
                },
                stable_diffusion::unet_2d::BlockConfig {
                    out_channels: 640,
                    use_cross_attn: Some(1),
                    attention_head_dim: 8,
                },
                stable_diffusion::unet_2d::BlockConfig {
                    out_channels: 1280,
                    use_cross_attn: Some(1),
                    attention_head_dim: 8,
                },
                stable_diffusion::unet_2d::BlockConfig {
                    out_channels: 1280,
                    use_cross_attn: None,
                    attention_head_dim: 8,
                },
            ],
            center_input_sample: false,
            flip_sin_to_cos: true,
            freq_shift: 0.,
            layers_per_block: 2,
            downsample_padding: 1,
            mid_block_scale_factor: 1.,
            norm_num_groups: 32,
            norm_eps: 1e-5,
            cross_attention_dim: 768,
            sliced_attention_size: None,
            use_linear_projection: false,
        };
        stable_diffusion::unet_2d::UNet2DConditionModel::new(
            vb.pp("model.diffusion_model"),
            4, 4, false, config
        )?
    };
    
    println!("✅ Model components loaded successfully");
    
    // Encode prompts
    println!("\n📝 Encoding prompts...");
    
    // Tokenize prompts
    let tokens = tokenizer.encode(&args.prompt)?;
    let neg_tokens = tokenizer.encode(&args.negative_prompt)?;
    
    // Pad to max length (77 for SD 1.5)
    let max_len = 77;
    let mut padded_tokens = tokens.clone();
    padded_tokens.resize(max_len, 49407); // pad token
    let mut padded_neg_tokens = neg_tokens.clone();
    padded_neg_tokens.resize(max_len, 49407);
    
    // Convert to tensors
    let tokens = Tensor::new(padded_tokens.as_slice(), &candle_device)?.unsqueeze(0)?;
    let neg_tokens = Tensor::new(padded_neg_tokens.as_slice(), &candle_device)?.unsqueeze(0)?;
    
    // Encode with CLIP
    let text_embeddings = text_encoder.forward(&tokens)?;
    let neg_embeddings = text_encoder.forward(&neg_tokens)?;
    
    // For classifier-free guidance
    let text_embeddings = if args.cfg_scale > 1.0 {
        Tensor::cat(&[&neg_embeddings, &text_embeddings], 0)?
    } else {
        text_embeddings
    };
    
    println!("✅ Prompts encoded");
    
    // Initialize scheduler
    let num_steps = args.steps;
    let scheduler = stable_diffusion::schedulers::ddim::DDIMScheduler::new(num_steps);
    
    // Generate initial latents
    println!("\n🎲 Generating initial latents...");
    let latents_shape = (1, 4, args.height / 8, args.width / 8);
    let mut latents = if let Some(seed) = args.seed {
        use rand::{SeedableRng, Rng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let latents_vec: Vec<f32> = (0..latents_shape.0 * latents_shape.1 * latents_shape.2 * latents_shape.3)
            .map(|_| rng.gen::<f32>() * 2.0 - 1.0)
            .collect();
        Tensor::from_vec(latents_vec, latents_shape, &candle_device)?
    } else {
        Tensor::randn(0.0f32, 1.0, latents_shape, &candle_device)?
    };
    
    // Scale initial noise by scheduler
    latents = (latents * scheduler.init_noise_sigma())?;
    
    // Denoising loop
    println!("\n🔄 Running denoising loop...");
    let timesteps = scheduler.timesteps();
    let total_steps = timesteps.len();
    
    for (i, &timestep) in timesteps.iter().enumerate() {
        print!("\r  Step {}/{}", i + 1, total_steps);
        use std::io::Write;
        std::io::stdout().flush()?;
        
        // Expand latents for classifier-free guidance
        let latent_model_input = if args.cfg_scale > 1.0 {
            Tensor::cat(&[&latents, &latents], 0)?
        } else {
            latents.clone()
        };
        
        // Scale model input by scheduler
        let latent_model_input = scheduler.scale_model_input(&latent_model_input, timestep)?;
        
        // Predict noise
        let noise_pred = unet.forward(
            &latent_model_input,
            timestep as f64,
            &text_embeddings
        )?;
        
        // Perform guidance
        let noise_pred = if args.cfg_scale > 1.0 {
            let noise_pred_chunks = noise_pred.chunk(2, 0)?;
            let noise_pred_uncond = &noise_pred_chunks[0];
            let noise_pred_text = &noise_pred_chunks[1];
            
            (noise_pred_uncond + ((noise_pred_text - noise_pred_uncond)? * args.cfg_scale as f64)?)?
        } else {
            noise_pred
        };
        
        // Scheduler step
        latents = scheduler.step(&noise_pred, timestep, &latents)?;
    }
    
    println!("\n✅ Denoising complete");
    
    // Decode latents to image
    println!("\n🖼️  Decoding latents to image...");
    let latents_scaled = (latents / 0.18215)?;
    let image = vae.decode(&latents_scaled)?;
    
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