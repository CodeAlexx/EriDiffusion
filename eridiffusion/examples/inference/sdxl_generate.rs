//! SDXL image generation example with eridiffusion-rs

use eridiffusion_core::{Device, ModelInputs};
use eridiffusion_models::{SDXLModel, ModelFactory, DiffusionModel};
use eridiffusion_inference::{InferencePipeline, InferenceConfig, SchedulerType, OutputType};
use candle_core::{DType, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::stable_diffusion::{self, clip};
use clap::Parser;
use std::path::PathBuf;
use image;

#[derive(Parser)]
struct Args {
    /// Text prompt for image generation
    #[arg(long, default_value = "A majestic lion in a savanna at sunset")]
    prompt: String,
    
    /// Negative prompt (what to avoid)
    #[arg(long, default_value = "")]
    negative_prompt: String,
    
    /// Image width
    #[arg(long, default_value = "1024")]
    width: usize,
    
    /// Image height  
    #[arg(long, default_value = "1024")]
    height: usize,
    
    /// Number of inference steps
    #[arg(long, default_value = "30")]
    steps: usize,
    
    /// Guidance scale (higher = more prompt adherence)
    #[arg(long, default_value = "7.5")]
    cfg_scale: f32,
    
    /// Random seed for reproducibility
    #[arg(long)]
    seed: Option<u64>,
    
    /// Output image path
    #[arg(long, default_value = "sdxl_output.png")]
    output: PathBuf,
    
    /// Use CPU instead of GPU
    #[arg(long)]
    cpu: bool,
    
    /// Path to SDXL model directory
    #[arg(long, default_value = "/home/alex/SwarmUI/Models/Stable-Diffusion/sdXL_v10.safetensors")]
    model_path: PathBuf,
}

fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    let args = Args::parse();
    
    println!("🎨 SDXL Image Generation with eridiffusion-rs\n");
    println!("Configuration:");
    println!("  Model: SDXL 1.0");
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
    
    // Load SDXL components from safetensors
    println!("\n⏳ Loading SDXL model components...");
    
    // Load the safetensors file
    let tensors = unsafe { candle_core::safetensors::load(&args.model_path, &candle_device)? };
    let vb = VarBuilder::from_tensors(tensors, DType::F16, &candle_device);
    
    // Create CLIP text encoders
    let text_encoder = {
        let config = clip::Config::sdxl();
        let text_model = clip::ClipTextTransformer::new(vb.pp("conditioner.embedders.0.transformer"), &config)?;
        text_model
    };
    
    let text_encoder_2 = {
        let config = clip::Config::sdxl2();  
        let text_model = clip::ClipTextTransformer::new(vb.pp("conditioner.embedders.1.model"), &config)?;
        text_model
    };
    
    // Create tokenizers
    // Check multiple possible locations for tokenizer files
    let tokenizer_path = if std::path::Path::new("/home/alex/SwarmUI/Models/CLIP/tokenizer.json").exists() {
        "/home/alex/SwarmUI/Models/CLIP/tokenizer.json"
    } else {
        "tokenizer.json" // Fallback to local
    };
    
    let tokenizer_2_path = if std::path::Path::new("/home/alex/SwarmUI/Models/CLIP/tokenizer_2.json").exists() {
        "/home/alex/SwarmUI/Models/CLIP/tokenizer_2.json"
    } else {
        "tokenizer_2.json" // Fallback to local
    };
    
    let tokenizer = clip::Tokenizer::from_file(tokenizer_path)?;
    let tokenizer_2 = clip::Tokenizer::from_file(tokenizer_2_path)?;
    
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
                    attention_head_dim: 5,
                },
                stable_diffusion::unet_2d::BlockConfig {
                    out_channels: 640,
                    use_cross_attn: Some(2),
                    attention_head_dim: 10,
                },
                stable_diffusion::unet_2d::BlockConfig {
                    out_channels: 1280,
                    use_cross_attn: Some(10),
                    attention_head_dim: 20,
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
            cross_attention_dim: 2048,
            sliced_attention_size: None,
            use_linear_projection: true,
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
    let tokens_2 = tokenizer_2.encode(&args.prompt)?;
    let neg_tokens = tokenizer.encode(&args.negative_prompt)?;
    let neg_tokens_2 = tokenizer_2.encode(&args.negative_prompt)?;
    
    // Convert to tensors
    let tokens = Tensor::new(tokens.as_slice(), &candle_device)?.unsqueeze(0)?;
    let tokens_2 = Tensor::new(tokens_2.as_slice(), &candle_device)?.unsqueeze(0)?;
    let neg_tokens = Tensor::new(neg_tokens.as_slice(), &candle_device)?.unsqueeze(0)?;
    let neg_tokens_2 = Tensor::new(neg_tokens_2.as_slice(), &candle_device)?.unsqueeze(0)?;
    
    // Encode with CLIP
    let text_embeddings = text_encoder.forward(&tokens)?;
    let text_embeddings_2 = text_encoder_2.forward(&tokens_2)?;
    let neg_embeddings = text_encoder.forward(&neg_tokens)?;
    let neg_embeddings_2 = text_encoder_2.forward(&neg_tokens_2)?;
    
    // Concatenate embeddings for SDXL
    let text_embeddings = Tensor::cat(&[&text_embeddings, &text_embeddings_2], 2)?;
    let neg_embeddings = Tensor::cat(&[&neg_embeddings, &neg_embeddings_2], 2)?;
    
    // For classifier-free guidance
    let text_embeddings = Tensor::cat(&[&neg_embeddings, &text_embeddings], 0)?;
    
    println!("✅ Prompts encoded");
    
    // Initialize scheduler
    let num_steps = args.steps;
    let scheduler = stable_diffusion::schedulers::ddim::DDIMScheduler::new(num_steps);
    
    // Generate initial latents
    println!("\n🎲 Generating initial latents...");
    let latents_shape = (1, 4, args.height / 8, args.width / 8);
    let mut latents = if let Some(seed) = args.seed {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        Tensor::randn(0.0f32, 1.0, latents_shape, &candle_device)?
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
        let latent_model_input = Tensor::cat(&[&latents, &latents], 0)?;
        
        // Scale model input by scheduler
        let latent_model_input = scheduler.scale_model_input(&latent_model_input, timestep)?;
        
        // Predict noise
        let noise_pred = unet.forward(
            &latent_model_input,
            timestep as f64,
            &text_embeddings
        )?;
        
        // Perform guidance
        let noise_pred_chunks = noise_pred.chunk(2, 0)?;
        let noise_pred_uncond = &noise_pred_chunks[0];
        let noise_pred_text = &noise_pred_chunks[1];
        
        let noise_pred = (noise_pred_uncond + ((noise_pred_text - noise_pred_uncond)? * args.cfg_scale as f64)?)?;
        
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