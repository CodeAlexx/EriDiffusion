#![cfg(feature = "legacy-bins")]

// NOTE: Legacy binary; gated behind the `legacy-bins` feature so it stays out of default builds. Enable with `--features legacy-bins` if you still rely on it.

#![cfg(feature = "samples")]

use clap::{Parser, Subcommand};
use eridiffusion::samplers::{FluxSampler, SD35Sampler, SDXLSampler, SamplingConfig};
use flame_core::device::Device;
use flame_core::Result;
use flame_core::{DType, Device};
use std::path::PathBuf;

// External sampling utility for SDXL, SD 3.5, and Flux models
// This can be used independently of training to generate images

#[derive(Parser)]
#[command(author, version, about = "Generate images with diffusion models", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate images with SDXL
    Sdxl {
        /// Text prompt
        #[arg(long)]
        prompt: String,

        /// Negative prompt
        #[arg(long, default_value = "")]
        negative_prompt: String,

        /// Path to SDXL model
        #[arg(
            long,
            default_value = "/home/alex/SwarmUI/Models/Stable-Diffusion/sdXL_v10.safetensors"
        )]
        model: PathBuf,

        /// Path to CLIP tokenizer
        #[arg(long, default_value = "/home/alex/SwarmUI/Models/CLIP/tokenizer.json")]
        tokenizer: PathBuf,

        /// Path to CLIP tokenizer 2
        #[arg(long, default_value = "/home/alex/SwarmUI/Models/CLIP/tokenizer_2.json")]
        tokenizer_2: PathBuf,

        /// Path to LoRA weights (optional)
        #[arg(long)]
        lora: Option<PathBuf>,

        /// Output directory
        #[arg(long, default_value = "./outputs")]
        output_dir: PathBuf,

        /// Number of inference steps
        #[arg(long, default_value_t = 30)]
        steps: usize,

        /// Guidance scale
        #[arg(long, default_value_t = 7.5)]
        cfg_scale: f64,

        /// Image width
        #[arg(long, default_value_t = 1024)]
        width: usize,

        /// Image height
        #[arg(long, default_value_t = 1024)]
        height: usize,

        /// Random seed (optional)
        #[arg(long)]
        seed: Option<u64>,

        /// Number of images to generate
        #[arg(long, default_value_t = 1)]
        num_images: usize,
    },

    /// Generate images with SD 3.5
    Sd35 {
        /// Text prompt
        #[arg(long)]
        prompt: String,

        /// Negative prompt
        #[arg(long, default_value = "")]
        negative_prompt: String,

        /// Model variant: medium, large, or large-turbo
        #[arg(long, default_value = "large")]
        variant: String,

        /// Path to SD 3.5 model
        #[arg(long)]
        model: Option<PathBuf>,

        /// Path to LoRA/LoKr weights (optional)
        #[arg(long)]
        adapter: Option<PathBuf>,

        /// Output directory
        #[arg(long, default_value = "./outputs")]
        output_dir: PathBuf,

        /// Number of inference steps
        #[arg(long, default_value_t = 28)]
        steps: usize,

        /// Guidance scale
        #[arg(long, default_value_t = 7.0)]
        cfg_scale: f64,

        /// Shift parameter
        #[arg(long, default_value_t = 3.0)]
        shift: f64,

        /// Image width
        #[arg(long, default_value_t = 1024)]
        width: usize,

        /// Image height
        #[arg(long, default_value_t = 1024)]
        height: usize,

        /// Random seed (optional)
        #[arg(long)]
        seed: Option<u64>,

        /// Number of images to generate
        #[arg(long, default_value_t = 1)]
        num_images: usize,
    },

    /// Generate images with Flux
    Flux {
        /// Text prompt
        #[arg(long)]
        prompt: String,

        /// Model variant: dev or schnell
        #[arg(long, default_value = "dev")]
        variant: String,

        /// Path to Flux model
        #[arg(long)]
        model: Option<PathBuf>,

        /// Path to LoRA weights (optional)
        #[arg(long)]
        lora: Option<PathBuf>,

        /// Output directory
        #[arg(long, default_value = "./outputs")]
        output_dir: PathBuf,

        /// Number of inference steps
        #[arg(long, default_value_t = 20)]
        steps: usize,

        /// Guidance scale (1.0 for schnell, 3.5 for dev)
        #[arg(long)]
        cfg_scale: Option<f64>,

        /// Image width
        #[arg(long, default_value_t = 1024)]
        width: usize,

        /// Image height
        #[arg(long, default_value_t = 1024)]
        height: usize,

        /// Random seed (optional)
        #[arg(long)]
        seed: Option<u64>,

        /// Number of images to generate
        #[arg(long, default_value_t = 1)]
        num_images: usize,

        /// Use INT8 quantization for memory efficiency
        #[arg(long)]
        int8: bool,
    },
}

fn main() -> flame_core::Result<()> {
    // Initialize logging
    env_logger::init_from_env(env_logger::Env::default().default_filter_or("info"));

    let cli = Cli::parse();
    let device = Device::cuda(0)?;

    match cli.command {
        Commands::Sdxl {
            prompt,
            negative_prompt,
            model,
            tokenizer,
            tokenizer_2,
            lora,
            output_dir,
            steps,
            cfg_scale,
            width,
            height,
            seed,
            num_images,
        } => {
            sample_sdxl(
                &prompt,
                &negative_prompt,
                &model,
                &tokenizer,
                &tokenizer_2,
                lora.as_deref(),
                &output_dir,
                steps,
                cfg_scale,
                width,
                height,
                seed,
                num_images,
                &device,
            )?;
        }

        Commands::Sd35 {
            prompt,
            negative_prompt,
            variant,
            model,
            adapter,
            output_dir,
            steps,
            cfg_scale,
            shift,
            width,
            height,
            seed,
            num_images,
        } => {
            sample_sd35(
                &prompt,
                &negative_prompt,
                &variant,
                model.as_deref(),
                adapter.as_deref(),
                &output_dir,
                steps,
                cfg_scale,
                shift,
                width,
                height,
                seed,
                num_images,
                &device,
            )?;
        }

        Commands::Flux {
            prompt,
            variant,
            model,
            lora,
            output_dir,
            steps,
            cfg_scale,
            width,
            height,
            seed,
            num_images,
            int8,
        } => {
            sample_flux(
                &prompt,
                &variant,
                model.as_deref(),
                lora.as_deref(),
                &output_dir,
                steps,
                cfg_scale,
                width,
                height,
                seed,
                num_images,
                int8,
                &device,
            )?;
        }
    }

    Ok(())
}

fn sample_sdxl(
    prompt: &str,
    negative_prompt: &str,
    model_path: &Path,
    tokenizer_path: &Path,
    tokenizer_2_path: &Path,
    lora_path: Option<&Path>,
    output_dir: &Path,
    steps: usize,
    cfg_scale: f64,
    width: usize,
    height: usize,
    seed: Option<u64>,
    num_images: usize,
    device: &Device,
) -> flame_core::Result<()> {
    println!("=== SDXL Sampling ===");
    println!("Prompt: {}", prompt);
    println!("Model: {}", model_path.display());

    // Create output directory
    std::fs::create_dir_all(output_dir)
        .map_err(|e| {
            flame_core::Error::InvalidOperation(format!("Failed to create directory: {}", e))
        })
        .map_err(|e| flame_core::Error::InvalidOperation(e.to_string()))?;

    // Initialize sampler
    let dtype = DType::F16;
    let config = SamplingConfig {
        prompts: vec![prompt.to_string()],
        negative_prompt: negative_prompt.to_string(),
        num_inference_steps: steps,
        guidance_scale: cfg_scale,
        width,
        height,
        seed: seed.unwrap_or(42),
    };

    let sampler =
        SDXLSampler::new(model_path, tokenizer_path, tokenizer_2_path, lora_path, device, dtype)?;

    // Generate images
    for i in 0..num_images {
        println!("\nGenerating image {}/{}...", i + 1, num_images);

        let image = sampler.sample(&config)?;
        let output_path = output_dir.join(format!("sdxl_image_{:03}.png", i));

        // Save image
        image.save(&output_path)?;
        println!("Saved: {}", output_path.display());
    }

    println!("\n✓ SDXL sampling complete!");
    Ok(())
}

fn sample_sd35(
    prompt: &str,
    negative_prompt: &str,
    variant: &str,
    model_path: Option<&Path>,
    adapter_path: Option<&Path>,
    output_dir: &Path,
    steps: usize,
    cfg_scale: f64,
    shift: f64,
    width: usize,
    height: usize,
    seed: Option<u64>,
    num_images: usize,
    device: &Device,
) -> flame_core::Result<()> {
    println!("=== SD 3.5 Sampling ===");
    println!("Prompt: {}", prompt);
    println!("Variant: {}", variant);

    // Create output directory
    std::fs::create_dir_all(output_dir)
        .map_err(|e| {
            flame_core::Error::InvalidOperation(format!("Failed to create directory: {}", e))
        })
        .map_err(|e| flame_core::Error::InvalidOperation(e.to_string()))?;

    // Determine model path based on variant
    let default_model = match variant {
        "medium" => "/home/alex/SwarmUI/Models/diffusion_models/sd3.5_medium.safetensors",
        "large" => "/home/alex/SwarmUI/Models/diffusion_models/sd3.5_large.safetensors",
        "large-turbo" => "/home/alex/SwarmUI/Models/diffusion_models/sd3.5_large_turbo.safetensors",
        _ => return Err(anyhow::anyhow!("Unknown variant: {}", variant)),
    };

    let model_path =
        model_path.map(|p| p.to_path_buf()).unwrap_or_else(|| PathBuf::from(default_model));

    println!("Model: {}", model_path.display());

    // Initialize sampler
    let dtype = DType::F16;
    let config = SamplingConfig {
        prompts: vec![prompt.to_string()],
        negative_prompt: negative_prompt.to_string(),
        num_inference_steps: steps,
        guidance_scale: cfg_scale,
        width,
        height,
        seed: seed.unwrap_or(42),
    };

    let sampler = SD35Sampler::new(&model_path, adapter_path, device, dtype, shift)?;

    // Generate images
    for i in 0..num_images {
        println!("\nGenerating image {}/{}...", i + 1, num_images);

        let image = sampler.sample(&config)?;
        let output_path = output_dir.join(format!("sd35_{}_image_{:03}.png", variant, i));

        // Save image
        image.save(&output_path)?;
        println!("Saved: {}", output_path.display());
    }

    println!("\n✓ SD 3.5 sampling complete!");
    Ok(())
}

fn sample_flux(
    prompt: &str,
    variant: &str,
    model_path: Option<&Path>,
    lora_path: Option<&Path>,
    output_dir: &Path,
    steps: usize,
    cfg_scale: Option<f64>,
    width: usize,
    height: usize,
    seed: Option<u64>,
    num_images: usize,
    int8: bool,
    device: &Device,
) -> flame_core::Result<()> {
    println!("=== Flux Sampling ===");
    println!("Prompt: {}", prompt);
    println!("Variant: {}", variant);

    // Create output directory
    std::fs::create_dir_all(output_dir)
        .map_err(|e| {
            flame_core::Error::InvalidOperation(format!("Failed to create directory: {}", e))
        })
        .map_err(|e| flame_core::Error::InvalidOperation(e.to_string()))?;

    // Determine model path and cfg scale based on variant
    let (default_model, default_cfg) = match variant {
        "schnell" => ("/home/alex/SwarmUI/Models/diffusion_models/flux1-schnell.safetensors", 1.0),
        "dev" => ("/home/alex/SwarmUI/Models/diffusion_models/flux1-dev.safetensors", 3.5),
        _ => return Err(anyhow::anyhow!("Unknown variant: {}", variant)),
    };

    let model_path =
        model_path.map(|p| p.to_path_buf()).unwrap_or_else(|| PathBuf::from(default_model));
    let cfg_scale = cfg_scale.unwrap_or(default_cfg);

    println!("Model: {}", model_path.display());
    if int8 {
        println!("Using INT8 quantization");
    }

    // Initialize sampler
    let dtype = if int8 { DType::U8 } else { DType::F16 };
    let config = SamplingConfig {
        prompts: vec![prompt.to_string()],
        negative_prompt: String::new(), // Flux doesn't use negative prompts
        num_inference_steps: steps,
        guidance_scale: cfg_scale,
        width,
        height,
        seed: seed.unwrap_or(42),
    };

    let sampler = FluxSampler::new(&model_path, lora_path, device, dtype, variant == "schnell")?;

    // Generate images
    for i in 0..num_images {
        println!("\nGenerating image {}/{}...", i + 1, num_images);

        let image = sampler.sample(&config)?;
        let output_path = output_dir.join(format!("flux_{}_image_{:03}.png", variant, i));

        // Save image
        image.save(&output_path)?;
        println!("Saved: {}", output_path.display());
    }

    println!("\n✓ Flux sampling complete!");
    Ok(())
}
