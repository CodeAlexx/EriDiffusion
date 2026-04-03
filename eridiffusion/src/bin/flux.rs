#![cfg(feature = "legacy-bins")]

// NOTE: Legacy binary; gated behind the `legacy-bins` feature so it stays out of default builds. Enable with `--features legacy-bins` if you still rely on it.

use clap::{Parser, Subcommand};
use flame_core::device::Device;
use flame_core::{DType, Result};
use serde_yaml;
use std::path::PathBuf;

/// Flux CLI - Unified tool for training and inference
#[derive(Parser)]
#[clap(name = "flux", about = "Flux diffusion model training and inference")]
struct Args {
    #[clap(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Train a LoRA adapter
    Train {
        /// Path to training config YAML
        #[clap(short, long)]
        config: PathBuf,

        /// Model variant (dev or schnell)
        #[clap(short, long, default_value = "dev")]
        variant: String,

        /// Output directory
        #[clap(short, long)]
        output: Option<PathBuf>,

        /// Use INT8 quantization for 24GB GPUs
        #[clap(long)]
        int8: bool,
    },

    /// Generate images  
    Generate {
        /// Text prompt
        #[clap(short, long)]
        prompt: String,

        /// Model variant (dev or schnell)
        #[clap(short, long, default_value = "dev")]
        variant: String,

        /// LoRA adapter path
        #[clap(long)]
        lora: Option<PathBuf>,

        /// LoRA strength
        #[clap(long, default_value = "1.0")]
        lora_scale: f32,

        /// Output image path
        #[clap(short, long, default_value = "output.png")]
        output: PathBuf,

        /// Number of inference steps
        #[clap(long, default_value = "20")]
        steps: usize,

        /// Guidance scale (3.5 for Dev, 1.0 for Schnell)
        #[clap(long)]
        cfg: Option<f64>,

        /// Image width
        #[clap(long, default_value = "1024")]
        width: usize,

        /// Image height  
        #[clap(long, default_value = "1024")]
        height: usize,
    },
}

fn main() -> flame_core::Result<()> {
    let args = Args::parse();

    match args.command {
        Commands::Train { config, variant, output, int8 } => {
            train_lora(config, variant, output, int8)
        }
        Commands::Generate {
            prompt,
            variant,
            lora,
            lora_scale,
            output,
            steps,
            cfg,
            width,
            height,
        } => generate_image(prompt, variant, lora, lora_scale, output, steps, cfg, width, height),
    }
}

fn train_lora(
    config: PathBuf,
    _variant: String,
    _output: Option<PathBuf>,
    _int8: bool,
) -> flame_core::Result<()> {
    // Delegate to the unified trainer entry that handles YAML and pipelines.
    eridiffusion::trainers::train_from_config(config)
}

fn generate_image(
    prompt: String,
    variant: String,
    lora: Option<PathBuf>,
    lora_scale: f32,
    output: PathBuf,
    steps: usize,
    cfg: Option<f64>,
    width: usize,
    height: usize,
) -> flame_core::Result<()> {
    use eridiffusion::inference::flux::generate_flux_image;

    println!("Generating Flux {} image...", variant);
    println!("Prompt: {}", prompt);

    // Default CFG based on variant
    let cfg_scale = cfg.unwrap_or(if variant == "schnell" { 1.0 } else { 3.5 });

    let device = Device::cuda(0)?;
    let dtype = DType::F16;

    // Generate image
    let lora_opt: Option<&str> = lora.as_deref().and_then(|p| p.to_str());
    generate_flux_image(
        &prompt, &variant, lora_opt, lora_scale, &output, steps, cfg_scale, width, height, device,
        dtype,
    )?;

    println!("Image saved to: {}", output.display());
    Ok(())
}
