#![cfg(feature = "legacy-bins")]

// NOTE: Legacy binary; gated behind the `legacy-bins` feature so it stays out of default builds. Enable with `--features legacy-bins` if you still rely on it.

use clap::{Parser, Subcommand};
use flame_core::device::Device;
use flame_core::Result;
use flame_core::{DType, Device};
use std::path::PathBuf;

/// SD 3.5 CLI - Unified tool for training and inference
#[derive(Parser)]
#[clap(name = "sd35", about = "Stable Diffusion 3.5 model training and inference")]
struct Args {
    #[clap(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Train a LoRA/LoKr adapter
    Train {
        /// Path to training config YAML
        #[clap(short, long)]
        config: PathBuf,

        /// Adapter type (lora or lokr)
        #[clap(long, default_value = "lokr")]
        adapter: String,

        /// Output directory
        #[clap(short, long)]
        output: Option<PathBuf>,
    },

    /// Generate images  
    Generate {
        /// Text prompt
        #[clap(short, long)]
        prompt: String,

        /// Negative prompt
        #[clap(short, long, default_value = "")]
        negative: String,

        /// Model variant (medium/large/large-turbo)
        #[clap(short, long, default_value = "large")]
        variant: String,

        /// LoRA/LoKr adapter path
        #[clap(long)]
        adapter: Option<PathBuf>,

        /// Adapter strength
        #[clap(long, default_value = "1.0")]
        adapter_scale: f32,

        /// Output image path
        #[clap(short, long, default_value = "output.png")]
        output: PathBuf,

        /// Number of inference steps
        #[clap(long, default_value = "28")]
        steps: usize,

        /// Guidance scale
        #[clap(long, default_value = "7.0")]
        cfg: f64,

        /// Shift parameter (1.0 for base, higher for Turbo)
        #[clap(long, default_value = "3.0")]
        shift: f64,
    },
}

fn main() -> flame_core::Result<()> {
    let args = Args::parse();

    match args.command {
        Commands::Train { config, adapter, output } => train_adapter(config, adapter, output),
        Commands::Generate {
            prompt,
            negative,
            variant,
            adapter,
            adapter_scale,
            output,
            steps,
            cfg,
            shift,
        } => generate_image(
            prompt,
            negative,
            variant,
            adapter,
            adapter_scale,
            output,
            steps,
            cfg,
            shift,
        ),
    }
}

fn train_adapter(
    config: PathBuf,
    adapter_type: String,
    output: Option<PathBuf>,
) -> flame_core::Result<()> {
    println!("Training SD 3.5 {} adapter...", adapter_type);
    println!("Config: {}", config.display());

    // Load config
    let config_str = std::fs::read_to_string(&config)
        .map_err(|e| {
            flame_core::Error::InvalidOperation(format!("Failed to read file: {}", e))
        })
        .map_err(|e| flame_core::Error::InvalidOperation(e.to_string()))?;

    // Set output directory
    let output_dir = output.unwrap_or_else(|| PathBuf::from("output"));
    std::fs::create_dir_all(&output_dir)
        .map_err(|e| {
            flame_core::Error::InvalidOperation(format!("Failed to create directory: {}", e))
        })
        .map_err(|e| flame_core::Error::InvalidOperation(e.to_string()))?;

    let device = Device::cuda(0)?;
    let dtype = DType::F16;

    match adapter_type.as_str() {
        "lora" => {
            // Use pipeline_sd35_lora when available
            println!("SD 3.5 LoRA training coming soon...");
        }
        "lokr" => {
            // Use pipeline_sd35_lokr when available
            println!("SD 3.5 LoKr training coming soon...");
        }
        _ => {
            eprintln!("Unknown adapter type: {}", adapter_type);
        }
    }

    Ok(())
}

fn generate_image(
    prompt: String,
    negative: String,
    variant: String,
    adapter: Option<PathBuf>,
    adapter_scale: f32,
    output: PathBuf,
    steps: usize,
    cfg: f64,
    shift: f64,
) -> flame_core::Result<()> {
    use eridiffusion::inference::sd35::generate_sd35_image;

    println!("Generating SD 3.5 image...");
    println!("Prompt: {}", prompt);
    println!("Variant: {}", variant);

    let device = Device::cuda(0)?;
    let dtype = DType::F16;

    // Generate image
    generate_sd35_image(
        &prompt,
        &negative,
        &variant,
        adapter.as_deref(),
        adapter_scale,
        &output,
        steps,
        cfg,
        shift,
        device,
        dtype,
    )?;

    println!("Image saved to: {}", output.display());
    Ok(())
}
