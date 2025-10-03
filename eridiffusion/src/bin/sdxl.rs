#![cfg(feature = "legacy-bins")]

// NOTE: Legacy binary; gated behind the `legacy-bins` feature so it stays out of default builds. Enable with `--features legacy-bins` if you still rely on it.

use clap::{Parser, Subcommand};
use flame_core::device::Device;
use flame_core::DType;
use flame_core::Result;
use serde_yaml;
use std::path::PathBuf;

/// SDXL CLI - Unified tool for training and inference
#[derive(Parser)]
#[clap(name = "sdxl", about = "SDXL diffusion model training and inference")]
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
        /// Output directory
        #[clap(short, long)]
        output: Option<PathBuf>,
        /// Resume from checkpoint
        #[clap(long)]
        resume: Option<PathBuf>,
    },
    /// Generate images
    Generate {
        #[clap(short, long)]
        prompt: String,
        #[clap(short, long, default_value = "")]
        negative: String,
        /// Base model path
        #[clap(short, long)]
        model: Option<PathBuf>,
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
        #[clap(long, default_value = "30")]
        steps: usize,
        /// Guidance scale
        #[clap(long, default_value = "7.5")]
        cfg: f64,
        /// Random seed
        #[clap(long)]
        seed: Option<u64>,
        /// Image width
        #[clap(long, default_value = "1024")]
        width: usize,
        /// Image height
        #[clap(long, default_value = "1024")]
        height: usize,
    },
    /// Inspect a model or LoRA
    Inspect { path: PathBuf },
}

fn main() -> flame_core::Result<()> {
    let args = Args::parse();
    match args.command {
        Commands::Train { config, output, resume } => train_lora(config, output, resume),
        Commands::Generate {
            prompt,
            negative,
            model,
            lora,
            lora_scale,
            output,
            steps,
            cfg,
            seed,
            width,
            height,
        } => generate_image(
            prompt, negative, model, lora, lora_scale, output, steps, cfg, seed, width, height,
        ),
        Commands::Inspect { path } => inspect_model(path),
    }
}

fn train_lora(
    config: PathBuf,
    _output: Option<PathBuf>,
    _resume: Option<PathBuf>,
) -> flame_core::Result<()> {
    // Unified training entry point which parses YAML and selects appropriate pipeline
    eridiffusion::trainers::train_from_config(config)
}

fn generate_image(
    prompt: String,
    negative: String,
    model: Option<PathBuf>,
    lora: Option<PathBuf>,
    lora_scale: f32,
    output: PathBuf,
    steps: usize,
    cfg: f64,
    seed: Option<u64>,
    width: usize,
    height: usize,
) -> flame_core::Result<()> {
    use eridiffusion::inference::sdxl::generate_sdxl_image;

    println!("Generating SDXL image...");
    println!("Prompt: {}", prompt);

    // Default model path
    let model_path = model.unwrap_or_else(|| {
        PathBuf::from(
            "/home/alex/SwarmUI/Models/diffusion_models/sd_xl_base_1.0_0.9vae.safetensors",
        )
    });

    let device = Device::cuda(0)?;
    let dtype = DType::F16;

    generate_sdxl_image(
        &prompt,
        &negative,
        &model_path,
        lora.as_deref(),
        lora_scale,
        &output,
        steps,
        cfg,
        seed,
        width,
        height,
        device,
        dtype,
    )?;

    println!("Image saved to: {}", output.display());
    Ok(())
}

fn inspect_model(path: PathBuf) -> flame_core::Result<()> {
    use safetensors::SafeTensors;

    println!("Inspecting: {}", path.display());

    // Read file and deserialize
    let data = std::fs::read(&path).map_err(|e| {
        flame_core::Error::InvalidOperation(format!("Failed to read file: {}", e))
    })?;
    let weights = SafeTensors::deserialize(&data).map_err(|e| {
        flame_core::Error::InvalidOperation(format!(
            "Failed to deserialize safetensors: {}",
            e
        ))
    })?;

    println!("\nFound {} tensors", weights.tensors().len());

    // Group by type
    let mut lora_weights = vec![];
    let mut other_weights = vec![];
    for (name, view) in weights.tensors() {
        if name.contains("lora") || name.contains("lokr") {
            lora_weights.push((name, view));
        } else {
            other_weights.push((name, view));
        }
    }

    if !lora_weights.is_empty() {
        println!("\nLoRA/LoKr weights:");
        for (name, view) in &lora_weights {
            println!("  {} -> shape: {:?}", name, view.shape());
        }
    }

    if !other_weights.is_empty() && other_weights.len() <= 20 {
        println!("\nOther weights:");
        for (name, view) in &other_weights {
            println!("  {} -> shape: {:?}", name, view.shape());
        }
    } else if !other_weights.is_empty() {
        println!("\nOther weights: {} tensors (too many to list)", other_weights.len());
    }

    Ok(())
}
