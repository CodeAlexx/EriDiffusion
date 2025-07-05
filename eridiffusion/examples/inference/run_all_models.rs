//! Comprehensive example demonstrating all working diffusion models in eridiffusion-rs

use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate images with SD 1.5
    Sd15 {
        #[arg(long, default_value = "A beautiful landscape")]
        prompt: String,
        #[arg(long, default_value = "sd15_output.png")]
        output: PathBuf,
    },
    
    /// Generate images with SDXL
    Sdxl {
        #[arg(long, default_value = "A futuristic cityscape")]
        prompt: String,
        #[arg(long, default_value = "sdxl_output.png")]
        output: PathBuf,
    },
    
    /// Generate images with SD 3.5
    Sd35 {
        #[arg(long, default_value = "A magical forest")]
        prompt: String,
        #[arg(long, default_value = "sd35_output.png")]
        output: PathBuf,
    },
    
    /// Generate images with Flux
    Flux {
        #[arg(long, default_value = "A cyberpunk scene")]
        prompt: String,
        #[arg(long, default_value = "flux_output.png")]
        output: PathBuf,
        #[arg(long)]
        schnell: bool,
    },
    
    /// List all available models
    List,
    
    /// Run benchmark on all models
    Benchmark {
        #[arg(long, default_value = "A test image")]
        prompt: String,
    },
}

fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    let cli = Cli::parse();
    
    println!("🎨 AI-Toolkit-RS - All Models Demo\n");
    
    match cli.command {
        Commands::Sd15 { prompt, output } => {
            println!("Running SD 1.5 generation...");
            run_sd15(&prompt, &output)?;
        }
        
        Commands::Sdxl { prompt, output } => {
            println!("Running SDXL generation...");
            run_sdxl(&prompt, &output)?;
        }
        
        Commands::Sd35 { prompt, output } => {
            println!("Running SD 3.5 generation...");
            run_sd35(&prompt, &output)?;
        }
        
        Commands::Flux { prompt, output, schnell } => {
            println!("Running Flux {} generation...", if schnell { "Schnell" } else { "Dev" });
            run_flux(&prompt, &output, schnell)?;
        }
        
        Commands::List => {
            list_available_models();
        }
        
        Commands::Benchmark { prompt } => {
            run_benchmark(&prompt)?;
        }
    }
    
    Ok(())
}

fn run_sd15(prompt: &str, output: &PathBuf) -> anyhow::Result<()> {
    use std::process::Command;
    
    let status = Command::new("cargo")
        .args(&[
            "run",
            "--example",
            "sd15_generate",
            "--",
            "--prompt",
            prompt,
            "--output",
            output.to_str().unwrap(),
        ])
        .status()?;
    
    if !status.success() {
        anyhow::bail!("SD 1.5 generation failed");
    }
    
    Ok(())
}

fn run_sdxl(prompt: &str, output: &PathBuf) -> anyhow::Result<()> {
    use std::process::Command;
    
    let status = Command::new("cargo")
        .args(&[
            "run",
            "--example",
            "sdxl_generate",
            "--",
            "--prompt",
            prompt,
            "--output",
            output.to_str().unwrap(),
        ])
        .status()?;
    
    if !status.success() {
        anyhow::bail!("SDXL generation failed");
    }
    
    Ok(())
}

fn run_sd35(prompt: &str, output: &PathBuf) -> anyhow::Result<()> {
    // SD 3.5 is already working via the candle integration
    use eridiffusion_models::sd3_candle::{SD3Args, Which};
    
    let args = SD3Args {
        prompt: prompt.to_string(),
        uncond_prompt: String::new(),
        cpu: false,
        tracing: false,
        use_flash_attn: false,
        height: 1024,
        width: 1024,
        which: Which::V3_5Large,
        num_inference_steps: Some(28),
        cfg_scale: Some(5.0),
        time_shift: 3.0,
        use_slg: false,
        seed: Some(42),
    };
    
    println!("Generating with SD 3.5...");
    eridiffusion_models::sd3_candle::run(args)?;
    
    // Move output file
    if std::path::Path::new("out.jpg").exists() {
        std::fs::rename("out.jpg", output)?;
        println!("✅ Image saved to: {}", output.display());
    }
    
    Ok(())
}

fn run_flux(prompt: &str, output: &PathBuf, schnell: bool) -> anyhow::Result<()> {
    use std::process::Command;
    
    let mut args = vec![
        "run",
        "--example",
        "flux_generate",
        "--",
        "--prompt",
        prompt,
        "--output",
        output.to_str().unwrap(),
    ];
    
    if schnell {
        args.push("--schnell");
    }
    
    let status = Command::new("cargo")
        .args(&args)
        .status()?;
    
    if !status.success() {
        anyhow::bail!("Flux generation failed");
    }
    
    Ok(())
}

fn list_available_models() {
    println!("📋 Available Models in eridiffusion-rs:\n");
    
    println!("✅ Working Models:");
    println!("  • SD 1.5 - Stable Diffusion 1.5");
    println!("  • SDXL - Stable Diffusion XL");
    println!("  • SD 3.5 - Stable Diffusion 3.5 (via Candle integration)");
    println!("  • Flux Dev/Schnell - Black Forest Labs Flux");
    
    println!("\n🚧 In Development:");
    println!("  • PixArt-α/Σ - Transformer-based diffusion");
    println!("  • AuraFlow - Flow matching model");
    println!("  • Lumina - Next-gen DiT model");
    println!("  • OmniGen v2 - Multi-modal generation");
    
    println!("\n📍 Model Locations:");
    println!("  • Models: /home/alex/SwarmUI/Models/");
    println!("  • CLIP: /home/alex/SwarmUI/Models/CLIP/");
    println!("  • VAE: /home/alex/SwarmUI/Models/VAE/");
}

fn run_benchmark(prompt: &str) -> anyhow::Result<()> {
    use std::time::Instant;
    
    println!("⏱️  Running benchmark on all models...\n");
    
    // SD 1.5 benchmark
    println!("Benchmarking SD 1.5...");
    let start = Instant::now();
    run_sd15(prompt, &PathBuf::from("benchmark_sd15.png"))?;
    let sd15_time = start.elapsed();
    
    // SDXL benchmark
    println!("\nBenchmarking SDXL...");
    let start = Instant::now();
    run_sdxl(prompt, &PathBuf::from("benchmark_sdxl.png"))?;
    let sdxl_time = start.elapsed();
    
    // SD 3.5 benchmark
    println!("\nBenchmarking SD 3.5...");
    let start = Instant::now();
    run_sd35(prompt, &PathBuf::from("benchmark_sd35.png"))?;
    let sd35_time = start.elapsed();
    
    // Results
    println!("\n📊 Benchmark Results:");
    println!("  • SD 1.5: {:.2}s", sd15_time.as_secs_f64());
    println!("  • SDXL: {:.2}s", sdxl_time.as_secs_f64());
    println!("  • SD 3.5: {:.2}s", sd35_time.as_secs_f64());
    
    // Clean up
    let _ = std::fs::remove_file("benchmark_sd15.png");
    let _ = std::fs::remove_file("benchmark_sdxl.png");
    let _ = std::fs::remove_file("benchmark_sd35.png");
    
    Ok(())
}