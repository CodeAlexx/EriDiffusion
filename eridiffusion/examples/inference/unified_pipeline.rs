//! Unified inference pipeline for all diffusion models

use eridiffusion_core::{Device, ModelInputs, ModelArchitecture};
use eridiffusion_models::{DiffusionModel, ModelFactory};
use eridiffusion_inference::{InferencePipeline, InferenceConfig, SchedulerType};
use candle_core::{DType, Tensor};
use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
struct Args {
    /// Model to use (sd15, sdxl, sd35, flux)
    #[arg(long, default_value = "sd35")]
    model: String,
    
    /// Text prompt
    #[arg(long, default_value = "A beautiful landscape")]
    prompt: String,
    
    /// Negative prompt
    #[arg(long, default_value = "")]
    negative_prompt: String,
    
    /// Image width
    #[arg(long)]
    width: Option<usize>,
    
    /// Image height
    #[arg(long)]
    height: Option<usize>,
    
    /// Number of inference steps
    #[arg(long)]
    steps: Option<usize>,
    
    /// Guidance scale
    #[arg(long)]
    cfg_scale: Option<f32>,
    
    /// Random seed
    #[arg(long)]
    seed: Option<u64>,
    
    /// Output path
    #[arg(long, default_value = "output.png")]
    output: PathBuf,
    
    /// Use CPU
    #[arg(long)]
    cpu: bool,
    
    /// Model path override
    #[arg(long)]
    model_path: Option<PathBuf>,
}

struct ModelConfig {
    architecture: ModelArchitecture,
    default_width: usize,
    default_height: usize,
    default_steps: usize,
    default_cfg: f32,
    model_path: PathBuf,
}

fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    let args = Args::parse();
    
    println!("🎨 Unified Diffusion Pipeline - eridiffusion-rs\n");
    
    // Get model configuration
    let model_config = match args.model.as_str() {
        "sd15" => ModelConfig {
            architecture: ModelArchitecture::SD15,
            default_width: 512,
            default_height: 512,
            default_steps: 50,
            default_cfg: 7.5,
            model_path: PathBuf::from("/home/alex/SwarmUI/Models/Stable-Diffusion/v1-5-pruned-emaonly.safetensors"),
        },
        "sdxl" => ModelConfig {
            architecture: ModelArchitecture::SDXL,
            default_width: 1024,
            default_height: 1024,
            default_steps: 30,
            default_cfg: 7.5,
            model_path: PathBuf::from("/home/alex/SwarmUI/Models/Stable-Diffusion/sdXL_v10.safetensors"),
        },
        "sd35" => ModelConfig {
            architecture: ModelArchitecture::SD35Large,
            default_width: 1024,
            default_height: 1024,
            default_steps: 28,
            default_cfg: 5.0,
            model_path: PathBuf::from("/home/alex/SwarmUI/Models/diffusion_models/sd3.5_large.safetensors"),
        },
        "flux" => ModelConfig {
            architecture: ModelArchitecture::FluxSchnell,
            default_width: 1024,
            default_height: 1024,
            default_steps: 4,
            default_cfg: 3.5,
            model_path: PathBuf::from("/home/alex/SwarmUI/Models/diffusion_models/flux1-schnell.safetensors"),
        },
        _ => anyhow::bail!("Unknown model: {}. Available: sd15, sdxl, sd35, flux", args.model),
    };
    
    // Apply defaults
    let width = args.width.unwrap_or(model_config.default_width);
    let height = args.height.unwrap_or(model_config.default_height);
    let steps = args.steps.unwrap_or(model_config.default_steps);
    let cfg_scale = args.cfg_scale.unwrap_or(model_config.default_cfg);
    let model_path = args.model_path.unwrap_or(model_config.model_path);
    
    println!("Configuration:");
    println!("  Model: {} ({})", args.model, format!("{:?}", model_config.architecture));
    println!("  Prompt: {}", args.prompt);
    println!("  Resolution: {}x{}", width, height);
    println!("  Steps: {}", steps);
    println!("  CFG Scale: {}", cfg_scale);
    println!("  Seed: {:?}", args.seed);
    println!("  Device: {}", if args.cpu { "CPU" } else { "CUDA" });
    println!("  Model Path: {}", model_path.display());
    
    // For now, delegate to the specific model implementation
    match args.model.as_str() {
        "sd35" => {
            // SD 3.5 uses the Candle integration
            use eridiffusion_models::sd3_candle::{SD3Args, Which};
            
            let sd3_args = SD3Args {
                prompt: args.prompt.clone(),
                uncond_prompt: args.negative_prompt.clone(),
                cpu: args.cpu,
                tracing: false,
                use_flash_attn: false,
                height,
                width,
                which: Which::V3_5Large,
                num_inference_steps: Some(steps),
                cfg_scale: Some(cfg_scale),
                time_shift: 3.0,
                use_slg: false,
                seed: args.seed,
            };
            
            println!("\n⏳ Running SD 3.5 generation...");
            eridiffusion_models::sd3_candle::run(sd3_args)?;
            
            // Move output file
            if std::path::Path::new("out.jpg").exists() {
                std::fs::rename("out.jpg", &args.output)?;
                println!("✅ Image saved to: {}", args.output.display());
            }
        }
        
        "sdxl" => {
            // Run SDXL example
            use std::process::Command;
            
            let mut cmd_args = vec![
                "run",
                "--example",
                "sdxl_generate",
                "--",
                "--prompt",
                &args.prompt,
                "--negative-prompt",
                &args.negative_prompt,
                "--width",
                &width.to_string(),
                "--height",
                &height.to_string(),
                "--steps",
                &steps.to_string(),
                "--cfg-scale",
                &cfg_scale.to_string(),
                "--output",
                args.output.to_str().unwrap(),
                "--model-path",
                model_path.to_str().unwrap(),
            ];
            
            if let Some(seed) = args.seed {
                cmd_args.push("--seed");
                cmd_args.push(&seed.to_string());
            }
            
            if args.cpu {
                cmd_args.push("--cpu");
            }
            
            println!("\n⏳ Running SDXL generation...");
            let status = Command::new("cargo")
                .args(&cmd_args)
                .status()?;
            
            if !status.success() {
                anyhow::bail!("SDXL generation failed");
            }
        }
        
        "sd15" => {
            // Run SD 1.5 example
            use std::process::Command;
            
            let mut cmd_args = vec![
                "run",
                "--example",
                "sd15_generate",
                "--",
                "--prompt",
                &args.prompt,
                "--negative-prompt",
                &args.negative_prompt,
                "--width",
                &width.to_string(),
                "--height",
                &height.to_string(),
                "--steps",
                &steps.to_string(),
                "--cfg-scale",
                &cfg_scale.to_string(),
                "--output",
                args.output.to_str().unwrap(),
                "--model-path",
                model_path.to_str().unwrap(),
            ];
            
            if let Some(seed) = args.seed {
                cmd_args.push("--seed");
                cmd_args.push(&seed.to_string());
            }
            
            if args.cpu {
                cmd_args.push("--cpu");
            }
            
            println!("\n⏳ Running SD 1.5 generation...");
            let status = Command::new("cargo")
                .args(&cmd_args)
                .status()?;
            
            if !status.success() {
                anyhow::bail!("SD 1.5 generation failed");
            }
        }
        
        "flux" => {
            // Run Flux example
            use std::process::Command;
            
            let mut cmd_args = vec![
                "run",
                "--example",
                "flux_generate",
                "--",
                "--prompt",
                &args.prompt,
                "--width",
                &width.to_string(),
                "--height",
                &height.to_string(),
                "--steps",
                &steps.to_string(),
                "--guidance",
                &cfg_scale.to_string(),
                "--output",
                args.output.to_str().unwrap(),
                "--model-path",
                model_path.to_str().unwrap(),
                "--schnell", // Default to schnell variant
            ];
            
            if let Some(seed) = args.seed {
                cmd_args.push("--seed");
                cmd_args.push(&seed.to_string());
            }
            
            if args.cpu {
                cmd_args.push("--cpu");
            }
            
            println!("\n⏳ Running Flux generation...");
            let status = Command::new("cargo")
                .args(&cmd_args)
                .status()?;
            
            if !status.success() {
                anyhow::bail!("Flux generation failed");
            }
        }
        
        _ => unreachable!(),
    }
    
    println!("\n🎉 Generation complete!");
    
    Ok(())
}