//! Generate SD 3.5 images using our eridiffusion-rs pipeline

use eridiffusion_models::sd3_candle::{SD3Args, Which};
use candle_core::Device;
use clap::Parser;

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "a lady at the beach")]
    prompt: String,
    
    #[arg(long, default_value = "768")]
    height: usize,
    
    #[arg(long, default_value = "768")]
    width: usize,
    
    #[arg(long, default_value = "20")]
    steps: usize,
    
    #[arg(long, default_value = "4.0")]
    cfg_scale: f64,
    
    #[arg(long, default_value = "42")]
    seed: u64,
    
    #[arg(long, default_value = "output.jpg")]
    output: String,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    
    println!("🎨 AI-Toolkit-RS SD 3.5 Generation Pipeline\n");
    
    // Create SD3 args for our pipeline
    let sd3_args = SD3Args {
        prompt: args.prompt.clone(),
        uncond_prompt: String::new(),
        cpu: false,
        tracing: false,
        use_flash_attn: false,
        height: args.height,
        width: args.width,
        which: Which::V3_5Large,
        num_inference_steps: Some(args.steps),
        cfg_scale: Some(args.cfg_scale),
        time_shift: 3.0,
        use_slg: false,
        seed: Some(args.seed),
    };
    
    println!("Configuration:");
    println!("  Model: SD 3.5 Large");
    println!("  Prompt: {}", sd3_args.prompt);
    println!("  Resolution: {}x{}", sd3_args.width, sd3_args.height);
    println!("  Steps: {}", args.steps);
    println!("  CFG Scale: {}", sd3_args.cfg_scale.unwrap());
    println!("  Seed: {}", sd3_args.seed.unwrap());
    println!("  Output: {}", args.output);
    
    // Run generation using our integrated pipeline
    println!("\nRunning generation...");
    let start = std::time::Instant::now();
    
    // Use the integrated Candle SD3 implementation
    eridiffusion_models::sd3_candle::run(sd3_args)?;
    
    let elapsed = start.elapsed();
    println!("\n✅ Generation complete in {:.2}s", elapsed.as_secs_f64());
    
    // Move output file to desired location
    if std::path::Path::new("out.jpg").exists() {
        std::fs::rename("out.jpg", &args.output)?;
        println!("📸 Image saved as: {}", args.output);
    } else {
        eprintln!("❌ Output file not found");
    }
    
    Ok(())
}