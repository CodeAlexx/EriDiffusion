//! SD3/SD3.5 pipeline example using our integrated pipeline

use eridiffusion_core::{Device, Result};
use eridiffusion_inference::{
    sd3_pipeline::{SD3Pipeline, SD3PipelineConfig, Scheduler},
    SD35ModelVariant,
};
use clap::Parser;
use std::path::PathBuf;
use candle_core::{DType, Tensor};
use image::{ImageBuffer, Rgb};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Model file path (sd3.5_large.safetensors)
    #[arg(long)]
    model_file: PathBuf,
    
    /// CLIP-G file path (for SD3.5)
    #[arg(long)]
    clip_g_file: Option<PathBuf>,
    
    /// CLIP-L file path (for SD3.5)
    #[arg(long)]
    clip_l_file: Option<PathBuf>,
    
    /// T5-XXL file path (for SD3.5)
    #[arg(long)]
    t5_file: Option<PathBuf>,
    
    /// Model variant
    #[arg(long, default_value = "large")]
    variant: String,
    
    /// Text prompt
    #[arg(long, default_value = "A rustic cabin in the woods during autumn, smoke rising from the chimney")]
    prompt: String,
    
    /// Image width
    #[arg(long, default_value = "1024")]
    width: usize,
    
    /// Image height
    #[arg(long, default_value = "1024")]
    height: usize,
    
    /// Number of inference steps
    #[arg(long, default_value = "28")]
    steps: usize,
    
    /// Guidance scale
    #[arg(long, default_value = "4.5")]
    guidance_scale: f32,
    
    /// Random seed
    #[arg(long)]
    seed: Option<u64>,
    
    /// Output image path
    #[arg(long, default_value = "sd3_output.png")]
    output: PathBuf,
    
    /// Use CPU
    #[arg(long)]
    cpu: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    let args = Args::parse();
    
    // Set device
    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::Cuda(0)
    };
    
    println!("Using device: {:?}", device);
    
    // Parse variant
    let variant = match args.variant.as_str() {
        "medium" => SD35ModelVariant::Medium,
        "large" => SD35ModelVariant::Large,
        "large-turbo" => SD35ModelVariant::LargeTurbo,
        _ => {
            eprintln!("Unknown variant: {}, using large", args.variant);
            SD35ModelVariant::Large
        }
    };
    
    // Create pipeline config
    let config = SD3PipelineConfig {
        model_variant: variant,
        scheduler: Scheduler::FlowMatch,
        guidance_scale: args.guidance_scale,
        num_inference_steps: args.steps,
    };
    
    println!("Loading SD3.5 pipeline from files...");
    
    // Create pipeline from files
    let pipeline = SD3Pipeline::from_files(
        config,
        &args.model_file,
        args.clip_g_file.as_deref(),
        args.clip_l_file.as_deref(),
        args.t5_file.as_deref(),
        device,
    )?;
    
    println!("Pipeline loaded successfully!");
    
    // Generate image
    println!("Generating image with prompt: {}", args.prompt);
    println!("Size: {}x{}, Steps: {}, Guidance: {}", 
        args.width, args.height, args.steps, args.guidance_scale);
    
    let start = std::time::Instant::now();
    
    let image_tensor = pipeline.generate(
        &args.prompt,
        args.width,
        args.height,
        args.guidance_scale,
        args.steps,
        args.seed,
    ).await?;
    
    let elapsed = start.elapsed();
    println!("Generation completed in {:.2}s", elapsed.as_secs_f32());
    
    // Save image
    save_image(&image_tensor, &args.output)?;
    println!("Image saved to: {:?}", args.output);
    
    Ok(())
}

fn save_image(tensor: &Tensor, path: &PathBuf) -> Result<()> {
    // Ensure tensor is on CPU
    let tensor = tensor.to_device(&candle_core::Device::Cpu)?;
    
    // Get dimensions (B, C, H, W)
    let (_, channels, height, width) = tensor.dims4()
        .map_err(|e| eridiffusion_core::Error::Runtime(format!("Invalid tensor shape: {}", e)))?;
    
    if channels != 3 {
        return Err(eridiffusion_core::Error::Runtime(format!(
            "Expected 3 channels, got {}",
            channels
        )));
    }
    
    // Convert to u8 and get data
    let data = tensor.to_dtype(DType::U8)?;
    let data_vec = data.flatten_all()?.to_vec1::<u8>()
        .map_err(|e| eridiffusion_core::Error::Runtime(format!("Failed to convert tensor: {}", e)))?;
    
    // Create image buffer
    let mut img = ImageBuffer::<Rgb<u8>, Vec<u8>>::new(width as u32, height as u32);
    
    // Copy data (tensor is in CHW format, need to convert to HWC)
    for y in 0..height {
        for x in 0..width {
            let r = data_vec[0 * height * width + y * width + x];
            let g = data_vec[1 * height * width + y * width + x];
            let b = data_vec[2 * height * width + y * width + x];
            img.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
        }
    }
    
    // Save image
    img.save(path)
        .map_err(|e| eridiffusion_core::Error::Runtime(format!("Failed to save image: {}", e)))?;
    
    Ok(())
}