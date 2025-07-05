//! SD3/SD3.5 image generation example

use eridiffusion_models::{SD3Model, sd3_candle::Which};
use anyhow::Result;
use candle_core::{Device, DType};
use std::path::Path;

fn main() -> Result<()> {
    println!("SD3/SD3.5 Image Generation Example");
    
    // Configuration
    let which = Which::V3_5Large;
    let prompt = "A cute rusty robot holding a candle torch in its hand, with glowing neon text \"LETS GO RUSTY\" displayed on its chest, bright background, high quality, 4k";
    let uncond_prompt = "";
    let height = 1024;
    let width = 1024;
    let num_inference_steps = 28;
    let cfg_scale = 4.0;
    let time_shift = 3.0;
    let seed = Some(42);
    let use_slg = false;
    
    // Set up device
    let device = if candle_core::utils::cuda_is_available() {
        println!("Using CUDA device");
        Device::new_cuda(0)?
    } else {
        println!("Using CPU device");
        Device::Cpu
    };
    
    // Create model
    println!("Creating SD3 model...");
    let mut model = SD3Model::new(which, device)?;
    
    // Load model files
    println!("Loading model weights...");
    
    // For SD3.5, we need to specify the individual model files
    // These paths should point to your actual model files
    let model_path = Path::new("/home/alex/SwarmUI/Models/Stable-Diffusion");
    
    match which {
        Which::V3_5Large | Which::V3_5LargeTurbo => {
            // SD3.5 Large uses separate files
            let model_file = model_path.join("sd3.5_large.safetensors");
            let clip_g_file = model_path.join("text_encoders/clip_g.safetensors");
            let clip_l_file = model_path.join("text_encoders/clip_l.safetensors");
            let t5_file = model_path.join("text_encoders/t5xxl_fp16.safetensors");
            
            model.load_from_files(
                &model_file,
                Some(&clip_g_file),
                Some(&clip_l_file),
                Some(&t5_file),
            )?;
        }
        Which::V3Medium => {
            // SD3 Medium uses a combined file
            let model_file = model_path.join("sd3_medium_incl_clips_t5xxlfp16.safetensors");
            model.load_from_files(&model_file, None, None, None)?;
        }
        Which::V3_5Medium => {
            // SD3.5 Medium
            let model_file = model_path.join("sd3.5_medium.safetensors");
            let clip_g_file = model_path.join("text_encoders/clip_g.safetensors");
            let clip_l_file = model_path.join("text_encoders/clip_l.safetensors");
            let t5_file = model_path.join("text_encoders/t5xxl_fp16.safetensors");
            
            model.load_from_files(
                &model_file,
                Some(&clip_g_file),
                Some(&clip_l_file),
                Some(&t5_file),
            )?;
        }
    }
    
    println!("Model loaded successfully!");
    
    // Generate image
    println!("Generating image with prompt: {}", prompt);
    println!("Settings: {}x{}, {} steps, CFG scale: {}", width, height, num_inference_steps, cfg_scale);
    
    let start_time = std::time::Instant::now();
    
    let image_tensor = model.generate(
        prompt,
        uncond_prompt,
        height,
        width,
        num_inference_steps,
        cfg_scale,
        time_shift,
        seed,
        use_slg,
    )?;
    
    let dt = start_time.elapsed().as_secs_f32();
    println!(
        "Generation complete in {:.2}s ({:.2} steps/s)",
        dt,
        num_inference_steps as f32 / dt
    );
    
    // Save the image
    println!("Saving image to output.jpg");
    candle_examples::save_image(&image_tensor.i(0)?, "output.jpg")?;
    
    println!("Done!");
    
    Ok(())
}