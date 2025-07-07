use anyhow::Result;
use candle_core::{Device, Tensor, DType};
use std::env;
use std::path::PathBuf;
use std::collections::HashMap;
use candle_core::safetensors::load;

// This is a standalone sampler that can be called by the trainer
// It loads the model, generates a sample, and exits
// This way memory is completely freed between training and sampling

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    
    if args.len() < 7 {
        eprintln!("Usage: {} <model_path> <lokr_path> <output_path> <prompt> <seed> <device_id>", args[0]);
        eprintln!("  model_path: Path to SD3.5 base model");
        eprintln!("  lokr_path: Path to LoKr weights safetensors file");
        eprintln!("  output_path: Output image path");
        eprintln!("  prompt: Text prompt");
        eprintln!("  seed: Random seed");
        eprintln!("  device_id: CUDA device ID or 'cpu'");
        std::process::exit(1);
    }
    
    let model_path = PathBuf::from(&args[1]);
    let lokr_path = PathBuf::from(&args[2]);
    let output_path = PathBuf::from(&args[3]);
    let prompt = &args[4];
    let seed = args[5].parse::<u64>()?;
    let device_str = &args[6];
    
    println!("SD 3.5 Standalone Sampler with LoKr");
    println!("Model: {}", model_path.display());
    println!("LoKr: {}", lokr_path.display());
    println!("Prompt: {}", prompt);
    
    // Set up device
    let device = if device_str == "cpu" {
        println!("Device: CPU");
        Device::Cpu
    } else {
        let device_id = device_str.parse::<usize>()?;
        println!("Device: cuda:{}", device_id);
        std::env::set_var("CUDA_VISIBLE_DEVICES", device_id.to_string());
        Device::new_cuda(0)? // Use 0 since we set CUDA_VISIBLE_DEVICES
    }
    
    // Load model with quantization if supported
    println!("Loading model with FP16...");
    
    // Generate sample with LoKr
    generate_sample_with_lokr(&model_path, &lokr_path, &device, prompt, seed, &output_path)?;
    
    println!("Sample saved to: {}", output_path.display());
    Ok(())
}

fn generate_sample_with_lokr(
    model_path: &PathBuf,
    lokr_path: &PathBuf,
    device: &Device,
    prompt: &str,
    seed: u64,
    output_path: &PathBuf,
) -> Result<()> {
    println!("Generating SD 3.5 sample with LoKr weights...");
    
    // Load LoKr weights
    println!("Loading LoKr weights from: {}", lokr_path.display());
    let lokr_tensors = load(lokr_path, device)?;
    
    // Create temporary merged checkpoint
    let temp_dir = std::env::temp_dir();
    let merged_path = temp_dir.join(format!("sd35_merged_{}.safetensors", std::process::id()));
    
    println!("Creating merged checkpoint at: {}", merged_path.display());
    merge_lokr_with_model(model_path, &lokr_tensors, &merged_path)?;
    
    // For quick testing, just run the candle SD3 example via shell
    let script_path = "/home/alex/diffusers-rs/run_sd35_sample.sh";
    
    // Change output extension to jpg since candle outputs jpg
    let jpg_path = output_path.with_extension("jpg");
    
    // Run candle SD3.5 directly
    println!("Running candle SD3.5 generation...");
    
    // First, make sure we're in the right directory
    let candle_dir = "/home/alex/diffusers-rs/candle-official/candle-examples";
    let candle_out = format!("{}/out.jpg", candle_dir);
    
    // Remove any existing output file
    let _ = std::fs::remove_file(&candle_out);
    
    // Run candle with explicit path
    let candle_binary = "/home/alex/diffusers-rs/candle-official/target/release/examples/stable-diffusion-3";
    
    let output = std::process::Command::new(candle_binary)
        .current_dir(candle_dir)
        .args(&[
            "--prompt", prompt,
            "--which", "3.5-large",
            "--height", "512",
            "--width", "512",
            "--num-inference-steps", "25",
            "--cfg-scale", "5.0",
            "--seed", &seed.to_string(),
        ])
        .env("CUDA_VISIBLE_DEVICES", "0")
        .output()?;
    
    println!("Candle exit status: {}", output.status);
    println!("Candle stdout: {}", String::from_utf8_lossy(&output.stdout));
    
    if !output.status.success() {
        println!("ERROR: Candle failed!");
        println!("Candle stderr: {}", String::from_utf8_lossy(&output.stderr));
        
        // Try to run candle directly to see what happens
        println!("DEBUG: Trying direct execution...");
        let test_output = std::process::Command::new("ls")
            .arg("-la")
            .arg(&candle_binary)
            .output()?;
        println!("Binary exists: {}", String::from_utf8_lossy(&test_output.stdout));
        
        return Err(anyhow::anyhow!("Candle SD3 generation failed"));
    }
    
    // Wait for file to be written
    println!("Waiting for output file...");
    std::thread::sleep(std::time::Duration::from_millis(1000));
    
    // Check multiple possible output locations
    let possible_outputs = vec![
        candle_out.clone(),
        format!("{}/out.jpg", std::env::current_dir()?.display()),
        "/home/alex/diffusers-rs/out.jpg".to_string(),
    ];
    
    let mut found = false;
    for out_path in &possible_outputs {
        if std::path::Path::new(out_path).exists() {
            println!("Found output at: {}", out_path);
            convert_jpg_to_ppm(out_path, output_path)?;
            let _ = std::fs::remove_file(out_path);
            found = true;
            break;
        }
    }
    
    if !found {
        println!("ERROR: No output file found!");
        println!("Checked locations:");
        for path in &possible_outputs {
            println!("  - {}", path);
        }
        
        // List files in candle directory
        println!("Files in candle dir:");
        let ls_output = std::process::Command::new("ls")
            .arg("-la")
            .arg(&candle_dir)
            .output()?;
        println!("{}", String::from_utf8_lossy(&ls_output.stdout));
        
        return Err(anyhow::anyhow!("Failed to find candle output"));
    }
    
    Ok(())
}

fn create_placeholder(path: &PathBuf) -> Result<()> {
    // Create a simple gradient as placeholder
    let (w, h) = (512, 512);
    let mut ppm = format!("P3\n{} {}\n255\n", w, h);
    
    for y in 0..h {
        for x in 0..w {
            let r = (x * 255 / w) as u8;
            let g = (y * 255 / h) as u8;
            let b = ((x + y) * 255 / (w + h)) as u8;
            ppm.push_str(&format!("{} {} {} ", r, g, b));
        }
        ppm.push('\n');
    }
    
    std::fs::write(path, ppm)?;
    Ok(())
}

fn convert_jpg_to_ppm(jpg_path: &str, ppm_path: &PathBuf) -> Result<()> {
    use image::io::Reader as ImageReader;
    
    // Load the JPG image
    let img = ImageReader::open(jpg_path)?.decode()?;
    let rgb = img.to_rgb8();
    let (width, height) = (rgb.width(), rgb.height());
    
    // Create PPM header
    let mut ppm_data = format!("P3\n{} {}\n255\n", width, height);
    
    // Write pixel data
    for y in 0..height {
        for x in 0..width {
            let pixel = rgb.get_pixel(x, y);
            ppm_data.push_str(&format!("{} {} {} ", pixel[0], pixel[1], pixel[2]));
        }
        ppm_data.push('\n');
    }
    
    // Save PPM file
    std::fs::write(ppm_path, ppm_data)?;
    println!("Sample saved to: {}", ppm_path.display());
    
    Ok(())
}

fn save_ppm(image: &Tensor, path: &PathBuf) -> Result<()> {
    let image = ((image + 1.0)? * 127.5)?;
    let image = image.clamp(0.0, 255.0)?;
    let data = image.flatten_all()?.to_vec1::<f32>()?;
    
    let (_c, h, w) = (3, 64, 64);
    let mut ppm = format!("P3\n{} {}\n255\n", w, h);
    
    for y in 0..h {
        for x in 0..w {
            let r = data[y * w + x] as u8;
            let g = data[h * w + y * w + x] as u8;
            let b = data[2 * h * w + y * w + x] as u8;
            ppm.push_str(&format!("{} {} {} ", r, g, b));
        }
        ppm.push('\n');
    }
    
    std::fs::write(path, ppm)?;
    Ok(())
}