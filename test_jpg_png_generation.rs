//! Standalone test for JPG/PNG generation
//! Demonstrates proper image output to /outputs/loraname/samples

use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing proper JPG/PNG image generation for diffusion models");
    println!("Output directory: outputs/<lora_name>/samples/");
    
    // Test configurations for each model
    let tests = vec![
        ("sdxl_lora", "jpg", 1024, vec![127, 191, 255]),      // SDXL uses JPG
        ("sd35_lora", "png", 1024, vec![255, 127, 191]),      // SD 3.5 uses PNG
        ("flux_lora", "jpg", 1024, vec![191, 255, 127]),      // Flux uses JPG
    ];
    
    for (lora_name, format, size, color) in tests {
        println!("\n{} Generating {} test image for {}", 
                 if format == "jpg" { "📷" } else { "🖼️" }, 
                 format.to_uppercase(), 
                 lora_name);
        
        // Create output directory structure
        let output_dir = PathBuf::from("outputs").join(lora_name).join("samples");
        std::fs::create_dir_all(&output_dir)?;
        
        // Generate test image
        let mut img = image::ImageBuffer::new(size, size);
        
        // Create a pattern that represents typical diffusion output
        for (x, y, pixel) in img.enumerate_pixels_mut() {
            // Simulate denoised image with gradients
            let r = ((x as f32 / size as f32) * color[0] as f32) as u8;
            let g = ((y as f32 / size as f32) * color[1] as f32) as u8;
            let b = (((x + y) as f32 / (2.0 * size as f32)) * color[2] as f32) as u8;
            
            *pixel = image::Rgb([r, g, b]);
        }
        
        // Save with correct format
        let filename = format!("sample_step000000_idx00.{}", format);
        let filepath = output_dir.join(&filename);
        
        match format {
            "jpg" => {
                img.save_with_format(&filepath, image::ImageFormat::Jpeg)?;
                println!("  ✓ Saved JPG to: {:?}", filepath);
            }
            "png" => {
                img.save_with_format(&filepath, image::ImageFormat::Png)?;
                println!("  ✓ Saved PNG to: {:?}", filepath);
            }
            _ => unreachable!(),
        }
        
        // Save metadata
        let metadata_path = filepath.with_extension("txt");
        let metadata = format!(
            "Model: {}\n\
             Prompt: a beautiful landscape painting\n\
             Negative Prompt: \n\
             Step: 0\n\
             CFG Scale: {}\n\
             Seed: 42\n\
             Resolution: {}x{}\n\
             Format: {}",
            lora_name,
            match lora_name {
                "sdxl_lora" => "7.5",
                "sd35_lora" => "5.0",
                "flux_lora" => "3.5",
                _ => "7.5",
            },
            size, size, format.to_uppercase()
        );
        std::fs::write(&metadata_path, metadata)?;
        println!("  ✓ Saved metadata to: {:?}", metadata_path);
    }
    
    println!("\n✅ All test images generated successfully!");
    println!("\nSummary:");
    println!("- SDXL: JPG format at outputs/sdxl_lora/samples/");
    println!("- SD 3.5: PNG format at outputs/sd35_lora/samples/");
    println!("- Flux: JPG format at outputs/flux_lora/samples/");
    println!("\nThese match the expected formats for each model type.");
    
    Ok(())
}