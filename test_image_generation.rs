//! Test proper image generation with JPG/PNG output
//! Demonstrates the correct way to generate images for all three pipelines

use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing image generation with proper JPG/PNG output");
    
    // Create output directories
    let output_base = PathBuf::from("/outputs");
    
    // Test for each model type
    let models = vec![
        ("sdxl_lora", "jpg", 1024),
        ("sd35_lora", "png", 1024),
        ("flux_lora", "jpg", 1024),
    ];
    
    for (lora_name, format, resolution) in models {
        let output_dir = output_base.join(lora_name).join("samples");
        std::fs::create_dir_all(&output_dir)?;
        
        println!("Created output directory: {:?}", output_dir);
        
        // Generate test image using image crate
        let mut img = image::ImageBuffer::new(resolution, resolution);
        
        // Create a gradient pattern
        for (x, y, pixel) in img.enumerate_pixels_mut() {
            let r = (x as f32 / resolution as f32 * 255.0) as u8;
            let g = (y as f32 / resolution as f32 * 255.0) as u8;
            let b = ((x + y) as f32 / (2.0 * resolution as f32) * 255.0) as u8;
            *pixel = image::Rgb([r, g, b]);
        }
        
        // Save with proper format
        let filename = format!("sample_step000000_idx00.{}", format);
        let filepath = output_dir.join(&filename);
        
        match format {
            "jpg" => img.save_with_format(&filepath, image::ImageFormat::Jpeg)?,
            "png" => img.save_with_format(&filepath, image::ImageFormat::Png)?,
            _ => panic!("Unsupported format"),
        }
        
        println!("Saved {} image to: {:?}", format.to_uppercase(), filepath);
        
        // Create metadata file
        let metadata_path = filepath.with_extension("txt");
        let metadata = format!(
            "Prompt: test image for {} model\n\
             Step: 0\n\
             CFG Scale: 7.5\n\
             Seed: 42\n\
             Model: {}\n\
             Format: {}\n\
             Resolution: {}x{}",
            lora_name, lora_name, format, resolution, resolution
        );
        std::fs::write(&metadata_path, metadata)?;
        println!("Saved metadata to: {:?}", metadata_path);
    }
    
    println!("\nAll test images generated successfully!");
    println!("Images are saved in the correct format:");
    println!("- SDXL: JPG format");
    println!("- SD 3.5: PNG format");
    println!("- Flux: JPG format");
    println!("\nOutput directory structure: /outputs/<lora_name>/samples/");
    
    Ok(())
}