use image::io::Reader as ImageReader;
use std::path::Path;
use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    
    if args.len() < 2 {
        println!("Usage: {} <ppm_file> [output_png]", args[0]);
        println!("\nConverts PPM files to PNG format");
        return Ok(());
    }
    
    let input_path = &args[1];
    let output_path = if args.len() > 2 {
        args[2].clone()
    } else {
        input_path.replace(".ppm", ".png")
    };
    
    println!("Converting {} -> {}", input_path, output_path);
    
    // Read PPM and save as PNG
    let img = ImageReader::open(input_path)?
        .decode()?;
    
    img.save(&output_path)?;
    
    println!("✓ Conversion complete!");
    
    // Show image info
    let (width, height) = (img.width(), img.height());
    println!("  Size: {}x{}", width, height);
    println!("  Saved to: {}", output_path);
    
    Ok(())
}