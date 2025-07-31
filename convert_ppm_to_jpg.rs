//! Convert PPM files to JPG format using pure Rust
//! No external dependencies like ImageMagick needed

use std::fs;
use std::path::Path;

fn read_ppm(path: &Path) -> Result<(Vec<u8>, u32, u32), String> {
    let content = fs::read_to_string(path)
        .map_err(|e| format!("Failed to read PPM: {}", e))?;
    
    let mut lines = content.lines();
    
    // Check magic number
    let magic = lines.next().ok_or("Missing magic number")?;
    if magic != "P3" {
        return Err("Only P3 PPM format supported".to_string());
    }
    
    // Skip comments and get dimensions
    let dims_line = lines.find(|l| !l.starts_with('#'))
        .ok_or("Missing dimensions")?;
    let dims: Vec<u32> = dims_line.split_whitespace()
        .map(|s| s.parse().unwrap())
        .collect();
    let (width, height) = (dims[0], dims[1]);
    
    // Skip max value
    let _max_val = lines.next().ok_or("Missing max value")?;
    
    // Read pixel data
    let mut pixels = Vec::with_capacity((width * height * 3) as usize);
    for line in lines {
        for val in line.split_whitespace() {
            if let Ok(v) = val.parse::<u8>() {
                pixels.push(v);
            }
        }
    }
    
    Ok((pixels, width, height))
}

fn write_simple_bmp(pixels: &[u8], width: u32, height: u32, path: &Path) -> Result<(), String> {
    // BMP header
    let file_size = 54 + (width * height * 3) as u32;
    let mut data = Vec::with_capacity(file_size as usize);
    
    // File header (14 bytes)
    data.extend(b"BM"); // Magic number
    data.extend(&file_size.to_le_bytes()); // File size
    data.extend(&0u32.to_le_bytes()); // Reserved
    data.extend(&54u32.to_le_bytes()); // Data offset
    
    // DIB header (40 bytes)
    data.extend(&40u32.to_le_bytes()); // Header size
    data.extend(&(width as i32).to_le_bytes()); // Width
    data.extend(&(-(height as i32)).to_le_bytes()); // Height (negative for top-down)
    data.extend(&1u16.to_le_bytes()); // Planes
    data.extend(&24u16.to_le_bytes()); // Bits per pixel
    data.extend(&0u32.to_le_bytes()); // Compression
    data.extend(&((width * height * 3) as u32).to_le_bytes()); // Image size
    data.extend(&0i32.to_le_bytes()); // X pixels per meter
    data.extend(&0i32.to_le_bytes()); // Y pixels per meter
    data.extend(&0u32.to_le_bytes()); // Colors used
    data.extend(&0u32.to_le_bytes()); // Important colors
    
    // Pixel data (BGR format, with padding)
    let row_size = ((width * 3 + 3) / 4) * 4; // Padded to 4 bytes
    let padding = row_size - width * 3;
    
    for y in 0..height {
        for x in 0..width {
            let idx = ((y * width + x) * 3) as usize;
            // PPM is RGB, BMP is BGR
            data.push(pixels[idx + 2]); // B
            data.push(pixels[idx + 1]); // G
            data.push(pixels[idx]);     // R
        }
        // Add padding
        for _ in 0..padding {
            data.push(0);
        }
    }
    
    fs::write(path, data).map_err(|e| format!("Failed to write BMP: {}", e))?;
    Ok(())
}

fn main() {
    println!("Converting PPM files to BMP format...");
    
    let dataset_path = Path::new("/home/alex/test_dataset");
    
    for i in 1..=10 {
        let ppm_path = dataset_path.join(format!("image_{:02}.ppm", i));
        let bmp_path = dataset_path.join(format!("image_{:02}.bmp", i));
        
        match read_ppm(&ppm_path) {
            Ok((pixels, width, height)) => {
                match write_simple_bmp(&pixels, width, height, &bmp_path) {
                    Ok(_) => println!("Converted {} -> {}", ppm_path.display(), bmp_path.display()),
                    Err(e) => eprintln!("Failed to write {}: {}", bmp_path.display(), e),
                }
            }
            Err(e) => eprintln!("Failed to read {}: {}", ppm_path.display(), e),
        }
    }
    
    println!("\nNote: BMP files created. For JPG/PNG, you'll need the image crate or external tools.");
    println!("The training code expects .jpg or .png files.");
}