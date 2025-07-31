//! Standalone test dataset creator
//! Compile with: rustc -O create_test_images.rs

use std::fs;
use std::path::PathBuf;

fn main() {
    println!("Creating test dataset with simple patterns...");
    
    let dataset_path = PathBuf::from("/home/alex/test_dataset");
    if let Err(e) = fs::create_dir_all(&dataset_path) {
        eprintln!("Failed to create directory: {}", e);
        return;
    }
    
    // Create simple test images using raw PPM format (no dependencies)
    let test_data = vec![
        ("gradient_blue", "A blue gradient test image"),
        ("gradient_red", "A red gradient test image"),
        ("pattern_stripes", "Striped pattern test image"),
        ("pattern_checkerboard", "Checkerboard pattern test"),
        ("solid_white", "Solid white test image"),
    ];
    
    for (i, (name, caption)) in test_data.iter().enumerate() {
        let image_path = dataset_path.join(format!("image_{:02}.ppm", i + 1));
        let caption_path = dataset_path.join(format!("image_{:02}.txt", i + 1));
        
        // Create 256x256 PPM image
        let mut ppm_data = String::from("P3\n256 256\n255\n");
        
        for y in 0u32..256 {
            for x in 0u32..256 {
                let (r, g, b) = match *name {
                    "gradient_blue" => (0, 0, y as u8),
                    "gradient_red" => (y as u8, 0, 0),
                    "pattern_stripes" => {
                        if (x / 16) % 2 == 0 { (255, 255, 255) } else { (0, 0, 0) }
                    }
                    "pattern_checkerboard" => {
                        if ((x / 16) + (y / 16)) % 2 == 0 { (255, 255, 255) } else { (0, 0, 0) }
                    }
                    "solid_white" => (255, 255, 255),
                    _ => (128, 128, 128),
                };
                ppm_data.push_str(&format!("{} {} {} ", r, g, b));
            }
            ppm_data.push('\n');
        }
        
        // Save PPM image
        if let Err(e) = fs::write(&image_path, &ppm_data) {
            eprintln!("Failed to write image {}: {}", image_path.display(), e);
        } else {
            println!("Created: {} ({})", image_path.display(), name);
        }
        
        // Save caption
        if let Err(e) = fs::write(&caption_path, caption) {
            eprintln!("Failed to write caption: {}", e);
        }
    }
    
    // Create additional test images with more patterns
    for i in 5..10 {
        let image_path = dataset_path.join(format!("image_{:02}.ppm", i + 1));
        let caption_path = dataset_path.join(format!("image_{:02}.txt", i + 1));
        
        let mut ppm_data = String::from("P3\n256 256\n255\n");
        
        for y in 0u32..256 {
            for x in 0u32..256 {
                let (r, g, b) = match i {
                    5 => (x as u8, y as u8, 0),  // Red-green gradient
                    6 => (0, x as u8, y as u8),  // Green-blue gradient
                    7 => (x as u8, 0, y as u8),  // Red-blue gradient
                    8 => {  // Circles
                        let dx = x as i32 - 128;
                        let dy = y as i32 - 128;
                        let dist = ((dx * dx + dy * dy) as f32).sqrt() as u8;
                        (dist, dist, dist)
                    }
                    9 => {  // Diagonal stripes
                        if ((x + y) / 20) % 2 == 0 { (255, 200, 0) } else { (0, 100, 255) }
                    }
                    _ => (128, 128, 128),
                };
                ppm_data.push_str(&format!("{} {} {} ", r, g, b));
            }
            ppm_data.push('\n');
        }
        
        let caption = match i {
            5 => "Red to green gradient diagonal",
            6 => "Green to blue gradient diagonal", 
            7 => "Red to blue gradient diagonal",
            8 => "Concentric circles pattern",
            9 => "Diagonal stripe pattern",
            _ => "Test pattern",
        };
        
        fs::write(&image_path, &ppm_data).unwrap();
        fs::write(&caption_path, caption).unwrap();
        println!("Created test image {}: {}", i + 1, caption);
    }
    
    println!("\nTest dataset created successfully!");
    println!("Location: {}", dataset_path.display());
    println!("Total images: 10 (PPM format)");
    println!("\nNote: PPM files can be converted to JPG/PNG using:");
    println!("  for f in *.ppm; do convert \"$f\" \"${{f%.ppm}}.jpg\"; done");
}