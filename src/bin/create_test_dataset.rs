//! Create a test dataset with real images for sampling tests
//! This generates actual images using Rust's image crate

use anyhow::{Result, Context};
use image::{ImageBuffer, Rgb, RgbImage, DynamicImage};
use std::path::PathBuf;
use std::fs;
use rand::Rng;

/// Generate a gradient image
fn create_gradient_image(width: u32, height: u32, color1: [u8; 3], color2: [u8; 3]) -> RgbImage {
    ImageBuffer::from_fn(width, height, |x, y| {
        let t = y as f32 / height as f32;
        let r = (color1[0] as f32 * (1.0 - t) + color2[0] as f32 * t) as u8;
        let g = (color1[1] as f32 * (1.0 - t) + color2[1] as f32 * t) as u8;
        let b = (color1[2] as f32 * (1.0 - t) + color2[2] as f32 * t) as u8;
        Rgb([r, g, b])
    })
}

/// Generate a pattern image
fn create_pattern_image(width: u32, height: u32, pattern_type: &str) -> RgbImage {
    let mut rng = rand::thread_rng();
    
    ImageBuffer::from_fn(width, height, |x, y| {
        match pattern_type {
            "checkerboard" => {
                let size = 32;
                let is_white = ((x / size) + (y / size)) % 2 == 0;
                if is_white {
                    Rgb([255, 255, 255])
                } else {
                    Rgb([0, 0, 0])
                }
            }
            "stripes" => {
                let size = 16;
                let is_light = (x / size) % 2 == 0;
                if is_light {
                    Rgb([200, 200, 255])
                } else {
                    Rgb([100, 100, 200])
                }
            }
            "circles" => {
                let cx = width / 2;
                let cy = height / 2;
                let dx = x as i32 - cx as i32;
                let dy = y as i32 - cy as i32;
                let dist = ((dx * dx + dy * dy) as f32).sqrt();
                let max_dist = ((cx * cx + cy * cy) as f32).sqrt();
                let t = (dist / max_dist).min(1.0);
                let intensity = (255.0 * (1.0 - t)) as u8;
                Rgb([intensity, intensity / 2, intensity])
            }
            "noise" => {
                let r = rng.gen_range(0..255);
                let g = rng.gen_range(0..255);
                let b = rng.gen_range(0..255);
                Rgb([r, g, b])
            }
            "waves" => {
                let wave_x = ((x as f32 / 20.0).sin() * 127.5 + 127.5) as u8;
                let wave_y = ((y as f32 / 20.0).cos() * 127.5 + 127.5) as u8;
                let combined = ((wave_x as u16 + wave_y as u16) / 2) as u8;
                Rgb([combined, combined / 2, 255 - combined])
            }
            _ => Rgb([128, 128, 128])
        }
    })
}

/// Generate a landscape-like image
fn create_landscape_image(width: u32, height: u32, scene_type: &str) -> RgbImage {
    ImageBuffer::from_fn(width, height, |x, y| {
        let horizon = height * 2 / 3;
        let y_norm = y as f32 / height as f32;
        
        match scene_type {
            "mountain" => {
                if y < horizon {
                    // Sky
                    let t = y as f32 / horizon as f32;
                    let r = (135.0 + 50.0 * t) as u8;  // Light blue to deeper blue
                    let g = (206.0 - 50.0 * t) as u8;
                    let b = (250.0 - 50.0 * t) as u8;
                    Rgb([r, g, b])
                } else {
                    // Mountains with snow peaks
                    let mountain_height = (x as f32 / 50.0).sin().abs() * 100.0;
                    let peak_y = horizon as f32 - mountain_height;
                    if (y as f32) < peak_y + 20.0 {
                        Rgb([255, 255, 255])  // Snow
                    } else {
                        Rgb([100, 80, 60])  // Mountain rock
                    }
                }
            }
            "ocean" => {
                if y < horizon {
                    // Sky with gradient
                    let t = y_norm;
                    Rgb([
                        (255.0 * (1.0 - t * 0.5)) as u8,
                        (200.0 * (1.0 - t * 0.3)) as u8,
                        (150.0 + 105.0 * t) as u8,
                    ])
                } else {
                    // Ocean with waves
                    let wave = ((x as f32 / 30.0 + y as f32 / 10.0).sin() * 20.0) as i32;
                    let blue_var = (200 + wave).clamp(150, 255) as u8;
                    Rgb([0, 100, blue_var])
                }
            }
            "forest" => {
                if y < height / 4 {
                    // Sky
                    Rgb([135, 206, 235])
                } else {
                    // Forest with varying greens
                    let tree_var = ((x as f32 / 10.0).sin() * 50.0) as i32;
                    let green = (150 + tree_var).clamp(100, 200) as u8;
                    Rgb([50, green, 30])
                }
            }
            "desert" => {
                if y < horizon {
                    // Desert sky
                    let t = y_norm;
                    Rgb([
                        (255.0 - 50.0 * t) as u8,
                        (220.0 - 70.0 * t) as u8,
                        (180.0 - 80.0 * t) as u8,
                    ])
                } else {
                    // Sand dunes
                    let dune = ((x as f32 / 40.0).sin() * 30.0) as i32;
                    let sand = (220 + dune).clamp(180, 255) as u8;
                    Rgb([sand, sand - 20, sand - 40])
                }
            }
            "city" => {
                if y < horizon / 2 {
                    // Night sky
                    Rgb([20, 20, 50])
                } else {
                    // Buildings with lights
                    let building = (x / 50) % 3 == 0;
                    if building && y > horizon / 2 && y < horizon {
                        let has_light = (x / 10 + y / 10) % 3 == 0;
                        if has_light {
                            Rgb([255, 255, 200])  // Window light
                        } else {
                            Rgb([40, 40, 60])  // Dark building
                        }
                    } else {
                        Rgb([30, 30, 40])  // Dark background
                    }
                }
            }
            _ => Rgb([128, 128, 128])
        }
    })
}

fn main() -> Result<()> {
    println!("Creating test dataset with real images...");
    
    let dataset_path = PathBuf::from("/home/alex/test_dataset");
    fs::create_dir_all(&dataset_path)?;
    
    // Define test images with various styles
    let test_images = vec![
        ("gradient_sunset", "A vibrant sunset gradient from orange to purple"),
        ("pattern_geometric", "Abstract geometric patterns in blue tones"),
        ("landscape_mountain", "Majestic mountain peaks with snow"),
        ("landscape_ocean", "Ocean waves under a clear sky"),
        ("landscape_forest", "Dense forest with tall trees"),
        ("pattern_waves", "Flowing wave patterns in multiple colors"),
        ("landscape_desert", "Desert dunes at golden hour"),
        ("landscape_city", "City skyline at night with lights"),
        ("pattern_circles", "Concentric circles with color gradients"),
        ("pattern_noise", "Abstract noise pattern for texture testing"),
    ];
    
    // Generate images
    for (i, (name, caption)) in test_images.iter().enumerate() {
        let img = match name {
            name if name.starts_with("gradient") => {
                create_gradient_image(512, 512, [255, 100, 0], [128, 0, 255])
            }
            name if name.starts_with("pattern") => {
                let pattern_type = if name.contains("geometric") { "checkerboard" }
                else if name.contains("waves") { "waves" }
                else if name.contains("circles") { "circles" }
                else { "noise" };
                create_pattern_image(512, 512, pattern_type)
            }
            name if name.starts_with("landscape") => {
                let scene = if name.contains("mountain") { "mountain" }
                else if name.contains("ocean") { "ocean" }
                else if name.contains("forest") { "forest" }
                else if name.contains("desert") { "desert" }
                else if name.contains("city") { "city" }
                else { "mountain" };
                create_landscape_image(512, 512, scene)
            }
            _ => create_pattern_image(512, 512, "noise")
        };
        
        // Save image
        let image_path = dataset_path.join(format!("image_{:02}.jpg", i + 1));
        img.save(&image_path)?;
        println!("Created: {} ({})", image_path.display(), name);
        
        // Save caption
        let caption_path = dataset_path.join(format!("image_{:02}.txt", i + 1));
        fs::write(&caption_path, caption)?;
        println!("  Caption: {}", caption);
    }
    
    // Also create some simple test images for quick testing
    println!("\nCreating additional simple test images...");
    
    for i in 1..=5 {
        let simple_img = match i {
            1 => create_gradient_image(512, 512, [0, 0, 255], [0, 255, 255]),
            2 => create_gradient_image(512, 512, [255, 0, 0], [255, 255, 0]),
            3 => create_gradient_image(512, 512, [0, 255, 0], [0, 255, 255]),
            4 => create_pattern_image(512, 512, "stripes"),
            5 => create_pattern_image(512, 512, "checkerboard"),
            _ => create_pattern_image(512, 512, "noise"),
        };
        
        let simple_caption = match i {
            1 => "blue to cyan gradient test image",
            2 => "red to yellow gradient test image",
            3 => "green to cyan gradient test image",
            4 => "blue stripe pattern test",
            5 => "black and white checkerboard",
            _ => "test image",
        };
        
        let image_path = dataset_path.join(format!("simple_{:02}.jpg", i));
        simple_img.save(&image_path)?;
        
        let caption_path = dataset_path.join(format!("simple_{:02}.txt", i));
        fs::write(&caption_path, simple_caption)?;
        
        println!("Created simple test image {}: {}", i, simple_caption);
    }
    
    println!("\nTest dataset created successfully!");
    println!("Location: {}", dataset_path.display());
    println!("Total images: 15");
    println!("\nYou can now test sampling with:");
    println!("  cargo run --release --bin test_sampling_integration");
    println!("Or run actual training test:");
    println!("  ./trainer config/test_all_sampling.yaml");
    
    Ok(())
}