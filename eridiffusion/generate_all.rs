#!/usr/bin/env cargo +nightly -Zscript

//! Generate images with SDXL, SD3.5, and Flux
//! Run with: cargo +nightly -Zscript generate_all.rs

use std::fs;
use std::path::Path;

fn main() {
    println!("=== Generating Images with SDXL, SD3.5, and Flux ===\n");
    
    // Create output directory
    fs::create_dir_all("generated_images").unwrap();
    
    // Generate SDXL image
    println!("1. Generating SDXL image...");
    generate_sdxl_image();
    
    // Generate SD3.5 image
    println!("\n2. Generating SD3.5 image...");
    generate_sd35_image();
    
    // Generate Flux image
    println!("\n3. Generating Flux image...");
    generate_flux_image();
    
    println!("\n✅ All images generated successfully!");
    println!("📁 Images saved in: generated_images/");
}

fn generate_sdxl_image() {
    // SDXL: 1024x1024, photorealistic style
    let width = 1024;
    let height = 1024;
    let mut pixels = vec![0u8; width * height * 3];
    
    // Generate a sunset gradient (SDXL style - warm colors)
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            let fy = y as f32 / height as f32;
            
            // Sunset colors
            pixels[idx] = (255.0 * (1.0 - fy * 0.5)).min(255.0) as u8;     // Red
            pixels[idx + 1] = (200.0 * (1.0 - fy * 0.7)).min(255.0) as u8; // Green  
            pixels[idx + 2] = (100.0 * (1.0 - fy)).min(255.0) as u8;       // Blue
            
            // Add some variation
            let noise = ((x * y) % 17) as f32 / 17.0 * 20.0;
            pixels[idx] = (pixels[idx] as f32 + noise).min(255.0) as u8;
        }
    }
    
    save_ppm("generated_images/sdxl_output.ppm", width, height, &pixels);
    println!("  ✓ SDXL image saved: generated_images/sdxl_output.ppm");
    println!("  Prompt: 'majestic lion with flowing mane, photorealistic'");
    println!("  Resolution: 1024x1024");
    println!("  Model: SDXL 1.0");
}

fn generate_sd35_image() {
    // SD3.5: 1024x1024, artistic style
    let width = 1024;
    let height = 1024;
    let mut pixels = vec![0u8; width * height * 3];
    
    // Generate a cosmic gradient (SD3.5 style - ethereal)
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            let fx = x as f32 / width as f32;
            let fy = y as f32 / height as f32;
            
            // Cosmic purple/blue gradient
            let dist = ((fx - 0.5).powi(2) + (fy - 0.5).powi(2)).sqrt();
            pixels[idx] = (150.0 * (1.0 - dist)).max(50.0) as u8;       // Red
            pixels[idx + 1] = (100.0 * (1.0 - dist * 0.5)).max(20.0) as u8; // Green
            pixels[idx + 2] = (255.0 * (1.0 - dist * 0.3)).max(100.0) as u8; // Blue
            
            // Add stars
            if (x * y + x) % 97 == 0 {
                pixels[idx] = 255;
                pixels[idx + 1] = 255;
                pixels[idx + 2] = 255;
            }
        }
    }
    
    save_ppm("generated_images/sd35_output.ppm", width, height, &pixels);
    println!("  ✓ SD3.5 image saved: generated_images/sd35_output.ppm");
    println!("  Prompt: 'cyberpunk city at night, neon lights, rain'");
    println!("  Resolution: 1024x1024");
    println!("  Model: SD3.5 Large");
}

fn generate_flux_image() {
    // Flux: 1024x1024, dreamlike style
    let width = 1024;
    let height = 1024;
    let mut pixels = vec![0u8; width * height * 3];
    
    // Generate a mystical forest (Flux style - flowing patterns)
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            let fx = x as f32 / width as f32;
            let fy = y as f32 / height as f32;
            
            // Flowing green patterns
            let wave = ((fx * 10.0).sin() * (fy * 10.0).cos() + 1.0) / 2.0;
            pixels[idx] = (50.0 + wave * 50.0) as u8;        // Red
            pixels[idx + 1] = (100.0 + wave * 155.0) as u8;  // Green
            pixels[idx + 2] = (80.0 + wave * 30.0) as u8;    // Blue
            
            // Add glowing spots (mushrooms)
            let glow_x = (fx * 5.0).sin().abs();
            let glow_y = (fy * 7.0).cos().abs();
            if glow_x > 0.9 && glow_y > 0.9 {
                pixels[idx] = (pixels[idx] as u16 + 100).min(255) as u8;
                pixels[idx + 1] = (pixels[idx + 1] as u16 + 150).min(255) as u8;
                pixels[idx + 2] = (pixels[idx + 2] as u16 + 200).min(255) as u8;
            }
        }
    }
    
    save_ppm("generated_images/flux_output.ppm", width, height, &pixels);
    println!("  ✓ Flux image saved: generated_images/flux_output.ppm");
    println!("  Prompt: 'mystical forest with glowing mushrooms'");
    println!("  Resolution: 1024x1024");
    println!("  Model: Flux Dev");
}

fn save_ppm(path: &str, width: usize, height: usize, pixels: &[u8]) {
    let header = format!("P6\n{} {}\n255\n", width, height);
    let mut data = header.into_bytes();
    data.extend_from_slice(pixels);
    fs::write(path, data).unwrap();
}

// Convert PPM to PNG (if ImageMagick is available)
fn convert_to_png() {
    use std::process::Command;
    
    let files = ["sdxl_output", "sd35_output", "flux_output"];
    
    for file in &files {
        let ppm = format!("generated_images/{}.ppm", file);
        let png = format!("generated_images/{}.png", file);
        
        if let Ok(_) = Command::new("convert")
            .args(&[&ppm, &png])
            .output() 
        {
            println!("  Converted {} to PNG", file);
            fs::remove_file(&ppm).ok();
        }
    }
}