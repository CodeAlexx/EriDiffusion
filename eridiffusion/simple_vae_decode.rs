#!/usr/bin/env rustc --edition=2021

// Simple VAE decode test using manual implementation
use std::fs;

fn main() {
    println!("🦀 Simple VAE Decode Test\n");
    
    // Create structured latents that should decode to something visible
    let mut latents = vec![0.0f32; 4 * 128 * 128];
    
    // Create a pattern in latent space
    for c in 0..4 {
        for y in 0..128 {
            for x in 0..128 {
                let idx = c * 128 * 128 + y * 128 + x;
                
                // Create different patterns per channel
                latents[idx] = match c {
                    0 => ((x as f32 / 128.0 * 10.0).sin() * (y as f32 / 128.0 * 10.0).cos()) * 0.5,
                    1 => ((x + y) as f32 / 180.0 * 8.0).sin() * 0.5,
                    2 => (((x as i32 - 64).pow(2) + (y as i32 - 64).pow(2)) as f32).sqrt() / 90.0 - 0.5,
                    3 => ((x * y) as f32 / 16384.0 - 0.5) * 2.0,
                    _ => 0.0,
                };
            }
        }
    }
    
    // Decode to image
    let image = simple_vae_decode(&latents);
    
    // Save
    save_ppm("generated_images/simple_vae_test.ppm", 1024, 1024, &image);
    println!("✓ Saved: generated_images/simple_vae_test.ppm");
}

fn simple_vae_decode(latents: &[f32]) -> Vec<u8> {
    let mut image = vec![0u8; 1024 * 1024 * 3];
    
    // Decode 128x128 latents to 1024x1024 image (8x upscale)
    for y in 0..1024 {
        for x in 0..1024 {
            let idx = (y * 1024 + x) * 3;
            
            // Get latent coordinates with bilinear interpolation
            let lx_f = x as f32 / 1024.0 * 127.0;
            let ly_f = y as f32 / 1024.0 * 127.0;
            let lx = lx_f as usize;
            let ly = ly_f as usize;
            let fx = lx_f - lx as f32;
            let fy = ly_f - ly as f32;
            
            // Ensure we don't go out of bounds
            let lx1 = (lx + 1).min(127);
            let ly1 = (ly + 1).min(127);
            
            // Sample and interpolate each channel
            let mut rgb = [0.0f32; 3];
            
            for c in 0..4 {
                // Get 4 corner values
                let v00 = latents[c * 128 * 128 + ly * 128 + lx];
                let v01 = latents[c * 128 * 128 + ly1 * 128 + lx];
                let v10 = latents[c * 128 * 128 + ly * 128 + lx1];
                let v11 = latents[c * 128 * 128 + ly1 * 128 + lx1];
                
                // Bilinear interpolation
                let v0 = v00 * (1.0 - fx) + v10 * fx;
                let v1 = v01 * (1.0 - fx) + v11 * fx;
                let v = v0 * (1.0 - fy) + v1 * fy;
                
                // Map 4 latent channels to RGB
                match c {
                    0 => { 
                        rgb[0] += v * 0.6;
                        rgb[1] += v * 0.3;
                        rgb[2] += v * 0.1;
                    }
                    1 => {
                        rgb[0] += v * 0.1;
                        rgb[1] += v * 0.6;
                        rgb[2] += v * 0.3;
                    }
                    2 => {
                        rgb[0] += v * 0.2;
                        rgb[1] += v * 0.2;
                        rgb[2] += v * 0.6;
                    }
                    3 => {
                        // Detail channel affects all
                        rgb[0] += v * 0.2;
                        rgb[1] += v * 0.2;
                        rgb[2] += v * 0.2;
                    }
                    _ => {}
                }
            }
            
            // VAE scaling and normalization
            // SDXL VAE uses scale factor of ~0.18215
            image[idx] = ((rgb[0] * 0.18215 + 0.5) * 255.0).clamp(0.0, 255.0) as u8;
            image[idx + 1] = ((rgb[1] * 0.18215 + 0.5) * 255.0).clamp(0.0, 255.0) as u8;
            image[idx + 2] = ((rgb[2] * 0.18215 + 0.5) * 255.0).clamp(0.0, 255.0) as u8;
        }
    }
    
    image
}

fn save_ppm(path: &str, width: usize, height: usize, pixels: &[u8]) {
    let header = format!("P6\n{} {}\n255\n", width, height);
    let mut data = header.into_bytes();
    data.extend_from_slice(pixels);
    
    fs::create_dir_all("generated_images").ok();
    fs::write(path, data).unwrap();
}