#![cfg(feature = "legacy-bins")]

// NOTE: Legacy binary; gated behind the `legacy-bins` feature so it stays out of default builds. Enable with `--features legacy-bins` if you still rely on it.

use flame_core::Result;
use image::{Rgb, RgbImage};
use std::path::PathBuf;

// Generate a test image to verify our image saving pipeline works
// This creates a simple "white swan on mars" visualization

fn main() -> flame_core::Result<()> {
    println!("=== Test Image Generator ===");
    println!("Creating 'a white swan on mars' test image");

    // Create output directory
    let output_dir = PathBuf::from("./outputs/test_images");
    std::fs::create_dir_all(&output_dir)
        .map_err(|e| {
            flame_core::Error::InvalidOperation(format!("Failed to create directory: {}", e))
        })
        .map_err(|e| flame_core::Error::InvalidOperation(e.to_string()))?;

    // Create image
    let width = 1024;
    let height = 1024;
    let mut img = RgbImage::new(width, height);

    // Create Mars-like background
    let center_x = width as f32 / 2.0;
    let center_y = height as f32 / 2.0;

    for y in 0..height {
        for x in 0..width {
            let dx = x as f32 - center_x;
            let dy = y as f32 - center_y;
            let dist = (dx * dx + dy * dy).sqrt();

            // Mars surface (reddish)
            let color = if dist < 400.0 {
                let r = (200.0 + dist * 0.1).min(255.0) as u8;
                let g = (100.0 + dist * 0.05).min(255.0) as u8;
                let b = (50.0 + dist * 0.02).min(255.0) as u8;
                Rgb([r, g, b])
            } else {
                // Space background
                Rgb([20, 10, 30])
            };

            img.put_pixel(x, y, color);
        }
    }

    // Add a simple white swan shape
    let swan_cx = width / 2;
    let swan_cy = height / 2;

    // Swan body (ellipse)
    for y in 0..height {
        for x in 0..width {
            let dx = (x as i32 - swan_cx as i32) as f32;
            let dy = (y as i32 - swan_cy as i32) as f32;

            // Body ellipse
            let body_check = (dx / 150.0).powi(2) + (dy / 100.0).powi(2);

            // Neck (smaller ellipse, offset up and right)
            let neck_dx = dx - 80.0;
            let neck_dy = dy + 60.0;
            let neck_check = (neck_dx / 40.0).powi(2) + (neck_dy / 80.0).powi(2);

            // Head (circle)
            let head_dx = dx - 120.0;
            let head_dy = dy + 100.0;
            let head_check = (head_dx.powi(2) + head_dy.powi(2)).sqrt() / 30.0;

            if body_check < 1.0 || neck_check < 1.0 || head_check < 1.0 {
                // Blend white swan with background
                let old_color = img.get_pixel(x, y);
                let blend = 0.9;
                let new_color = Rgb([
                    (old_color[0] as f32 * (1.0 - blend) + 255.0 * blend) as u8,
                    (old_color[1] as f32 * (1.0 - blend) + 255.0 * blend) as u8,
                    (old_color[2] as f32 * (1.0 - blend) + 255.0 * blend) as u8,
                ]);
                img.put_pixel(x, y, new_color);
            }
        }
    }

    // Add some "Martian terrain" texture
    for y in (500..900).step_by(20) {
        for x in 0..width {
            let wave = ((x as f32 * 0.01 + y as f32 * 0.005).sin() * 10.0) as i32;
            let py = (y as i32 + wave).clamp(0, height as i32 - 1) as u32;

            let old_color = img.get_pixel(x, py);
            let darker = Rgb([
                (old_color[0] as f32 * 0.8) as u8,
                (old_color[1] as f32 * 0.8) as u8,
                (old_color[2] as f32 * 0.8) as u8,
            ]);
            img.put_pixel(x, py, darker);
        }
    }

    // Save the image
    let output_path = output_dir.join("white_swan_on_mars_test.png");
    img.save(&output_path)?;

    println!("\n✓ Test image saved to: {:?}", output_path);
    println!("  Size: {}x{}", width, height);
    println!("\nThis demonstrates our image generation pipeline works.");
    println!("Real SDXL would replace the simple drawing with neural network output.");

    Ok(())
}
