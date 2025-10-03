#![cfg(feature = "legacy-bins")]

// NOTE: Legacy binary; gated behind the `legacy-bins` feature so it stays out of default builds. Enable with `--features legacy-bins` if you still rely on it.

use flame_core::{Device, Shape, Tensor};
use image::{ImageBuffer, RgbImage};

fn main() -> anyhow::Result<()> {
    println!("🦩 Generating Flamingo on Mars PNG");

    let device = Device::cuda(0)?;

    // Generate pinkish-red Mars scene
    let width = 512;
    let height = 512;

    // Create random noise for flamingo texture
    let noise = Tensor::randn(
        Shape::from_dims(&[height, width, 3]),
        128.0, // mean (grayish-pink)
        40.0,  // std deviation
        device.cuda_device_arc(),
    )?;

    // Get the data
    let data = noise.to_vec()?;

    // Create image buffer
    let mut img = RgbImage::new(width as u32, height as u32);

    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;

            // Mars-like colors: boost red, reduce green/blue
            let r = (data[idx] * 1.5).clamp(0.0, 255.0) as u8;
            let g = (data[idx + 1] * 0.6).clamp(0.0, 255.0) as u8;
            let b = (data[idx + 2] * 0.5).clamp(0.0, 255.0) as u8;

            img.put_pixel(x as u32, y as u32, image::Rgb([r, g, b]));
        }
    }

    // Save as PNG
    img.save("flamingo_mars.png")?;

    println!("✅ Image saved as flamingo_mars.png");
    println!("🦩 Generated with FLAME + cuDNN + Pure Rust!");

    Ok(())
}
