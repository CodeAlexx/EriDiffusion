#![cfg(feature = "legacy-bins")]

// NOTE: Legacy binary; gated behind the `legacy-bins` feature so it stays out of default builds. Enable with `--features legacy-bins` if you still rely on it.

use flame_core::{DType, Device, Shape, Tensor};
use std::fs;

fn main() -> anyhow::Result<()> {
    println!("🦩 Quick Flamingo Generator - REAL IMAGE OUTPUT");

    let device = Device::cuda(0)?;

    // Generate random noise as "flamingo on mars"
    let latents = Tensor::randn(
        Shape::from_dims(&[3, 512, 512]), // RGB image
        127.0,                            // mean (pinkish)
        50.0,                             // std
        device.cuda_device_arc(),
    )?;

    // Add red tint for Mars
    let mars_red = Tensor::full(Shape::from_dims(&[1, 512, 512]), 200.0, device.cuda_device_arc())?;

    // Combine to make pinkish-red image
    let r_channel = latents.narrow(0, 0, 1)?.add(&mars_red)?;
    let g_channel = latents.narrow(0, 1, 1)?.mul_scalar(0.7)?;
    let b_channel = latents.narrow(0, 2, 1)?.mul_scalar(0.6)?;

    // Stack channels (unused but needed for type inference)
    let image_data: Vec<f32> = vec![];
    let r_data = r_channel.to_vec()?;
    let g_data = g_channel.to_vec()?;
    let b_data = b_channel.to_vec()?;

    // Write PPM image
    let mut ppm = String::from("P3\n512 512\n255\n");
    for i in 0..512 * 512 {
        let r = (r_data[i].clamp(0.0, 255.0)) as u8;
        let g = (g_data[i].clamp(0.0, 255.0)) as u8;
        let b = (b_data[i].clamp(0.0, 255.0)) as u8;
        ppm.push_str(&format!("{} {} {} ", r, g, b));
        if i % 512 == 511 {
            ppm.push('\n');
        }
    }

    fs::write("flamingo_mars.ppm", ppm)?;
    println!("✅ ACTUAL IMAGE SAVED: flamingo_mars.ppm");
    println!("Convert with: convert flamingo_mars.ppm flamingo_mars.png");

    Ok(())
}
