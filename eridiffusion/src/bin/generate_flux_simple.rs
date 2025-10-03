#![cfg(feature = "legacy-bins")]

// NOTE: Legacy binary; gated behind the `legacy-bins` feature so it stays out of default builds. Enable with `--features legacy-bins` if you still rely on it.

use eridiffusion::inference::flux::generate_flux_image;
use flame_core::{DType, Device};
use std::path::Path;

fn main() -> anyhow::Result<()> {
    println!("🚀 Flux Image Generation - Simple");
    println!("==================================");

    // Setup device
    let device = Device::cuda(0)?;
    let dtype = DType::F16;

    // Generate the image
    let prompt = "a flamingo on mars";
    let variant = "dev"; // Use flux-dev
    let output_path = Path::new("flamingo_on_mars_flux.png");
    let steps = 20;
    let cfg_scale = 1.0;
    let width = 1024;
    let height = 1024;

    println!("\n🎨 Generating: {}", prompt);
    println!("  Model: flux-{}", variant);
    println!("  Resolution: {}x{}", width, height);
    println!("  Steps: {}", steps);
    println!("  CFG: {}", cfg_scale);
    println!("  Output: {:?}", output_path);

    generate_flux_image(
        prompt,
        variant,
        None, // No LoRA
        1.0,  // LoRA scale (unused)
        output_path,
        steps,
        cfg_scale,
        width,
        height,
        device,
        dtype,
    )?;

    println!("\n✅ Image saved successfully!");
    println!("🎉 Generation complete!");

    Ok(())
}
