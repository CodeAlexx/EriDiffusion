#![cfg(feature = "legacy-bins")]

// NOTE: Legacy binary; gated behind the `legacy-bins` feature so it stays out of default builds. Enable with `--features legacy-bins` if you still rely on it.

use eridiffusion::inference::generate_flux_image;
use flame_core::{DType, Device};
use std::path::Path;

fn main() -> anyhow::Result<()> {
    println!("🦩 Generating Flamingo on Mars with Flux-Dev");
    println!("{}", "=".repeat(50));

    // Setup device with cuDNN enabled
    let device = Device::cuda(0)?;
    let dtype = DType::F16;

    // Parameters as requested
    let prompt = "a flamingo on mars";
    let variant = "schnell"; // Using schnell temporarily due to memory constraints
    let steps = 4; // Schnell only needs 4 steps
    let cfg_scale = 1.0; // CFG 1.0 as requested
    let width = 512; // Start with 512x512 for memory
    let height = 512;
    let output_path = Path::new("flamingo_on_mars.ppm");

    println!("Configuration:");
    println!("  Model: flux1-{}", variant);
    println!("  Prompt: {}", prompt);
    println!("  Steps: {}", steps);
    println!("  CFG Scale: {}", cfg_scale);
    println!("  Resolution: {}x{}", width, height);
    println!("  Output: {}", output_path.display());
    println!();

    // Now ACTUALLY generate the image
    println!("Starting generation...");
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

    println!("\n✅ Image generated successfully!");
    println!("📁 Output saved to: {}", output_path.display());
    println!("💡 Convert to PNG with: convert {} flamingo_on_mars.png", output_path.display());

    Ok(())
}
