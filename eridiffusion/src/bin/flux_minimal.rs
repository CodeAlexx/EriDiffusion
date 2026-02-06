#![cfg(feature = "legacy-bins")]

// NOTE: Legacy binary; gated behind the `legacy-bins` feature so it stays out of default builds. Enable with `--features legacy-bins` if you still rely on it.

use flame_core::device::Device;
use flame_core::{DType, Result};
use std::{env, path::Path};

fn main() -> Result<()> {
    println!("Flux Minimal Inference Test");
    println!("===========================");

    // Test basic device creation
    let device = Device::cuda(0)?;
    println!("✓ CUDA device created");

    // Test that we can at least import the inference module
    use eridiffusion::inference::flux::generate_flux_image;

    let variant = env::var("FLUX_VARIANT").unwrap_or_else(|_| "dev".to_string());
    println!("Model variant: {}", variant);

    let prompt = "A red sports car on a mountain road";
    let output_path = Path::new("test_flux_minimal.png");

    println!("Generating image...");
    println!("Prompt: {}", prompt);

    // Try to generate with minimal settings
    generate_flux_image(
        prompt,
        &variant,
        None, // No LoRA
        1.0,  // LoRA scale (unused)
        output_path,
        4,   // Just 4 steps for speed
        1.0, // CFG scale
        512, // Width
        512, // Height
        device,
        DType::F16,
    )?;

    println!("✓ Image generated successfully!");
    println!("Output: {}", output_path.display());

    Ok(())
}
