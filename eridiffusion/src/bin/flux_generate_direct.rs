#![cfg(feature = "legacy-bins")]

// NOTE: Legacy binary; gated behind the `legacy-bins` feature so it stays out of default builds. Enable with `--features legacy-bins` if you still rely on it.

use flame_core::{DType, Device};
use std::path::Path;

fn main() -> anyhow::Result<()> {
    println!("🚀 Flux-dev Image Generation");
    println!("============================");

    // Setup device
    let device = Device::cuda(0)?;

    // Call the existing generate_flux_image function
    eridiffusion::inference::flux::generate_flux_image(
        "a flamingo on mars", // prompt
        "dev",                // variant - using dev as requested
        None,                 // no LoRA
        1.0,                  // LoRA scale (unused)
        Path::new("flamingo_on_mars_flux.png"),
        20,   // steps
        1.0,  // CFG scale
        1024, // width
        1024, // height
        device,
        DType::F16, // Use F16 for memory efficiency
    )?;

    println!("\n✅ Image generated successfully!");
    println!("📸 Output: flamingo_on_mars_flux.png");

    Ok(())
}
