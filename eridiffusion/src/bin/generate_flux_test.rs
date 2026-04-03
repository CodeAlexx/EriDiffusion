#![cfg(feature = "legacy-bins")]

// NOTE: Legacy binary; gated behind the `legacy-bins` feature so it stays out of default builds. Enable with `--features legacy-bins` if you still rely on it.

use flame_core::{DType, Device};

fn main() -> anyhow::Result<()> {
    println!("🦩 Flux Flamingo Generator");
    println!("==========================");

    // For now, just test that we can load the libraries
    let device = Device::cuda(0)?;
    println!("✅ CUDA device initialized");

    println!("\n⚠️  Full Flux inference requires implementing CPU offloading");
    println!("    The model is 23GB but we only have 24GB GPU memory");
    println!("    We need to use the FluxLayerStreamer from training code");

    println!("\n📝 TODO: Use the same memory-efficient loading as training:");
    println!("    - FluxLayerStreamer for streaming layers");
    println!("    - CPU offloading for non-critical weights");
    println!("    - Gradient checkpointing techniques");

    Ok(())
}
