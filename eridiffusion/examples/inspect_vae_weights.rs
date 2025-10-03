use eridiffusion::cuda_device;
use eridiffusion::loaders::WeightLoader;
use std::path::Path;

fn main() -> flame_core::Result<()> {
    println!("=== Inspecting VAE Weights ===");

    let device = cuda_device(0)?;
    let vae_path = Path::new("/home/alex/SwarmUI/Models/VAE/ae.safetensors");

    println!("Loading weights from: {:?}", vae_path);
    let wl = WeightLoader::from_safetensors(vae_path, device)?;

    println!("\nTotal weights: {}", wl.weights.len());
    println!("\nAll weight keys:");

    let mut keys: Vec<_> = wl.weights.keys().collect();
    keys.sort();

    for key in &keys {
        println!("  {}", key);
    }

    println!("\nKeys containing 'quant':");
    for key in &keys {
        if key.contains("quant") {
            println!("  {} -> shape: {:?}", key, wl.weights[*key].shape());
        }
    }

    println!("\nKeys containing 'conv':");
    for key in &keys {
        if key.contains("conv") && !key.contains("quant") {
            println!("  {}", key);
            if keys.len() < 50 {
                // Only show shapes if not too many keys
                println!("    shape: {:?}", wl.weights[*key].shape());
            }
        }
    }

    Ok(())
}
