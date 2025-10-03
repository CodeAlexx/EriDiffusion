#![cfg(feature = "legacy-bins")]

// NOTE: Legacy binary; gated behind the `legacy-bins` feature so it stays out of default builds. Enable with `--features legacy-bins` if you still rely on it.

use crate::loaders::WeightLoader::from_safetensors;
use flame_core::device::Device;
use flame_core::{CudaDevice, Result};
use std::collections::HashMap;

fn main(device: &CudaDevice) -> flame_core::Result<()> {
    println!("=== LoRA Weight Inspector ===");

    let lora_path = "/home/alex/diffusers-rs/Cyberpunk_Anime_sdxl.safetensors";
    let device = Device::cuda(0)?;

    println!("Loading LoRA from: {}", lora_path);

    // Load the safetensors file
    let tensors = load_safetensors(lora_path, &device)?;

    println!("\nFound {} tensors in LoRA file", tensors.len());

    // Group by type
    let mut lora_down_count = 0;
    let mut lora_up_count = 0;
    let mut text_encoder_loras = 0;
    let mut unet_loras = 0;

    // Analyze the weights
    println!("\nLoRA weight analysis:");
    println!("{}", "=".repeat(80));

    for (name, tensor) in &tensors {
        let shape = tensor.shape();

        if name.contains("lora_down") {
            lora_down_count += 1;
        } else if name.contains("lora_up") {
            lora_up_count += 1;
        }

        if name.contains("text_encoder") || name.contains("te1") || name.contains("te2") {
            text_encoder_loras += 1;
        } else if name.contains("unet")
            || name.contains("down_blocks")
            || name.contains("mid_block")
            || name.contains("up_blocks")
        {
            unet_loras += 1;
        }

        // Print first few for inspection
        if lora_down_count + lora_up_count <= 10 {
            println!("{}: {:?}", name, shape);
        }
    }

    println!("\nSummary:");
    println!("  LoRA down projections: {}", lora_down_count);
    println!("  LoRA up projections: {}", lora_up_count);
    println!("  Text encoder LoRAs: {}", text_encoder_loras);
    println!("  UNet LoRAs: {}", unet_loras);

    // Check for common LoRA patterns
    let mut attention_loras = HashMap::new();
    for (name, _) in &tensors {
        if name.contains("attn") || name.contains("attention") {
            let parts: Vec<&str> = name.split('.').collect();
            for part in parts {
                if part.contains("to_k")
                    || part.contains("to_v")
                    || part.contains("to_q")
                    || part.contains("to_out")
                {
                    *attention_loras.entry(part).or_insert(0) += 1;
                }
            }
        }
    }

    println!("\nAttention layer targets:");
    for (target, count) in attention_loras {
        println!("  {}: {} layers", target, count / 2); // Divide by 2 for up/down pairs
    }

    // Example weight names for documentation
    println!("\nExample weight names:");
    let mut examples = vec![];
    for (name, _) in tensors.iter().take(5) {
        examples.push(name.clone());
    }
    examples.sort();
    for name in examples {
        println!("  {}", name);
    }

    Ok(())
}
