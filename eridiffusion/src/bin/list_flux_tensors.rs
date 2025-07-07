//! List all tensor names in Flux model

use anyhow::Result;
use safetensors::SafeTensors;
use std::fs::File;
use std::io::Read;

fn main() -> Result<()> {
    let model_path = "/home/alex/SwarmUI/Models/diffusion_models/flux1-dev.safetensors";
    
    // Read the file
    let mut file = File::open(model_path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    
    // Parse safetensors
    let tensors = SafeTensors::deserialize(&buffer)?;
    let mut names: Vec<_> = tensors.names().into_iter().map(|s| s.to_string()).collect();
    names.sort();
    
    // Print all double_blocks.0 tensors
    println!("double_blocks.0 tensors:");
    for name in &names {
        if name.starts_with("double_blocks.0.") {
            println!("  {}", name);
        }
    }
    
    // Print all single_blocks.0 tensors
    println!("\nsingle_blocks.0 tensors:");
    for name in &names {
        if name.starts_with("single_blocks.0.") {
            println!("  {}", name);
        }
    }
    
    // Print img_mod tensors
    println!("\nimg_mod related tensors:");
    for name in &names {
        if name.contains("img_mod") {
            println!("  {}", name);
        }
    }
    
    // Check for attention tensor naming
    println!("\nAttention tensor naming check:");
    let has_to_q = names.iter().any(|n| n.contains("to_q"));
    let has_to_k = names.iter().any(|n| n.contains("to_k"));
    let has_to_v = names.iter().any(|n| n.contains("to_v"));
    let has_qkv = names.iter().any(|n| n.contains("qkv"));
    
    println!("  Has to_q tensors: {}", has_to_q);
    println!("  Has to_k tensors: {}", has_to_k);
    println!("  Has to_v tensors: {}", has_to_v);
    println!("  Has qkv tensors: {}", has_qkv);
    
    // Show some attention examples
    println!("\nAttention tensor examples:");
    for name in &names {
        if name.contains("attn") && (name.contains(".weight") || name.contains(".bias")) {
            println!("  {}", name);
            if names.len() > 1000 && name.contains("double_blocks.1") {
                break; // Just show first block
            }
        }
    }
    
    Ok(())
}