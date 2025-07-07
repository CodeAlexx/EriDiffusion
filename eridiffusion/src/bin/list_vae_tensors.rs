//! List tensor names in VAE file

use anyhow::Result;
use safetensors::SafeTensors;
use std::fs::File;
use std::io::Read;

fn main() -> Result<()> {
    let vae_path = "/home/alex/SwarmUI/Models/VAE/ae.safetensors";
    println!("Checking VAE tensors in: {}", vae_path);
    
    // Read the file
    let mut file = File::open(vae_path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    
    // Parse safetensors
    let tensors = SafeTensors::deserialize(&buffer)?;
    let names: Vec<_> = tensors.names();
    
    println!("\nTotal tensors: {}", names.len());
    
    // Look for encoder/decoder patterns
    println!("\nFirst 30 tensor names:");
    for (i, name) in names.iter().take(30).enumerate() {
        println!("  {}: {}", i, name);
    }
    
    // Check for specific patterns
    println!("\nChecking patterns:");
    let has_encoder = names.iter().any(|n| n.contains("encoder"));
    let has_decoder = names.iter().any(|n| n.contains("decoder"));
    let has_down_blocks = names.iter().any(|n| n.contains("down_blocks"));
    let has_up_blocks = names.iter().any(|n| n.contains("up_blocks"));
    
    println!("  Has 'encoder' prefix: {}", has_encoder);
    println!("  Has 'decoder' prefix: {}", has_decoder);
    println!("  Has 'down_blocks': {}", has_down_blocks);
    println!("  Has 'up_blocks': {}", has_up_blocks);
    
    // Check for conv_in
    let conv_in_tensors: Vec<_> = names.iter().filter(|n| n.contains("conv_in")).collect();
    if !conv_in_tensors.is_empty() {
        println!("\nconv_in tensors:");
        for t in conv_in_tensors.iter().take(5) {
            println!("  {}", t);
        }
    }
    
    Ok(())
}