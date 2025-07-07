//! Debug VAE decoder structure

use safetensors::SafeTensors;
use std::fs::File;
use std::io::Read;
use anyhow::Result;

fn main() -> Result<()> {
    let path = "/home/alex/SwarmUI/Models/VAE/ae.safetensors";
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    
    let tensors = SafeTensors::deserialize(&buffer)?;
    let keys: Vec<String> = tensors.names().into_iter().map(|s| s.to_string()).collect();
    
    // Check decoder up blocks
    println!("=== Decoder Up Blocks ===");
    let mut up_blocks = std::collections::HashSet::new();
    for key in &keys {
        if key.starts_with("decoder.up.") && key.contains(".block.") {
            let parts: Vec<&str> = key.split('.').collect();
            if parts.len() > 3 {
                let block_idx = parts[2];
                up_blocks.insert(block_idx.to_string());
            }
        }
    }
    let mut up_blocks: Vec<_> = up_blocks.into_iter().collect();
    up_blocks.sort();
    println!("Up blocks found: {:?}", up_blocks);
    
    // Check for upsample layers
    println!("\n=== Upsample Layers ===");
    for key in &keys {
        if key.contains("upsample") && key.starts_with("decoder.") {
            println!("{}", key);
        }
    }
    
    // Check block 0 structure
    println!("\n=== Decoder Up Block 0 Structure ===");
    for key in &keys {
        if key.starts_with("decoder.up.0.") {
            println!("{}", key);
        }
    }
    
    // Check block 3 (last block) structure
    println!("\n=== Decoder Up Block 3 Structure ===");
    for key in &keys {
        if key.starts_with("decoder.up.3.") {
            println!("{}", key);
        }
    }
    
    Ok(())
}