//! Quick VAE tensor name inspector

use safetensors::SafeTensors;
use std::fs::File;
use std::io::Read;
use anyhow::Result;

fn inspect_ae_safetensors(path: &str) -> Result<()> {
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    
    let tensors = SafeTensors::deserialize(&buffer)?;
    let mut keys: Vec<String> = tensors.names().into_iter().map(|s| s.to_string()).collect();
    keys.sort();
    
    println!("=== ae.safetensors tensor names ===");
    println!("Total tensors: {}", keys.len());
    
    // Group by component
    let encoder_keys: Vec<&String> = keys.iter().filter(|k| k.contains("encoder")).collect();
    let decoder_keys: Vec<&String> = keys.iter().filter(|k| k.contains("decoder")).collect();
    let other_keys: Vec<&String> = keys.iter()
        .filter(|k| !k.contains("encoder") && !k.contains("decoder"))
        .collect();
    
    println!("\nEncoder tensors: {}", encoder_keys.len());
    for key in encoder_keys.iter().take(10) {
        println!("  {}", key);
    }
    if encoder_keys.len() > 10 {
        println!("  ... and {} more", encoder_keys.len() - 10);
    }
    
    println!("\nDecoder tensors: {}", decoder_keys.len());
    for key in decoder_keys.iter().take(10) {
        println!("  {}", key);
    }
    if decoder_keys.len() > 10 {
        println!("  ... and {} more", decoder_keys.len() - 10);
    }
    
    println!("\nOther tensors: {}", other_keys.len());
    for key in &other_keys {
        println!("  {}", key);
    }
    
    // Look for specific patterns
    println!("\n=== Key patterns ===");
    
    // Check for conv_norm_out
    let norm_out_keys: Vec<&String> = keys.iter().filter(|k| k.contains("norm_out")).collect();
    println!("\nNorm out layers:");
    for key in &norm_out_keys {
        println!("  {}", key);
    }
    
    // Check block structure
    let block_keys: Vec<&String> = keys.iter().filter(|k| k.contains("block")).collect();
    println!("\nSample block tensors:");
    for key in block_keys.iter().take(5) {
        println!("  {}", key);
    }
    
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <path/to/ae.safetensors>", args[0]);
        eprintln!("Example: {} /home/alex/SwarmUI/Models/VAE/ae.safetensors", args[0]);
        std::process::exit(1);
    }
    
    inspect_ae_safetensors(&args[1])?;
    Ok(())
}