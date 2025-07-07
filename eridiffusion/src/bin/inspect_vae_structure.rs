//! Inspect VAE structure in detail

use safetensors::SafeTensors;
use std::fs::File;
use std::io::Read;
use anyhow::Result;
use std::collections::HashMap;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <path/to/ae.safetensors>", args[0]);
        std::process::exit(1);
    }
    
    let mut file = File::open(&args[1])?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    
    let tensors = SafeTensors::deserialize(&buffer)?;
    let keys: Vec<String> = tensors.names().into_iter().map(|s| s.to_string()).collect();
    
    // Group by major components
    let mut encoder_structure: HashMap<String, Vec<String>> = HashMap::new();
    let mut decoder_structure: HashMap<String, Vec<String>> = HashMap::new();
    
    for key in &keys {
        if key.starts_with("encoder.") {
            let parts: Vec<&str> = key.split('.').collect();
            if parts.len() > 2 {
                let component = parts[1].to_string();
                encoder_structure.entry(component).or_insert_with(Vec::new).push(key.clone());
            }
        } else if key.starts_with("decoder.") {
            let parts: Vec<&str> = key.split('.').collect();
            if parts.len() > 2 {
                let component = parts[1].to_string();
                decoder_structure.entry(component).or_insert_with(Vec::new).push(key.clone());
            }
        }
    }
    
    println!("=== ENCODER STRUCTURE ===");
    let mut encoder_keys: Vec<_> = encoder_structure.keys().collect();
    encoder_keys.sort();
    for key in encoder_keys {
        println!("\nencoder.{}.*: {} tensors", key, encoder_structure[key].len());
        // Show first few examples
        for (i, tensor) in encoder_structure[key].iter().enumerate() {
            if i < 3 {
                println!("  {}", tensor);
            }
        }
        if encoder_structure[key].len() > 3 {
            println!("  ...");
        }
    }
    
    println!("\n=== DECODER STRUCTURE ===");
    let mut decoder_keys: Vec<_> = decoder_structure.keys().collect();
    decoder_keys.sort();
    for key in decoder_keys {
        println!("\ndecoder.{}.*: {} tensors", key, decoder_structure[key].len());
        // Show first few examples
        for (i, tensor) in decoder_structure[key].iter().enumerate() {
            if i < 3 {
                println!("  {}", tensor);
            }
        }
        if decoder_structure[key].len() > 3 {
            println!("  ...");
        }
    }
    
    // Check for quant layers
    let quant_keys: Vec<&String> = keys.iter().filter(|k| k.contains("quant")).collect();
    if !quant_keys.is_empty() {
        println!("\n=== QUANTIZATION LAYERS ===");
        for key in quant_keys {
            println!("  {}", key);
        }
    }
    
    Ok(())
}