//! Inspect actual tensor names in Flux model file

use anyhow::Result;
use std::env;

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let model_path = if args.len() > 1 {
        &args[1]
    } else {
        "/home/alex/SwarmUI/Models/diffusion_models/flux1-schnell.safetensors"
    };
    
    println!("Inspecting: {}", model_path);
    
    // Just read the header to get tensor names without loading full tensors
    let file = std::fs::File::open(model_path)?;
    let mut reader = std::io::BufReader::new(file);
    
    // Read header length (first 8 bytes)
    let mut header_len_bytes = [0u8; 8];
    use std::io::Read;
    reader.read_exact(&mut header_len_bytes)?;
    let header_len = u64::from_le_bytes(header_len_bytes) as usize;
    
    println!("Header length: {} bytes", header_len);
    
    // Read header JSON
    let mut header_bytes = vec![0u8; header_len];
    reader.read_exact(&mut header_bytes)?;
    let header_str = String::from_utf8(header_bytes)?;
    
    // Parse as JSON to get tensor names
    let header: serde_json::Value = serde_json::from_str(&header_str)?;
    
    if let Some(obj) = header.as_object() {
        let mut tensor_names: Vec<String> = obj.keys()
            .filter(|k| !k.starts_with("__"))  // Skip metadata keys
            .cloned()
            .collect();
        tensor_names.sort();
        
        println!("\nTotal tensors: {}", tensor_names.len());
        
        // Check attention naming
        let has_to_q = tensor_names.iter().any(|n| n.contains("to_q"));
        let has_to_k = tensor_names.iter().any(|n| n.contains("to_k"));
        let has_to_v = tensor_names.iter().any(|n| n.contains("to_v"));
        let has_qkv = tensor_names.iter().any(|n| n.contains("qkv"));
        
        println!("\nAttention tensor naming:");
        println!("  Has to_q: {}", has_to_q);
        println!("  Has to_k: {}", has_to_k);
        println!("  Has to_v: {}", has_to_v);
        println!("  Has qkv: {}", has_qkv);
        
        // Show first attention tensors
        println!("\nFirst attention-related tensors:");
        let mut count = 0;
        for name in &tensor_names {
            if name.contains("attn") && (name.contains("weight") || name.contains("bias")) {
                println!("  {}", name);
                count += 1;
                if count >= 10 {
                    break;
                }
            }
        }
        
        // Show double_blocks.0 structure
        println!("\ndouble_blocks.0 tensors:");
        for name in &tensor_names {
            if name.starts_with("double_blocks.0.") {
                println!("  {}", name);
            }
        }
        
        // Show single_blocks structure
        println!("\nsingle_blocks tensors (first few):");
        let mut count = 0;
        for name in &tensor_names {
            if name.starts_with("single_blocks.") {
                println!("  {}", name);
                count += 1;
                if count >= 20 {
                    break;
                }
            }
        }
    }
    
    Ok(())
}