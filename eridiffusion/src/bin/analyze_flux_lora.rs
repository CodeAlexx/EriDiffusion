//! Analyze Flux LoRA model structure

use anyhow::Result;
use safetensors::SafeTensors;
use std::fs::File;
use std::io::Read;
use std::collections::HashMap;

fn main() -> Result<()> {
    let lora_path = "/home/alex/diffusers-rs/KristinKreuk_flux_lora_v3.safetensors";
    println!("Analyzing Flux LoRA model: {}", lora_path);
    
    // Read the file
    let mut file = File::open(lora_path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    
    // Parse safetensors
    let tensors = SafeTensors::deserialize(&buffer)?;
    let names: Vec<_> = tensors.names().into_iter().map(|s| s.to_string()).collect();
    
    println!("\nTotal LoRA tensors: {}", names.len());
    
    // Group by layer type
    let mut layer_groups: HashMap<String, Vec<String>> = HashMap::new();
    
    for name in &names {
        // Extract the layer type (double_blocks, single_blocks, etc.)
        let parts: Vec<&str> = name.split('.').collect();
        if parts.len() > 2 {
            let group = format!("{}.{}", parts[0], parts[1]);
            layer_groups.entry(group).or_insert_with(Vec::new).push(name.clone());
        }
    }
    
    println!("\nLayer groups:");
    let mut groups: Vec<_> = layer_groups.keys().cloned().collect();
    groups.sort();
    for group in &groups {
        println!("  {}: {} tensors", group, layer_groups[&group].len());
    }
    
    // Analyze LoRA structure
    println!("\nLoRA tensor patterns:");
    let mut lora_up_count = 0;
    let mut lora_down_count = 0;
    let mut unique_targets = std::collections::HashSet::new();
    
    for name in &names {
        if name.contains("lora_up") {
            lora_up_count += 1;
            // Extract target layer
            let target = name.replace(".lora_up.weight", "").replace(".lora_up", "");
            unique_targets.insert(target);
        } else if name.contains("lora_down") {
            lora_down_count += 1;
        }
    }
    
    println!("  LoRA up tensors: {}", lora_up_count);
    println!("  LoRA down tensors: {}", lora_down_count);
    println!("  Unique target layers: {}", unique_targets.len());
    
    // Show example tensor names and shapes
    println!("\nExample LoRA tensors (first 20):");
    for (i, name) in names.iter().take(20).enumerate() {
        let tensor = tensors.tensor(name)?;
        println!("  {}: {} - shape: {:?}", i, name, tensor.shape());
    }
    
    // Find specific patterns
    println!("\nAttention LoRA tensors:");
    for name in &names {
        if name.contains("attn") && name.contains("lora") {
            let tensor = tensors.tensor(name)?;
            println!("  {} - shape: {:?}", name, tensor.shape());
            if names.len() > 100 && name.contains("double_blocks.0") {
                break; // Just show first block as example
            }
        }
    }
    
    // Check for rank
    println!("\nAnalyzing LoRA rank:");
    let mut ranks = std::collections::HashSet::new();
    for name in &names {
        if name.contains("lora_down") {
            let tensor = tensors.tensor(name)?;
            let shape = tensor.shape();
            if shape.len() == 2 {
                ranks.insert(shape[1]); // rank is second dimension of down projection
            }
        }
    }
    println!("  Detected ranks: {:?}", ranks);
    
    // Check naming convention
    println!("\nNaming convention analysis:");
    if names[0].contains("transformer.") {
        println!("  Uses 'transformer.' prefix");
    } else if names[0].contains("model.diffusion_model.") {
        println!("  Uses 'model.diffusion_model.' prefix");
    } else {
        println!("  Direct layer names (no prefix)");
    }
    
    Ok(())
}