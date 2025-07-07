//! Minimal working example of Flux LoRA saving in AI-Toolkit format

use anyhow::Result;
use safetensors::{serialize_to_file, View, Dtype};
use std::collections::HashMap;
use std::borrow::Cow;
use std::path::Path;

/// Simple tensor wrapper for demo
struct DemoTensor {
    data: Vec<f32>,
    shape: Vec<usize>,
}

impl DemoTensor {
    fn new_random(shape: Vec<usize>) -> Self {
        let size: usize = shape.iter().product();
        let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.001).collect();
        Self { data, shape }
    }
}

impl View for &DemoTensor {
    fn dtype(&self) -> Dtype { Dtype::F32 }
    fn shape(&self) -> &[usize] { &self.shape }
    fn data(&self) -> Cow<[u8]> {
        let bytes: Vec<u8> = self.data.iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        Cow::Owned(bytes)
    }
    fn data_len(&self) -> usize { self.data.len() * 4 }
}

fn main() -> Result<()> {
    println!("=== Flux LoRA AI-Toolkit Format Demo ===\n");
    
    // Configuration
    let rank = 32;
    let hidden_size = 3072;
    
    println!("Creating LoRA weights:");
    println!("- Rank: {}", rank);
    println!("- Hidden size: {}", hidden_size);
    println!("- Format: AI-Toolkit (separate Q,K,V)\n");
    
    // Create demo tensors
    let mut tensors = HashMap::new();
    
    // Double block 0 - Image attention
    println!("Creating double_blocks.0.img_attn weights:");
    for target in ["to_q", "to_k", "to_v"] {
        let key_a = format!("transformer.double_blocks.0.img_attn.{}.lora_A", target);
        let key_b = format!("transformer.double_blocks.0.img_attn.{}.lora_B", target);
        
        println!("  {} [{}, {}]", key_a, hidden_size, rank);
        println!("  {} [{}, {}]", key_b, rank, hidden_size);
        
        tensors.insert(key_a, DemoTensor::new_random(vec![hidden_size, rank]));
        tensors.insert(key_b, DemoTensor::new_random(vec![rank, hidden_size]));
    }
    
    // Single block 0 - Self attention
    println!("\nCreating single_blocks.0.attn weights:");
    for target in ["to_q", "to_k", "to_v"] {
        let key_a = format!("transformer.single_blocks.0.attn.{}.lora_A", target);
        let key_b = format!("transformer.single_blocks.0.attn.{}.lora_B", target);
        
        tensors.insert(key_a, DemoTensor::new_random(vec![hidden_size, rank]));
        tensors.insert(key_b, DemoTensor::new_random(vec![rank, hidden_size]));
    }
    
    // Prepare for saving
    let tensor_refs: Vec<(&str, &DemoTensor)> = tensors.iter()
        .map(|(k, v)| (k.as_str(), v))
        .collect();
    
    // Metadata
    let mut metadata = HashMap::new();
    metadata.insert("format".to_string(), "ai-toolkit".to_string());
    metadata.insert("base_model".to_string(), "flux.1-dev".to_string());
    metadata.insert("rank".to_string(), rank.to_string());
    metadata.insert("alpha".to_string(), rank.to_string());
    
    // Save
    let output_path = Path::new("flux_lora_aikotoolkit_format.safetensors");
    serialize_to_file(tensor_refs, &Some(metadata), output_path)?;
    
    println!("\n✓ Saved to: {:?}", output_path);
    println!("\nKey points:");
    println!("1. Uses 'transformer.' prefix");
    println!("2. Separate to_q, to_k, to_v (not combined qkv)");
    println!("3. Uses lora_A and lora_B naming");
    println!("4. Compatible with SimpleTuner and AI-Toolkit");
    
    Ok(())
}