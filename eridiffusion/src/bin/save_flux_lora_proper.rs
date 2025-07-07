//! Save Flux LoRA weights with AI-Toolkit compatible naming
//! This shows the proper way to save LoRA weights that can be loaded by AI-Toolkit

use anyhow::Result;
use safetensors::{serialize_to_file, View, Dtype as SafeDtype};
use std::collections::HashMap;
use std::borrow::Cow;
use std::path::Path;

/// Simple tensor for demonstration
struct SimpleTensor {
    shape: Vec<usize>,
    data: Vec<u8>,
    dtype: SafeDtype,
}

impl SimpleTensor {
    /// Create a new tensor filled with zeros
    fn zeros(shape: Vec<usize>, dtype: SafeDtype) -> Self {
        let n_elements: usize = shape.iter().product();
        let byte_size = n_elements * dtype.size();
        let data = vec![0u8; byte_size];
        Self { shape, data, dtype }
    }
    
    /// Create a new tensor with small random values (simulated)
    fn randn(shape: Vec<usize>, dtype: SafeDtype) -> Self {
        let n_elements: usize = shape.iter().product();
        let byte_size = n_elements * dtype.size();
        // For demo purposes, just create data with a pattern
        let mut data = vec![0u8; byte_size];
        for i in 0..byte_size {
            data[i] = (i % 256) as u8;
        }
        Self { shape, data, dtype }
    }
}

impl View for &SimpleTensor {
    fn dtype(&self) -> SafeDtype {
        self.dtype
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> Cow<[u8]> {
        Cow::Borrowed(&self.data)
    }

    fn data_len(&self) -> usize {
        self.data.len()
    }
}

fn main() -> Result<()> {
    println!("Creating Flux LoRA weights with AI-Toolkit naming convention...\n");
    
    // Configuration
    let rank = 32;
    let alpha = 32.0;
    let hidden_size = 3072;
    let mlp_hidden = hidden_size * 4; // 12288
    let dtype = SafeDtype::F16; // Use F16 for smaller file size
    
    // Store all tensors
    let mut tensors: HashMap<String, SimpleTensor> = HashMap::new();
    
    // Create LoRA weights for first few blocks as example
    println!("Creating LoRA tensors...");
    
    // Double blocks (Flux has 19 double blocks)
    for block_idx in 0..2 {  // Just first 2 for demo
        // Image attention
        for target in ["to_q", "to_k", "to_v"] {
            let key_a = format!("transformer.double_blocks.{}.img_attn.{}.lora_A", block_idx, target);
            tensors.insert(key_a, SimpleTensor::randn(vec![hidden_size, rank], dtype));
            
            let key_b = format!("transformer.double_blocks.{}.img_attn.{}.lora_B", block_idx, target);
            tensors.insert(key_b, SimpleTensor::zeros(vec![rank, hidden_size], dtype));
        }
        
        // Text attention
        for target in ["to_q", "to_k", "to_v"] {
            let key_a = format!("transformer.double_blocks.{}.txt_attn.{}.lora_A", block_idx, target);
            tensors.insert(key_a, SimpleTensor::randn(vec![hidden_size, rank], dtype));
            
            let key_b = format!("transformer.double_blocks.{}.txt_attn.{}.lora_B", block_idx, target);
            tensors.insert(key_b, SimpleTensor::zeros(vec![rank, hidden_size], dtype));
        }
        
        // Image MLP
        let key = format!("transformer.double_blocks.{}.img_mlp.0.lora_A", block_idx);
        tensors.insert(key, SimpleTensor::randn(vec![hidden_size, rank], dtype));
        
        let key = format!("transformer.double_blocks.{}.img_mlp.0.lora_B", block_idx);
        tensors.insert(key, SimpleTensor::zeros(vec![rank, mlp_hidden], dtype));
        
        let key = format!("transformer.double_blocks.{}.img_mlp.2.lora_A", block_idx);
        tensors.insert(key, SimpleTensor::randn(vec![mlp_hidden, rank], dtype));
        
        let key = format!("transformer.double_blocks.{}.img_mlp.2.lora_B", block_idx);
        tensors.insert(key, SimpleTensor::zeros(vec![rank, hidden_size], dtype));
        
        // Text MLP
        let key = format!("transformer.double_blocks.{}.txt_mlp.0.lora_A", block_idx);
        tensors.insert(key, SimpleTensor::randn(vec![hidden_size, rank], dtype));
        
        let key = format!("transformer.double_blocks.{}.txt_mlp.0.lora_B", block_idx);
        tensors.insert(key, SimpleTensor::zeros(vec![rank, mlp_hidden], dtype));
        
        let key = format!("transformer.double_blocks.{}.txt_mlp.2.lora_A", block_idx);
        tensors.insert(key, SimpleTensor::randn(vec![mlp_hidden, rank], dtype));
        
        let key = format!("transformer.double_blocks.{}.txt_mlp.2.lora_B", block_idx);
        tensors.insert(key, SimpleTensor::zeros(vec![rank, hidden_size], dtype));
    }
    
    // Single blocks (Flux has 38 single blocks)
    for block_idx in 0..2 {  // Just first 2 for demo
        // Self attention
        for target in ["to_q", "to_k", "to_v"] {
            let key_a = format!("transformer.single_blocks.{}.attn.{}.lora_A", block_idx, target);
            tensors.insert(key_a, SimpleTensor::randn(vec![hidden_size, rank], dtype));
            
            let key_b = format!("transformer.single_blocks.{}.attn.{}.lora_B", block_idx, target);
            tensors.insert(key_b, SimpleTensor::zeros(vec![rank, hidden_size], dtype));
        }
        
        // MLP
        let key = format!("transformer.single_blocks.{}.mlp.0.lora_A", block_idx);
        tensors.insert(key, SimpleTensor::randn(vec![hidden_size, rank], dtype));
        
        let key = format!("transformer.single_blocks.{}.mlp.0.lora_B", block_idx);
        tensors.insert(key, SimpleTensor::zeros(vec![rank, mlp_hidden], dtype));
        
        let key = format!("transformer.single_blocks.{}.mlp.2.lora_A", block_idx);
        tensors.insert(key, SimpleTensor::randn(vec![mlp_hidden, rank], dtype));
        
        let key = format!("transformer.single_blocks.{}.mlp.2.lora_B", block_idx);
        tensors.insert(key, SimpleTensor::zeros(vec![rank, hidden_size], dtype));
    }
    
    println!("Created {} LoRA tensors", tensors.len());
    
    // Prepare metadata
    let mut metadata = HashMap::new();
    metadata.insert("format".to_string(), "pt".to_string());
    metadata.insert("lora_type".to_string(), "flux".to_string());
    metadata.insert("rank".to_string(), rank.to_string());
    metadata.insert("alpha".to_string(), alpha.to_string());
    metadata.insert("base_model".to_string(), "flux.1-dev".to_string());
    
    // Convert to format needed by serialize_to_file
    // We need to create a vector with string references
    let mut tensor_data: Vec<(&str, &SimpleTensor)> = Vec::new();
    let mut sorted_keys: Vec<_> = tensors.keys().collect();
    sorted_keys.sort();
    
    for key in &sorted_keys {
        tensor_data.push((key.as_str(), tensors.get(*key).unwrap()));
    }
    
    // Save to file
    let output_path = Path::new("/home/alex/diffusers-rs/eridiffusion/flux_lora_demo.safetensors");
    println!("\nSaving to: {:?}", output_path);
    
    serialize_to_file(tensor_data, &Some(metadata), output_path)?;
    
    println!("\nSuccessfully saved Flux LoRA weights!");
    println!("\nFirst 10 tensor names:");
    for (name, _) in tensor_data.iter().take(10) {
        println!("  {}", name);
    }
    
    println!("\nNOTE: This is a demo file with dummy weights.");
    println!("In actual training, these would be the learned LoRA parameters.");
    
    Ok(())
}