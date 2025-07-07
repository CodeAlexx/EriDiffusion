//! Example of saving Flux LoRA weights with AI-Toolkit compatible naming

use anyhow::Result;
use candle_core::{Device, Tensor, DType};
use safetensors::{serialize_to_file, View};
use std::collections::HashMap;
use std::borrow::Cow;
use std::path::Path;

/// Wrapper to make Candle tensors work with safetensors
#[derive(Clone, Copy)]
struct TensorView<'a> {
    tensor: &'a Tensor,
}

impl<'a> View for TensorView<'a> {
    fn dtype(&self) -> safetensors::Dtype {
        match self.tensor.dtype() {
            DType::F16 => safetensors::Dtype::F16,
            DType::F32 => safetensors::Dtype::F32,
            DType::BF16 => safetensors::Dtype::BF16,
            DType::I64 => safetensors::Dtype::I64,
            DType::U8 => safetensors::Dtype::U8,
            DType::U32 => safetensors::Dtype::U32,
            _ => panic!("Unsupported dtype"),
        }
    }

    fn shape(&self) -> &[usize] {
        self.tensor.dims()
    }

    fn data(&self) -> Cow<[u8]> {
        // Convert tensor to bytes
        let data = match self.tensor.dtype() {
            DType::F32 => {
                let vec = self.tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                bytemuck::cast_slice(&vec).to_vec()
            },
            DType::F16 => {
                let vec = self.tensor.flatten_all().unwrap().to_vec1::<half::f16>().unwrap();
                bytemuck::cast_slice(&vec).to_vec()
            },
            DType::BF16 => {
                let vec = self.tensor.flatten_all().unwrap().to_vec1::<half::bf16>().unwrap();
                bytemuck::cast_slice(&vec).to_vec()
            },
            _ => panic!("Unsupported dtype for data conversion"),
        };
        Cow::Owned(data)
    }

    fn data_len(&self) -> usize {
        let n_elements: usize = self.tensor.dims().iter().product();
        n_elements * self.tensor.dtype().size_in_bytes()
    }
}

fn main() -> Result<()> {
    println!("Example: Saving Flux LoRA weights with AI-Toolkit naming");
    
    // Initialize device
    let device = Device::Cpu;
    let dtype = DType::F32;
    
    // LoRA configuration
    let rank = 32;
    let alpha = 32.0;
    
    // Example dimensions from Flux model
    let hidden_size = 3072;
    
    // Create example LoRA weights for a single layer
    // Following AI-Toolkit naming: transformer.double_blocks.0.img_attn.to_q.lora_A
    let mut tensors_to_save: Vec<(String, TensorView)> = Vec::new();
    
    // Double blocks (example: block 0)
    for block_idx in 0..1 {
        // Image attention
        for target in ["to_q", "to_k", "to_v"].iter() {
            // lora_A: [hidden_size, rank]
            let lora_a = Tensor::randn(0.0, 0.02, (hidden_size, rank), &device)?;
            let key_a = format!("transformer.double_blocks.{}.img_attn.{}.lora_A", block_idx, target);
            tensors_to_save.push((key_a, TensorView { tensor: &lora_a }));
            
            // lora_B: [rank, hidden_size]
            let lora_b = Tensor::zeros((rank, hidden_size), dtype, &device)?;
            let key_b = format!("transformer.double_blocks.{}.img_attn.{}.lora_B", block_idx, target);
            tensors_to_save.push((key_b, TensorView { tensor: &lora_b }));
        }
        
        // Text attention
        for target in ["to_q", "to_k", "to_v"].iter() {
            let lora_a = Tensor::randn(0.0, 0.02, (hidden_size, rank), &device)?;
            let key_a = format!("transformer.double_blocks.{}.txt_attn.{}.lora_A", block_idx, target);
            tensors_to_save.push((key_a, TensorView { tensor: &lora_a }));
            
            let lora_b = Tensor::zeros((rank, hidden_size), dtype, &device)?;
            let key_b = format!("transformer.double_blocks.{}.txt_attn.{}.lora_B", block_idx, target);
            tensors_to_save.push((key_b, TensorView { tensor: &lora_b }));
        }
        
        // MLPs
        for mlp_type in ["img_mlp", "txt_mlp"].iter() {
            let mlp_hidden = hidden_size * 4; // Flux uses 4x hidden for MLP
            
            // lora_A for first projection
            let lora_a = Tensor::randn(0.0, 0.02, (hidden_size, rank), &device)?;
            let key_a = format!("transformer.double_blocks.{}.{}.0.lora_A", block_idx, mlp_type);
            tensors_to_save.push((key_a, TensorView { tensor: &lora_a }));
            
            // lora_B for first projection
            let lora_b = Tensor::zeros((rank, mlp_hidden), dtype, &device)?;
            let key_b = format!("transformer.double_blocks.{}.{}.0.lora_B", block_idx, mlp_type);
            tensors_to_save.push((key_b, TensorView { tensor: &lora_b }));
            
            // lora_A for second projection
            let lora_a2 = Tensor::randn(0.0, 0.02, (mlp_hidden, rank), &device)?;
            let key_a2 = format!("transformer.double_blocks.{}.{}.2.lora_A", block_idx, mlp_type);
            tensors_to_save.push((key_a2, TensorView { tensor: &lora_a2 }));
            
            // lora_B for second projection
            let lora_b2 = Tensor::zeros((rank, hidden_size), dtype, &device)?;
            let key_b2 = format!("transformer.double_blocks.{}.{}.2.lora_B", block_idx, mlp_type);
            tensors_to_save.push((key_b2, TensorView { tensor: &lora_b2 }));
        }
    }
    
    // Single blocks (example: block 0)
    for block_idx in 0..1 {
        // Self attention
        for target in ["to_q", "to_k", "to_v"].iter() {
            let lora_a = Tensor::randn(0.0, 0.02, (hidden_size, rank), &device)?;
            let key_a = format!("transformer.single_blocks.{}.attn.{}.lora_A", block_idx, target);
            tensors_to_save.push((key_a, TensorView { tensor: &lora_a }));
            
            let lora_b = Tensor::zeros((rank, hidden_size), dtype, &device)?;
            let key_b = format!("transformer.single_blocks.{}.attn.{}.lora_B", block_idx, target);
            tensors_to_save.push((key_b, TensorView { tensor: &lora_b }));
        }
        
        // MLP
        let mlp_hidden = hidden_size * 4;
        
        let lora_a = Tensor::randn(0.0, 0.02, (hidden_size, rank), &device)?;
        let key_a = format!("transformer.single_blocks.{}.mlp.0.lora_A", block_idx);
        tensors_to_save.push((key_a, TensorView { tensor: &lora_a }));
        
        let lora_b = Tensor::zeros((rank, mlp_hidden), dtype, &device)?;
        let key_b = format!("transformer.single_blocks.{}.mlp.0.lora_B", block_idx);
        tensors_to_save.push((key_b, TensorView { tensor: &lora_b }));
        
        let lora_a2 = Tensor::randn(0.0, 0.02, (mlp_hidden, rank), &device)?;
        let key_a2 = format!("transformer.single_blocks.{}.mlp.2.lora_A", block_idx);
        tensors_to_save.push((key_a2, TensorView { tensor: &lora_a2 }));
        
        let lora_b2 = Tensor::zeros((rank, hidden_size), dtype, &device)?;
        let key_b2 = format!("transformer.single_blocks.{}.mlp.2.lora_B", block_idx);
        tensors_to_save.push((key_b2, TensorView { tensor: &lora_b2 }));
    }
    
    // Prepare metadata
    let mut metadata = HashMap::new();
    metadata.insert("format".to_string(), "pt".to_string());
    metadata.insert("lora_type".to_string(), "flux".to_string());
    metadata.insert("rank".to_string(), rank.to_string());
    metadata.insert("alpha".to_string(), alpha.to_string());
    
    // Convert to the format expected by serialize_to_file
    let tensor_map: Vec<(&str, TensorView)> = tensors_to_save
        .iter()
        .map(|(k, v)| (k.as_str(), *v))
        .collect();
    
    // Save to file
    let output_path = Path::new("/home/alex/diffusers-rs/eridiffusion/flux_lora_example.safetensors");
    println!("Saving {} tensors to {:?}", tensor_map.len(), output_path);
    
    serialize_to_file(tensor_map, &Some(metadata), output_path)?;
    
    println!("Successfully saved Flux LoRA weights!");
    println!("\nExample tensor names:");
    for (name, _) in tensor_map.iter().take(10) {
        println!("  {}", name);
    }
    
    Ok(())
}