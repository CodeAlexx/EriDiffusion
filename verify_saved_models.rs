#!/usr/bin/env rust-script
//! Verify saved SD 3.5 LoKr models and caches

use safetensors::{SafeTensors, serialize};
use std::fs;
use std::path::Path;
use std::collections::HashMap;

fn verify_safetensors(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    println!("\nVerifying: {}", path.display());
    
    let data = fs::read(path)?;
    let tensors = SafeTensors::deserialize(&data)?;
    
    println!("  Format: Valid SafeTensors");
    println!("  Number of tensors: {}", tensors.len());
    
    // SafeTensors metadata is not directly accessible in the public API
    // We'll check the tensors instead
    
    // List tensor names and shapes
    println!("  Tensors:");
    let mut total_params = 0;
    for (name, tensor_view) in tensors.tensors() {
        let shape = tensor_view.shape();
        let dtype = tensor_view.dtype();
        let num_elements: usize = shape.iter().product();
        total_params += num_elements;
        
        println!("    {} - shape: {:?}, dtype: {:?}, elements: {}", 
                 name, shape, dtype, num_elements);
    }
    
    println!("  Total parameters: {}", total_params);
    
    // Verify LoKr structure
    let mut lokr_layers = HashMap::new();
    for (name, _) in tensors.tensors() {
        if name.contains("lora_down") || name.contains("lora_up") {
            let parts: Vec<&str> = name.split('.').collect();
            if parts.len() >= 4 {
                let layer_name = parts[..parts.len()-2].join(".");
                lokr_layers.entry(layer_name).or_insert(Vec::new()).push(name.clone());
            }
        }
    }
    
    println!("  LoKr layers found: {}", lokr_layers.len());
    for (layer, weights) in lokr_layers.iter() {
        println!("    {}: {} weights", layer, weights.len());
    }
    
    Ok(())
}

fn check_optimizer_state(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    println!("\nChecking optimizer state: {}", path.display());
    
    let data = fs::read(path)?;
    println!("  File size: {} bytes", data.len());
    
    // For PyTorch .pt files, we can at least check they're not empty
    if data.len() > 100 {
        println!("  Status: File exists and has content");
        
        // Check for PyTorch magic number (ZIP file)
        if data.len() >= 4 && &data[0..2] == b"PK" {
            println!("  Format: Valid PyTorch file (ZIP archive)");
        }
    } else {
        println!("  Status: File too small, might be corrupted");
    }
    
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("====================================");
    println!("SD 3.5 LoKr Model Verification");
    println!("====================================");
    
    // Check for saved models
    let output_dirs = vec![
        "output/sd35_lokr_real_gpu",
        "output/sd35_lokr_gpu",
        "output/rachv1Sd3",
    ];
    
    for dir in output_dirs {
        if !Path::new(dir).exists() {
            continue;
        }
        
        println!("\nChecking directory: {}", dir);
        
        // Check safetensors files
        let entries = fs::read_dir(dir)?;
        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            
            if let Some(ext) = path.extension() {
                match ext.to_str() {
                    Some("safetensors") => {
                        if let Err(e) = verify_safetensors(&path) {
                            println!("  ERROR verifying {}: {}", path.display(), e);
                        }
                    }
                    Some("pt") => {
                        if let Err(e) = check_optimizer_state(&path) {
                            println!("  ERROR checking {}: {}", path.display(), e);
                        }
                    }
                    _ => {}
                }
            }
        }
        
        // Check checkpoints subdirectory
        let checkpoint_dir = Path::new(dir).join("checkpoints");
        if checkpoint_dir.exists() {
            println!("\nChecking checkpoints in: {}", checkpoint_dir.display());
            
            let entries = fs::read_dir(checkpoint_dir)?;
            for entry in entries {
                let entry = entry?;
                let path = entry.path();
                
                if let Some(ext) = path.extension() {
                    match ext.to_str() {
                        Some("safetensors") => {
                            if let Err(e) = verify_safetensors(&path) {
                                println!("  ERROR verifying {}: {}", path.display(), e);
                            }
                        }
                        Some("pt") => {
                            if let Err(e) = check_optimizer_state(&path) {
                                println!("  ERROR checking {}: {}", path.display(), e);
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }
    
    // Create test save to verify our saving works
    println!("\n\nTesting safetensors save functionality...");
    test_safetensors_save()?;
    
    println!("\n✓ Verification complete!");
    Ok(())
}

fn test_safetensors_save() -> Result<(), Box<dyn std::error::Error>> {
    use candle_core::{Device, Tensor, DType};
    
    let device = Device::cuda_if_available(0)?;
    
    // Create test tensors like LoKr weights
    let mut tensors = HashMap::new();
    
    // Simulate LoKr structure
    for i in (0..38).step_by(4) {
        let lora_down = Tensor::randn(0.0, 0.01, &[1536, 64], &device)?
            .to_dtype(DType::F32)?;
        let lora_up = Tensor::zeros(&[64, 1536], DType::F32, &device)?;
        
        tensors.insert(
            format!("joint_blocks.{}.x_block.attn.to_q.lora_down.weight", i),
            lora_down.clone(),
        );
        tensors.insert(
            format!("joint_blocks.{}.x_block.attn.to_q.lora_up.weight", i),
            lora_up.clone(),
        );
        
        tensors.insert(
            format!("joint_blocks.{}.x_block.attn.to_v.lora_down.weight", i),
            lora_down,
        );
        tensors.insert(
            format!("joint_blocks.{}.x_block.attn.to_v.lora_up.weight", i),
            lora_up,
        );
    }
    
    // Add metadata
    let metadata = HashMap::from([
        ("format".to_string(), "pt".to_string()),
        ("type".to_string(), "lokr".to_string()),
        ("rank".to_string(), "64".to_string()),
        ("alpha".to_string(), "64.0".to_string()),
        ("base_model".to_string(), "sd3.5-large".to_string()),
    ]);
    
    // Save test file
    let test_path = "output/test_lokr_save.safetensors";
    fs::create_dir_all("output")?;
    
    let data = serialize(tensors, &Some(metadata))?;
    fs::write(test_path, data)?;
    
    println!("Created test save at: {}", test_path);
    
    // Verify the test save
    verify_safetensors(Path::new(test_path))?;
    
    // Clean up
    fs::remove_file(test_path)?;
    
    Ok(())
}