//! Quick check of model structure

use anyhow::Result;
use std::collections::HashMap;

fn main() -> Result<()> {
    println!("Quick model structure check...\n");
    
    // Check main Flux model
    let flux_path = "/home/alex/SwarmUI/Models/diffusion_models/flux1-dev.safetensors";
    if std::path::Path::new(flux_path).exists() {
        println!("✓ Flux model found at: {}", flux_path);
        let metadata = std::fs::metadata(flux_path)?;
        println!("  Size: {:.2} GB", metadata.len() as f64 / 1e9);
    } else {
        println!("✗ Flux model not found at expected path");
    }
    
    // Check VAE
    let vae_path = "/home/alex/SwarmUI/Models/VAE/ae.safetensors";
    if std::path::Path::new(vae_path).exists() {
        println!("\n✓ VAE found at: {}", vae_path);
        let metadata = std::fs::metadata(vae_path)?;
        println!("  Size: {:.2} MB", metadata.len() as f64 / 1e6);
    } else {
        println!("\n✗ VAE not found at expected path");
    }
    
    // Check text encoders
    let t5_path = "/home/alex/SwarmUI/Models/clip/t5xxl_fp16.safetensors";
    if std::path::Path::new(t5_path).exists() {
        println!("\n✓ T5 encoder found at: {}", t5_path);
        let metadata = std::fs::metadata(t5_path)?;
        println!("  Size: {:.2} GB", metadata.len() as f64 / 1e9);
    } else {
        println!("\n✗ T5 encoder not found at expected path");
    }
    
    // Check for CLIP
    let clip_dir = "/home/alex/SwarmUI/Models/clip";
    if std::path::Path::new(clip_dir).is_dir() {
        println!("\n✓ CLIP directory found at: {}", clip_dir);
        
        // List CLIP files
        let entries = std::fs::read_dir(clip_dir)?;
        let mut clip_files = Vec::new();
        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("safetensors") {
                clip_files.push(path);
            }
        }
        
        if !clip_files.is_empty() {
            println!("  Found {} CLIP model files:", clip_files.len());
            for file in &clip_files[..5.min(clip_files.len())] {
                println!("    - {}", file.file_name().unwrap().to_string_lossy());
            }
            if clip_files.len() > 5 {
                println!("    ... and {} more", clip_files.len() - 5);
            }
        }
    }
    
    println!("\n\nBased on the error 'cannot find tensor encoder.down_blocks.0.resnets.0.norm1.weight',");
    println!("it appears the code is looking for VAE tensors in the main Flux model.");
    println!("The VAE should be loaded separately from: {}", vae_path);
    
    Ok(())
}