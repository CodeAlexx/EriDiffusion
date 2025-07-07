//! Utility to merge LoKr weights with base model for inference
//! This creates a new checkpoint with LoKr weights merged in

use anyhow::Result;
use candle_core::{Device, DType, Tensor};
use candle_core::safetensors::{load, save};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about = "Merge LoKr weights with base SD3.5 model")]
struct Args {
    /// Path to base SD3.5 model
    #[arg(long)]
    base_model: PathBuf,
    
    /// Path to LoKr weights
    #[arg(long)]
    lokr_weights: PathBuf,
    
    /// Output path for merged model
    #[arg(long)]
    output: PathBuf,
    
    /// Scale factor for LoKr weights
    #[arg(long, default_value = "1.0")]
    scale: f32,
    
    /// Device to use for merging
    #[arg(long, default_value = "cuda:0")]
    device: String,
}

/// Load LoKr weights and extract W1, W2 matrices
fn load_lokr_weights(path: &Path, device: &Device) -> Result<HashMap<String, (Tensor, Tensor)>> {
    let tensors = load(path, device)?;
    let mut lokr_weights = HashMap::new();
    
    // Group tensors by layer name
    let mut layer_tensors: HashMap<String, HashMap<String, Tensor>> = HashMap::new();
    
    for (name, tensor) in tensors {
        // Parse tensor name: "layer_name.lokr_w1" or "layer_name.w1"
        let parts: Vec<&str> = name.split('.').collect();
        if parts.len() >= 2 {
            let layer_name = parts[..parts.len()-1].join(".");
            let weight_type = parts[parts.len()-1];
            
            layer_tensors.entry(layer_name.clone())
                .or_insert_with(HashMap::new)
                .insert(weight_type.to_string(), tensor);
        }
    }
    
    // Extract W1 and W2 for each layer
    for (layer_name, weights) in layer_tensors {
        let w1 = weights.get("lokr_w1")
            .or_else(|| weights.get("w1"))
            .ok_or_else(|| anyhow::anyhow!("Missing w1 for layer {}", layer_name))?;
            
        let w2 = weights.get("lokr_w2")
            .or_else(|| weights.get("w2"))
            .ok_or_else(|| anyhow::anyhow!("Missing w2 for layer {}", layer_name))?;
            
        lokr_weights.insert(layer_name, (w1.clone(), w2.clone()));
    }
    
    println!("Loaded {} LoKr layers", lokr_weights.len());
    Ok(lokr_weights)
}

/// Compute LoKr weight delta: scale * W1 @ W2
fn compute_lokr_delta(w1: &Tensor, w2: &Tensor, scale: f32) -> Result<Tensor> {
    let delta = w1.matmul(w2)?;
    Ok(delta.affine(scale as f64, 0.0)?)
}

/// Find the base model weight key for a given LoKr layer
fn find_base_weight_key(lokr_name: &str, base_weights: &HashMap<String, Tensor>) -> Option<String> {
    // Map LoKr layer names to base model weight names
    // This mapping depends on the specific model architecture
    
    // Direct mapping attempts
    let candidates = vec![
        format!("{}.weight", lokr_name),
        format!("model.diffusion_model.{}.weight", lokr_name),
        format!("model.diffusion_model.joint_blocks.{}.weight", lokr_name),
        format!("model.diffusion_model.transformer_blocks.{}.weight", lokr_name),
    ];
    
    for candidate in candidates {
        if base_weights.contains_key(&candidate) {
            return Some(candidate);
        }
    }
    
    // Try pattern matching
    for (key, _) in base_weights {
        if key.contains(lokr_name) && key.ends_with(".weight") {
            return Some(key.clone());
        }
    }
    
    None
}

/// Merge LoKr weights into base model
fn merge_weights(
    base_path: &Path,
    lokr_weights: HashMap<String, (Tensor, Tensor)>,
    scale: f32,
    device: &Device,
) -> Result<HashMap<String, Tensor>> {
    println!("Loading base model weights...");
    let mut base_weights = load(base_path, device)?;
    
    println!("Merging LoKr weights with scale {}...", scale);
    let mut merged_count = 0;
    let mut skipped_layers = Vec::new();
    
    for (lokr_name, (w1, w2)) in &lokr_weights {
        // Find corresponding base weight
        if let Some(base_key) = find_base_weight_key(lokr_name, &base_weights) {
            if let Some(base_weight) = base_weights.get_mut(&base_key) {
                // Compute LoKr delta
                let delta = compute_lokr_delta(w1, w2, scale)?;
                
                // Check dimensions match
                if base_weight.dims() == delta.dims() {
                    // Merge: base_weight = base_weight + delta
                    *base_weight = base_weight.add(&delta)?;
                    merged_count += 1;
                    println!("  Merged: {} -> {}", lokr_name, base_key);
                } else {
                    println!("  WARNING: Dimension mismatch for {}: base {:?} vs delta {:?}", 
                        lokr_name, base_weight.dims(), delta.dims());
                    skipped_layers.push(lokr_name.clone());
                }
            }
        } else {
            println!("  WARNING: No matching base weight found for {}", lokr_name);
            skipped_layers.push(lokr_name.clone());
        }
    }
    
    println!("\nMerge complete:");
    println!("  Merged layers: {}", merged_count);
    println!("  Skipped layers: {}", skipped_layers.len());
    if !skipped_layers.is_empty() {
        println!("  Skipped: {:?}", skipped_layers);
    }
    
    Ok(base_weights)
}

fn main() -> Result<()> {
    let args = Args::parse();
    
    // Parse device
    let device = if args.device == "cpu" {
        Device::Cpu
    } else if args.device.starts_with("cuda:") {
        let id = args.device[5..].parse::<usize>()?;
        Device::new_cuda(id)?
    } else {
        Device::new_cuda(0)?
    };
    
    println!("Merging LoKr weights into base model");
    println!("Base model: {}", args.base_model.display());
    println!("LoKr weights: {}", args.lokr_weights.display());
    println!("Output: {}", args.output.display());
    println!("Device: {:?}", device);
    
    // Load LoKr weights
    let lokr_weights = load_lokr_weights(&args.lokr_weights, &device)?;
    
    // Merge with base model
    let merged_weights = merge_weights(&args.base_model, lokr_weights, args.scale, &device)?;
    
    // Save merged model
    println!("\nSaving merged model to {}...", args.output.display());
    
    // Create output directory if needed
    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    
    // Save weights
    save(&merged_weights, &args.output)?;
    
    println!("Done! Merged model saved to {}", args.output.display());
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_lokr_delta_computation() -> Result<()> {
        let device = Device::Cpu;
        
        // Create test tensors
        let w1 = Tensor::randn(0.0f32, 0.02, &[768, 16], &device)?;
        let w2 = Tensor::randn(0.0f32, 0.02, &[16, 768], &device)?;
        
        // Compute delta
        let delta = compute_lokr_delta(&w1, &w2, 1.0)?;
        
        assert_eq!(delta.dims(), &[768, 768]);
        
        Ok(())
    }
}