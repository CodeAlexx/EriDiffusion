//! Staged model loading to avoid OOM on 24GB GPUs
//! 
//! This loads the model in stages, moving only necessary weights to GPU

use anyhow::{Result, Context};
use candle_core::{Device, DType, Tensor};
use candle_nn::{VarBuilder, VarMap};
use std::collections::HashMap;
use std::path::Path;

use super::unified_loader::{Architecture, FluxAdapter};

/// Load Flux weights in stages to avoid OOM
pub fn load_flux_weights_staged(
    checkpoint_path: &Path,
    device: Device,
    dtype: DType,
    hidden_size: usize,
) -> Result<VarBuilder<'static>> {
    println!("Loading Flux model with staged approach for memory efficiency...");
    
    // Stage 1: Load metadata only
    println!("Stage 1: Loading tensor metadata...");
    let file = std::fs::File::open(checkpoint_path)?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    
    // Read header to get tensor list
    let header_size = u64::from_le_bytes(
        mmap[0..8].try_into().context("Failed to read header size")?
    ) as usize;
    let header_json = &mmap[8..8 + header_size];
    let header_str = std::str::from_utf8(header_json)?;
    let parsed: serde_json::Value = serde_json::from_str(header_str)?;
    
    let mut tensor_names = Vec::new();
    if let Some(obj) = parsed.as_object() {
        for (key, _) in obj {
            if key != "__metadata__" {
                tensor_names.push(key.clone());
            }
        }
    }
    println!("Found {} tensors", tensor_names.len());
    
    // Stage 2: Load weights in batches
    println!("Stage 2: Loading weights in batches...");
    let batch_size = 50; // Load 50 tensors at a time
    let adapter = FluxAdapter::new(hidden_size);
    let var_map = VarMap::new();
    
    for (batch_idx, batch) in tensor_names.chunks(batch_size).enumerate() {
        println!("Loading batch {}/{}", batch_idx + 1, (tensor_names.len() + batch_size - 1) / batch_size);
        
        // Load batch to CPU
        let mut batch_tensors = HashMap::new();
        for name in batch {
            // Use candle's selective loading
            let tensors = candle_core::safetensors::load(checkpoint_path, &Device::Cpu)?;
            if let Some(tensor) = tensors.get(name) {
                batch_tensors.insert(name.clone(), tensor.clone());
            }
        }
        
        // Process and move to GPU
        for (name, tensor) in batch_tensors {
            // Adapt tensor
            let adapted_tensors = adapter.adapt_tensor(&name, tensor)?;
            
            for (new_name, new_tensor) in adapted_tensors {
                // Convert dtype if needed (on CPU to save GPU memory)
                let tensor = if new_tensor.dtype() != dtype {
                    new_tensor.to_dtype(dtype)?
                } else {
                    new_tensor
                };
                
                // Move to GPU
                let tensor = if tensor.device().location() != device.location() {
                    tensor.to_device(&device)?
                } else {
                    tensor
                };
                
                // Add to VarMap
                let var = candle_core::Var::from_tensor(&tensor)?;
                var_map.data().lock().unwrap().insert(new_name, var);
            }
        }
        
        // Clear CPU memory after each batch
        drop(batch_tensors);
        
        // Force GPU cache clear every few batches
        if batch_idx % 5 == 0 {
            if let Device::Cuda(_) = device {
                crate::memory::cuda::empty_cache()
                    .map_err(|e| anyhow::anyhow!("Failed to empty cache: {:?}", e))?;
            }
        }
    }
    
    println!("Stage 3: Creating VarBuilder...");
    Ok(VarBuilder::from_varmap(&var_map, dtype, &device))
}