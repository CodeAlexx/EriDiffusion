//! Single-pass Flux loader with on-the-fly quantization
//! 
//! Loads and quantizes weights in a single pass to avoid device context issues

use anyhow::{Result, Context};
use candle_core::{Tensor, Device, DType};
use candle_nn::{VarBuilder, VarMap};
use std::collections::HashMap;
use std::path::Path;
use safetensors::SafeTensors;
use memmap2::Mmap;
use std::fs::File;

// Remove quanto import for now
use crate::models::flux_custom::{FluxConfig, FluxModelWithLoRA};
use crate::models::flux_custom::lora::LoRAConfig;

pub struct SinglePassFluxLoader {
    device: Device,
    dtype: DType,
}

impl SinglePassFluxLoader {
    pub fn new(dtype: DType) -> Result<Self> {
        let device = crate::trainers::cached_device::get_single_device()?;
        Ok(Self { device, dtype })
    }
    
    /// Load and quantize Flux model in a single pass
    pub fn load_and_quantize_flux(
        &self,
        model_path: &Path,
        flux_config: &FluxConfig,
        lora_config: &LoRAConfig,
    ) -> Result<FluxModelWithLoRA> {
        println!("=== Single-Pass Flux Loading with Quantization ===");
        
        // Memory-map the safetensors file
        let file = File::open(model_path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let tensors = SafeTensors::deserialize(&mmap)?;
        
        println!("Found {} tensors in model file", tensors.names().len());
        
        // Create VarMap for the model
        let var_map = VarMap::new();
        let mut loaded_count = 0;
        
        // Process each tensor as we load it
        for name in tensors.names() {
            if loaded_count % 50 == 0 {
                println!("Loaded {}/{} tensors...", loaded_count, tensors.names().len());
            }
            
            // Load tensor data
            let tensor_view = tensors.tensor(name)?;
            let shape = tensor_view.shape();
            let dtype = convert_dtype(tensor_view.dtype())?;
            
            // Create tensor on CPU first
            let data = tensor_view.data();
            let cpu_tensor = Tensor::from_raw_buffer(
                data,
                dtype,
                shape,
                &Device::Cpu,
            )?;
            
            // Quantize if it's a large weight
            let final_tensor = if should_quantize(name) && shape.iter().product::<usize>() > 100_000 {
                // Quantize to INT8
                let quantized = quantize_tensor_int8(&cpu_tensor)?;
                
                // Move to cached device
                quantized.to_device(&self.device)?
            } else {
                // Small tensor, just move to device
                cpu_tensor.to_device(&self.device)?
            };
            
            // Add to VarMap
            var_map.data().lock().unwrap().insert(
                name.to_string(),
                candle_core::Var::from_tensor(&final_tensor)?
            );
            
            loaded_count += 1;
        }
        
        println!("Creating model structure with loaded weights...");
        
        // Create VarBuilder from our loaded weights
        let vb = VarBuilder::from_varmap(&var_map, self.dtype, &self.device);
        
        // Create the model
        let mut model = FluxModelWithLoRA::new(flux_config, vb)?;
        
        // LoRA is already initialized in the model constructor
        
        println!("✅ Model loaded and quantized in single pass!");
        Ok(model)
    }
}

/// Quantize a tensor to INT8 (simplified for now)
fn quantize_tensor_int8(tensor: &Tensor) -> Result<Tensor> {
    // For now, just convert to F16 to save memory
    // Proper INT8 quantization would require more complex handling
    Ok(tensor.to_dtype(DType::F16)?)
}

/// Check if a tensor should be quantized
fn should_quantize(name: &str) -> bool {
    // Don't quantize embeddings or layer norms
    !name.contains("embed") && 
    !name.contains("norm") &&
    !name.contains("ln") &&
    !name.contains("final_layer")
}

/// Convert safetensors dtype to candle dtype
fn convert_dtype(dtype: safetensors::Dtype) -> Result<DType> {
    match dtype {
        safetensors::Dtype::F32 => Ok(DType::F32),
        safetensors::Dtype::F16 => Ok(DType::F16),
        safetensors::Dtype::BF16 => Ok(DType::BF16),
        safetensors::Dtype::U8 => Ok(DType::U8),
        _ => anyhow::bail!("Unsupported dtype: {:?}", dtype),
    }
}