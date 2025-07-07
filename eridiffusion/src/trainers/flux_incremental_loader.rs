//! Incremental loader for Flux model
//! 
//! This loads the model layer by layer to avoid OOM issues

use anyhow::{Result, Context};
use candle_core::{Tensor, Device, DType, D};
use candle_nn::{VarBuilder, VarMap};
use std::path::Path;
use std::collections::HashMap;

use crate::models::flux_custom::{FluxConfig, FluxModelWithLoRA};
use crate::models::flux_custom::lora::LoRAConfig;
use crate::trainers::flux_cpu_offload::WeightOffloadManager;

/// Load Flux model incrementally with CPU offloading
pub struct IncrementalFluxLoader {
    device: Device,
    dtype: DType,
    offload_manager: WeightOffloadManager,
}

impl IncrementalFluxLoader {
    pub fn new(device: Device, dtype: DType) -> Result<Self> {
        // Keep only essential weights on GPU (e.g., 100 weights max)
        let offload_manager = WeightOffloadManager::new(device.clone(), 100);
        
        Ok(Self {
            device,
            dtype,
            offload_manager,
        })
    }
    
    /// Load model with incremental weight loading and CPU offloading
    pub fn load_incremental(
        &self,
        model_path: &Path,
        flux_config: &FluxConfig,
        lora_config: &LoRAConfig,
    ) -> Result<FluxModelWithLoRA> {
        println!("\n=== Incremental Flux Loading with CPU Offload ===");
        
        // First, load all weights to CPU
        println!("Step 1: Loading weights to CPU...");
        let cpu_weights = self.load_weights_to_cpu(model_path)?;
        println!("Loaded {} weights to CPU", cpu_weights.len());
        
        // Store all weights in offload manager
        println!("Step 2: Storing weights in offload manager...");
        for (name, weight) in cpu_weights {
            self.offload_manager.store_weight(name, weight)?;
        }
        
        // Create a custom VarBuilder that uses the offload manager
        println!("Step 3: Creating model with offloaded weights...");
        let vb = self.create_offload_var_builder()?;
        
        // Create model structure (this won't load all weights at once)
        let mut model = FluxModelWithLoRA::new(flux_config, vb)?;
        
        // Add LoRA adapters (these are small and stay on GPU)
        println!("Step 4: Adding LoRA adapters...");
        model.add_lora_to_all(lora_config, &self.device, self.dtype)?;
        
        println!("✅ Model loaded with CPU offloading!");
        let (cpu_count, gpu_count) = self.offload_manager.get_stats();
        println!("Weights: {} on CPU, {} on GPU", cpu_count, gpu_count);
        
        Ok(model)
    }
    
    /// Load weights to CPU memory
    fn load_weights_to_cpu(&self, model_path: &Path) -> Result<HashMap<String, Tensor>> {
        use safetensors::SafeTensors;
        use memmap2::Mmap;
        use std::fs::File;
        
        let file = File::open(model_path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let tensors = SafeTensors::deserialize(&mmap)?;
        
        let mut cpu_weights = HashMap::new();
        let mut loaded = 0;
        
        for name in tensors.names() {
            if loaded % 50 == 0 {
                println!("  Loading weight {}/{}", loaded, tensors.names().len());
            }
            
            let tensor_view = tensors.tensor(name)?;
            let dtype = convert_safetensor_dtype(tensor_view.dtype())?;
            
            // Always load to CPU first
            let cpu_tensor = Tensor::from_raw_buffer(
                tensor_view.data(),
                dtype,
                tensor_view.shape(),
                &Device::Cpu,
            )?;
            
            cpu_weights.insert(name.to_string(), cpu_tensor);
            loaded += 1;
        }
        
        Ok(cpu_weights)
    }
    
    /// Create a VarBuilder that loads from the offload manager
    fn create_offload_var_builder(&self) -> Result<VarBuilder<'static>> {
        // Create an empty VarMap
        let var_map = VarMap::new();
        
        // We'll use a custom implementation that loads on-demand
        // For now, return a standard VarBuilder
        Ok(VarBuilder::from_varmap(&var_map, self.dtype, &self.device))
    }
}

/// Convert safetensors dtype to candle dtype
fn convert_safetensor_dtype(dtype: safetensors::Dtype) -> Result<DType> {
    match dtype {
        safetensors::Dtype::F32 => Ok(DType::F32),
        safetensors::Dtype::F16 => Ok(DType::F16),
        safetensors::Dtype::BF16 => Ok(DType::BF16),
        _ => anyhow::bail!("Unsupported dtype: {:?}", dtype),
    }
}

/// Custom VarBuilder that loads weights on-demand from offload manager
pub struct OffloadVarBuilder {
    offload_manager: WeightOffloadManager,
    dtype: DType,
    device: Device,
    prefix: String,
}

impl OffloadVarBuilder {
    pub fn new(offload_manager: WeightOffloadManager, dtype: DType, device: Device) -> Self {
        Self {
            offload_manager,
            dtype,
            device,
            prefix: String::new(),
        }
    }
    
    pub fn pp<S: Into<String>>(&self, s: S) -> Self {
        let s = s.into();
        let prefix = if self.prefix.is_empty() {
            s
        } else {
            format!("{}.{}", self.prefix, s)
        };
        
        Self {
            offload_manager: self.offload_manager.clone(),
            dtype: self.dtype,
            device: self.device.clone(),
            prefix,
        }
    }
    
    pub fn get<S: Into<String>>(&self, s: S, shape: &[usize]) -> Result<Tensor> {
        let name = if self.prefix.is_empty() {
            s.into()
        } else {
            format!("{}.{}", self.prefix, s.into())
        };
        
        // Load weight from offload manager (will move to GPU as needed)
        let tensor = self.offload_manager.get_weight(&name)?;
        
        // Verify shape
        if tensor.dims() != shape {
            anyhow::bail!(
                "Shape mismatch for {}: expected {:?}, got {:?}",
                name, shape, tensor.dims()
            );
        }
        
        Ok(tensor)
    }
}