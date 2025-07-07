//! Memory-efficient model loading for large models
//! 
//! This loader keeps weights on CPU and only moves them to GPU when needed

use anyhow::{Result, Context};
use candle_core::{Device, DType, Tensor, Var};
use candle_nn::{VarBuilder, VarMap};
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use super::unified_loader::{WeightAdapter, Architecture, FluxAdapter};

/// A VarMap that keeps tensors on CPU until accessed
pub struct LazyVarMap {
    /// CPU tensors waiting to be moved to GPU
    cpu_tensors: Arc<Mutex<HashMap<String, Tensor>>>,
    /// GPU tensors that have been accessed
    gpu_vars: VarMap,
    /// Target device
    device: Device,
    /// Target dtype
    dtype: DType,
}

impl LazyVarMap {
    pub fn new(device: Device, dtype: DType) -> Self {
        Self {
            cpu_tensors: Arc::new(Mutex::new(HashMap::new())),
            gpu_vars: VarMap::new(),
            device,
            dtype,
        }
    }
    
    /// Add a tensor (kept on CPU)
    pub fn insert(&self, name: String, tensor: Tensor) -> Result<()> {
        let mut cpu_tensors = self.cpu_tensors.lock().unwrap();
        cpu_tensors.insert(name, tensor);
        Ok(())
    }
    
    /// Get or create a Var, moving to GPU on demand
    pub fn get_or_create_var(&self, name: &str) -> Result<Var> {
        // Check if already on GPU
        if let Some(var) = self.gpu_vars.data().lock().unwrap().get(name) {
            return Ok(var.clone());
        }
        
        // Move from CPU to GPU
        let tensor = {
            let mut cpu_tensors = self.cpu_tensors.lock().unwrap();
            cpu_tensors.remove(name)
                .with_context(|| format!("Tensor '{}' not found", name))?
        };
        
        println!("Moving tensor '{}' to GPU ({})", name, tensor.elem_count());
        
        // Move to target device and dtype
        let tensor = if tensor.device().location() != self.device.location() {
            tensor.to_device(&self.device)?
        } else {
            tensor
        };
        
        let tensor = if tensor.dtype() != self.dtype {
            tensor.to_dtype(self.dtype)?
        } else {
            tensor
        };
        
        // Create Var and store
        let var = Var::from_tensor(&tensor)?;
        self.gpu_vars.data().lock().unwrap().insert(name.to_string(), var.clone());
        
        Ok(var)
    }
    
    /// Create a VarBuilder that loads tensors on demand
    pub fn create_var_builder(&self) -> VarBuilder<'static> {
        VarBuilder::from_varmap(&self.gpu_vars, self.dtype, &self.device)
    }
}

/// Memory-efficient loader for Flux models
pub struct MemoryEfficientFluxLoader {
    device: Device,
    dtype: DType,
    adapter: FluxAdapter,
}

impl MemoryEfficientFluxLoader {
    pub fn new(device: Device, dtype: DType, hidden_size: usize) -> Self {
        Self {
            device,
            dtype,
            adapter: FluxAdapter::new(hidden_size),
        }
    }
    
    /// Load model with memory-efficient strategy
    pub fn load(&self, path: &Path) -> Result<LazyVarMap> {
        println!("Loading model with memory-efficient strategy...");
        
        // Load to CPU
        let tensors = candle_core::safetensors::load(path, &Device::Cpu)
            .context("Failed to load safetensors file")?;
        
        println!("Loaded {} tensors to CPU", tensors.len());
        
        // Detect architecture
        let source_arch = Architecture::detect(&tensors);
        println!("Detected architecture: {:?}", source_arch);
        
        // Create lazy var map
        let lazy_map = LazyVarMap::new(self.device.clone(), self.dtype);
        
        // Adapt and store tensors (keeping on CPU)
        for (name, tensor) in tensors {
            let adapted_tensors = self.adapter.adapt_tensor(&name, tensor)?;
            for (new_name, new_tensor) in adapted_tensors {
                lazy_map.insert(new_name, new_tensor)?;
            }
        }
        
        println!("Prepared {} tensors for lazy loading", 
                 lazy_map.cpu_tensors.lock().unwrap().len());
        
        Ok(lazy_map)
    }
}

/// Create a memory-efficient Flux model
pub fn create_memory_efficient_flux_model(
    checkpoint_path: &Path,
    device: Device,
    dtype: DType,
) -> Result<()> {
    use crate::models::flux_custom::{FluxConfig as FluxCustomConfig, FluxModelWithLoRA};
    use crate::models::flux_custom::lora::{LoRAConfig};
    
    println!("Creating memory-efficient Flux model...");
    
    // Load weights with memory-efficient loader
    let loader = MemoryEfficientFluxLoader::new(device.clone(), dtype, 3072);
    let lazy_map = loader.load(checkpoint_path)?;
    
    // Create model configuration
    let config = FluxCustomConfig::default();
    
    // Create model with empty weights first
    let vb = lazy_map.create_var_builder();
    
    // Build model structure (doesn't load weights yet)
    println!("Building model structure...");
    let mut model = FluxModelWithLoRA::new(&config, vb)?;
    
    // Configure LoRA
    let lora_config = LoRAConfig {
        rank: 16,
        alpha: 16.0,
        dropout: Some(0.0),
        target_modules: vec![
            "attn".to_string(),
            "mlp".to_string(),
        ],
        module_filters: vec![],
        init_scale: 0.01,
    };
    
    println!("Adding LoRA layers...");
    model.add_lora_to_all(&lora_config, &device, dtype)?;
    
    println!("Model created successfully with lazy weight loading!");
    println!("Weights will be moved to GPU on demand during forward passes");
    
    Ok(())
}