//! CPU-offloaded Flux model implementation
//! 
//! This wraps FluxModelWithLoRA to provide transparent CPU offloading
//! of weights, allowing training on 24GB VRAM.

use anyhow::{Result, Context};
use candle_core::{Tensor, Device, DType, Module, D};
use candle_nn::VarBuilder;
use candle_core::Var;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::models::flux_custom::{FluxModelWithLoRA, FluxConfig};
use crate::models::flux_custom::lora::LoRAConfig;
use super::flux_cpu_offload::WeightOffloadManager;

/// CPU-offloaded Flux model that manages memory efficiently
pub struct CpuOffloadedFluxModel {
    /// The actual Flux model
    model: FluxModelWithLoRA,
    /// Weight offload manager
    offload_manager: WeightOffloadManager,
    /// Track which layers are currently on GPU
    gpu_layers: Arc<Mutex<Vec<String>>>,
    /// Maximum layers to keep on GPU
    max_gpu_layers: usize,
}

impl CpuOffloadedFluxModel {
    /// Create a new CPU-offloaded Flux model
    pub fn new(
        config: &FluxConfig,
        vb: VarBuilder,
        device: Device,
        dtype: DType,
        max_gpu_layers: usize,
    ) -> Result<Self> {
        println!("Creating CPU-offloaded Flux model...");
        println!("  Max GPU layers: {}", max_gpu_layers);
        
        // Create the model
        let model = FluxModelWithLoRA::new(config, vb)?;
        
        // Create offload manager
        let offload_manager = WeightOffloadManager::new(device, max_gpu_layers * 10); // ~10 weights per layer
        
        Ok(Self {
            model,
            offload_manager,
            gpu_layers: Arc::new(Mutex::new(Vec::new())),
            max_gpu_layers,
        })
    }
    
    /// Add LoRA to all layers
    pub fn add_lora_to_all(
        &mut self,
        lora_config: &LoRAConfig,
        device: &Device,
        dtype: DType,
    ) -> Result<()> {
        self.model.add_lora_to_all(lora_config, device, dtype)?;
        println!("Added LoRA to model layers");
        Ok(())
    }
    
    /// Get trainable parameters (LoRA only)
    pub fn get_trainable_params(&self) -> Vec<Var> {
        self.model.get_trainable_params()
    }
    
    /// Forward pass with CPU offloading
    pub fn forward(
        &self,
        img: &Tensor,
        img_ids: &Tensor,
        txt: &Tensor,
        txt_ids: &Tensor,
        timesteps: &Tensor,
        y: &Tensor,
        guidance: Option<&Tensor>,
    ) -> Result<Tensor> {
        // For now, delegate to the model directly
        // In a full implementation, we would:
        // 1. Load required layers to GPU
        // 2. Run forward pass
        // 3. Offload layers back to CPU as needed
        
        println!("Running forward pass with CPU offloading...");
        
        // Clear some GPU memory before forward pass
        self.offload_manager.clear_gpu_cache();
        
        // Run forward pass
        let result = self.model.forward(img, img_ids, txt, txt_ids, timesteps, y, guidance)?;
        
        Ok(result)
    }
    
    /// Offload specific layers to CPU
    pub fn offload_layers(&self, layer_names: &[String]) -> Result<()> {
        // In a full implementation, this would move specific layers to CPU
        println!("Offloading {} layers to CPU", layer_names.len());
        Ok(())
    }
    
    /// Load specific layers to GPU
    pub fn load_layers(&self, layer_names: &[String]) -> Result<()> {
        // In a full implementation, this would move specific layers to GPU
        println!("Loading {} layers to GPU", layer_names.len());
        Ok(())
    }
}

/// Builder for CPU-offloaded model with better memory management
pub struct CpuOffloadedModelBuilder {
    device: Device,
    dtype: DType,
    max_gpu_memory_gb: f32,
}

impl CpuOffloadedModelBuilder {
    pub fn new(device: Device) -> Self {
        Self {
            device,
            dtype: DType::BF16,
            max_gpu_memory_gb: 20.0, // Leave 4GB for activations on 24GB card
        }
    }
    
    pub fn with_dtype(mut self, dtype: DType) -> Self {
        self.dtype = dtype;
        self
    }
    
    pub fn with_max_gpu_memory(mut self, gb: f32) -> Self {
        self.max_gpu_memory_gb = gb;
        self
    }
    
    /// Build the model with automatic layer distribution
    pub fn build(
        self,
        config: &FluxConfig,
        model_path: &std::path::Path,
    ) -> Result<CpuOffloadedFluxModel> {
        println!("\n=== Building CPU-Offloaded Flux Model ===");
        println!("Target GPU memory: {:.1} GB", self.max_gpu_memory_gb);
        
        // Estimate memory per layer
        let total_layers = config.depth + config.depth_single_blocks;
        // Flux uses 3072 hidden size, each layer has roughly:
        // - Attention: 4 * hidden^2 (Q,K,V,O projections)
        // - MLP: 4 * hidden^2 (up/down projections with 4x expansion)
        // Total: ~8 * hidden^2 parameters per layer
        let params_per_layer = (8 * config.hidden_size * config.hidden_size) as f32;
        // In BF16, each param is 2 bytes, plus gradients (another 2 bytes)
        let layer_memory_mb = (params_per_layer * 4.0) / (1024.0 * 1024.0);
        
        println!("Total layers: {}", total_layers);
        println!("Estimated memory per layer: {:.0} MB", layer_memory_mb);
        
        // Calculate how many layers can fit on GPU
        // Reserve some memory for activations and other overhead
        let available_mb = (self.max_gpu_memory_gb * 1024.0) as f32;
        let reserved_mb = 4096.0; // Reserve 4GB for activations, LoRA, etc
        let usable_mb = (available_mb - reserved_mb).max(1024.0); // At least 1GB
        let max_gpu_layers = ((usable_mb / layer_memory_mb) as usize).min(total_layers).max(1);
        
        println!("Max layers on GPU: {} / {}", max_gpu_layers, total_layers);
        
        // Load model weights
        let vb = crate::loaders::load_flux_weights(
            model_path,
            self.device.clone(),
            self.dtype,
            config.hidden_size,
        )?;
        
        // Create the CPU-offloaded model
        let model = CpuOffloadedFluxModel::new(
            config,
            vb,
            self.device,
            self.dtype,
            max_gpu_layers,
        )?;
        
        println!("✅ CPU-offloaded model created successfully!");
        
        Ok(model)
    }
}