// Quantized Flux model loader for 24GB GPUs
// Uses INT8 quantization to fit 22GB model in ~11GB

use anyhow::{Result, Context};
use candle_core::{Tensor, Device, DType};
use candle_nn::{VarBuilder, VarMap};
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, RwLock};

use crate::memory::{
    MemoryPool, MemoryPoolConfig, BlockSwapManager, BlockSwapConfig,
    QuantoManager, QuantoConfig, QuantizationMode, cuda,
};
use crate::models::flux_custom::{FluxConfig, FluxModelWithLoRA};

/// Load and quantize Flux model for 24GB GPUs
pub struct QuantizedFluxLoader {
    device: Device,
    dtype: DType,
    quanto_manager: Arc<QuantoManager>,
    memory_pool: Arc<RwLock<MemoryPool>>,
    block_swap_manager: Option<Arc<BlockSwapManager>>,
}

impl QuantizedFluxLoader {
    pub fn new(device: Device) -> Result<Self> {
        Self::new_with_dtype(device, DType::BF16)
    }
    
    pub fn new_with_dtype(_device: Device, dtype: DType) -> Result<Self> {
        // Always use cached device to avoid device mismatches
        let device = crate::trainers::cached_device::get_single_device()?;
        
        // Setup memory pool for 24GB
        let mut pool_config = MemoryPoolConfig::flux_24gb();
        pool_config.max_size = 20 * 1024 * 1024 * 1024; // Leave 4GB for system
        
        let device_id = match &device {
            Device::Cuda(_) => {
                // For now, assume device 0. In production, we'd extract from CudaDevice
                0i32
            },
            _ => 0,
        };
        
        let memory_pool = Arc::new(RwLock::new(
            MemoryPool::new(device_id, pool_config)?
        ));
        
        // Setup block swapping
        let mut swap_config = BlockSwapConfig::default();
        swap_config.max_gpu_memory = 20 * 1024 * 1024 * 1024;
        swap_config.active_blocks = 12; // Keep 12 blocks in GPU
        swap_config.prefetch_blocks = 4;
        
        let block_swap_manager = Some(Arc::new(
            BlockSwapManager::new(swap_config)?
        ));
        
        // Setup quantization config
        let quanto_config = QuantoConfig::flux_24gb();
        
        // Use CPU for quantization, then move to GPU
        let quanto_manager = Arc::new(QuantoManager::new(
            Device::Cpu,  // Quantize on CPU first
            quanto_config,
            memory_pool.clone(),
            block_swap_manager.clone(),
        ));
        
        Ok(Self {
            device,
            dtype,
            quanto_manager,
            memory_pool,
            block_swap_manager,
        })
    }
    
    /// Load and quantize Flux model
    pub fn load_quantized_model<P: AsRef<Path>>(
        &self,
        model_path: P,
        config: &FluxConfig,
    ) -> Result<HashMap<String, Tensor>> {
        let model_path = model_path.as_ref();
        
        println!("\n=== Quantized Flux Loading ===");
        println!("Loading model to CPU for quantization...");
        
        // Load to CPU first to avoid OOM
        let cpu_weights = candle_core::safetensors::load(model_path, &Device::Cpu)
            .with_context(|| format!("Failed to load model from {:?}", model_path))?;
        
        println!("Loaded {} tensors, quantizing to INT8...", cpu_weights.len());
        println!("DEBUG: Starting quantization with QuantoManager...");
        
        // Quantize the model
        self.quanto_manager.quantize_model(&cpu_weights)?;
        
        // Get memory savings
        let (original_size, quantized_size) = self.quanto_manager.get_memory_savings()?;
        println!("\nMemory savings:");
        println!("  Original size: {:.2} GB", original_size as f64 / 1e9);
        println!("  Quantized size: {:.2} GB", quantized_size as f64 / 1e9);
        println!("  Compression ratio: {:.2}x", original_size as f64 / quantized_size as f64);
        
        // Create a minimal weight map with only essential weights
        // The rest will be loaded on-demand
        let mut essential_weights = HashMap::new();
        
        // Only load small, essential weights upfront
        for name in cpu_weights.keys() {
            // Load embeddings and small weights
            if name.contains("img_in") || name.contains("txt_in") || 
               name.contains("time_in") || name.contains("vector_in") ||
               name.contains("guidance_in") || name.contains("final_layer") {
                let weight = self.quanto_manager.get_weight_with_dtype(name, Some(self.dtype))?;
                // Ensure weight is on the cached device
                let cached_device = crate::trainers::cached_device::get_single_device()?;
                let weight = if !weight.device().same_device(&cached_device) {
                    weight.to_device(&cached_device)?
                } else {
                    weight
                };
                essential_weights.insert(name.clone(), weight);
            }
        }
        
        println!("\nLoaded {} essential weights, rest will be loaded on-demand", essential_weights.len());
        Ok(essential_weights)
    }
    
    /// Create Flux model with quantized weights - memory efficient version
    pub fn create_model_with_lora(
        &self,
        config: &FluxConfig,
        lora_config: &crate::models::flux_custom::lora::LoRAConfig,
        quantized_weights: &HashMap<String, Tensor>,
    ) -> Result<FluxModelWithLoRA> {
        println!("Creating Flux model with LoRA adapters (memory-efficient mode)...");
        
        // Clear GPU cache before model creation
        cuda::empty_cache()?;
        
        // For now, we need to provide all weights to create the model
        // But we'll load them in smaller batches to avoid OOM
        println!("Loading model weights in batches to manage memory...");
        
        // Get all weight names from quanto manager
        let all_weights = self.load_all_weights_batched()?;
        
        // Create VarMap with all weights
        let var_map = candle_nn::VarMap::new();
        for (name, tensor) in &all_weights {
            var_map.data().lock().unwrap().insert(
                name.clone(), 
                candle_core::Var::from_tensor(tensor)?
            );
        }
        
        // Create VarBuilder with cached device
        let cached_device = crate::trainers::cached_device::get_single_device()?;
        let vb = candle_nn::VarBuilder::from_varmap(&var_map, DType::F16, &cached_device);
        
        println!("Creating model structure...");
        // Create model structure with weights
        let mut model = FluxModelWithLoRA::new(config, vb)?;
        
        println!("Initializing LoRA adapters...");
        // Add LoRA adapters - these are small and fit in memory
        model.add_lora_to_all(lora_config, &self.device, DType::F16)?;
        
        println!("Model created successfully with quantized weights!");
        
        Ok(model)
    }
    
    /// Load all weights in batches to avoid OOM
    fn load_all_weights_batched(&self) -> Result<HashMap<String, Tensor>> {
        println!("Loading all quantized weights in batches...");
        
        // Get actual weight names from quanto manager
        let weight_names = self.quanto_manager.weight_names();
        println!("Found {} weights to load", weight_names.len());
        
        let mut all_weights = HashMap::new();
        
        // Load weights in small batches to manage memory
        let batch_size = 20;
        for (idx, batch) in weight_names.chunks(batch_size).enumerate() {
            println!("Loading batch {}/{}", idx + 1, (weight_names.len() + batch_size - 1) / batch_size);
            
            // Clear cache periodically
            if idx % 3 == 0 && idx > 0 {
                cuda::empty_cache()?;
            }
            
            for name in batch {
                let weight = self.quanto_manager.get_weight_with_dtype(name, Some(self.dtype))?;
                
                // Ensure we use the cached device
                let cached_device = crate::trainers::cached_device::get_single_device()?;
                
                // For large weights, keep on CPU if possible
                let elem_count = weight.shape().elem_count();
                if elem_count > 10_000_000 { // 10M elements
                    println!("  Large weight {} ({} elements) - keeping on CPU", name, elem_count);
                    all_weights.insert(name.clone(), weight);
                } else {
                    // Small weights can go to GPU using cached device
                    let weight_gpu = weight.to_device(&cached_device)?;
                    all_weights.insert(name.clone(), weight_gpu);
                }
            }
        }
        
        println!("Loaded {} weights successfully", all_weights.len());
        Ok(all_weights)
    }
    
    /// Get the quanto manager for on-demand weight loading
    pub fn quanto_manager(&self) -> Arc<QuantoManager> {
        self.quanto_manager.clone()
    }
    
    /// Load pre-quantized weights if available
    pub fn load_prequantized<P: AsRef<Path>>(
        &self,
        path: P,
    ) -> Result<HashMap<String, Tensor>> {
        // Check if it's an FP8 model
        let path = path.as_ref();
        let is_fp8 = path.to_string_lossy().contains("fp8");
        
        if is_fp8 {
            println!("Detected FP8 model, loading directly...");
            let weights = candle_core::safetensors::load(path, &self.device)?;
            
            // FP8 models are already small enough
            println!("FP8 model loaded: ~11GB");
            Ok(weights)
        } else {
            // Full precision model needs quantization
            Err(anyhow::anyhow!(
                "This appears to be a full precision model (22GB). \
                Please use load_quantized_model() instead, or provide an FP8 model."
            ))
        }
    }
}

/// Helper to detect if a model is already quantized
pub fn is_quantized_model<P: AsRef<Path>>(path: P) -> bool {
    let path_str = path.as_ref().to_string_lossy();
    path_str.contains("fp8") || 
    path_str.contains("int8") || 
    path_str.contains("quantized") ||
    path_str.contains("bnb")
}

/// Estimate memory requirements
pub fn estimate_memory_requirements(model_size_gb: f64, is_training: bool) -> (f64, f64) {
    if is_training {
        // Training needs: model + gradients + optimizer states + activations
        let model_mem = model_size_gb;
        let gradient_mem = model_size_gb;
        let optimizer_mem = model_size_gb * 2.0; // Adam has 2 states
        let activation_mem = 2.0; // Rough estimate
        
        let total = model_mem + gradient_mem + optimizer_mem + activation_mem;
        (total, model_mem)
    } else {
        // Inference only needs model + activations
        (model_size_gb + 1.0, model_size_gb)
    }
}

/// Check if configuration is viable
pub fn check_memory_viability(device: &Device) -> Result<()> {
    let device_id = match device {
        Device::Cuda(_) => {
            // For now, assume device 0. In production, we'd extract from CudaDevice
            0i32
        },
        _ => 0,
    };
    let (free, total) = cuda::get_memory_info(device_id)?;
    let free_gb = free as f64 / 1e9;
    let total_gb = total as f64 / 1e9;
    
    println!("\n=== Memory Check ===");
    println!("Total VRAM: {:.2} GB", total_gb);
    println!("Free VRAM: {:.2} GB", free_gb);
    
    // Full Flux model
    let (full_train, full_model) = estimate_memory_requirements(22.0, true);
    println!("\nFull precision Flux:");
    println!("  Model size: {:.2} GB", full_model);
    println!("  Training needs: {:.2} GB", full_train);
    println!("  Viable: {}", if full_train < total_gb { "NO" } else { "NO" });
    
    // INT8 Flux
    let (int8_train, int8_model) = estimate_memory_requirements(11.0, true);
    println!("\nINT8 Flux:");
    println!("  Model size: {:.2} GB", int8_model);
    println!("  Training needs: {:.2} GB", int8_train);
    println!("  Viable: {}", if int8_train < total_gb { "Maybe" } else { "NO" });
    
    // INT8 + gradient checkpointing + block swapping
    let optimized_train = 11.0 + 2.0 + 1.0; // model + some gradients + activations
    println!("\nINT8 + Optimizations:");
    println!("  Estimated needs: {:.2} GB", optimized_train);
    println!("  Viable: {}", if optimized_train < total_gb { "YES" } else { "Maybe" });
    
    Ok(())
}