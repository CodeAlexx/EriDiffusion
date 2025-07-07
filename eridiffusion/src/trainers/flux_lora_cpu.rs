//! CPU-offloaded Flux LoRA training for 24GB GPUs
//! 
//! This implementation keeps the base model on CPU and only moves
//! activations to GPU as needed during forward passes.

use anyhow::{Result, Context};
use candle_core::{Tensor, DType, Device, Module, Var};
use candle_nn::{VarMap, Linear, linear};
use std::collections::HashMap;

/// Wrapper for CPU-offloaded Flux model
pub struct CPUOffloadedFlux {
    /// Base model weights (kept on CPU)
    base_weights: HashMap<String, Tensor>,
    /// LoRA weights (on GPU)
    lora_weights: VarMap,
    /// Model configuration
    config: FluxConfig,
    /// GPU device for computations
    gpu_device: Device,
    /// CPU device for storage
    cpu_device: Device,
}

#[derive(Clone)]
pub struct FluxConfig {
    pub num_layers: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub head_dim: usize,
}

impl CPUOffloadedFlux {
    pub fn new(
        base_weights_path: &std::path::Path,
        config: FluxConfig,
        gpu_device: Device,
    ) -> Result<Self> {
        println!("Loading Flux model with CPU offloading...");
        
        // Load base weights to CPU
        let base_weights = candle_core::safetensors::load(base_weights_path, &Device::Cpu)?;
        println!("Loaded {} tensors to CPU", base_weights.len());
        
        // Create LoRA weights on GPU
        let lora_weights = VarMap::new();
        
        Ok(Self {
            base_weights,
            lora_weights,
            config,
            gpu_device,
            cpu_device: Device::Cpu,
        })
    }
    
    /// Get a weight tensor, moving to GPU only when needed
    fn get_weight(&self, name: &str) -> Result<Tensor> {
        if let Some(tensor) = self.base_weights.get(name) {
            // Move to GPU on demand - create a copy on GPU
            let gpu_tensor = match tensor.device().location() {
                candle_core::DeviceLocation::Cpu => {
                    // CPU to GPU copy
                    tensor.to_dtype(tensor.dtype())?.to_device(&self.gpu_device)?
                }
                _ => tensor.clone(),  // Already on GPU
            };
            Ok(gpu_tensor)
        } else {
            Err(anyhow::anyhow!("Weight {} not found", name))
        }
    }
    
    /// Compute attention with CPU offloading and LoRA
    fn attention_layer(
        &self,
        x: &Tensor,
        layer_idx: usize,
        block_type: &str,
        alpha: f32,
    ) -> Result<Tensor> {
        let prefix = format!("{}.{}", block_type, layer_idx);
        
        // Debug: print what we're looking for
        println!("DEBUG: Looking for weights with prefix: {}", prefix);
        
        // Flux uses combined qkv tensor, not separate q, k, v
        // Try combined first, then fall back to separate
        let (q, k, v) = if let Ok(qkv_weight) = self.get_weight(&format!("{}.attn.qkv.weight", prefix)) {
            // Combined QKV tensor - split it
            let qkv = x.matmul(&qkv_weight.t()?)?;
            let hidden_size = self.config.hidden_size;
            
            // Split into Q, K, V
            let q = qkv.narrow(1, 0, hidden_size)?;
            let k = qkv.narrow(1, hidden_size, hidden_size)?;
            let v = qkv.narrow(1, hidden_size * 2, hidden_size)?;
            
            (q, k, v)
        } else {
            // Try separate tensors (fallback)
            let q_weight = self.get_weight(&format!("{}.attn.to_q.weight", prefix))?;
            let k_weight = self.get_weight(&format!("{}.attn.to_k.weight", prefix))?;
            let v_weight = self.get_weight(&format!("{}.attn.to_v.weight", prefix))?;
            
            // Compute Q, K, V with LoRA
            let q = self.apply_lora(x, &q_weight, &format!("{}.attn.to_q", prefix), alpha)?;
            let k = self.apply_lora(x, &k_weight, &format!("{}.attn.to_k", prefix), alpha)?;
            let v = self.apply_lora(x, &v_weight, &format!("{}.attn.to_v", prefix), alpha)?;
            
            (q, k, v)
        };
        
        // Clear GPU cache
        if let Device::Cuda(_) = &self.gpu_device {
            crate::memory::cuda::empty_cache()
                .map_err(|e| anyhow::anyhow!("Failed to empty cache: {:?}", e))?;
        }
        
        // Compute attention (simplified)
        let scores = q.matmul(&k.t()?)?;
        let attention = candle_nn::ops::softmax(&scores, 1)?;
        let out = attention.matmul(&v)?;
        
        // Output projection - try different naming conventions
        let o_weight = self.get_weight(&format!("{}.attn.to_out.0.weight", prefix))
            .or_else(|_| self.get_weight(&format!("{}.attn.norm.query.weight", prefix)))
            .or_else(|_| self.get_weight(&format!("{}.attn.proj.weight", prefix)))?;
            
        let result = self.apply_lora(&out, &o_weight, &format!("{}.attn.to_out.0", prefix), alpha)?;
        
        Ok(result)
    }
    
    /// Simplified layer processing for demonstration
    fn attention_layer_flux(
        &self,
        x: &Tensor,
        prefix: &str,  // e.g., "double_blocks.0.img_attn"
        _alpha: f32,
    ) -> Result<Tensor> {
        // For demonstration purposes, just load a weight and return input
        // This shows CPU->GPU transfer is working
        
        // Try to load any weight from this block to demonstrate transfer
        let weight_name = format!("{}.qkv.weight", prefix);
        if let Ok(weight) = self.get_weight(&weight_name) {
            println!("Successfully loaded {} to GPU (shape: {:?})", weight_name, weight.shape());
            // Just return the input unchanged for now
            Ok(x.clone())
        } else {
            // If that weight doesn't exist, try another
            let weight_name = format!("{}.proj.weight", prefix);
            if let Ok(weight) = self.get_weight(&weight_name) {
                println!("Successfully loaded {} to GPU (shape: {:?})", weight_name, weight.shape());
            }
            Ok(x.clone())
        }
    }
    
    /// Simple MLP layer (placeholder for now)
    fn mlp_layer(&self, x: &Tensor, prefix: &str, _alpha: f32) -> Result<Tensor> {
        // For now, just return the input
        // In a full implementation, we'd process the MLP layers
        Ok(x.clone())
    }
    
    /// Forward pass with CPU offloading - simplified for demonstration
    pub fn forward_offloaded(
        &self,
        img: &Tensor,
        _txt: &Tensor,
        _timestep: &Tensor,
        lora_alpha: f32,
    ) -> Result<Tensor> {
        println!("Running forward pass with CPU offloading...");
        println!("Input shape: {:?}", img.shape());
        
        let mut output = img.clone();
        
        // Process a few blocks to demonstrate CPU->GPU transfer
        for i in 0..5 {  // Just do 5 blocks for demo
            // Load some weights from CPU to GPU
            let img_prefix = format!("double_blocks.{}.img_attn", i);
            output = self.attention_layer_flux(&output, &img_prefix, lora_alpha)?;
            
            // Show memory usage
            if let Device::Cuda(_device_id) = &self.gpu_device {
                // TODO: Get device ordinal properly
                if let Ok(allocated) = crate::memory::cuda::memory_allocated(Some(0)) {
                    println!("GPU memory after block {}: {:.2} MB", i, allocated as f64 / 1024.0 / 1024.0);
                }
            }
            
            // Clear GPU memory
            if let Device::Cuda(_) = &self.gpu_device {
                crate::memory::cuda::empty_cache()
                    .map_err(|e| anyhow::anyhow!("Failed to empty cache: {:?}", e))?;
            }
            println!("Processed double block {}/5", i + 1);
        }
        
        println!("CPU offloading demonstration complete!");
        println!("In a real implementation, all 57 blocks would be processed.");
        
        // Return the output unchanged (for demo)
        Ok(output)
    }
    
    /// Create from already loaded weights
    pub fn new_with_weights(
        base_weights: HashMap<String, Tensor>,
        config: FluxConfig,
        gpu_device: Device,
    ) -> Result<Self> {
        println!("Creating CPU-offloaded Flux model with pre-loaded weights...");
        println!("Total tensors: {}", base_weights.len());
        
        // Debug: print first few tensor names
        println!("DEBUG: First 10 tensor names:");
        for (i, name) in base_weights.keys().take(10).enumerate() {
            println!("  {}: {}", i, name);
        }
        
        // Print double_blocks.0 tensors
        println!("DEBUG: double_blocks.0 tensors:");
        for name in base_weights.keys() {
            if name.starts_with("double_blocks.0.") {
                println!("  {}", name);
            }
        }
        
        // Calculate total memory
        let total_params: usize = base_weights.values()
            .map(|t| t.elem_count())
            .sum();
        println!("Total parameters: {:.2}B", total_params as f64 / 1e9);
        
        let lora_weights = VarMap::new();
        
        Ok(Self {
            base_weights,
            lora_weights,
            config,
            gpu_device,
            cpu_device: Device::Cpu,
        })
    }
    
    /// Initialize LoRA weights on GPU
    pub fn initialize_lora(&mut self, rank: usize, alpha: f32) -> Result<()> {
        println!("Initializing LoRA weights with rank={}, alpha={}", rank, alpha);
        
        let vb = candle_nn::VarBuilder::from_varmap(&self.lora_weights, DType::F32, &self.gpu_device);
        
        // Create LoRA weights for key layers
        // For Flux, we target the attention layers in double and single blocks
        let mut lora_count = 0;
        
        // Double blocks (0-18)
        for i in 0..19 {
            // Image attention QKV and projection
            let name = format!("double_blocks.{}.img_attn.qkv", i);
            self.create_lora_pair(&vb, &name, self.config.hidden_size * 3, rank)?;
            lora_count += 2;
            
            let name = format!("double_blocks.{}.img_attn.proj", i);
            self.create_lora_pair(&vb, &name, self.config.hidden_size, rank)?;
            lora_count += 2;
            
            // Text attention QKV and projection
            let name = format!("double_blocks.{}.txt_attn.qkv", i);
            self.create_lora_pair(&vb, &name, self.config.hidden_size * 3, rank)?;
            lora_count += 2;
            
            let name = format!("double_blocks.{}.txt_attn.proj", i);
            self.create_lora_pair(&vb, &name, self.config.hidden_size, rank)?;
            lora_count += 2;
        }
        
        // Single blocks (0-37)
        for i in 0..38 {
            // QKV projection (combined)
            let name = format!("single_blocks.{}.attn.qkv", i);
            self.create_lora_pair(&vb, &name, self.config.hidden_size * 3, rank)?;
            lora_count += 2;
            
            // Output projection
            let name = format!("single_blocks.{}.attn.proj", i);
            self.create_lora_pair(&vb, &name, self.config.hidden_size, rank)?;
            lora_count += 2;
        }
        
        println!("Created {} LoRA weight pairs", lora_count / 2);
        Ok(())
    }
    
    /// Create a LoRA weight pair (down and up projections)
    fn create_lora_pair(
        &self,
        vb: &candle_nn::VarBuilder,
        base_name: &str,
        hidden_size: usize,
        rank: usize,
    ) -> Result<()> {
        // LoRA down projection: hidden_size -> rank
        let _down = vb.get_with_hints(
            (rank, hidden_size),
            &format!("{}.lora_down", base_name),
            candle_nn::Init::Randn { mean: 0.0, stdev: 0.01 },
        )?;
        
        // LoRA up projection: rank -> hidden_size (initialized to zero)
        let _up = vb.get_with_hints(
            (hidden_size, rank),
            &format!("{}.lora_up", base_name),
            candle_nn::Init::Const(0.0),
        )?;
        
        Ok(())
    }
    
    /// Get trainable LoRA parameters
    pub fn get_lora_params(&self) -> Vec<Var> {
        self.lora_weights.all_vars()
    }
    
    /// Apply LoRA to a linear layer
    fn apply_lora(&self, x: &Tensor, base_weight: &Tensor, lora_name: &str, alpha: f32) -> Result<Tensor> {
        // Base computation
        let base_out = x.matmul(&base_weight.t()?)?;
        
        // Check if we have LoRA weights for this layer
        let down_name = format!("{}.lora_down", lora_name);
        let up_name = format!("{}.lora_up", lora_name);
        
        if let Some(vars) = self.lora_weights.data().lock().ok() {
            if let (Some(down_var), Some(up_var)) = (vars.get(&down_name), vars.get(&up_name)) {
                // Apply LoRA: x @ W + alpha * x @ W_down @ W_up
                let lora_out = x.matmul(&down_var.t()?)?
                    .matmul(&up_var.t()?)?;
                
                let scaled = (lora_out * (alpha as f64))?;
                Ok((base_out + scaled)?)
            } else {
                Ok(base_out)
            }
        } else {
            Ok(base_out)
        }
    }
}

/// Create a CPU-offloaded Flux model for 24GB GPUs
pub fn create_cpu_offloaded_flux(
    model_path: &std::path::Path,
    device: Device,
) -> Result<CPUOffloadedFlux> {
    let config = FluxConfig {
        num_layers: 57,  // 19 double + 38 single
        hidden_size: 3072,
        num_heads: 24,
        head_dim: 128,
    };
    
    CPUOffloadedFlux::new(model_path, config, device)
}