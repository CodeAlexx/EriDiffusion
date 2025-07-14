//! LoRA (Low-Rank Adaptation) implementation for Flux

use candle_core::{Device, DType, Module, Result, Tensor, D, Var};
use candle_nn::{linear, Linear, VarBuilder};
use std::collections::HashMap;
use std::path::Path;
use serde::{Serialize, Deserialize};

// Import GPU LoRA operations
#[cfg(feature = "cuda-backward")]
use candle_core::lora_backward_ops::LoRABackwardOps;

/// Trait for models that support LoRA injection
pub trait LoRACompatible {
    fn get_lora_targets(&self) -> Vec<LoRATarget>;
    fn apply_lora(&mut self, config: &LoRAConfig) -> Result<()>;
    fn get_trainable_params(&self) -> Vec<Var>;
    fn save_lora_weights(&self, path: &Path) -> Result<()>;
    fn load_lora_weights(&mut self, path: &Path) -> Result<()>;
}

/// Describes where LoRA can be applied
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoRATarget {
    pub name: String,
    pub module_type: ModuleType,
    pub in_features: usize,
    pub out_features: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ModuleType {
    Attention,
    MLP,
    CrossAttention,
    Projection,
}

/// Generic LoRA configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoRAConfig {
    pub rank: usize,
    pub alpha: f32,
    pub dropout: Option<f32>,
    pub target_modules: Vec<String>,
    pub module_filters: Vec<ModuleType>,
    pub init_scale: f32,
}

impl Default for LoRAConfig {
    fn default() -> Self {
        Self {
            rank: 16,
            alpha: 16.0,
            dropout: None,
            target_modules: vec![],
            module_filters: vec![ModuleType::Attention, ModuleType::MLP],
            init_scale: 0.01,
        }
    }
}

/// Linear layer with optional LoRA
#[derive(Debug)]
pub struct LinearWithLoRA {
    base: Linear,
    lora: Option<LoRAModule>,
    name: String,
    trainable: bool,
}

impl LinearWithLoRA {
    pub fn new(
        in_features: usize,
        out_features: usize,
        name: String,
        vb: VarBuilder,
    ) -> Result<Self> {
        let base = linear(in_features, out_features, vb)?;
        Ok(Self { 
            base, 
            lora: None, 
            name,
            trainable: false,
        })
    }
    
    pub fn add_lora(
        &mut self,
        rank: usize,
        alpha: f32,
        init_scale: f32,
        device: &Device,
        dtype: DType,
    ) -> Result<()> {
        let (in_features, out_features) = {
            let weight_shape = self.base.weight().dims2()?;
            (weight_shape.1, weight_shape.0)
        };
        
        self.lora = Some(LoRAModule::new(
            in_features,
            out_features,
            rank,
            alpha,
            init_scale,
            device,
            dtype,
        )?);
        self.trainable = true;
        
        Ok(())
    }
    
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let base_out = self.base.forward(x)?;
        
        if self.trainable && self.lora.is_some() {
            let lora_out = self.lora.as_ref().unwrap().forward(x)?;
            base_out.add(&lora_out)
        } else {
            Ok(base_out)
        }
    }
    
    pub fn get_trainable_params(&self) -> Vec<&Var> {
        self.lora.as_ref()
            .map(|l| l.get_trainable_params())
            .unwrap_or_default()
    }
    
    pub fn save_weights(&self, tensors: &mut HashMap<String, Tensor>) -> Result<()> {
        if let Some(lora) = &self.lora {
            tensors.insert(
                format!("{}.lora_a", self.name),
                lora.lora_a.as_tensor().clone(),
            );
            tensors.insert(
                format!("{}.lora_b", self.name),
                lora.lora_b.as_tensor().clone(),
            );
        }
        Ok(())
    }
    
    pub fn load_weights(&mut self, tensors: &HashMap<String, Tensor>) -> Result<()> {
        if let Some(lora) = &mut self.lora {
            if let Some(lora_a) = tensors.get(&format!("{}.lora_a", self.name)) {
                // Update the Var with new tensor values
                let _ = lora.lora_a.set(lora_a);
            }
            if let Some(lora_b) = tensors.get(&format!("{}.lora_b", self.name)) {
                let _ = lora.lora_b.set(lora_b);
            }
        }
        Ok(())
    }
    
    /// GPU-accelerated backward pass for LinearWithLoRA
    pub fn backward_gpu(&self, grad_output: &Tensor, input: &Tensor) -> Result<Option<(Tensor, Tensor)>> {
        if let Some(lora) = &self.lora {
            Ok(Some(lora.backward_gpu(grad_output, input)?))
        } else {
            Ok(None)
        }
    }
}

/// LoRA module implementation
#[derive(Debug)]
pub struct LoRAModule {
    pub lora_a: Var,
    pub lora_b: Var,
    scale: f32,
    rank: usize,
    alpha: f32,
}

impl LoRAModule {
    pub fn new(
        in_features: usize,
        out_features: usize,
        rank: usize,
        alpha: f32,
        init_scale: f32,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let scale = alpha / (rank as f32);
        
        // Kaiming uniform initialization for lora_a
        let bound = (3.0_f32 / in_features as f32).sqrt() * init_scale;
        // Use randn and scale instead of rand_uniform
        let lora_a_tensor = Tensor::randn(0f32, 1f32, (rank, in_features), device)?
            .affine(bound as f64, 0.0)?
            .to_dtype(dtype)?;
        let lora_a = Var::from_tensor(&lora_a_tensor)?;
        
        // Zero initialization for lora_b
        let lora_b = Var::from_tensor(&Tensor::zeros(
            (out_features, rank), 
            dtype, 
            device
        )?)?;
        
        Ok(Self {
            lora_a,
            lora_b,
            scale,
            rank,
            alpha,
        })
    }
    
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [batch, seq_len, in_features]
        // lora_a: [rank, in_features] -> after transpose: [in_features, rank]
        // lora_b: [out_features, rank] -> after transpose: [rank, out_features]
        
        // Get input dimensions
        let x_shape = x.dims();
        let x_ndims = x_shape.len();
        
        // Handle both 2D and 3D inputs
        let (reshaped_x, original_shape) = if x_ndims == 3 {
            let batch_size = x_shape[0];
            let seq_len = x_shape[1];
            let in_features = x_shape[2];
            // Reshape to 2D for matmul: [batch * seq_len, in_features]
            let reshaped = x.reshape((batch_size * seq_len, in_features))?;
            (reshaped, Some((batch_size, seq_len)))
        } else {
            (x.clone(), None)
        };
        
        // First projection: x @ lora_a.T
        let lora_a_t = self.lora_a.as_tensor().t()?;
        let h = reshaped_x.matmul(&lora_a_t)?; // [batch * seq_len, rank]
        
        // Second projection: h @ lora_b.T  
        let lora_b_t = self.lora_b.as_tensor().t()?;
        let out = h.matmul(&lora_b_t)?; // [batch * seq_len, out_features]
        
        // Apply scaling
        let scaled = out.affine(self.scale as f64, 0.0)?;
        
        // Reshape back to original dimensions if needed
        if let Some((batch_size, seq_len)) = original_shape {
            let out_features = self.lora_b.as_tensor().dim(0)?;
            scaled.reshape((batch_size, seq_len, out_features))
        } else {
            Ok(scaled)
        }
    }
    
    pub fn get_trainable_params(&self) -> Vec<&Var> {
        vec![&self.lora_a, &self.lora_b]
    }
    
    /// GPU-accelerated backward pass for LoRA
    #[cfg(feature = "cuda-backward")]
    pub fn backward_gpu(&self, grad_output: &Tensor, input: &Tensor) -> Result<(Tensor, Tensor)> {
        // Use GPU-optimized LoRA backward kernel
        LoRABackwardOps::backward(grad_output, input, self.lora_a.as_tensor(), self.lora_b.as_tensor(), self.scale)
    }
    
    /// CPU backward pass for LoRA
    #[cfg(not(feature = "cuda-backward"))]
    pub fn backward_gpu(&self, grad_output: &Tensor, input: &Tensor) -> Result<(Tensor, Tensor)> {
        // Standard backward computation
        // grad_a = grad_output @ lora_b.T @ input.T * scale
        // grad_b = grad_output.T @ input @ lora_a.T * scale
        
        // For 3D tensors, we need to handle batch dimension
        let input_shape = input.dims();
        let grad_shape = grad_output.dims();
        
        // Reshape to 2D if needed
        let (input_2d, grad_2d) = if input_shape.len() == 3 {
            let batch_seq = input_shape[0] * input_shape[1];
            let in_features = input_shape[2];
            let out_features = grad_shape[2];
            (
                input.reshape((batch_seq, in_features))?,
                grad_output.reshape((batch_seq, out_features))?
            )
        } else {
            (input.clone(), grad_output.clone())
        };
        
        // Compute gradients
        // grad_a = (grad_2d.T @ input_2d).T * scale
        let grad_a_temp = grad_2d.matmul(&self.lora_b.as_tensor())?;
        let grad_a = grad_a_temp.t()?.matmul(&input_2d)?.t()?.affine(self.scale as f64, 0.0)?;
        
        // grad_b = grad_2d.T @ (input_2d @ lora_a.T) * scale  
        let input_lora_a = input_2d.matmul(&self.lora_a.as_tensor().t()?)?;
        let grad_b = grad_2d.t()?.matmul(&input_lora_a)?.affine(self.scale as f64, 0.0)?;
        
        Ok((grad_a, grad_b))
    }
}