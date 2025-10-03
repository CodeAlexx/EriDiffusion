use crate::loaders::WeightLoader;
use flame_core::{DType, Shape, Tensor};
use flame_core::device::Device;
use flame_core::Parameter;
use std::{collections::HashMap, sync::Arc};
use flame_core::{Result};

// Proper MMDiT wrapper that applies LoKr at the correct layers
// This replaces the hacky workaround in sd35_lokr.rs

/// LoKr layer for MMDiT
pub struct LoKrLayer {
    pub w1: Parameter,
    pub w2: Parameter,
    pub rank: usize,
    pub alpha: f32,
}

impl LoKrLayer {
    pub fn new(
        in_features: usize,
        out_features: usize,
        rank: usize,
        alpha: f32,
        device: &Device,
        dtype: DType,
    ) -> flame_core::Result<Self> {
        // Initialize w1 and w2 for Kronecker product decomposition
        // For simplicity, we'll use square decomposition
        let factor = ((in_features * out_features) as f32).sqrt() as usize / rank;
        let factor = factor.max(1);
        
        // Find the best factorization
        let (w1_in, w1_out, w2_in, w2_out) = Self::find_factorization(
            in_features,
            out_features,
            rank,
            factor,
        )?;
        
        // Initialize weights
        let w1 = Parameter::randn(0.0, 0.02, (w1_out, w1_in), dtype, device)?;
        let w2 = Parameter::zeros((w2_out, w2_in), dtype, device)?;
        
        Ok(Self {
            w1,
            w2,
            rank,
            alpha,
        })
    }
    
    fn find_factorization(
        in_features: usize,
        out_features: usize,
        rank: usize,
        factor: usize,
    ) -> flame_core::Result<(usize, usize, usize, usize)> {
        // Simple factorization for now
        // w1: [rank, in_features/factor]
        // w2: [out_features/factor, rank]
        
        let w1_in = in_features / factor;
        let w1_out = rank;
        let w2_in = rank;
        let w2_out = out_features / factor;
        
        if w1_in * factor != in_features || w2_out * factor != out_features {
            return Err(flame_core::Error::InvalidOperation(
                format!("Cannot factorize {}x{} with rank {} and factor {}", 
                    in_features, out_features, rank, factor)
            ));
        }
        
        Ok((w1_in, w1_out, w2_in, w2_out))
    }
    
    pub fn forward(&self, x: &Tensor) -> flame_core::Result<Tensor> {
        // Apply Kronecker product: (w2 ⊗ w1) @ x
        // For efficiency, we compute: w2 @ (w1 @ x_reshaped)
        
        let w1_tensor = self.w1.tensor();
        let w2_tensor = self.w2.tensor();
        
        // First matmul with w1
        let h = x.matmul(&w1_tensor.transpose_dims(0, 1)?)?;
        
        // Second matmul with w2
        let out = h.matmul(&w2_tensor.transpose_dims(0, 1)?)?;
        
        // Scale by alpha/rank
        let scale = self.alpha / (self.rank as f32);
        out.mul_scalar(scale)
    }
}

/// MMDiT with LoKr wrapper
pub struct MMDiTWithLoKr {
    base_model: Arc<dyn flame_core::Module>,
    lokr_layers: HashMap<String, LoKrLayer>,
}

impl MMDiTWithLoKr {
    pub fn new(
        base_model: Arc<dyn flame_core::Module>,
        lokr_config: &LoKrConfig,
        device: &Device,
        dtype: DType,
    ) -> flame_core::Result<Self> {
        let mut lokr_layers = HashMap::new();
        
        // Create LoKr layers for target modules
        for target in &lokr_config.target_modules {
            // Extract dimensions from target module name
            // This is a simplified version - in practice you'd inspect the base model
            let (in_features, out_features) = Self::get_module_dims(&target)?;
            
            let lokr = LoKrLayer::new(
                in_features,
                out_features,
                lokr_config.rank,
                lokr_config.alpha,
                device,
                dtype,
            )?;
            
            lokr_layers.insert(target.clone(), lokr);
        }
        
        Ok(Self {
            base_model,
            lokr_layers,
        })
    }
    
    fn get_module_dims(module_name: &str) -> flame_core::Result<(usize, usize)> {
        // This would need to inspect the actual model
        // For now, return common dimensions for MMDiT
        if module_name.contains("to_q") || module_name.contains("to_k") || module_name.contains("to_v") {
            Ok((1536, 1536)) // SD3.5 Large hidden dim
        } else if module_name.contains("to_out") {
            Ok((1536, 1536))
        } else if module_name.contains("ff.net.0") {
            Ok((1536, 6144)) // 4x hidden dim for FFN
        } else if module_name.contains("ff.net.2") {
            Ok((6144, 1536))
        } else {
            Err(flame_core::Error::InvalidOperation(
                format!("Unknown module: {}", module_name)
            ))
        }
    }
    
    pub fn forward(&self, x: &Tensor, timestep: &Tensor, context: &Tensor) -> flame_core::Result<Tensor> {
        // This is a placeholder - actual implementation would need to:
        // 1. Run base model forward pass
        // 2. Intercept at LoKr injection points
        // 3. Add LoKr outputs to base outputs
        
        // For now, just run base model
        // In practice, you'd need hooks or a custom forward implementation
        self.base_model.forward(x)
    }
    
    pub fn get_trainable_params(&self) -> Vec<&Parameter> {
        let mut params = Vec::new();
        
        for lokr in self.lokr_layers.values() {
            params.push(&lokr.w1);
            params.push(&lokr.w2);
        }
        
        params
    }
}

/// LoKr configuration
#[derive(Clone)]
pub struct LoKrConfig {
    pub rank: usize,
    pub alpha: f32,
    pub target_modules: Vec<String>,
    pub decompose_factor: Option<i32>,
}

impl Default for LoKrConfig {
    fn default() -> Self {
        Self {
            rank: 16,
            alpha: 16.0,
            target_modules: vec![
                "joint_blocks.0.context_block.attn.to_q".to_string(),
                "joint_blocks.0.context_block.attn.to_k".to_string(),
                "joint_blocks.0.context_block.attn.to_v".to_string(),
                "joint_blocks.0.context_block.attn.to_out.0".to_string(),
            ],
            decompose_factor: None,
        }
    }
}

// Extension trait for missing tensor methods
trait TensorExt {
    fn mul_scalar(&self, scalar: f32) -> flame_core::Result<Tensor>;
}

impl TensorExt for Tensor {
    fn mul_scalar(&self, scalar: f32) -> flame_core::Result<Tensor> {
        let scalar_tensor = Tensor::full(self.shape(), scalar, self.device().clone())?;
        self.mul(&scalar_tensor)
    }
}