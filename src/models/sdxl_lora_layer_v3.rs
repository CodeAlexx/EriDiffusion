//! SDXL LoRA layers with proper gradient tracking
//! This version uses Var for trainable parameters to enable backpropagation

use candle_core::{Device, DType, Tensor, Module, Result, D, Var};
use candle_nn::{VarBuilder, Dropout, Linear, Conv2d, Conv2dConfig};
use serde_json;
use std::collections::HashMap;

/// LoRA Linear layer implementation with gradient tracking
#[derive(Clone)]
pub struct LoRALinearV3 {
    base_layer: Linear,
    lora_a: Var,  // Trainable variable
    lora_b: Var,  // Trainable variable
    scale: f32,
    dropout: Option<Dropout>,
    merged: bool,
    rank: usize,
    in_features: usize,
    out_features: usize,
}

impl LoRALinearV3 {
    pub fn new(
        vb: &VarBuilder,
        in_features: usize,
        out_features: usize,
        rank: usize,
        alpha: f32,
        dropout_rate: f32,
    ) -> Result<Self> {
        // Base layer (frozen during training)
        let base_layer = Linear::new(
            vb.get((out_features, in_features), "weight")?,
            vb.get(out_features, "bias").ok(),
        );
        
        // LoRA A matrix (down projection) - create as Var for gradient tracking
        let init_scale = 0.01;
        let bound = (3.0_f32 / in_features as f32).sqrt() * init_scale;
        let lora_a_tensor = Tensor::randn(0f32, 1f32, (rank, in_features), vb.device())?
            .affine(bound as f64, 0.0)?
            .to_dtype(vb.dtype())?;
        let lora_a = Var::from_tensor(&lora_a_tensor)?;
        
        // LoRA B matrix (up projection) - zero initialized
        let lora_b_tensor = Tensor::zeros((out_features, rank), vb.dtype(), vb.device())?;
        let lora_b = Var::from_tensor(&lora_b_tensor)?;
        
        // Scaling factor
        let scale = alpha / rank as f32;
        
        // Optional dropout
        let dropout = if dropout_rate > 0.0 {
            Some(Dropout::new(dropout_rate))
        } else {
            None
        };
        
        Ok(Self {
            base_layer,
            lora_a,
            lora_b,
            scale,
            dropout,
            merged: false,
            rank,
            in_features,
            out_features,
        })
    }
    
    /// Create a new LoRA layer without a base layer (for new layers)
    pub fn new_without_base(
        vb: &VarBuilder,
        in_features: usize,
        out_features: usize,
        rank: usize,
        alpha: f32,
        dropout_rate: f32,
    ) -> Result<Self> {
        // Create a zero-initialized base layer
        let base_weight = Tensor::zeros((out_features, in_features), vb.dtype(), vb.device())?;
        let base_layer = Linear::new(base_weight, None);
        
        // LoRA matrices as Vars
        let init_scale = 0.01;
        let bound = (3.0_f32 / in_features as f32).sqrt() * init_scale;
        let lora_a_tensor = Tensor::randn(0f32, 1f32, (rank, in_features), vb.device())?
            .affine(bound as f64, 0.0)?
            .to_dtype(vb.dtype())?;
        let lora_a = Var::from_tensor(&lora_a_tensor)?;
        
        let lora_b_tensor = Tensor::zeros((out_features, rank), vb.dtype(), vb.device())?;
        let lora_b = Var::from_tensor(&lora_b_tensor)?;
        
        let scale = alpha / rank as f32;
        
        let dropout = if dropout_rate > 0.0 {
            Some(Dropout::new(dropout_rate))
        } else {
            None
        };
        
        Ok(Self {
            base_layer,
            lora_a,
            lora_b,
            scale,
            dropout,
            merged: false,
            rank,
            in_features,
            out_features,
        })
    }
    
    /// Get trainable parameters (LoRA A and B matrices)
    pub fn get_trainable_params(&self) -> Vec<&Var> {
        vec![&self.lora_a, &self.lora_b]
    }
    
    /// Get both LoRA weights as tensors for saving
    pub fn lora_weights(&self) -> (&Tensor, &Tensor) {
        (self.lora_a.as_tensor(), self.lora_b.as_tensor())
    }
    
    /// Get LoRA B weight
    pub fn lora_b_weight(&self) -> &Tensor {
        self.lora_b.as_tensor()
    }
}

impl Module for LoRALinearV3 {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Base forward pass
        let base_output = self.base_layer.forward(x)?;
        
        if self.merged {
            return Ok(base_output);
        }
        
        // LoRA forward pass
        let lora_input = if let Some(dropout) = &self.dropout {
            dropout.forward(x, true)?
        } else {
            x.clone()
        };
        
        // Get input dimensions for proper reshaping
        let x_shape = lora_input.dims();
        let x_ndims = x_shape.len();
        
        // Handle both 2D and 3D inputs
        let (reshaped_x, original_shape) = if x_ndims == 3 {
            let batch_size = x_shape[0];
            let seq_len = x_shape[1];
            let in_features = x_shape[2];
            // Reshape to 2D for matmul: [batch * seq_len, in_features]
            let reshaped = lora_input.reshape((batch_size * seq_len, in_features))?;
            (reshaped, Some((batch_size, seq_len)))
        } else {
            (lora_input.clone(), None)
        };
        
        // x @ A^T @ B^T * scale
        let lora_a_t = self.lora_a.as_tensor().t()?;
        let h = reshaped_x.matmul(&lora_a_t)?; // [batch * seq_len, rank]
        
        let lora_b_t = self.lora_b.as_tensor().t()?;
        let lora_output = h.matmul(&lora_b_t)?; // [batch * seq_len, out_features]
        
        // Reshape back if needed
        let lora_output = if let Some((batch_size, seq_len)) = original_shape {
            lora_output.reshape((batch_size, seq_len, self.out_features))?
        } else {
            lora_output
        };
        
        // Apply scaling
        let lora_output = lora_output.mul(&Tensor::new(self.scale, lora_output.device())?)?;
        
        // Add base and LoRA outputs
        base_output.add(&lora_output)
    }
}

/// Configuration for LoRA
#[derive(Debug, Clone)]
pub struct LoRAConfig {
    pub rank: usize,
    pub alpha: f32,
    pub dropout: f32,
    pub target_modules: Vec<String>,
    pub modules_to_save: Vec<String>,
}

/// Container for all LoRA layers in a model
pub struct LoRAModelV3 {
    pub lora_layers: HashMap<String, LoRALinearV3>,
    pub config: LoRAConfig,
}

impl LoRAModelV3 {
    pub fn new(config: LoRAConfig) -> Self {
        Self {
            lora_layers: HashMap::new(),
            config,
        }
    }
    
    /// Add a LoRA layer
    pub fn add_layer(&mut self, name: String, layer: LoRALinearV3) {
        self.lora_layers.insert(name, layer);
    }
    
    /// Get all trainable parameters
    pub fn get_trainable_params(&self) -> Vec<&Var> {
        let mut params = Vec::new();
        for (_, layer) in &self.lora_layers {
            params.extend(layer.get_trainable_params());
        }
        params
    }
    
    /// Get all LoRA layers
    pub fn get_all_lora_layers(&self) -> Vec<(String, &LoRALinearV3)> {
        self.lora_layers.iter()
            .map(|(name, layer)| (name.clone(), layer))
            .collect()
    }
    
    /// Save LoRA weights to file
    pub fn save_pretrained(&self, path: &str) -> Result<()> {
        use std::fs;
        use candle_core::safetensors;
        
        // Create directory if it doesn't exist
        fs::create_dir_all(path)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to create directory: {}", e)))?;
        
        // Collect all tensors to save
        let mut tensors = HashMap::new();
        
        // Save all LoRA layer weights
        for (name, layer) in &self.lora_layers {
            let (lora_a, lora_b) = layer.lora_weights();
            tensors.insert(format!("{}.lora_a.weight", name), lora_a.clone());
            tensors.insert(format!("{}.lora_b.weight", name), lora_b.clone());
        }
        
        // Save tensors to safetensors file
        let weights_path = format!("{}/adapter_model.safetensors", path);
        safetensors::save(&tensors, &weights_path)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to save weights: {}", e)))?;
        
        // Save LoRA configuration
        let config_path = format!("{}/adapter_config.json", path);
        let config_json = serde_json::json!({
            "rank": self.config.rank,
            "alpha": self.config.alpha,
            "dropout": self.config.dropout,
            "target_modules": self.config.target_modules,
            "modules_to_save": self.config.modules_to_save,
            "lora_layers": self.lora_layers.keys().collect::<Vec<_>>(),
        });
        
        fs::write(&config_path, serde_json::to_string_pretty(&config_json).unwrap())
            .map_err(|e| candle_core::Error::Msg(format!("Failed to write config: {}", e)))?;
        
        println!("Saved LoRA weights to: {}", weights_path);
        println!("Saved LoRA config to: {}", config_path);
        
        Ok(())
    }
}