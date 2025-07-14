use candle_core::{Device, DType, Tensor, Module, Result, D};
use candle_nn::{VarBuilder, Dropout, Linear, Conv2d, Conv2dConfig};
use serde_json;

/// LoRA Linear layer implementation
#[derive(Clone)]
pub struct LoRALinear {
    base_layer: Linear,
    lora_a: Linear,
    lora_b: Linear,
    scale: f32,
    dropout: Option<Dropout>,
    merged: bool,
}

impl LoRALinear {
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
        
        // LoRA A matrix (down projection)
        let lora_a = Linear::new(
            vb.pp("lora_a").get_with_hints((rank, in_features), "weight", candle_nn::init::DEFAULT_KAIMING_NORMAL)?,
            None,
        );
        
        // LoRA B matrix (up projection)
        let lora_b = Linear::new(
            vb.pp("lora_b").get_with_hints((out_features, rank), "weight", candle_nn::init::ZERO)?,
            None,
        );
        
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
        
        // LoRA matrices
        let lora_a = Linear::new(
            vb.pp("lora_a").get_with_hints((rank, in_features), "weight", candle_nn::init::DEFAULT_KAIMING_NORMAL)?,
            None,
        );
        
        let lora_b = Linear::new(
            vb.pp("lora_b").get_with_hints((out_features, rank), "weight", candle_nn::init::ZERO)?,
            None,
        );
        
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
        })
    }
    
    /// Merge LoRA weights into base layer
    pub fn merge(&mut self) -> Result<()> {
        if self.merged {
            return Ok(());
        }
        
        // Get LoRA weight: B @ A * scale
        let lora_weight = self.lora_b.weight()
            .matmul(&self.lora_a.weight())?
            .mul(&Tensor::new(self.scale, self.lora_b.weight().device())?)?;
        
        // Add to base weight
        let new_weight = self.base_layer.weight().add(&lora_weight)?;
        
        // Update base layer
        self.base_layer = Linear::new(
            new_weight,
            self.base_layer.bias().cloned(),
        );
        
        self.merged = true;
        Ok(())
    }
    
    /// Unmerge LoRA weights from base layer
    pub fn unmerge(&mut self) -> Result<()> {
        if !self.merged {
            return Ok(());
        }
        
        // Get LoRA weight: B @ A * scale
        let lora_weight = self.lora_b.weight()
            .matmul(&self.lora_a.weight())?
            .mul(&Tensor::new(self.scale, self.lora_b.weight().device())?)?;
        
        // Subtract from base weight
        let new_weight = self.base_layer.weight().sub(&lora_weight)?;
        
        // Update base layer
        self.base_layer = Linear::new(
            new_weight,
            self.base_layer.bias().cloned(),
        );
        
        self.merged = false;
        Ok(())
    }
    
    /// Get LoRA A weight for gradient flow testing
    pub fn lora_a_weight(&self) -> &Tensor {
        self.lora_a.weight()
    }
    
    /// Get LoRA B weight
    pub fn lora_b_weight(&self) -> &Tensor {
        self.lora_b.weight()
    }
    
    /// Get both LoRA weights for optimizer
    pub fn lora_weights(&self) -> (&Tensor, &Tensor) {
        (self.lora_a.weight(), self.lora_b.weight())
    }
}

impl Module for LoRALinear {
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
        
        // x @ A^T @ B^T * scale
        let lora_output = self.lora_a.forward(&lora_input)?;
        let lora_output = self.lora_b.forward(&lora_output)?;
        let lora_output = lora_output.mul(&Tensor::new(self.scale, lora_output.device())?)?;
        
        // Add base and LoRA outputs
        base_output.add(&lora_output)
    }
}

/// LoRA Conv2d layer implementation
#[derive(Clone)]
pub struct LoRAConv2d {
    base_layer: Conv2d,
    lora_a: Conv2d,
    lora_b: Conv2d,
    scale: f32,
    dropout: Option<Dropout>,
    merged: bool,
}

impl LoRAConv2d {
    pub fn new(
        vb: &VarBuilder,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        rank: usize,
        alpha: f32,
        dropout_rate: f32,
        config: Conv2dConfig,
    ) -> Result<Self> {
        // Base convolutional layer
        let base_layer = Conv2d::new(
            vb.get((out_channels, in_channels, kernel_size, kernel_size), "weight")?,
            vb.get(out_channels, "bias").ok(),
            config,
        );
        
        // LoRA down projection (1x1 conv)
        let lora_a_config = Conv2dConfig {
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: 1,
            cudnn_fwd_algo: None,
        };
        
        let lora_a = Conv2d::new(
            vb.pp("lora_a").get_with_hints(
                (rank, in_channels, 1, 1),
                "weight",
                candle_nn::init::DEFAULT_KAIMING_NORMAL
            )?,
            None,
            lora_a_config,
        );
        
        // LoRA up projection (kernel_size conv)
        let lora_b_config = config;
        
        let lora_b = Conv2d::new(
            vb.pp("lora_b").get_with_hints(
                (out_channels, rank, kernel_size, kernel_size),
                "weight",
                candle_nn::init::ZERO
            )?,
            None,
            lora_b_config,
        );
        
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
        })
    }
}

impl Module for LoRAConv2d {
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
        
        // Apply LoRA transformations
        let lora_output = self.lora_a.forward(&lora_input)?;
        let lora_output = self.lora_b.forward(&lora_output)?;
        let lora_output = lora_output.mul(&Tensor::new(self.scale, lora_output.device())?)?;
        
        // Add base and LoRA outputs
        base_output.add(&lora_output)
    }
}

impl LoRAConv2d {
    /// Get LoRA A weight
    pub fn lora_a_weight(&self) -> &Tensor {
        self.lora_a.weight()
    }
    
    /// Get LoRA B weight
    pub fn lora_b_weight(&self) -> &Tensor {
        self.lora_b.weight()
    }
    
    /// Get both LoRA weights for optimizer
    pub fn lora_weights(&self) -> (&Tensor, &Tensor) {
        (self.lora_a.weight(), self.lora_b.weight())
    }
}

/// LoRA attention module for more complex attention mechanisms
pub struct LoRAAttention {
    to_q: LoRALinear,
    to_k: LoRALinear,
    to_v: LoRALinear,
    to_out: LoRALinear,
    heads: usize,
    dim_head: usize,
}

impl LoRAAttention {
    pub fn new(
        vb: &VarBuilder,
        dim: usize,
        heads: usize,
        dim_head: usize,
        rank: usize,
        alpha: f32,
        dropout: f32,
    ) -> Result<Self> {
        let inner_dim = heads * dim_head;
        
        let to_q = LoRALinear::new(
            &vb.pp("to_q"),
            dim,
            inner_dim,
            rank,
            alpha,
            dropout,
        )?;
        
        let to_k = LoRALinear::new(
            &vb.pp("to_k"),
            dim,
            inner_dim,
            rank,
            alpha,
            dropout,
        )?;
        
        let to_v = LoRALinear::new(
            &vb.pp("to_v"),
            dim,
            inner_dim,
            rank,
            alpha,
            dropout,
        )?;
        
        let to_out = LoRALinear::new(
            &vb.pp("to_out"),
            inner_dim,
            dim,
            rank,
            alpha,
            dropout,
        )?;
        
        Ok(Self {
            to_q,
            to_k,
            to_v,
            to_out,
            heads,
            dim_head,
        })
    }
    
    pub fn forward(&self, x: &Tensor, context: Option<&Tensor>) -> Result<Tensor> {
        let (b, n, _) = x.dims3()?;
        let h = self.heads;
        let d = self.dim_head;
        
        // Compute Q, K, V
        let q = self.to_q.forward(x)?;
        
        let context = context.unwrap_or(x);
        let k = self.to_k.forward(context)?;
        let v = self.to_v.forward(context)?;
        
        // Reshape for multi-head attention
        let q = q.reshape(&[b, n, h, d])?.transpose(1, 2)?; // [b, h, n, d]
        let k = k.reshape(&[b, n, h, d])?.transpose(1, 2)?; // [b, h, n, d]
        let v = v.reshape(&[b, n, h, d])?.transpose(1, 2)?; // [b, h, n, d]
        
        // Scaled dot-product attention
        let scale = (d as f32).sqrt();
        let scores = q.matmul(&k.transpose(D::Minus1, D::Minus2)?)?
            .div(&Tensor::new(scale, q.device())?)?;
        
        let attn = candle_nn::ops::softmax(&scores, D::Minus1)?;
        
        // Apply attention to values
        let out = attn.matmul(&v)?; // [b, h, n, d]
        
        // Reshape back
        let out = out.transpose(1, 2)?.reshape(&[b, n, h * d])?;
        
        // Final projection
        self.to_out.forward(&out)
    }
}

/// Helper to inject LoRA into existing model layers
pub struct LoRAInjector;

impl LoRAInjector {
    /// Find and replace Linear layers with LoRA versions
    pub fn inject_lora_linear(
        module_name: &str,
        original_weight: &Tensor,
        original_bias: Option<&Tensor>,
        vb: &VarBuilder,
        rank: usize,
        alpha: f32,
        dropout: f32,
    ) -> Result<LoRALinear> {
        let (out_features, in_features) = original_weight.dims2()?;
        
        // Create base layer with original weights
        let base_layer = Linear::new(
            original_weight.clone(),
            original_bias.cloned(),
        );
        
        // Create LoRA matrices
        let lora_a = Linear::new(
            vb.pp(&format!("{}_lora_a", module_name))
                .get_with_hints((rank, in_features), "weight", candle_nn::init::DEFAULT_KAIMING_NORMAL)?,
            None,
        );
        
        let lora_b = Linear::new(
            vb.pp(&format!("{}_lora_b", module_name))
                .get_with_hints((out_features, rank), "weight", candle_nn::init::ZERO)?,
            None,
        );
        
        let scale = alpha / rank as f32;
        
        let dropout = if dropout > 0.0 {
            Some(Dropout::new(dropout))
        } else {
            None
        };
        
        Ok(LoRALinear {
            base_layer,
            lora_a,
            lora_b,
            scale,
            dropout,
            merged: false,
        })
    }
    
    /// Check if a module name should have LoRA applied
    pub fn should_apply_lora(module_name: &str, target_modules: &[String]) -> bool {
        target_modules.iter().any(|target| {
            module_name.contains(target) || module_name.ends_with(target)
        })
    }
}

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct LoRAConfig {
    pub rank: usize,
    pub alpha: f32,
    pub dropout: f32,
    pub target_modules: Vec<String>,
    pub modules_to_save: Vec<String>,
}

/// Enum to hold different types of LoRA layers while preserving type information
#[derive(Clone)]
pub enum LoRALayer {
    Linear(LoRALinear),
    Conv2d(LoRAConv2d),
}

impl LoRALayer {
    /// Get A and B weight tensors for saving
    pub fn get_weights(&self) -> (Tensor, Tensor) {
        match self {
            LoRALayer::Linear(linear) => (
                linear.lora_a.weight().clone(),
                linear.lora_b.weight().clone(),
            ),
            LoRALayer::Conv2d(conv) => (
                conv.lora_a.weight().clone(),
                conv.lora_b.weight().clone(),
            ),
        }
    }
    
    /// Count parameters in this layer
    pub fn num_params(&self) -> usize {
        match self {
            LoRALayer::Linear(linear) => {
                let a_params = linear.lora_a.weight().dims().iter().product::<usize>();
                let b_params = linear.lora_b.weight().dims().iter().product::<usize>();
                a_params + b_params
            }
            LoRALayer::Conv2d(conv) => {
                let a_params = conv.lora_a.weight().dims().iter().product::<usize>();
                let b_params = conv.lora_b.weight().dims().iter().product::<usize>();
                a_params + b_params
            }
        }
    }
}

impl Module for LoRALayer {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            LoRALayer::Linear(linear) => linear.forward(x),
            LoRALayer::Conv2d(conv) => conv.forward(x),
        }
    }
}

/// LoRA model wrapper for easier management
pub struct LoRAModel {
    pub lora_layers: HashMap<String, LoRALayer>,
    pub config: LoRAConfig,
}

impl LoRAModel {
    pub fn new(config: LoRAConfig) -> Self {
        Self {
            lora_layers: HashMap::new(),
            config,
        }
    }
    
    /// Add a LoRA layer
    pub fn add_layer(&mut self, name: String, layer: LoRALayer) {
        self.lora_layers.insert(name, layer);
    }
    
    /// Get total number of trainable parameters
    pub fn num_trainable_params(&self) -> usize {
        self.lora_layers.values().map(|layer| layer.num_params()).sum()
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
            let (lora_a, lora_b) = layer.get_weights();
            tensors.insert(format!("{}.lora_a.weight", name), lora_a);
            tensors.insert(format!("{}.lora_b.weight", name), lora_b);
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
    
    /// Load LoRA weights from file
    pub fn from_pretrained(path: &str, device: &Device, dtype: DType) -> Result<Self> {
        use std::fs;
        use candle_core::safetensors;
        
        // Load configuration
        let config_path = format!("{}/adapter_config.json", path);
        let config_content = fs::read_to_string(&config_path)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to read config from {}: {}", config_path, e)))?;
        
        let config_json: serde_json::Value = serde_json::from_str(&config_content)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to parse config JSON: {}", e)))?;
        
        // Extract configuration values
        let rank = config_json["rank"].as_u64()
            .ok_or_else(|| candle_core::Error::Msg("Missing 'rank' in config".to_string()))? as usize;
        let alpha = config_json["alpha"].as_f64()
            .ok_or_else(|| candle_core::Error::Msg("Missing 'alpha' in config".to_string()))? as f32;
        let dropout = config_json["dropout"].as_f64()
            .unwrap_or(0.0) as f32;
        
        let target_modules = config_json["target_modules"].as_array()
            .ok_or_else(|| candle_core::Error::Msg("Missing 'target_modules' in config".to_string()))?
            .iter()
            .filter_map(|v| v.as_str().map(String::from))
            .collect();
        
        let modules_to_save = config_json["modules_to_save"].as_array()
            .unwrap_or(&Vec::new())
            .iter()
            .filter_map(|v| v.as_str().map(String::from))
            .collect();
        
        let lora_layer_names: Vec<String> = config_json["lora_layers"].as_array()
            .unwrap_or(&Vec::new())
            .iter()
            .filter_map(|v| v.as_str().map(String::from))
            .collect();
        
        let config = LoRAConfig {
            rank,
            alpha,
            dropout,
            target_modules,
            modules_to_save,
        };
        
        // Load weights
        let weights_path = format!("{}/adapter_model.safetensors", path);
        let tensors = safetensors::load(&weights_path, device)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to load weights from {}: {}", weights_path, e)))?;
        
        // Create model
        let mut model = Self::new(config);
        
        // Note: We cannot reconstruct the actual LoRA layers without knowing their dimensions
        // This would require additional metadata in the saved config
        // For now, we just load the config and weights are available in `tensors`
        println!("Loaded LoRA config from: {}", path);
        println!("Loaded {} tensors from: {}", tensors.len(), weights_path);
        println!("NOTE: Layer reconstruction requires additional dimension metadata");
        
        Ok(model)
    }
}
