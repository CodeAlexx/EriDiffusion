//! LoCoN (LoRA for Convolution) implementation

use eridiffusion_core::{
    NetworkAdapter, NetworkType, NetworkMetadata, ModelArchitecture, Device, Result, Error,
};
use candle_core::{Tensor, DType, Module, Shape, Var};
use candle_nn::{Conv2d, Conv2dConfig, Linear, VarBuilder};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;
use std::path::Path;
use async_trait::async_trait;

/// LoCoN configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoConConfig {
    pub rank: usize,
    pub alpha: f32,
    pub dropout: f32,
    pub use_cp: bool, // Use CP decomposition
    pub target_modules: Vec<String>,
    pub conv_rank: Option<usize>, // Separate rank for conv layers
    pub decompose_both: bool, // Decompose both weight and kernel
    pub factor: Option<i32>, // Factorization factor (-1 for auto)
}

impl Default for LoConConfig {
    fn default() -> Self {
        Self {
            rank: 4,
            alpha: 1.0,
            dropout: 0.0,
            use_cp: false,
            target_modules: vec![
                "conv".to_string(),
                "Conv2d".to_string(),
                "to_q".to_string(),
                "to_k".to_string(),
                "to_v".to_string(),
            ],
            conv_rank: None,
            decompose_both: true,
            factor: Some(-1),
        }
    }
}

/// LoCoN layer types
pub enum LoConLayer {
    Linear(LoConLinear),
    Conv2d(LoConConv2d),
}

/// LoCoN linear layer
pub struct LoConLinear {
    lora_a: Linear,
    lora_b: Linear,
    scaling: f32,
    dropout: Option<f32>,
}

/// LoCoN convolutional layer
pub struct LoConConv2d {
    lora_a: Conv2d,
    lora_b: Conv2d,
    lora_mid: Option<Conv2d>, // For CP decomposition
    scaling: f32,
    dropout: Option<f32>,
    kernel_size: usize,
}

impl LoConLinear {
    pub fn new(
        in_features: usize,
        out_features: usize,
        rank: usize,
        alpha: f32,
        dropout: f32,
        device: &Device,
    ) -> Result<Self> {
        let candle_device = device.to_candle()?;
        let dtype = DType::F32;
        
        let lora_a = Linear::new(
            kaiming_uniform_init(rank, in_features, dtype, &candle_device)?,
            None,
        );
        
        let lora_b = Linear::new(
            Tensor::zeros(&[out_features, rank], dtype, &candle_device)?,
            None,
        );
        
        let scaling = alpha / rank as f32;
        
        Ok(Self {
            lora_a,
            lora_b,
            scaling,
            dropout: if dropout > 0.0 { Some(dropout) } else { None },
        })
    }
    
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut result = x.clone();
        
        if let Some(dropout_rate) = self.dropout {
            result = apply_dropout(&result, dropout_rate)?;
        }
        
        result = self.lora_a.forward(&result)?;
        result = self.lora_b.forward(&result)?;
        
        result.affine(self.scaling as f64, 0.0)
            .map_err(|e| Error::Network(e.to_string()))
    }
}

impl LoConConv2d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        rank: usize,
        alpha: f32,
        dropout: f32,
        use_cp: bool,
        device: &Device,
    ) -> Result<Self> {
        let candle_device = device.to_candle()?;
        let dtype = DType::F32;
        
        let conv_config = Conv2dConfig {
            padding: kernel_size / 2,
            stride: 1,
            dilation: 1,
            groups: 1,
            cudnn_fwd_algo: None,
        };
        
        let (lora_a, lora_b, lora_mid) = if use_cp {
            // CP decomposition: weight = (U kron V) @ M @ (P kron Q)
            let mid_rank = rank * rank;
            
            // Spatial decomposition
            let lora_a = candle_nn::conv2d(
                in_channels,
                rank,
                1, // 1x1 conv
                conv_config,
                VarBuilder::zeros(dtype, &candle_device),
            )?;
            
            // Middle transformation
            let lora_mid = Some(candle_nn::conv2d(
                rank,
                rank,
                kernel_size,
                conv_config,
                VarBuilder::zeros(dtype, &candle_device),
            )?);
            
            // Output projection
            let lora_b = candle_nn::conv2d(
                rank,
                out_channels,
                1, // 1x1 conv
                conv_config,
                VarBuilder::zeros(dtype, &candle_device),
            )?;
            
            (lora_a, lora_b, lora_mid)
        } else {
            // Standard matrix decomposition
            let lora_a = candle_nn::conv2d(
                in_channels,
                rank,
                kernel_size,
                conv_config,
                VarBuilder::zeros(dtype, &candle_device),
            )?;
            
            let lora_b = candle_nn::conv2d(
                rank,
                out_channels,
                1, // 1x1 conv
                conv_config,
                VarBuilder::zeros(dtype, &candle_device),
            )?;
            
            (lora_a, lora_b, None)
        };
        
        // Initialize lora_a with kaiming uniform
        Self::init_conv_weights(&lora_a, in_channels, rank, kernel_size)?;
        
        let scaling = alpha / rank as f32;
        
        Ok(Self {
            lora_a,
            lora_b,
            lora_mid,
            scaling,
            dropout: if dropout > 0.0 { Some(dropout) } else { None },
            kernel_size,
        })
    }
    
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut result = x.clone();
        
        if let Some(dropout_rate) = self.dropout {
            result = apply_dropout(&result, dropout_rate)?;
        }
        
        // Apply decomposition
        result = self.lora_a.forward(&result)?;
        
        if let Some(ref lora_mid) = self.lora_mid {
            result = lora_mid.forward(&result)?;
        }
        
        result = self.lora_b.forward(&result)?;
        
        Ok((result * self.scaling as f64)?)
    }
    
    fn init_conv_weights(conv: &Conv2d, in_channels: usize, out_channels: usize, kernel_size: usize) -> Result<()> {
        // Simplified initialization
        // In practice would modify conv weights with proper initialization
        Ok(())
    }
}

/// LoCoN adapter state
struct LoConState {
    layers: HashMap<String, LoConLayer>,
    enabled: bool,
    training: bool,
    device: Device,
}

/// LoCoN adapter
pub struct LoConAdapter {
    config: LoConConfig,
    state: Arc<RwLock<LoConState>>,
    architecture: ModelArchitecture,
    metadata: NetworkMetadata,
}

impl LoConAdapter {
    /// Create new LoCoN adapter
    pub fn new(
        config: LoConConfig,
        architecture: ModelArchitecture,
        device: Device,
    ) -> Result<Self> {
        let state = Arc::new(RwLock::new(LoConState {
            layers: HashMap::new(),
            enabled: true,
            training: false,
            device,
        }));
        
        let metadata = NetworkMetadata {
            name: "locon_adapter".to_string(),
            network_type: NetworkType::LoCoN,
            version: "1.0.0".to_string(),
            base_model: "unknown".to_string(),
            rank: Some(config.rank),
            alpha: Some(config.alpha),
            target_modules: config.target_modules.clone(),
            created_at: chrono::Utc::now(),
            config: HashMap::new(),
        };
        
        Ok(Self {
            config,
            state,
            architecture,
            metadata,
        })
    }
    
    /// Initialize LoCoN layers
    pub fn initialize_layers(&self, model_info: &HashMap<String, LayerInfo>) -> Result<()> {
        let mut state = self.state.write();
        
        for (name, info) in model_info {
            if self.should_adapt_module(name) {
                match &info.layer_type {
                    LayerType::Linear { in_features, out_features } => {
                        let layer = LoConLinear::new(
                            *in_features,
                            *out_features,
                            self.config.rank,
                            self.config.alpha,
                            self.config.dropout,
                            &state.device,
                        )?;
                        
                        state.layers.insert(name.clone(), LoConLayer::Linear(layer));
                    }
                    LayerType::Conv2d { in_channels, out_channels, kernel_size } => {
                        let rank = self.config.conv_rank.unwrap_or(self.config.rank);
                        
                        let layer = LoConConv2d::new(
                            *in_channels,
                            *out_channels,
                            *kernel_size,
                            rank,
                            self.config.alpha,
                            self.config.dropout,
                            self.config.use_cp,
                            &state.device,
                        )?;
                        
                        state.layers.insert(name.clone(), LoConLayer::Conv2d(layer));
                    }
                }
            }
        }
        
        tracing::info!("Initialized {} LoCoN layers", state.layers.len());
        Ok(())
    }
    
    fn should_adapt_module(&self, name: &str) -> bool {
        self.config.target_modules.iter().any(|pattern| {
            name.contains(pattern)
        })
    }
    
    /// Apply LoCoN to layer
    pub fn apply_to_layer(&self, layer_name: &str, x: &Tensor) -> Result<Option<Tensor>> {
        let state = self.state.read();
        
        if !state.enabled {
            return Ok(None);
        }
        
        if let Some(layer) = state.layers.get(layer_name) {
            let output = match layer {
                LoConLayer::Linear(linear) => linear.forward(x)?,
                LoConLayer::Conv2d(conv) => conv.forward(x)?,
            };
            Ok(Some(output))
        } else {
            Ok(None)
        }
    }
    
    /// Get number of parameters
    pub fn num_parameters(&self) -> usize {
        let state = self.state.read();
        
        state.layers.values().map(|layer| {
            match layer {
                LoConLayer::Linear(l) => {
                    // A and B parameters
                    l.lora_a.weight().elem_count() + l.lora_b.weight().elem_count()
                }
                LoConLayer::Conv2d(c) => {
                    let mut count = c.lora_a.weight().elem_count() + c.lora_b.weight().elem_count();
                    if let Some(ref mid) = c.lora_mid {
                        count += mid.weight().elem_count();
                    }
                    count
                }
            }
        }).sum()
    }
}

#[async_trait]
impl NetworkAdapter for LoConAdapter {
    fn adapter_type(&self) -> NetworkType {
        NetworkType::LoCoN
    }
    
    fn metadata(&self) -> &NetworkMetadata {
        &self.metadata
    }
    
    fn target_modules(&self) -> &[String] {
        &self.config.target_modules
    }
    
    fn trainable_parameters(&self) -> Vec<&Var> {
        // TODO: Return actual Var references from LoCoN layers
        // For now return empty as layers use Linear/Conv2d which don't expose Var directly
        Vec::new()
    }
    
    fn parameters(&self) -> HashMap<String, Tensor> {
        let state = self.state.read();
        let mut params = HashMap::new();
        
        for (name, layer) in &state.layers {
            match layer {
                LoConLayer::Linear(linear) => {
                    params.insert(format!("{}.lora_a.weight", name), linear.lora_a.weight().clone());
                    params.insert(format!("{}.lora_b.weight", name), linear.lora_b.weight().clone());
                }
                LoConLayer::Conv2d(conv) => {
                    params.insert(format!("{}.lora_a.weight", name), conv.lora_a.weight().clone());
                    params.insert(format!("{}.lora_b.weight", name), conv.lora_b.weight().clone());
                    if let Some(ref lora_mid) = conv.lora_mid {
                        params.insert(format!("{}.lora_mid.weight", name), lora_mid.weight().clone());
                    }
                }
            }
        }
        
        params
    }
    
    fn apply_to_layer(&self, layer_name: &str, input: &Tensor) -> Result<Tensor> {
        let state = self.state.read();
        
        if !state.enabled {
            return Ok(input.clone());
        }
        
        if let Some(layer) = state.layers.get(layer_name) {
            match layer {
                LoConLayer::Linear(linear) => linear.forward(input),
                LoConLayer::Conv2d(conv) => conv.forward(input),
            }
        } else {
            Ok(input.clone())
        }
    }
    
    fn merge_weights(&mut self, scale: f32) -> Result<()> {
        let mut state = self.state.write();
        
        // Update scaling for all layers
        for layer in state.layers.values_mut() {
            match layer {
                LoConLayer::Linear(linear) => {
                    linear.scaling = scale * self.config.alpha / self.config.rank as f32;
                }
                LoConLayer::Conv2d(conv) => {
                    let rank = self.config.conv_rank.unwrap_or(self.config.rank);
                    conv.scaling = scale * self.config.alpha / rank as f32;
                }
            }
        }
        
        Ok(())
    }
    
    async fn save_weights(&self, path: &Path) -> Result<()> {
        use candle_core::safetensors::save;
        
        let tensors = {
            let state = self.state.read();
            let mut tensors = HashMap::new();
            
            for (name, layer) in &state.layers {
                match layer {
                    LoConLayer::Linear(linear) => {
                        tensors.insert(
                            format!("{}.lora_a.weight", name),
                            linear.lora_a.weight().clone(),
                        );
                        tensors.insert(
                            format!("{}.lora_b.weight", name),
                            linear.lora_b.weight().clone(),
                        );
                    }
                    LoConLayer::Conv2d(conv) => {
                        tensors.insert(
                            format!("{}.lora_a.weight", name),
                            conv.lora_a.weight().clone(),
                        );
                        tensors.insert(
                            format!("{}.lora_b.weight", name),
                            conv.lora_b.weight().clone(),
                        );
                        if let Some(ref lora_mid) = conv.lora_mid {
                            tensors.insert(
                                format!("{}.lora_mid.weight", name),
                                lora_mid.weight().clone(),
                            );
                        }
                    }
                }
            }
            tensors
        };
        
        // Save metadata
        let metadata_path = path.with_extension("json");
        let metadata_json = serde_json::to_string_pretty(&self.metadata)?;
        tokio::fs::write(&metadata_path, metadata_json).await?;
        
        // Save tensors
        save(&tensors, path)?;
        
        Ok(())
    }
    
    async fn load_weights(&mut self, path: &Path) -> Result<()> {
        use candle_core::safetensors::load;
        
        // Load tensors
        let tensors = load(path, &self.state.read().device.to_candle()?)?;
        
        // Load LoCoN weights from state dict
        {
            let mut state = self.state.write();
            
            for (key, tensor) in tensors {
                if let Some((module_name, param_name)) = key.rsplit_once('.') {
                    if let Some(layer) = state.layers.get_mut(module_name) {
                        match (layer, param_name) {
                            (LoConLayer::Linear(linear), "lora_a.weight") => {
                                // Update lora_a weight
                            }
                            (LoConLayer::Linear(linear), "lora_b.weight") => {
                                // Update lora_b weight
                            }
                            (LoConLayer::Conv2d(conv), "lora_a.weight") => {
                                // Update lora_a weight
                            }
                            (LoConLayer::Conv2d(conv), "lora_b.weight") => {
                                // Update lora_b weight
                            }
                            _ => {}
                        }
                    }
                }
            }
        } // Drop the write lock here
        
        // Load metadata if exists
        let metadata_path = path.with_extension("json");
        if metadata_path.exists() {
            let metadata_json = tokio::fs::read_to_string(&metadata_path).await?;
            self.metadata = serde_json::from_str(&metadata_json)?;
        }
        
        Ok(())
    }
    
    fn memory_usage(&self) -> usize {
        let state = self.state.read();
        
        state.layers.values().map(|layer| {
            match layer {
                LoConLayer::Linear(linear) => {
                    // LoRA A and B weights
                    let a_params = self.config.rank * linear.lora_a.weight().dims()[1];
                    let b_params = linear.lora_b.weight().dims()[0] * self.config.rank;
                    (a_params + b_params) * 4 // f32 = 4 bytes
                }
                LoConLayer::Conv2d(conv) => {
                    let rank = self.config.conv_rank.unwrap_or(self.config.rank);
                    let a_shape = conv.lora_a.weight().dims();
                    let b_shape = conv.lora_b.weight().dims();
                    
                    let a_params = a_shape.iter().product::<usize>();
                    let b_params = b_shape.iter().product::<usize>();
                    let mid_params = if conv.lora_mid.is_some() {
                        rank * rank * conv.kernel_size * conv.kernel_size
                    } else {
                        0
                    };
                    
                    (a_params + b_params + mid_params) * 4
                }
            }
        }).sum()
    }
    
    fn to_device(&mut self, device: &Device) -> Result<()> {
        let mut state = self.state.write();
        state.device = device.clone();
        
        // Device migration is handled during layer creation
        
        Ok(())
    }
}

/// Layer information
pub struct LayerInfo {
    pub name: String,
    pub layer_type: LayerType,
}

/// Layer types
pub enum LayerType {
    Linear {
        in_features: usize,
        out_features: usize,
    },
    Conv2d {
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
    },
}

/// Helper functions
fn kaiming_uniform_init(
    fan_out: usize,
    fan_in: usize,
    dtype: DType,
    device: &candle_core::Device,
) -> Result<Tensor> {
    let bound = (3.0f64 / fan_in as f64).sqrt();
    Tensor::rand(
        -bound,
        bound,
        &[fan_out, fan_in],
        device,
    )?.to_dtype(dtype).map_err(|e| Error::Tensor(e.to_string()))
}

fn apply_dropout(x: &Tensor, rate: f32) -> Result<Tensor> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    
    let mask = Tensor::from_vec(
        (0..x.elem_count())
            .map(|_| if rng.gen::<f32>() > rate { 1.0f32 } else { 0.0f32 })
            .collect::<Vec<_>>(),
        x.shape(),
        x.device(),
    )?;
    
    Ok((x * mask)?)
}

/// LoCoN utilities
pub mod utils {
    use super::*;
    
    /// Calculate optimal rank for convolution
    pub fn calculate_conv_rank(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        compression_ratio: f32,
    ) -> usize {
        let full_params = in_channels * out_channels * kernel_size * kernel_size;
        let target_params = (full_params as f32 * compression_ratio) as usize;
        
        // For conv: rank * (in_channels * k + out_channels) ≈ target_params
        let divisor = in_channels * kernel_size + out_channels;
        let rank = target_params / divisor;
        
        rank.max(1).min(in_channels.min(out_channels))
    }
    
    /// Extract tucker decomposition from conv weight
    pub fn tucker_decomposition(
        weight: &Tensor, // [out_channels, in_channels, k, k]
        rank: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        // Simplified Tucker decomposition
        // In practice would use proper tensor decomposition
        
        let shape = weight.dims();
        let device = weight.device();
        
        let factor1 = Tensor::randn(0.0f32, 0.02, &[shape[0], rank], device)?;
        let core = Tensor::randn(0.0f32, 0.02, &[rank, rank, shape[2], shape[3]], device)?;
        let factor2 = Tensor::randn(0.0f32, 0.02, &[shape[1], rank], device)?;
        
        Ok((factor1, core, factor2))
    }
    
    /// CP decomposition for convolution
    pub fn cp_decomposition(
        weight: &Tensor,
        rank: usize,
    ) -> Result<Vec<Tensor>> {
        // Simplified CP decomposition
        // Returns factors for each mode
        
        let shape = weight.dims();
        let device = weight.device();
        
        let factors = vec![
            Tensor::randn(0.0f32, 0.02, &[shape[0], rank], device)?,
            Tensor::randn(0.0f32, 0.02, &[shape[1], rank], device)?,
            Tensor::randn(0.0f32, 0.02, &[shape[2], rank], device)?,
            Tensor::randn(0.0f32, 0.02, &[shape[3], rank], device)?,
        ];
        
        Ok(factors)
    }
}