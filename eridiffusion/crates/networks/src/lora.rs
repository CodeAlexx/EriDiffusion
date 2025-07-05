//! LoRA (Low-Rank Adaptation) implementation

use eridiffusion_core::{
    NetworkAdapter, NetworkType, NetworkMetadata, ModelArchitecture, Device, Result, Error,
};
use candle_core::{Tensor, DType, Module, Shape, Var};
use candle_nn::{Linear, VarBuilder};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;
use std::path::Path;
use async_trait::async_trait;

/// LoRA configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoRAConfig {
    pub rank: usize,
    pub alpha: f32,
    pub dropout: f32,
    pub target_modules: Vec<String>,
    pub use_bias: bool,
    pub fan_in_fan_out: bool,
    pub merge_weights: bool,
    pub init_weights: bool,
    pub use_rslora: bool,
    pub use_dora: bool,
    pub rank_pattern: HashMap<String, usize>,
    pub alpha_pattern: HashMap<String, f32>,
}

impl Default for LoRAConfig {
    fn default() -> Self {
        Self {
            rank: 4,
            alpha: 1.0,
            dropout: 0.0,
            target_modules: vec![
                "to_q".to_string(),
                "to_k".to_string(),
                "to_v".to_string(),
                "to_out.0".to_string(),
            ],
            use_bias: false,
            fan_in_fan_out: false,
            merge_weights: false,
            init_weights: true,
            use_rslora: false,
            use_dora: false,
            rank_pattern: HashMap::new(),
            alpha_pattern: HashMap::new(),
        }
    }
}

/// LoRA layer
pub struct LoRALayer {
    lora_a: Linear,
    lora_b: Linear,
    scaling: f32,
    dropout: Option<f32>,
    merged: bool,
    rank: usize,
    in_features: usize,
    out_features: usize,
}

impl LoRALayer {
    /// Create new LoRA layer
    pub fn new(
        in_features: usize,
        out_features: usize,
        rank: usize,
        alpha: f32,
        dropout: f32,
        use_bias: bool,
        device: &Device,
    ) -> Result<Self> {
        let candle_device = device.to_candle()?;
        let dtype = DType::F32;
        
        // Create low-rank matrices
        let lora_a = Linear::new(
            Self::kaiming_uniform_init(rank, in_features, dtype, &candle_device)?,
            if use_bias {
                Some(Tensor::zeros(&[rank], dtype, &candle_device)?)
            } else {
                None
            },
        );
        
        let lora_b = Linear::new(
            Tensor::zeros(&[out_features, rank], dtype, &candle_device)?,
            if use_bias {
                Some(Tensor::zeros(&[out_features], dtype, &candle_device)?)
            } else {
                None
            },
        );
        
        let scaling = alpha / rank as f32;
        
        Ok(Self {
            lora_a,
            lora_b,
            scaling,
            dropout: if dropout > 0.0 { Some(dropout) } else { None },
            merged: false,
            rank,
            in_features,
            out_features,
        })
    }
    
    /// Forward pass
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Apply dropout if configured
        let input = if let Some(dropout_rate) = self.dropout {
            if self.training() {
                self.apply_dropout(x, dropout_rate)?
            } else {
                x.clone()
            }
        } else {
            x.clone()
        };
        
        // Apply LoRA: (x @ A) @ B * scaling
        let a_out = self.lora_a.forward(&input)?;
        let b_out = self.lora_b.forward(&a_out)?;
        
        // Apply scaling efficiently using affine
        b_out.affine(self.scaling as f64, 0.0)
            .map_err(|e| Error::Network(e.to_string()))
    }
    
    /// Merge LoRA weights into base layer
    pub fn merge(&mut self, base_weight: &Tensor) -> Result<Tensor> {
        if self.merged {
            return Ok(base_weight.clone());
        }
        
        // Compute LoRA weight: B @ A * scaling
        let lora_weight = self.get_weight()?;
        
        // Merge with base weight
        let merged_weight = (base_weight + &lora_weight)?;
        
        self.merged = true;
        Ok(merged_weight)
    }
    
    /// Unmerge LoRA weights
    pub fn unmerge(&mut self, merged_weight: &Tensor) -> Result<Tensor> {
        if !self.merged {
            return Ok(merged_weight.clone());
        }
        
        // Compute LoRA weight
        let lora_weight = self.get_weight()?;
        
        // Subtract from merged weight
        let base_weight = (merged_weight - &lora_weight)?;
        
        self.merged = false;
        Ok(base_weight)
    }
    
    /// Get LoRA weight matrix
    pub fn get_weight(&self) -> Result<Tensor> {
        // B @ A * scaling
        let a_weight = &self.lora_a.weight();
        let b_weight = &self.lora_b.weight();
        
        let weight = b_weight.matmul(&a_weight.t()?)?;
        Ok((weight * self.scaling as f64)?)
    }
    
    /// Apply dropout
    fn apply_dropout(&self, x: &Tensor, rate: f32) -> Result<Tensor> {
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
    
    /// Check if in training mode
    fn training(&self) -> bool {
        // Would be set by parent adapter
        true
    }
    
    /// Kaiming uniform initialization
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
}

/// LoRA adapter state
struct LoRAState {
    layers: HashMap<String, LoRALayer>,
    enabled: bool,
    training: bool,
    device: Device,
}

/// LoRA adapter
pub struct LoRAAdapter {
    config: LoRAConfig,
    state: Arc<RwLock<LoRAState>>,
    architecture: ModelArchitecture,
    metadata: NetworkMetadata,
}

impl LoRAAdapter {
    /// Create new LoRA adapter
    pub fn new(
        config: LoRAConfig,
        architecture: ModelArchitecture,
        device: Device,
    ) -> Result<Self> {
        let state = Arc::new(RwLock::new(LoRAState {
            layers: HashMap::new(),
            enabled: true,
            training: false,
            device,
        }));
        
        let metadata = NetworkMetadata {
            name: "lora_adapter".to_string(),
            network_type: NetworkType::LoRA,
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
    
    /// Initialize LoRA layers for model
    pub fn initialize_layers(&self, model_state_dict: &HashMap<String, TensorInfo>) -> Result<()> {
        let mut state = self.state.write();
        
        for (name, info) in model_state_dict {
            if self.should_adapt_module(name) {
                let rank = self.get_rank_for_module(name);
                let alpha = self.get_alpha_for_module(name);
                
                if info.shape.len() >= 2 {
                    let out_features = info.shape[0];
                    let in_features = info.shape[1];
                    
                    let lora_layer = LoRALayer::new(
                        in_features,
                        out_features,
                        rank,
                        alpha,
                        self.config.dropout,
                        self.config.use_bias,
                        &state.device,
                    )?;
                    
                    state.layers.insert(name.clone(), lora_layer);
                    
                    tracing::debug!(
                        "Initialized LoRA for {}: {}x{} (rank={})",
                        name, in_features, out_features, rank
                    );
                }
            }
        }
        
        tracing::info!("Initialized {} LoRA layers", state.layers.len());
        Ok(())
    }
    
    /// Check if module should be adapted
    fn should_adapt_module(&self, name: &str) -> bool {
        self.config.target_modules.iter().any(|pattern| {
            name.contains(pattern)
        })
    }
    
    /// Get rank for specific module
    fn get_rank_for_module(&self, name: &str) -> usize {
        for (pattern, rank) in &self.config.rank_pattern {
            if name.contains(pattern) {
                return *rank;
            }
        }
        self.config.rank
    }
    
    /// Get alpha for specific module
    fn get_alpha_for_module(&self, name: &str) -> f32 {
        for (pattern, alpha) in &self.config.alpha_pattern {
            if name.contains(pattern) {
                return *alpha;
            }
        }
        self.config.alpha
    }
    
    /// Apply LoRA to specific layer with base output
    pub fn apply_with_base(&self, layer_name: &str, x: &Tensor, base_output: &Tensor) -> Result<Tensor> {
        let state = self.state.read();
        
        if !state.enabled {
            return Ok(base_output.clone());
        }
        
        if let Some(lora_layer) = state.layers.get(layer_name) {
            let lora_output = lora_layer.forward(x)?;
            Ok((base_output + &lora_output)?)
        } else {
            Ok(base_output.clone())
        }
    }
    
    /// Merge all LoRA weights
    pub fn merge_weights(&self, model_state_dict: &mut HashMap<String, Tensor>) -> Result<()> {
        let mut state = self.state.write();
        
        for (name, lora_layer) in &mut state.layers {
            if let Some(base_weight) = model_state_dict.get(name) {
                let merged_weight = lora_layer.merge(base_weight)?;
                model_state_dict.insert(name.clone(), merged_weight);
            }
        }
        
        Ok(())
    }
    
    /// Unmerge all LoRA weights
    pub fn unmerge_weights(&self, model_state_dict: &mut HashMap<String, Tensor>) -> Result<()> {
        let mut state = self.state.write();
        
        for (name, lora_layer) in &mut state.layers {
            if let Some(merged_weight) = model_state_dict.get(name) {
                let base_weight = lora_layer.unmerge(merged_weight)?;
                model_state_dict.insert(name.clone(), base_weight);
            }
        }
        
        Ok(())
    }
    
    /// Get total number of parameters
    pub fn num_parameters(&self) -> usize {
        let state = self.state.read();
        
        state.layers.values().map(|layer| {
            let a_params = layer.rank * layer.in_features;
            let b_params = layer.out_features * layer.rank;
            a_params + b_params
        }).sum()
    }
    
    /// Get parameter reduction ratio
    pub fn compression_ratio(&self, base_parameters: usize) -> f32 {
        let lora_params = self.num_parameters();
        lora_params as f32 / base_parameters as f32
    }
}

#[async_trait]
impl NetworkAdapter for LoRAAdapter {
    fn adapter_type(&self) -> NetworkType {
        NetworkType::LoRA
    }
    
    fn metadata(&self) -> &NetworkMetadata {
        &self.metadata
    }
    
    fn target_modules(&self) -> &[String] {
        &self.config.target_modules
    }
    
    fn trainable_parameters(&self) -> Vec<&Var> {
        // TODO: Return actual Var references from LoRA layers
        // For now return empty as layers use Linear which doesn't expose Var directly
        Vec::new()
    }
    
    fn parameters(&self) -> HashMap<String, Tensor> {
        let state = self.state.read();
        let mut params = HashMap::new();
        
        for (name, layer) in &state.layers {
            // Clone LoRA A and B weights
            params.insert(
                format!("{}.lora_a.weight", name), 
                layer.lora_a.weight().clone()
            );
            params.insert(
                format!("{}.lora_b.weight", name), 
                layer.lora_b.weight().clone()
            );
            
            // Clone biases if present
            if let Some(bias) = layer.lora_a.bias() {
                params.insert(format!("{}.lora_a.bias", name), bias.clone());
            }
            if let Some(bias) = layer.lora_b.bias() {
                params.insert(format!("{}.lora_b.bias", name), bias.clone());
            }
        }
        
        params
    }
    
    fn apply_to_layer(&self, layer_name: &str, input: &Tensor) -> Result<Tensor> {
        let state = self.state.read();
        
        if !state.enabled {
            return Ok(input.clone());
        }
        
        if let Some(lora_layer) = state.layers.get(layer_name) {
            lora_layer.forward(input)
        } else {
            // If this layer doesn't have LoRA, return input unchanged
            Ok(input.clone())
        }
    }
    
    fn merge_weights(&mut self, scale: f32) -> Result<()> {
        let mut state = self.state.write();
        
        // Update scaling for all layers
        for layer in state.layers.values_mut() {
            layer.scaling = scale * self.config.alpha / layer.rank as f32;
        }
        
        Ok(())
    }
    
    async fn save_weights(&self, path: &Path) -> Result<()> {
        use candle_core::safetensors::save;
        
        let tensors = {
            let state = self.state.read();
            let mut tensors = HashMap::new();
            
            for (name, layer) in &state.layers {
                tensors.insert(
                    format!("{}.lora_a.weight", name),
                    layer.lora_a.weight().clone(),
                );
                tensors.insert(
                    format!("{}.lora_b.weight", name),
                    layer.lora_b.weight().clone(),
                );
                
                if let Some(bias) = layer.lora_a.bias() {
                    tensors.insert(
                        format!("{}.lora_a.bias", name),
                        bias.clone(),
                    );
                }
                if let Some(bias) = layer.lora_b.bias() {
                    tensors.insert(
                        format!("{}.lora_b.bias", name),
                        bias.clone(),
                    );
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
        let device = self.state.read().device.clone();
        let tensors = load(path, &device.to_candle()?)?;
        
        // Parse loaded tensors and create layers
        {
            let mut state = self.state.write();
            
            let layer_keys: Vec<String> = tensors.keys()
                .filter(|k| k.ends_with(".lora_a.weight"))
                .map(|k| k.trim_end_matches(".lora_a.weight").to_string())
                .collect();
                
            for layer_name in layer_keys {
                let a_key = format!("{}.lora_a.weight", layer_name);
                let b_key = format!("{}.lora_b.weight", layer_name);
                
                if let (Some(a_weight), Some(b_weight)) = (tensors.get(&a_key), tensors.get(&b_key)) {
                    // Create new LoRA layer from loaded weights
                    let a_shape = a_weight.dims();
                    let b_shape = b_weight.dims();
                    
                    let layer = LoRALayer::new(
                        a_shape[1],  // in_features
                        b_shape[0], // out_features
                        a_shape[0],   // rank
                        self.get_alpha_for_module(&layer_name),
                        self.config.dropout,
                        self.config.use_bias,
                        &state.device,
                    )?;
                    
                    // TODO: Actually set the loaded weights into the layer
                    // This requires modifying LoRALayer to accept pre-initialized weights
                    
                    state.layers.insert(layer_name, layer);
                }
            }
        }
        
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
            let a_size = layer.rank * layer.in_features * 4; // f32 = 4 bytes
            let b_size = layer.out_features * layer.rank * 4;
            let bias_size = if self.config.use_bias {
                (layer.rank + layer.out_features) * 4
            } else {
                0
            };
            a_size + b_size + bias_size
        }).sum()
    }
    
    fn to_device(&mut self, device: &Device) -> Result<()> {
        let mut state = self.state.write();
        state.device = device.clone();
        
        // TODO: Move all layers to new device
        // This requires recreating tensors on the new device
        
        Ok(())
    }
}

/// Tensor info for initialization
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: DType,
}

/// LoRA utilities
pub mod utils {
    use super::*;
    
    /// Calculate optimal rank based on module size
    pub fn calculate_adaptive_rank(
        in_features: usize,
        out_features: usize,
        compression_ratio: f32,
    ) -> usize {
        let full_params = in_features * out_features;
        let target_params = (full_params as f32 * compression_ratio) as usize;
        
        // rank * (in_features + out_features) ≈ target_params
        let rank = target_params / (in_features + out_features);
        
        rank.max(1).min(in_features.min(out_features))
    }
    
    /// SVD-based LoRA initialization
    pub async fn svd_init_lora(
        weight: &Tensor,
        rank: usize,
    ) -> Result<(Tensor, Tensor)> {
        // Simplified - would use actual SVD
        let shape = weight.dims();
        let device = weight.device();
        
        let u = Tensor::randn(0.0f32, 0.02, &[shape[0], rank], device)?;
        let v = Tensor::randn(0.0f32, 0.02, &[rank, shape[1]], device)?;
        
        Ok((u, v))
    }
    
    /// Extract LoRA from fine-tuned model
    pub async fn extract_lora(
        base_model_weights: &HashMap<String, Tensor>,
        finetuned_weights: &HashMap<String, Tensor>,
        rank: usize,
        config: &LoRAConfig,
    ) -> Result<HashMap<String, (Tensor, Tensor)>> {
        let mut lora_weights = HashMap::new();
        
        for (name, base_weight) in base_model_weights {
            if let Some(finetuned_weight) = finetuned_weights.get(name) {
                // Calculate weight difference
                let diff = (finetuned_weight - base_weight)?;
                
                // Perform low-rank decomposition
                let (lora_a, lora_b) = svd_init_lora(&diff, rank).await?;
                
                lora_weights.insert(name.clone(), (lora_a, lora_b));
            }
        }
        
        Ok(lora_weights)
    }
    
    /// Quantize LoRA weights
    pub fn quantize_lora(
        lora_weights: &HashMap<String, (Tensor, Tensor)>,
        bits: usize,
    ) -> Result<HashMap<String, (Tensor, Tensor)>> {
        // Simplified quantization
        let mut quantized = HashMap::new();
        
        for (name, (a, b)) in lora_weights {
            // Would apply actual quantization
            quantized.insert(name.clone(), (a.clone(), b.clone()));
        }
        
        Ok(quantized)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    
    #[test]
    fn test_lora_layer() -> Result<()> {
        let device = Device::Cpu;
        let ai_device = eridiffusion_core::Device::CPU;
        
        let layer = LoRALayer::new(
            768,  // in_features
            768,  // out_features
            4,    // rank
            1.0,  // alpha
            0.0,  // dropout
            false, // use_bias
            &ai_device,
        )?;
        
        let x = Tensor::randn(0.0f32, 1.0, &[2, 768], &device)?;
        let output = layer.forward(&x)?;
        
        assert_eq!(output.dims(), &[2, 768]);
        Ok(())
    }
    
    #[test]
    fn test_rank_calculation() {
        let rank = utils::calculate_adaptive_rank(768, 768, 0.01);
        assert!(rank > 0);
        assert!(rank < 768);
    }
}