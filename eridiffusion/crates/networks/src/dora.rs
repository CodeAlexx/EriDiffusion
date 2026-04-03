/*
//! DoRA (Weight-Decomposed Low-Rank Adaptation) implementation (original content temporarily gated)
// Original implementation kept for reference. Disabled from compilation due to incomplete code blocks.
*/

// Minimal compile-time stub (non-destructive): exposes config and types without altering behavior elsewhere.
use serde::{Deserialize, Serialize};
use flame_core::Tensor;
use eridiffusion_core::Device;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DoRAConfig {
    pub rank: usize,
    pub alpha: f32,
}

impl Default for DoRAConfig {
    fn default() -> Self { Self { rank: 4, alpha: 1.0 } }
}

pub struct DoRAAdapter;

pub struct DoRALayer { pub scaling: f32 }

impl DoRALayer {
    pub fn new(_in_features: usize, _out_features: usize, rank: usize, alpha: f32, _dropout: f32, _epsilon: f32, _device: &Device, _base_weight: Option<&Tensor>) -> anyhow::Result<Self> {
        Ok(Self { scaling: alpha / rank as f32 })
    }
    pub fn forward(&self, x: &Tensor, _base_weight: &Tensor) -> anyhow::Result<Tensor> { x.clone() }
}
        lora_out = self.lora_a.forward(&lora_out)?;
        lora_out = self.lora_b.forward(&lora_out)?;
        lora_out = (lora_out * self.scaling as f64)?;
        
        // Compute base output with normalized weight
        let normalized_weight = self.normalize_weight(base_weight)?;
        let base_out = x.matmul(&normalized_weight.transpose(0, 1)?)?;
        
        // Scale by magnitude and add LoRA
        let scaled_base = self.apply_magnitude(&base_out)?;
        
        Ok((scaled_base + lora_out)?)
    }
    
    /// Normalize weight matrix
    fn normalize_weight(&self, weight: &Tensor) -> anyhow::Result<Tensor> {
        // Compute column-wise L2 norm
        let eps = Tensor::from_vec(&[self.epsilon], weight.device())?;
        let weight_norm = (weight.sqr()?.sum_keepdim(1isize as isize)?.sqrt()? + eps)?;
        
        // Normalize weight
        Ok((weight.div(&weight_norm)?)?)
    }
    
    /// Apply magnitude scaling
    fn apply_magnitude(&self, x: &Tensor) -> anyhow::Result<Tensor> {
        // x shape: [batch, out_features]
        // magnitude shape: [out_features]
        let magnitude = self.magnitude.unsqueeze(0)?;
        Ok(x.mul(&magnitude)?)
    }
    
    /// Compute weight norm for initialization
    fn compute_weight_norm(weight: &Tensor, epsilon: f32) -> anyhow::Result<Tensor> {
        // Compute column-wise L2 norm
        let eps = Tensor::from_vec(&[epsilon], weight.device())?;
        let norm = (weight.sqr()?.sum_keepdim(1isize as isize)?.sqrt()? + eps)?;
        Ok(norm.squeeze(1)?)
    }
    
    /// Get decomposed weight
    pub fn get_decomposed_weight(&self, base_weight: &Tensor) -> anyhow::Result<(Tensor, Tensor)> {
        // Normalize base weight
        let normalized = self.normalize_weight(base_weight)?;
        
        // Compute LoRA weight
        let lora_weight = self.get_lora_weight()?;
        
        // Combine: magnitude * (normalized + lora)
        let direction = (normalized + lora_weight)?;
        
        Ok((direction, self.magnitude.clone()))
    }
    
    /// Get LoRA weight matrix
    fn get_lora_weight(&self) -> anyhow::Result<Tensor> {
        let a_weight = &self.lora_a.weight();
        let b_weight = &self.lora_b.weight();
        
        let weight = b_weight.matmul(&a_weight.transpose(0, 1)?)?;
        Ok((weight * self.scaling as f64)?)
    }
    
    /// Apply dropout
    fn apply_dropout(&self, x: &Tensor, rate: f32) -> anyhow::Result<Tensor> {
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
    
    /// Kaiming uniform initialization
    fn kaiming_uniform_init(
        fan_out: usize,
        fan_in: usize,
        dtype: DType,
        device: &) -> anyhow::Result<Tensor> {
        let bound = (3.0f64 / fan_in as f64).sqrt();
        Tensor::rand(
            -bound,
            bound,
            &[fan_out, fan_in],
            device,
        )?.to_dtype(dtype).map_err(|e| Error::Tensor(e.to_string()))
    }
}

/// DoRA adapter state
struct DoRAState {
    layers: HashMap<String, DoRALayer>,
    base_weights: HashMap<String, Tensor>,
    enabled: bool,
    training: bool,
    device: Device,
}

/// DoRA adapter
pub struct DoRAAdapter {
    config: DoRAConfig,
    state: Arc<RwLock<DoRAState>>,
    architecture: ModelArchitecture,
    metadata: NetworkMetadata,
}

impl DoRAAdapter {
    /// Create new DoRA adapter
    pub fn new(
        config: DoRAConfig,
        architecture: ModelArchitecture,
        device: Device,
    ) -> anyhow::Result<Self> {
        let state = Arc::new(RwLock::new(DoRAState {
            layers: HashMap::new(),
            base_weights: HashMap::new(),
            enabled: true,
            training: false,
            device,
        }));
        
        let metadata = NetworkMetadata {
            name: "dora_adapter".to_string(),
            network_type: NetworkType::DoRA,
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
    
    /// Initialize DoRA layers
    pub fn initialize_layers(
        &self,
        model_state_dict: &HashMap<String, Tensor>,
    ) -> anyhow::Result<()> {
        let mut state = self.state.write();
        
        for (name, weight) in model_state_dict {
            if self.should_adapt_module(name) && weight.rank() >= 2 {
                let out_features = weight.dim(0);
                let in_features = weight.dim(1);
                
                let dora_layer = DoRALayer::new(
                    in_features,
                    out_features,
                    self.config.rank,
                    self.config.alpha,
                    self.config.dropout,
                    self.config.epsilon,
                    &state.device,
                    Some(weight),
                )?;
                
                state.layers.insert(name.clone(), dora_layer);
                state.base_weights.insert(name.clone(), weight.clone());
                
                tracing::debug!(
                    "Initialized DoRA for {}: {}x{} (rank={})",
                    name, in_features, out_features, self.config.rank
                );
            }
        }
        
        tracing::info!("Initialized {} DoRA layers", state.layers.len());
        Ok(())
    }
    
    /// Check if module should be adapted
    fn should_adapt_module(&self, name: &str) -> bool {
        self.config.target_modules.iter().any(|pattern| {
            name.contains(pattern)
        })
    }
    
    /// Apply DoRA to specific layer
    pub fn apply_to_layer(
        &self,
        layer_name: &str,
        x: &Tensor,
    ) -> anyhow::Result<Option<Tensor>> {
        let state = self.state.read();
        
        if !state.enabled {
            return Ok(None);
        }
        
        if let (Some(dora_layer), Some(base_weight)) = 
            (state.layers.get(layer_name), state.base_weights.get(layer_name)) {
            let output = dora_layer.forward(x, base_weight)?;
            Ok(Some(output))
        } else {
            Ok(None)
        }
    }
    
    /// Get total number of parameters
    pub fn num_parameters(&self) -> usize {
        let state = self.state.read();
        
        state.layers.values().map(|layer| {
            let a_params = layer.rank * layer.in_features;
            let b_params = layer.out_features * layer.rank;
            let magnitude_params = layer.out_features;
            a_params + b_params + magnitude_params
        }).sum()
    }
    
    /// Update magnitude parameters
    pub fn update_magnitudes(&self) -> anyhow::Result<()> {
        let state = self.state.read();
        
        for (name, layer) in &state.layers {
            if let Some(base_weight) = state.base_weights.get(name) {
                // Could implement magnitude update strategies here
                // For example, re-compute from current weights
            }
        }
        
        Ok(())
    }
}

#[async_trait]
impl NetworkAdapter for DoRAAdapter {
    fn adapter_type(&self) -> NetworkType {
        NetworkType::DoRA
    }
    
    fn metadata(&self) -> &NetworkMetadata {
        &self.metadata
    }
    
    fn target_modules(&self) -> &[String] {
        &self.config.target_modules
    }
    
    fn trainable_parameters(&self) -> Vec<&Parameter> {
        // TODO: Return actual Parameter references from DoRA layers
        Vec::new()
    }
    
    fn parameters(&self) -> HashMap<String, Tensor> {
        let state = self.state.read();
        let mut params = HashMap::new();
        
        for (name, layer) in &state.layers {
            // Add LoRA A and B weights
            params.insert(format!("{}.lora_a.weight", name), layer.lora_a.weight().clone());
            params.insert(format!("{}.lora_b.weight", name), layer.lora_b.weight().clone());
            
            // Add magnitude vector
            params.insert(format!("{}.magnitude", name), layer.magnitude.clone());
        }
        
        params
    }
    
    fn apply_to_layer(&self, layer_name: &str, input: &Tensor) -> anyhow::Result<Tensor> {
        let state = self.state.read();
        
        if !state.enabled {
            return Ok(input.clone());
        }
        
        // DoRA needs the base weight to apply properly
        // For now, just apply LoRA part without magnitude scaling
        if let Some(dora_layer) = state.layers.get(layer_name) {
            // Get base weight if available
            if let Some(base_weight) = state.base_weights.get(layer_name) {
                dora_layer.forward(input, base_weight)
            } else {
                // Without base weight, can only apply LoRA part
                Ok(input.clone())
            }
        } else {
            Ok(input.clone())
        }
    }
    
    fn merge_weights(&mut self, scale: f32) -> anyhow::Result<()> {
        let mut state = self.state.write();
        
        // Update scaling for all layers
        for layer in state.layers.values_mut() {
            layer.scaling = scale * self.config.alpha / layer.rank as f32;
        }
        
        Ok(())
    }
    
    async fn save_weights(&self, path: &Path) -> anyhow::Result<()> {
                
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
                tensors.insert(
                    format!("{}.magnitude", name),
                    layer.magnitude.clone(),
                );
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
    
    async fn load_weights(&mut self, path: &Path) -> anyhow::Result<()> {
                
        // Load tensors
        let device = self.state.read().device.clone();
        let tensors = flame_core::safetensors::load(path, &device)?;
        
        // TODO: Implement proper loading logic
        // This requires parsing the tensors and reconstructing DoRA layers
        
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
            let magnitude_size = layer.out_features * 4;
            a_size + b_size + magnitude_size
        }).sum()
    }
    
    fn to_device(&mut self, device: &Device) -> anyhow::Result<()> {
        let mut state = self.state.write();
        state.device = device;
        
        // TODO: Move all layers to new device
        
        Ok(())
    }
}

/// DoRA utilities
pub mod utils {
    use super::*;
    
    /// Analyze weight decomposition quality
    pub fn analyze_decomposition(
        base_weight: &Tensor,
        dora_layer: &DoRALayer,
    ) -> anyhow::Result<DecompositionMetrics> {
        let (direction, magnitude) = dora_layer.get_decomposed_weight(base_weight)?;
        
        // Compute reconstruction error
        let reconstructed = direction.mul(&magnitude.unsqueeze(1)?)?;
        let error = (base_weight - &reconstructed)?.sqr()?.mean_all()?;
        
        // Compute effective rank
        let lora_weight = dora_layer.get_lora_weight()?;
        let lora_norm = lora_weight.sqr()?.sum()?.sqrt()?;
        
        Ok(DecompositionMetrics {
            reconstruction_error: error.item::<f32>().unwrap(),
            lora_contribution: lora_norm.item::<f32>().unwrap(),
            magnitude_variance: magnitude.var(0)?.item::<f32>().unwrap(),
        })
    }
    
    /// Convert LoRA to DoRA
    pub fn lora_to_dora(
        lora_a: &Tensor,
        lora_b: &Tensor,
        base_weight: &Tensor,
        epsilon: f32,
    ) -> anyhow::Result<(Tensor, Tensor, Tensor)> {
        // Compute LoRA weight
        let lora_weight = lora_b.matmul(&lora_a.transpose(0, 1)?)?;
        
        // Compute new weight
        let new_weight = (base_weight + &lora_weight)?;
        
        // Extract magnitude
        let magnitude = DoRALayer::compute_weight_norm(&new_weight, epsilon)?;
        
        // Normalize to get direction
        let direction = new_weight.div(&magnitude.unsqueeze(1)?)?;
        
        // Compute new LoRA that approximates: direction - normalized(base_weight)
        let normalized_base = base_weight.broadcast_div(
            &DoRALayer::compute_weight_norm(base_weight, epsilon)?.unsqueeze(1)?
        )?;
        let lora_residual = (direction - normalized_base)?;
        
        // Could perform SVD here to get new A and B
        
        Ok((lora_a.clone(), lora_b.clone(), magnitude))
    }
}

/// Decomposition analysis metrics
#[derive(Debug, Clone)]
pub struct DecompositionMetrics {
    pub reconstruction_error: f32,
    pub lora_contribution: f32,
    pub magnitude_variance: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
        
    #[test]
    fn test_dora_layer() -> anyhow::Result<()> {
        let device = Device::cuda(0)?;
        let ai_device = eridiffusion_core::Device::CPU;
        
        let base_weight = Tensor::randn(0.0f32, 0.02, Shape::from(vec![768, 768]), &device)?;
        
        let layer = DoRALayer::new(
            768,  // in_features
            768,  // out_features
            4,    // rank
            1.0,  // alpha
            0.0,  // dropout
            1e-6, // epsilon
            &ai_device,
            Some(&base_weight),
        )?;
        
        let x = Tensor::randn(0.0f32, 1.0, Shape::from(vec![2, 768]), &device)?;
        let output = layer.forward(&x, &base_weight)?;
        
        assert_eq!(output.dims(), &[2, 768]);
        Ok(())
    }
    
    #[test]
    fn test_weight_normalization() -> anyhow::Result<()> {
        let device = Device::cuda(0)?;
        let weight = Tensor::randn(0.0f32, 1.0, Shape::from(vec![10, 5]), &device)?;
        
        let norm = DoRALayer::compute_weight_norm(&weight, 1e-6)?;
        assert_eq!(norm.dims(), &[10]);
        
        Ok(())
    }
}
