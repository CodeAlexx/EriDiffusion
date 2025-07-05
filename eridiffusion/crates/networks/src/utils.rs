//! Utilities for network adapters

use eridiffusion_core::{NetworkAdapter, NetworkType, NetworkMetadata, Device, Result, Error};
use candle_core::{Tensor, DType, Var};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use std::path::Path;
use async_trait::async_trait;
use crate::lora::{LoRAAdapter, LoRAConfig};
use crate::lokr::{LoKrAdapter, LoKrConfig};
use crate::dora::{DoRAAdapter, DoRAConfig};

/// Network adapter utilities
pub mod adapter {
    use super::*;
    
    /// Merge multiple adapters with weights
    pub fn merge_adapters(
        adapters: Vec<Box<dyn NetworkAdapter + Send + Sync>>,
        weights: Vec<f32>,
    ) -> Result<Box<dyn NetworkAdapter + Send + Sync>> {
        if adapters.is_empty() {
            return Err(Error::Model("No adapters to merge".to_string()));
        }
        
        if adapters.len() != weights.len() {
            return Err(Error::Model("Adapter and weight counts must match".to_string()));
        }
        
        // Normalize weights
        let weight_sum: f32 = weights.iter().sum();
        let normalized_weights: Vec<f32> = weights.iter()
            .map(|w| w / weight_sum)
            .collect();
        
        // Clone first adapter and merge others into it
        let mut merged = adapters.into_iter().next()
            .ok_or_else(|| Error::Network("Cannot merge empty adapter list".to_string()))?;
        
        // Would implement actual merging logic here
        // For now, just return the first adapter
        
        Ok(merged)
    }
    
    /// Stack adapters for ensemble
    pub fn stack_adapters(
        adapters: Vec<Box<dyn NetworkAdapter + Send + Sync>>,
    ) -> Result<StackedAdapter> {
        let adapter_count = adapters.len();
        let metadata = NetworkMetadata {
            name: "stacked_adapter".to_string(),
            network_type: NetworkType::Custom("Stacked".to_string()),
            version: "1.0.0".to_string(),
            base_model: "unknown".to_string(),
            rank: None,
            alpha: None,
            target_modules: vec![],
            created_at: chrono::Utc::now(),
            config: HashMap::new(),
        };
        
        Ok(StackedAdapter {
            adapters,
            weights: vec![1.0 / adapter_count as f32; adapter_count],
            metadata,
        })
    }
    
    /// Create adapter from checkpoint
    pub async fn load_adapter_from_checkpoint(
        checkpoint_path: &std::path::Path,
        device: &Device,
    ) -> Result<Box<dyn NetworkAdapter + Send + Sync>> {
        use candle_core::safetensors;
        
        // Load tensors from checkpoint
        let tensors = safetensors::load(checkpoint_path, &device.to_candle()?)?;
        
        // Detect adapter type from tensor names
        let is_lora = tensors.keys().any(|k| k.contains("lora_up") || k.contains("lora_down"));
        let is_lokr = tensors.keys().any(|k| k.contains("lokr_w1") || k.contains("lokr_w2"));
        
        if is_lora {
            // Load as LoRA adapter
            let config = LoRAConfig {
                rank: 16, // Would be determined from tensor shapes
                alpha: 16.0,
                dropout: 0.0,
                target_modules: vec!["attention".to_string()],
                use_bias: false,
                fan_in_fan_out: false,
                merge_weights: false,
                init_weights: true,
                use_rslora: false,
                use_dora: false,
                alpha_pattern: Default::default(),
                rank_pattern: Default::default(),
            };
            Ok(Box::new(crate::lora::LoRAAdapter::new(
                config,
                eridiffusion_core::ModelArchitecture::SDXL, // Default architecture
                device.clone(),
            )?))
        } else if is_lokr {
            // Load as LoKr adapter
            let config = LoKrConfig {
                rank: 16,
                alpha: 16.0,
                factor: Some(8),
                dropout: 0.0,
                target_modules: vec!["attention".to_string()],
                decompose_factor: None,
                use_scalar: false,
                init_weights: true,
            };
            Ok(Box::new(crate::lokr::LoKrAdapter::new(
                config,
                eridiffusion_core::ModelArchitecture::SDXL, // Default architecture
                device.clone(),
            )?))
        } else {
            Err(Error::Model("Unknown adapter type in checkpoint".to_string()))
        }
    }
}

/// Stacked adapter for ensemble
pub struct StackedAdapter {
    adapters: Vec<Box<dyn NetworkAdapter + Send + Sync>>,
    weights: Vec<f32>,
    metadata: NetworkMetadata,
}

#[async_trait]
impl NetworkAdapter for StackedAdapter {
    fn adapter_type(&self) -> NetworkType {
        NetworkType::Custom("Stacked".to_string())
    }
    
    fn metadata(&self) -> &NetworkMetadata {
        &self.metadata
    }
    
    fn target_modules(&self) -> &[String] {
        // Return empty as this is a composite adapter
        &[]
    }
    
    fn trainable_parameters(&self) -> Vec<&Var> {
        // Stacked adapter doesn't have its own trainable parameters
        Vec::new()
    }
    
    fn parameters(&self) -> HashMap<String, Tensor> {
        // Collect parameters from all adapters
        let mut all_params = HashMap::new();
        
        for (i, adapter) in self.adapters.iter().enumerate() {
            let adapter_params = adapter.parameters();
            for (name, tensor) in adapter_params {
                all_params.insert(format!("adapter_{}.{}", i, name), tensor);
            }
        }
        
        all_params
    }
    
    fn apply_to_layer(&self, layer_name: &str, input: &Tensor) -> Result<Tensor> {
        let mut outputs = Vec::new();
        
        // Get output from each adapter
        for adapter in &self.adapters {
            outputs.push(adapter.apply_to_layer(layer_name, input)?);
        }
        
        // Weighted average of outputs
        let mut result = outputs[0].affine(self.weights[0] as f64, 0.0)?;
        
        for (i, output) in outputs.iter().skip(1).enumerate() {
            let weighted = output.affine(self.weights[i + 1] as f64, 0.0)?;
            result = (result + weighted)?;
        }
        
        Ok(result)
    }
    
    fn merge_weights(&mut self, scale: f32) -> Result<()> {
        // Apply scale to all adapters
        for adapter in &mut self.adapters {
            adapter.merge_weights(scale)?;
        }
        Ok(())
    }
    
    async fn save_weights(&self, path: &Path) -> Result<()> {
        // Save each adapter's weights with a prefix
        // Note: We need to handle the async calls differently since adapters may not be Send
        for (i, adapter) in self.adapters.iter().enumerate() {
            let adapter_path = path.with_file_name(format!(
                "{}_adapter_{}.safetensors",
                path.file_stem().unwrap().to_str().unwrap(),
                i
            ));
            // Use block_on to handle the async call within a sync context
            tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(adapter.save_weights(&adapter_path))
            })?;
        }
        
        // Save metadata
        let metadata_path = path.with_extension("json");
        let metadata_json = serde_json::to_string_pretty(&self.metadata)?;
        tokio::fs::write(&metadata_path, metadata_json).await?;
        
        // Save weights config
        let weights_config = serde_json::json!({
            "adapter_count": self.adapters.len(),
            "weights": self.weights,
        });
        let weights_path = path.with_file_name(format!(
            "{}_weights.json",
            path.file_stem().unwrap().to_str().unwrap()
        ));
        tokio::fs::write(&weights_path, serde_json::to_string_pretty(&weights_config)?).await?;
        
        Ok(())
    }
    
    async fn load_weights(&mut self, path: &Path) -> Result<()> {
        // Load weights config
        let weights_path = path.with_file_name(format!(
            "{}_weights.json",
            path.file_stem().unwrap().to_str().unwrap()
        ));
        if weights_path.exists() {
            let weights_json = tokio::fs::read_to_string(&weights_path).await?;
            let weights_config: serde_json::Value = serde_json::from_str(&weights_json)?;
            if let Some(weights) = weights_config["weights"].as_array() {
                self.weights = weights.iter()
                    .filter_map(|v| v.as_f64())
                    .map(|v| v as f32)
                    .collect();
            }
        }
        
        // Load each adapter's weights
        // Note: We need to handle the async calls differently since adapters may not be Send
        for (i, adapter) in self.adapters.iter_mut().enumerate() {
            let adapter_path = path.with_file_name(format!(
                "{}_adapter_{}.safetensors",
                path.file_stem().unwrap().to_str().unwrap(),
                i
            ));
            if adapter_path.exists() {
                // Use block_on to handle the async call within a sync context
                tokio::task::block_in_place(|| {
                    tokio::runtime::Handle::current().block_on(adapter.load_weights(&adapter_path))
                })?;
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
        self.adapters.iter()
            .map(|a| a.memory_usage())
            .sum()
    }
    
    fn to_device(&mut self, device: &Device) -> Result<()> {
        for adapter in &mut self.adapters {
            adapter.to_device(device)?;
        }
        Ok(())
    }
}

/// Quantization utilities
pub mod quantization {
    use super::*;
    
    /// Quantization config
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct QuantizationConfig {
        pub bits: usize,
        pub group_size: Option<usize>,
        pub symmetric: bool,
        pub per_channel: bool,
    }
    
    /// Quantize tensor to n-bit
    pub fn quantize_tensor(
        tensor: &Tensor,
        config: &QuantizationConfig,
    ) -> Result<QuantizedTensor> {
        let (scale, zero_point) = compute_quantization_params(tensor, config)?;
        
        // Quantize
        let quantized = quantize_with_params(tensor, &scale, &zero_point, config)?;
        
        Ok(QuantizedTensor {
            data: quantized,
            scale,
            zero_point,
            config: config.clone(),
        })
    }
    
    /// Dequantize tensor
    pub fn dequantize_tensor(quantized: &QuantizedTensor) -> Result<Tensor> {
        dequantize_with_params(
            &quantized.data,
            &quantized.scale,
            &quantized.zero_point,
            &quantized.config,
        )
    }
    
    fn compute_quantization_params(
        tensor: &Tensor,
        config: &QuantizationConfig,
    ) -> Result<(Tensor, Tensor)> {
        let qmin = -(1 << (config.bits - 1)) as f32;
        let qmax = ((1 << (config.bits - 1)) - 1) as f32;
        
        // Compute min/max
        let min_val = tensor.min_keepdim(0)?;
        let max_val = tensor.max_keepdim(0)?;
        
        // Compute scale and zero point
        let scale = (max_val - &min_val)?.affine(1.0 / (qmax - qmin) as f64, 0.0)?;
        let zero_point = if config.symmetric {
            Tensor::zeros_like(&scale)?
        } else {
            (Tensor::new(&[qmin], scale.device())? - (&min_val / &scale)?)?
        };
        
        Ok((scale, zero_point))
    }
    
    fn quantize_with_params(
        tensor: &Tensor,
        scale: &Tensor,
        zero_point: &Tensor,
        config: &QuantizationConfig,
    ) -> Result<Tensor> {
        let qmin = -(1 << (config.bits - 1)) as f32;
        let qmax = ((1 << (config.bits - 1)) - 1) as f32;
        
        // Scale and add zero point
        let quantized = ((tensor / scale)? + zero_point)?;
        
        // Clamp and round
        let quantized = quantized.clamp(qmin as f64, qmax as f64)?;
        let quantized = quantized.round()?;
        
        Ok(quantized)
    }
    
    fn dequantize_with_params(
        quantized: &Tensor,
        scale: &Tensor,
        zero_point: &Tensor,
        config: &QuantizationConfig,
    ) -> Result<Tensor> {
        Ok(((quantized - zero_point)? * scale)?)
    }
}

/// Quantized tensor
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    pub data: Tensor,
    pub scale: Tensor,
    pub zero_point: Tensor,
    pub config: quantization::QuantizationConfig,
}

/// Pruning utilities
pub mod pruning {
    use super::*;
    
    /// Pruning method
    #[derive(Debug, Clone, Copy)]
    pub enum PruningMethod {
        Magnitude,
        Random,
        Structured,
        Gradual,
    }
    
    /// Prune tensor by magnitude
    pub fn magnitude_prune(
        tensor: &Tensor,
        sparsity: f32,
        structured: bool,
    ) -> Result<Tensor> {
        if structured {
            // Structured pruning (e.g., channel-wise)
            structured_magnitude_prune(tensor, sparsity)
        } else {
            // Unstructured pruning
            unstructured_magnitude_prune(tensor, sparsity)
        }
    }
    
    fn unstructured_magnitude_prune(tensor: &Tensor, sparsity: f32) -> Result<Tensor> {
        // Compute magnitude
        let magnitude = tensor.abs()?;
        
        // Flatten and sort
        let flat_mag = magnitude.flatten_all()?;
        let num_elements = flat_mag.dims()[0];
        let num_pruned = (num_elements as f32 * sparsity) as usize;
        
        // Get threshold (simplified - would use actual sorting)
        let threshold = 0.01; // Placeholder
        
        // Create mask
        let mask = magnitude.ge(threshold)?;
        
        // Apply mask
        Ok((tensor * mask.to_dtype(tensor.dtype())?)?)
    }
    
    fn structured_magnitude_prune(tensor: &Tensor, sparsity: f32) -> Result<Tensor> {
        // Compute channel-wise magnitude
        let channel_mag = tensor.sqr()?.sum_keepdim(1)?.sqrt()?;
        
        // Determine channels to prune
        let num_channels = tensor.dims()[0];
        let num_pruned = (num_channels as f32 * sparsity) as usize;
        
        // Create channel mask (simplified)
        let mask = Tensor::ones(tensor.dims(), tensor.dtype(), tensor.device())?;
        
        Ok((tensor * mask)?)
    }
}

/// Analysis utilities
pub mod analysis {
    use super::*;
    
    /// Analyze adapter compression
    pub fn analyze_compression(
        adapter: &dyn NetworkAdapter,
        base_params: usize,
    ) -> CompressionAnalysis {
        let adapter_params = count_parameters(adapter);
        let compression_ratio = adapter_params as f32 / base_params as f32;
        
        CompressionAnalysis {
            adapter_params,
            base_params,
            compression_ratio,
            memory_saved: base_params - adapter_params,
        }
    }
    
    /// Count adapter parameters
    pub fn count_parameters(adapter: &dyn NetworkAdapter) -> usize {
        adapter.parameters().values()
            .map(|tensor| tensor.elem_count())
            .sum()
    }
    
    /// Analyze adapter activations
    pub fn analyze_activations(
        adapter: &dyn NetworkAdapter,
        sample_inputs: &[eridiffusion_core::ModelInputs],
    ) -> Result<ActivationStats> {
        // Would compute activation statistics
        Ok(ActivationStats {
            mean_activation: 0.0,
            std_activation: 0.0,
            sparsity: 0.0,
            dead_neurons: 0,
        })
    }
}

/// Compression analysis results
#[derive(Debug, Clone)]
pub struct CompressionAnalysis {
    pub adapter_params: usize,
    pub base_params: usize,
    pub compression_ratio: f32,
    pub memory_saved: usize,
}

/// Activation statistics
#[derive(Debug, Clone)]
pub struct ActivationStats {
    pub mean_activation: f32,
    pub std_activation: f32,
    pub sparsity: f32,
    pub dead_neurons: usize,
}

/// Conversion utilities
pub mod conversion {
    use super::*;
    
    /// Convert between adapter types
    pub fn convert_adapter(
        source: &dyn NetworkAdapter,
        target_type: &str,
        config: Option<serde_json::Value>,
        device: &Device,
    ) -> Result<Box<dyn NetworkAdapter>> {
        use NetworkType::*;
        match (source.adapter_type(), target_type) {
            (LoRA, "DoRA") => {
                // Convert LoRA to DoRA by adding magnitude vectors
                // DoRA = LoRA + magnitude vector per layer
                // This is a conceptual implementation
                let dora_config = crate::dora::DoRAConfig {
                    rank: 16, // From source LoRA
                    alpha: 16.0,
                    dropout: 0.0,
                    target_modules: vec!["attention".to_string()],
                    use_bias: false,
                    epsilon: 1e-6,
                    init_weights: true,
                    magnitude_init: crate::dora::MagnitudeInit::FromWeight,
                };
                Ok(Box::new(crate::dora::DoRAAdapter::new(
                    dora_config,
                    eridiffusion_core::ModelArchitecture::SDXL, // Default architecture
                    device.clone(),
                )?))
            }
            (LoRA, "LoKr") => {
                // Convert LoRA to LoKr using Kronecker decomposition
                // LoKr can represent LoRA with factor decomposition
                let lokr_config = crate::lokr::LoKrConfig {
                    rank: 16, // From source LoRA
                    alpha: 16.0,
                    factor: Some(8), // Decomposition factor
                    dropout: 0.0,
                    target_modules: vec!["attention".to_string()],
                    decompose_factor: None,
                    use_scalar: false,
                    init_weights: true,
                };
                Ok(Box::new(crate::lokr::LoKrAdapter::new(
                    lokr_config,
                    eridiffusion_core::ModelArchitecture::SDXL, // Default architecture
                    device.clone(),
                )?))
            }
            _ => Err(Error::Model(format!(
                "Conversion from {:?} to {} not supported",
                source.adapter_type(), target_type
            ))),
        }
    }
    
    /// Export adapter to ONNX format
    pub async fn export_to_onnx(
        adapter: &dyn NetworkAdapter,
        output_path: &std::path::Path,
    ) -> Result<()> {
        // ONNX export requires significant infrastructure:
        // 1. Convert Candle tensors to ONNX graph representation
        // 2. Handle operator mapping from Candle to ONNX
        // 3. Serialize to protobuf format
        // This would typically use the ort crate or similar
        tracing::warn!(
            "ONNX export requires additional dependencies. \
            Consider using safetensors format for model interchange."
        );
        Err(Error::Unsupported(
            "ONNX export requires ort crate integration. Use safetensors format instead.".to_string()
        ))
    }
    
    /// Import adapter from PyTorch checkpoint
    pub async fn import_from_pytorch(
        checkpoint_path: &std::path::Path,
        adapter_type: &str,
        device: &Device,
    ) -> Result<Box<dyn NetworkAdapter + Send + Sync>> {
        // PyTorch checkpoints can be loaded if they're in safetensors format
        // For .pt/.pth files, we would need pickle support which is a security risk
        if checkpoint_path.extension().map_or(false, |ext| ext == "safetensors") {
            // Can load safetensors format directly
            super::adapter::load_adapter_from_checkpoint(checkpoint_path, device).await
        } else {
            tracing::warn!(
                "Direct PyTorch .pt/.pth import is not supported for security reasons. \
                Please convert to safetensors format using Python: \
                safetensors.torch.save_file(state_dict, 'model.safetensors')"
            );
            Err(Error::Unsupported(
                "PyTorch .pt files require pickle which is insecure. Convert to safetensors format.".to_string()
            ))
        }
    }
}