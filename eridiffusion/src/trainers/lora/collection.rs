use flame_core::device::Device;
use flame_core::Parameter;
use flame_core::Result;
use flame_core::{DType, Tensor};
use log::debug;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::adapter::{LoRAAdapter, SimpleLoRA};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoRAConfig {
    pub rank: usize,
    pub alpha: f32,
    pub dropout: Option<f32>,
    pub target_modules: Vec<String>,
    pub dtype: String,
}

impl Default for LoRAConfig {
    fn default() -> Self {
        Self {
            rank: 16,
            alpha: 16.0,
            dropout: None,
            target_modules: vec![
                "to_q".to_string(),
                "to_k".to_string(),
                "to_v".to_string(),
                "to_out.0".to_string(),
            ],
            dtype: "f32".to_string(),
        }
    }
}

/// Collection of LoRA adapters for a model
#[derive(Clone)]
pub struct LoRACollection {
    pub rank: usize,
    pub alpha: f32,
    pub dtype: DType,
    pub adapters: HashMap<String, SimpleLoRA>,
    pub config: LoRAConfig,
}

impl LoRACollection {
    pub fn new(config: LoRAConfig, device: &Device) -> flame_core::Result<Self> {
        let dtype = match config.dtype.as_str() {
            "f16" => DType::F16,
            "bf16" => DType::BF16,
            _ => DType::F32,
        };

        Ok(Self { rank: config.rank, alpha: config.alpha, dtype, adapters: HashMap::new(), config })
    }

    /// Add a LoRA adapter for a specific layer
    pub fn add_adapter(
        &mut self,
        name: &str,
        in_features: usize,
        out_features: usize,
        device: &Device,
    ) -> flame_core::Result<()> {
        let adapter =
            SimpleLoRA::new(in_features, out_features, self.rank, self.alpha, device, self.dtype)?;

        self.adapters.insert(name.to_string(), adapter);
        Ok(())
    }

    /// Get adapter by name
    pub fn get_adapter(&self, name: &str) -> Option<&SimpleLoRA> {
        self.adapters.get(name)
    }

    /// Get mutable adapter by name
    pub fn get_adapter_mut(&mut self, name: &str) -> Option<&mut SimpleLoRA> {
        self.adapters.get_mut(name)
    }

    /// Get all trainable parameters
    pub fn parameters(&self) -> Vec<&Parameter> {
        let mut params = Vec::new();
        for adapter in self.adapters.values() {
            params.extend(adapter.parameters());
        }
        params
    }

    /// Get all trainable parameters (owned)
    pub fn trainable_parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        for adapter in self.adapters.values() {
            params.push(adapter.down.clone());
            params.push(adapter.up.clone());
        }
        params
    }

    /// Get adapter count
    pub fn adapter_count(&self) -> usize {
        self.adapters.len()
    }

    /// Get merged weights for inference
    pub fn get_merged_weights(&self) -> flame_core::Result<HashMap<String, Tensor>> {
        let mut merged_weights = HashMap::new();

        for (name, adapter) in &self.adapters {
            // Compute the LoRA weight: alpha/rank * down @ up
            let scale = self.alpha / self.rank as f32;
            let lora_weight =
                adapter.down.tensor()?.matmul(&adapter.up.tensor()?)?.mul_scalar(scale)?;

            // Store the merged weight with the adapter name
            merged_weights.insert(name.clone(), lora_weight);
        }

        Ok(merged_weights)
    }

    /// Apply LoRA adapter to a tensor with base weights
    pub fn apply(
        &self,
        adapter_name: &str,
        input: &Tensor,
        base_weight: &Tensor,
        base_bias: Option<&Tensor>,
    ) -> flame_core::Result<Tensor> {
        // Apply base transformation first
        let mut output = input.matmul(base_weight)?;
        if let Some(bias) = base_bias {
            output = output.add(bias)?;
        }

        // Add LoRA if adapter exists
        if let Some(adapter) = self.get_adapter(adapter_name) {
            let lora_output = adapter.forward(input)?;
            output = output.add(&lora_output)?;
        }

        Ok(output)
    }

    /// Create LoRACollection from SDXL weights
    pub fn from_sdxl_weights(
        weights: &HashMap<String, Tensor>,
        rank: usize,
        alpha: f32,
        device: &Device,
        dtype: DType,
    ) -> flame_core::Result<Self> {
        let config = LoRAConfig {
            rank,
            alpha,
            dropout: Some(0.0),
            target_modules: vec!["attn".to_string()],
            dtype: match dtype {
                DType::F16 => "f16".to_string(),
                DType::BF16 => "bf16".to_string(),
                _ => "f32".to_string(),
            },
        };

        let mut collection = Self::new(config, device)?;

        // Pattern: Find all attention projection layers
        for (name, weight) in weights {
            if name.contains("attn")
                & (name.contains("to_k")
                    || name.contains("to_v")
                    || name.contains("to_q")
                    || name.contains("to_out.0"))
            {
                let shape = weight.shape();
                if shape.rank() >= 2 {
                    let out_features = shape.dims()[0];
                    let in_features = shape.dims()[0];

                    collection.add_adapter(name, in_features, out_features, device)?;
                    debug!("Created LoRA adapter for {}: {}x{}", name, in_features, out_features);
                }
            }
        }

        Ok(collection)
    }

    /// Create LoRA adapters for SDXL UNet
    pub fn create_sdxl_adapters(&mut self, device: &Device) -> flame_core::Result<()> {
        // Attention blocks dimensions for SDXL
        let attention_dims = [
            (320, 320),   // input blocks 1, 2
            (640, 640),   // input blocks 4, 5
            (1280, 1280), // input blocks 7, 8
            (1280, 1280), // middle block
            (1280, 1280), // output blocks 0, 1, 2
            (640, 640),   // output blocks 3, 4, 5
            (320, 320),   // output blocks 6, 7, 8
        ];

        // Create adapters for each attention layer
        let blocks = [
            ("input_blocks", vec![1, 2, 4, 5, 7, 8]),
            ("middle_block", vec![1]),
            ("output_blocks", vec![0, 1, 2, 3, 4, 5, 6, 7, 8]),
        ];

        for (block_type, indices) in blocks.iter() {
            for &idx in indices {
                let dim_idx = match block_type {
                    &"input_blocks" => {
                        if idx <= 2 {
                            0
                        } else if idx <= 5 {
                            1
                        } else {
                            2
                        }
                    }
                    &"middle_block" => 3,
                    &"output_blocks" => {
                        if idx <= 2 {
                            4
                        } else if idx <= 5 {
                            5
                        } else {
                            6
                        }
                    }
                    _ => 0,
                };

                let (in_dim, out_dim) = attention_dims[dim_idx];

                // Create adapters for each attention module
                for module in &["to_q", "to_k", "to_v", "to_out.0"] {
                    let name =
                        format!("{}.{}.1.transformer_blocks.0.attn1.{}", block_type, idx, module);
                    self.add_adapter(&name, in_dim, out_dim, device)?;

                    // Also add cross-attention adapters
                    let cross_name =
                        format!("{}.{}.1.transformer_blocks.0.attn2.{}", block_type, idx, module);
                    self.add_adapter(&cross_name, in_dim, out_dim, device)?;
                }
            }
        }

        Ok(())
    }

    /// Save LoRA weights to safetensors format
    pub fn save_weights(&self, path: &std::path::Path) -> flame_core::Result<()> {
        use safetensors::{serialize, tensor::TensorView, Dtype as SafeDtype};

        let mut tensors = HashMap::new();

        for (name, adapter) in &self.adapters {
            // Convert to CPU for saving
            let down_cpu = adapter.down.tensor()?;
            let up_cpu = adapter.up.tensor()?;

            // Add to tensors map
            let down_name = format!("{}.lora_down.weight", name);
            let up_name = format!("{}.lora_up.weight", name);

            let down_data = down_cpu.to_vec1::<f32>()?;
            let up_data = up_cpu.to_vec1::<f32>()?;

            let down_shape = down_cpu.shape().dims().to_vec();
            let up_shape = up_cpu.shape().dims().to_vec();

            // Convert f32 vectors to byte slices
            let down_bytes = unsafe {
                std::slice::from_raw_parts(down_data.as_ptr() as *const u8, down_data.len() * 4)
            };
            let up_bytes = unsafe {
                std::slice::from_raw_parts(up_data.as_ptr() as *const u8, up_data.len() * 4)
            };

            tensors.insert(
                down_name,
                TensorView::new(SafeDtype::F32, down_shape, down_bytes)
                    .map_err(|e| flame_core::Error::InvalidOperation(e.to_string()))?,
            );

            tensors.insert(
                up_name,
                TensorView::new(SafeDtype::F32, up_shape, up_bytes)
                    .map_err(|e| flame_core::Error::InvalidOperation(e.to_string()))?,
            );
        }

        // Add metadata
        let metadata = HashMap::from([
            ("format".to_string(), "pt".to_string()),
            ("rank".to_string(), self.rank.to_string()),
            ("alpha".to_string(), self.alpha.to_string()),
        ]);

        let data = serialize(tensors, &Some(metadata))
            .map_err(|e| flame_core::Error::InvalidOperation(e.to_string()))?;
        std::fs::write(path, data)
            .map_err(|e| flame_core::Error::InvalidOperation(e.to_string()))?;

        Ok(())
    }
}
