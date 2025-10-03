use crate::loaders::WeightLoader;
use flame_core::{DType, Shape, Tensor, Parameter};
use flame_core::device::Device;
use flame_core::optimizers::{Adam, SGD};
use crate::networks::lora::LoRALinear;
use flame_core::{Result};

use std::collections::HashMap;
pub struct SDXLLoRAUNet {
    inner: super::sdxl_unet_complete::UNet2DConditionModel,
    lora_layers: HashMap<String, LoRALinear>,
    device: Device,
    dtype: DType,
}
pub struct LoRAUNetBuilder {
    unet: super::sdxl_unet_complete::UNet2DConditionModel,
    device: Device,
    dtype: DType,
    lora_config: LoRAConfig,
}
pub struct LoRAConfig {
    pub rank: usize,
pub alpha: f32,
pub dropout: f32,
}

// FLAME uses flame_core::device::Device instead of Device

// FLAME uses flame_core::device::Device instead of Device

// FLAME uses flame_core::device::Device instead of Device

/// SDXL UNet with LoRA injection support

// WeightLoader implementation is in crate::loaders::WeightLoader

#[derive(Clone)]
pub struct PrefixedWeightLoader {
    loader: WeightLoader,
prefix: String,
}

impl PrefixedWeightLoader {
pub fn get(&self, key: &str) -> Result<&Tensor> {
let full_key = format!("{}.{}", self.prefix, key);
self.loader.get(&full_key)
}

pub fn tensor(&self, key: &str, shape: &[usize]) -> Result<Tensor> {
let full_key = format!("{}.{}", self.prefix, key);
self.loader.tensor(&full_key, shape)
}

pub fn pp(&self, prefix: &str) -> PrefixedWeightLoader {
PrefixedWeightLoader {
loader: self.loader.clone(),
            prefix: format!("{}.{}", self.prefix, prefix),
}
}
}

impl SDXLLoRAUNet {
pub fn new(
inner: super::sdxl_unet_complete::UNet2DConditionModel,
device: Device,
dtype: DType,
) -> Self {
Self {
inner,
lora_layers: HashMap::new(),
device,
dtype,
}

/// Add a LoRA layer for a specific module
pub fn add_lora_layer(&mut self, name: String, layer: LoRALinear) -> Result<()> {
self.lora_layers.insert(name, layer);
Ok(())
}

/// Get mutable access to LoRA layers  
pub fn lora_layers_mut(&mut self) -> &mut HashMap<String, LoRALinear> {
    &mut self.lora_layers
}

/// Forward pass with LoRA modifications
pub fn forward_with_lora(
&self,
sample: &Tensor,
timestep: f64,
encoder_hidden_states: &Tensor,
) -> Result<Tensor> {
// For now, we'll use a hook-based approach
// In a full implementation, we'd need to traverse the UNet structure
// and inject LoRA at the right points

// Convert inputs to F32 for FLAME compatibility;
let sample_f32 = sample.to_dtype(DType::F32)?;
let encoder_hidden_states_f32 = encoder_hidden_states.to_dtype(DType::F32)?;

// We need to implement a custom forward that applies LoRA
// For now, let's do a simplified version that modifies attention
let result = self.forward_with_hooks(&sample_f32, timestep, &encoder_hidden_states_f32)?;

// Convert back to original dtype
if result.dtype() != sample.dtype() {
Ok(result.to_dtype(sample.dtype())?)
} else {
Ok(result)
}

/// Custom forward implementation with LoRA hooks
fn forward_with_hooks(
&self,
sample: &Tensor,
timestep: f64,
encoder_hidden_states: &Tensor,
) -> Result<Tensor> {
// This is a simplified implementation
// In reality, we'd need to intercept specific layers

// For now, use the standard forward and we'll implement proper hooks later
Ok(self.inner.forward(sample, timestep, encoder_hidden_states)?)
}

/// Apply LoRA to attention output;
pub fn apply_lora_to_attention(
&self,
module_name: &str,
input: &Tensor,
original_output: &Tensor,
) -> Result<Tensor> {
if let Some(lora_layer) = self.lora_layers.get(module_name) {
// Apply LoRA: output = original_output + lora_scale * lora(input)
let lora_output = lora_layer.forward(input)?;
Ok(original_output.add(&lora_output)?)
} else {
Ok(original_output.clone())
}
}

    }
}

/// Builder for creating LoRA-injected UNet

impl LoRAUNetBuilder {
    pub fn new(
        unet: super::sdxl_unet_complete::UNet2DConditionModel,
        device: Device,
        dtype: DType,
        lora_config: LoRAConfig,
    ) -> Self {
        Self {
            unet,
            device,
            dtype,
            lora_config,
        }
    }

    /// Build the LoRA-injected UNet
    pub fn build(self, device: &Device) -> Result<SDXLLoRAUNet> {
        let device = self.device;
        let dtype = self.dtype;
        let lora_config = self.lora_config.clone();

        let mut lora_unet = SDXLLoRAUNet::new(self.unet, device.clone(), dtype);

        // Create LoRA layers for target modules
        for (i, module_name) in lora_config.target_modules.iter().enumerate() {
            let layer_name = format!("lora_unet_{}_{}", module_name.replace(".", "_"), i);

            // Get dimensions based on module type - hardcode for now
            let (in_features, out_features) = Self::get_module_dimensions_static(module_name);

            let lora_layer = LoRALinear::new(
                in_features,
                out_features,
                lora_config.rank,
                lora_config.alpha,
                lora_config.dropout,
                &device,
            )?;

            lora_unet.add_lora_layer(module_name.clone(), lora_layer)?;
        }

        Ok(lora_unet)
    }

    /// Get input/output dimensions for a module
    fn get_module_dimensions_static(module_name: &str) -> (usize, usize) {
        // SDXL attention dimensions
        match module_name {
            name if name.contains("to_k") || name.contains("to_v") => {
                if name.contains("down_blocks.0") {
                    (320, 320)
                } else if name.contains("down_blocks.1") {
                    (640, 640)
                } else if name.contains("down_blocks.2") || name.contains("mid_block") {
                    (1280, 1280)
                } else if name.contains("up_blocks.0") {
                    (1280, 1280)
                } else if name.contains("up_blocks.1") {
                    (640, 640)
                } else if name.contains("up_blocks.2") {
                    (320, 320)
                } else {
                    (1280, 1280) // default
                }
            }
            name if name.contains("to_q") => {
                // to_q dimensions match to_k/to_v
                Self::get_module_dimensions_static(&name.replace("to_q", "to_k"))
            }
            name if name.contains("to_out") => {
                // to_out dimensions match input
                Self::get_module_dimensions_static(&name.replace("to_out", "to_k"))
            }
            _ => (1280, 1280), // default fallback
        }
    }
}

impl crate::flame_training::FLAMEModel for SDXLLoRAUNet {
    fn parameters(&self) -> Vec<&Parameter> {
        // Collect all parameters from the model
        let mut params = Vec::new();
        // TODO: Add actual parameter collection from inner UNet and LoRA layers
        params
    }

    fn named_parameters(&self) -> HashMap<String, &Parameter> {
        // Collect named parameters
        let mut params = HashMap::new();
        // TODO: Add actual named parameter collection from inner UNet and LoRA layers
        params
    }
}
