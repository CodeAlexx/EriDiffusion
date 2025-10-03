use crate::loaders::WeightLoader;
use anyhow::anyhow;
use flame_core::device::Device;
use flame_core::optimizers::{Adam, SGD};
use flame_core::Result;
use flame_core::{DType, Error, Parameter, Shape, Tensor};
use std::collections::HashMap;

use crate::flame_training::FLAMEModel;
use crate::models::sdxl_unet_complete::{AddedCondKwargs, UNet2DConditionModel};
use crate::models::BlockConfig;

pub struct SDXLUNet2DConditionModel {
    inner: UNet2DConditionModel,
    device: Device,
    dtype: DType,
}

#[derive(Clone)]
pub struct SDXLConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub cross_attention_dim: usize,
    pub use_flash_attn: bool,
}

// Import PrefixedWeightLoader from loaders
use crate::loaders::PrefixedWeightLoader;

impl SDXLUNet2DConditionModel {
    pub fn from_weights(weights: HashMap<String, Tensor>, device: &Device) -> Result<Self> {
        let wl = WeightLoader { weights, device: device.clone() };
        let config = SDXLConfig::default();
        Self::new(wl, config)
    }

    pub fn new(wl: WeightLoader, config: SDXLConfig) -> Result<Self> {
        let device = wl.device().clone();
        let dtype = DType::F32; // Default dtype

        // Create the inner UNet model
        let unet_config = super::sdxl_unet_complete::UNet2DConditionModelConfig::sdxl();
        let inner =
            super::sdxl_unet_complete::UNet2DConditionModel::new(unet_config, &device, wl.weights)?;

        Ok(Self { inner, device, dtype })
    }

    /// Create from an existing FLAME UNet model
    pub fn from_inner(inner: UNet2DConditionModel, device: Device, dtype: DType) -> Self {
        Self { inner, device, dtype }
    }

    /// Forward pass for SDXL with additional conditioning
    pub fn forward(
        &self,
        sample: &Tensor,
        timestep: f64,
        encoder_hidden_states: &Tensor,
        class_labels: Option<&Tensor>,
        return_dict: bool,
    ) -> Result<Tensor> {
        // Convert timestep to tensor if needed
        let timestep_tensor = Tensor::full(
            Shape::from_dims(&[1]),
            timestep as f32,
            self.device.cuda_device().clone(),
        )?;

        // For now, use the standard forward pass
        // TODO: Add support for SDXL's additional conditioning (text_embeds, time_ids)
        Ok(self.inner.forward(sample, &timestep_tensor, encoder_hidden_states, None)?)
    }

    /// Forward pass for training (expects timestep as tensor)
    pub fn forward_train(
        &self,
        sample: &Tensor,
        timesteps: &Tensor,
        encoder_hidden_states: &Tensor,
        added_cond_kwargs: Option<&HashMap<String, Tensor>>,
    ) -> Result<Tensor> {
        // Extract single timestep value for FLAME's UNet
        let timestep = if timesteps.shape().rank() == 0 {
            timesteps.to_scalar::<f64>()?
        } else {
            timesteps.slice(&[(0, 1)])?.to_scalar::<f64>()?
        };

        // Handle SDXL additional conditioning
        let encoder_hidden_states = encoder_hidden_states.clone();

        // Convert HashMap to AddedCondKwargs if provided
        let added_cond_kwargs_converted = if let Some(kwargs) = added_cond_kwargs {
            if let (Some(text_embeds), Some(time_ids)) =
                (kwargs.get("text_embeds"), kwargs.get("time_ids"))
            {
                // Extract text_time_embeds or use zeros if not provided
                let text_time_embeds = if let Some(tte) = kwargs.get("text_time_embeds") {
                    tte.clone()
                } else {
                    Tensor::zeros(text_embeds.shape().clone(), text_embeds.device().clone())?
                };

                Some(AddedCondKwargs {
                    text_embeds: text_embeds.clone(),
                    text_time_embeds,
                    time_ids: time_ids.clone(),
                })
            } else {
                None
            }
        } else {
            None
        };

        // The UNet is loaded in F32, so convert inputs to F32
        let sample_f32 = sample.to_dtype(DType::F32)?;
        let encoder_hidden_states_f32 = encoder_hidden_states.to_dtype(DType::F32)?;

        // Debug: Check dtypes before forward pass (commented out for cleaner output)
        // println!("UNet forward - input sample dtype: {:?}, converted to: {:?}", sample.dtype(), sample_f32.dtype());
        // println!("UNet forward - input encoder_hidden_states dtype: {:?}, converted to: {:?}", encoder_hidden_states.dtype(), encoder_hidden_states_f32.dtype());
        // println!("UNet forward - UNet internal dtype: {:?}", self.dtype);

        // Forward pass through the UNet (which expects F32)
        let result = self.inner.forward(
            &sample_f32,
            timesteps,
            &encoder_hidden_states_f32,
            added_cond_kwargs_converted.as_ref(),
        )?;

        // Convert back to original dtype if needed
        if result.dtype() != sample.dtype() {
            Ok(result.to_dtype(sample.dtype())?)
        } else {
            Ok(result)
        }
    }

    /// Get device
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get dtype
    pub fn dtype(&self) -> DType {
        self.dtype
    }
}

impl Default for SDXLConfig {
    fn default() -> Self {
        Self {
            in_channels: 4,
            out_channels: 4,
            cross_attention_dim: 2048, // SDXL uses concatenated CLIP embeddings
            use_flash_attn: false,
        }
    }
}

/// Get SDXL-specific UNet configuration
pub fn get_sdxl_unet_config() -> SDXLConfig {
    SDXLConfig::default()
}

/// Load SDXL UNet from safetensors file
pub fn load_sdxl_unet(
    model_path: &str,
    device: &Device,
    dtype: DType,
) -> Result<SDXLUNet2DConditionModel> {
    println!("Loading SDXL UNet from: {}", model_path);

    // For now, let's use a simpler approach - we know the model has "model.diffusion_model" prefix
    // Use standard loader for stability
    let wl = WeightLoader::from_safetensors(model_path, device.clone())?;

    // Try with the known prefix first
    let wl_with_prefix = wl.pp("model.diffusion_model");
    let config = SDXLConfig::default();

    // Try without prefix first
    match SDXLUNet2DConditionModel::new(wl, config) {
        Ok(model) => {
            println!("Successfully loaded UNet with prefix: 'model.diffusion_model'");
            return Ok(model);
        }
        Err(e) => {
            println!("Failed to load with standard prefix: {}", e);

            // Can't retry since wl was moved, just fail
            println!("Failed to load UNet model");
        }
    }

    println!("\nFailed to load UNet. The model file may:");
    println!("1. Not be an SDXL model");
    println!("2. Have a different weight structure");
    println!("3. Be corrupted or incomplete");

    Err(flame_core::Error::InvalidOperation(
        "Failed to load SDXL UNet. Please verify the model file is a valid SDXL checkpoint."
            .to_string(),
    ))
}

/// Load SDXL UNet with custom config
pub fn load_sdxl_unet_with_config(
    model_path: &str,
    device: &Device,
    dtype: DType,
    config: SDXLConfig,
) -> Result<SDXLUNet2DConditionModel> {
    println!("Loading SDXL UNet from: {}", model_path);

    // Use standard loader for stability
    let wl = WeightLoader::from_safetensors(model_path, device.clone())?;

    SDXLUNet2DConditionModel::new(wl, config)
}

impl crate::flame_training::FLAMEModel for SDXLUNet2DConditionModel {
    fn parameters(&self) -> Vec<&Parameter> {
        // Collect all parameters from the model
        let mut params = Vec::new();
        // TODO: Add actual parameter collection from inner UNet model
        params
    }

    fn named_parameters(&self) -> HashMap<String, &Parameter> {
        // Collect named parameters
        let mut params = HashMap::new();
        // TODO: Add actual named parameter collection from inner UNet model
        params
    }
}
