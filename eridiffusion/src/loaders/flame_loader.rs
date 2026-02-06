use crate::loaders::WeightLoader;
use crate::models::text_encoder_complete::CLIPTextEncoder;
use crate::models::vae_complete::{AutoEncoderKL, VAEConfig};
use anyhow::Context;
use flame_core::device::Device;
use flame_core::optimizers::{Adam, SGD};
use flame_core::{DType, Result, Shape, Tensor};
use half::{bf16, f16};
use std::{collections::HashMap, path::Path, sync::Arc};

pub struct PrefixedWeightLoader {
    loader: WeightLoader,
    prefix: String,
}
pub struct FlameWeightLoader;

/// High-level checkpoint loader for FLAME models
pub struct FlameCheckpointLoader {
    device: flame_core::device::Device,
}

// FLAME weight loader for safetensors format

// FLAME uses flame_core::device::Device instead of Device
// bf16 and f16 are already imported from half crate above

// WeightLoader implementation is in crate::loaders::WeightLoader

impl PrefixedWeightLoader {
    pub fn get(&self, key: &str) -> flame_core::Result<&Tensor> {
        let full_key = format!("{}.{}", self.prefix, key);
        self.loader.get(&full_key)
    }

    pub fn tensor(&self, key: &str, shape: &[usize]) -> flame_core::Result<Tensor> {
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

impl FlameCheckpointLoader {
    pub fn new(device: flame_core::device::Device) -> Self {
        Self { device }
    }

    pub fn load_vae(&self, path: &Path) -> flame_core::Result<AutoEncoderKL> {
        // Load VAE weights and create model
        let weights = self.load_safetensors(path)?;
        AutoEncoderKL::from_weights(weights, VAEConfig::default(), self.device.clone())
    }

    pub fn load_clip_text_encoder(&self, path: &Path) -> flame_core::Result<CLIPTextEncoder> {
        // Load CLIP weights and create model
        let _weights = self.load_safetensors(path)?;
        // TODO: Implement CLIPTextEncoder construction from weights
        Err(flame_core::Error::InvalidOperation(
            "CLIPTextEncoder construction not yet implemented".into(),
        ))
    }

    fn load_safetensors(&self, path: &Path) -> flame_core::Result<HashMap<String, Tensor>> {
        crate::loaders::WeightLoader::from_safetensors(path, self.device.clone())
            .map(|loader| loader.weights)
    }
}

impl FlameWeightLoader {
    /// Load safetensors file into FLAME tensors
    pub fn load_safetensors(
        path: &str,
        device: &Device,
    ) -> flame_core::Result<HashMap<String, Tensor>> {
        crate::loaders::WeightLoader::from_safetensors(path, device.clone())
            .map(|loader| loader.weights)
    }

    /// Load specific tensors by prefix
    pub fn load_with_prefix(
        path: &str,
        prefix: &str,
        device: &Device,
    ) -> flame_core::Result<HashMap<String, Tensor>> {
        let all_tensors = Self::load_safetensors(path, device)?;

        Ok(all_tensors.into_iter().filter(|(name, _)| name.starts_with(prefix)).collect())
    }

    /// Load UNet weights
    pub fn load_unet_weights(
        path: &str,
        device: &Device,
    ) -> flame_core::Result<HashMap<String, Tensor>> {
        Self::load_with_prefix(path, "model.diffusion_model.", device)
    }

    /// Load VAE weights
    pub fn load_vae_weights(
        path: &str,
        device: &Device,
    ) -> flame_core::Result<HashMap<String, Tensor>> {
        // Try different VAE prefixes
        let mut weights = Self::load_with_prefix(path, "first_stage_model.", device)?;
        if weights.is_empty() {
            weights = Self::load_with_prefix(path, "vae.", device)?;
        }
        Ok(weights)
    }

    /// Load text encoder weights
    pub fn load_text_encoder_weights(
        path: &str,
        device: &Device,
    ) -> flame_core::Result<HashMap<String, Tensor>> {
        Self::load_with_prefix(path, "cond_stage_model.", device)
    }
}
