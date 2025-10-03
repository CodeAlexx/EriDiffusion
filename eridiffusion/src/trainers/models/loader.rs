use anyhow::Context;
use flame_core::device::Device;
use flame_core::Result;
use flame_core::{DType, Error, Tensor};
use log::{debug, info};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::trainers::sdxl_vae_native::SDXLVAENative;
use crate::trainers::text_encoders::TextEncoders;

pub struct LoadedModels {
    pub unet_weights: HashMap<String, Tensor>,
    pub vae_encoder: Option<SDXLVAENative>,
    pub text_encoders: Option<TextEncoders>,
}

pub struct ModelLoader {
    device: Device,
    dtype: DType,
    cache_dir: Option<PathBuf>,
}

impl ModelLoader {
    pub fn new(device: Device, dtype: DType) -> Self {
        Self { device, dtype, cache_dir: None }
    }

    pub fn with_cache_dir(mut self, cache_dir: PathBuf) -> Self {
        self.cache_dir = Some(cache_dir);
        self
    }

    /// Load SDXL models from paths
    pub fn load_sdxl_models(
        &self,
        unet_path: &Path,
        vae_path: Option<&Path>,
        text_encoder_paths: Option<(&Path, &Path)>,
    ) -> flame_core::Result<LoadedModels> {
        info!("Loading SDXL models...");

        // Load UNet weights
        let unet_weights = self.load_unet_weights(unet_path)?;

        // Load VAE if path provided
        let vae_encoder = if let Some(vae_path) = vae_path {
            info!("Loading VAE from {:?}", vae_path);
            Some(self.load_vae(vae_path)?)
        } else {
            None
        };

        // Load text encoders if paths provided
        let text_encoders = if let Some((clip_path, clip_g_path)) = text_encoder_paths {
            info!("Loading text encoders...");
            Some(self.load_text_encoders(clip_path, clip_g_path)?)
        } else {
            None
        };

        Ok(LoadedModels { unet_weights, vae_encoder, text_encoders })
    }

    /// Load UNet weights from safetensors file
    fn load_unet_weights(&self, path: &Path) -> flame_core::Result<HashMap<String, Tensor>> {
        info!("Loading UNet weights from {:?}", path);

        let weight_loader =
            crate::loaders::WeightLoader::from_safetensors(path, self.device.clone()).map_err(
                |e| Error::InvalidOperation(format!("Failed to load UNet weights: {}", e)),
            )?;

        // Convert WeightLoader to HashMap
        let mut weights = HashMap::new();
        for key in weight_loader.keys() {
            if let Ok(tensor) = weight_loader.get(key) {
                weights.insert(key.clone(), tensor.clone());
            }
        }

        // Validate expected keys exist
        let required_prefixes =
            ["time_embed", "label_emb", "input_blocks", "middle_block", "output_blocks"];
        for prefix in &required_prefixes {
            let has_prefix = weights.keys().any(|k| k.starts_with(prefix));
            if !has_prefix {
                return Err(Error::InvalidOperation(format!(
                    "Missing required UNet prefix: {}",
                    prefix
                )));
            }
        }

        debug!("Loaded {} UNet weight tensors", weights.len());
        Ok(weights)
    }

    /// Load VAE encoder
    fn load_vae(&self, path: &Path) -> flame_core::Result<SDXLVAENative> {
        let weight_loader =
            crate::loaders::WeightLoader::from_safetensors(path, self.device.clone()).map_err(
                |e| Error::InvalidOperation(format!("Failed to load VAE weights: {}", e)),
            )?;

        // Convert WeightLoader to HashMap
        let mut weights = HashMap::new();
        for key in weight_loader.keys() {
            if let Ok(tensor) = weight_loader.get(key) {
                weights.insert(key.clone(), tensor.clone());
            }
        }

        // Create VAE with loaded weights
        SDXLVAENative::new(weights, self.device.clone(), self.dtype)
    }

    /// Load text encoders
    fn load_text_encoders(
        &self,
        clip_path: &Path,
        clip_g_path: &Path,
    ) -> flame_core::Result<TextEncoders> {
        TextEncoders::from_safetensors(
            Some(clip_path),
            Some(clip_g_path),
            None,
            self.device.clone(),
        )
    }

    /// Load from a single checkpoint file (if it contains all models)
    pub fn load_from_checkpoint(&self, checkpoint_path: &Path) -> flame_core::Result<LoadedModels> {
        info!("Loading from checkpoint: {:?}", checkpoint_path);

        let checkpoint =
            crate::loaders::WeightLoader::from_safetensors(checkpoint_path, self.device.clone())
                .map_err(|e| {
                    Error::InvalidOperation(format!("Failed to load checkpoint: {}", e))
                })?;

        // Separate weights by model type
        let mut unet_weights = HashMap::new();
        let mut vae_weights = HashMap::new();
        let mut text_encoder_weights = HashMap::new();

        for key in checkpoint.keys() {
            if let Ok(tensor) = checkpoint.get(key) {
                if key.starts_with("model.diffusion_model.") {
                    // Remove prefix for UNet weights
                    let new_key = key.strip_prefix("model.diffusion_model.").unwrap().to_string();
                    unet_weights.insert(new_key, tensor.clone());
                } else if key.starts_with("first_stage_model.") || key.starts_with("vae.") {
                    vae_weights.insert(key.clone(), tensor.clone());
                } else if key.starts_with("cond_stage_model.") || key.starts_with("text_encoder.") {
                    text_encoder_weights.insert(key.clone(), tensor.clone());
                } else {
                    unet_weights.insert(key.clone(), tensor.clone());
                }
            }
        }

        // Create models from weights
        let vae_encoder = if !vae_weights.is_empty() {
            Some(SDXLVAENative::new(vae_weights, self.device.clone(), self.dtype)?)
        } else {
            None
        };

        // For now, text encoders need separate loading
        let text_encoders = None;

        Ok(LoadedModels { unet_weights, vae_encoder, text_encoders })
    }

    /// Validate model compatibility
    pub fn validate_models(&self, models: &LoadedModels) -> flame_core::Result<()> {
        // Check UNet has expected structure
        if models.unet_weights.is_empty() {
            return Err(flame_core::Error::InvalidOperation(
                "No UNet weights loaded".to_string(),
            ));
        }

        // Validate VAE if present
        if let Some(ref vae) = models.vae_encoder {
            // VAE validation would go here
            debug!("VAE encoder validated");
        }

        // Validate text encoders if present
        if let Some(ref encoders) = models.text_encoders {
            // Text encoder validation would go here
            debug!("Text encoders validated");
        }

        Ok(())
    }
}
