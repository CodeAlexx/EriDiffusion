use crate::models::text_encoder_complete::CLIPConfig as Config;
use crate::models::text_encoder_complete::CLIPTextEncoder;
use crate::models::vae_complete::AutoEncoderKL;
use crate::models::{
    BlockConfig, UNet2DConditionModel, UNet2DConfig as UNet2DConditionModelConfig,
};
use flame_core::device::Device;
use flame_core::{DType, Error, Result, Tensor};
use std::path::Path;

// Re-export vae module for compatibility
pub mod vae {
    pub use crate::models::vae::*;
}

/// Stable Diffusion configuration compatibility layer
#[derive(Clone)]
pub struct StableDiffusionConfig {
    pub clip: Config,
    pub clip2: Option<Config>,
    pub width: usize,
    pub height: usize,
    pub vocab_size: usize,
    pub sliced_attention_size: Option<usize>,
}

impl Default for StableDiffusionConfig {
    fn default() -> Self {
        Self {
            clip: Config {
                vocab_size: 49408,
                hidden_size: 768,
                intermediate_size: 3072,
                num_hidden_layers: 12,
                num_attention_heads: 12,
                max_position_embeddings: 77,
                layer_norm_eps: 1e-5,
                hidden_act: "quick_gelu".to_string(),
                projection_dim: Some(768),
                pad_token_id: 1,
            },
            clip2: None,
            width: 512,
            height: 512,
            vocab_size: 49408,
            sliced_attention_size: None,
        }
    }
}

impl StableDiffusionConfig {
    pub fn v2_1(
        sliced_attention_size: Option<usize>,
        height: Option<usize>,
        width: Option<usize>,
    ) -> Self {
        Self {
            clip: Config {
                vocab_size: 49408,
                hidden_size: 1024,
                intermediate_size: 4096,
                num_hidden_layers: 23,
                num_attention_heads: 16,
                max_position_embeddings: 77,
                layer_norm_eps: 1e-5,
                hidden_act: "gelu".to_string(),
                projection_dim: Some(512),
                pad_token_id: 1,
            },
            clip2: None,
            width: width.unwrap_or(768),
            height: height.unwrap_or(768),
            vocab_size: 49408,
            sliced_attention_size,
        }
    }

    pub fn sdxl(
        sliced_attention_size: Option<usize>,
        height: Option<usize>,
        width: Option<usize>,
    ) -> Self {
        Self {
            // CLIP-L config
            clip: Config {
                vocab_size: 49408,
                hidden_size: 768,
                intermediate_size: 3072,
                num_hidden_layers: 12,
                num_attention_heads: 12,
                max_position_embeddings: 77,
                layer_norm_eps: 1e-5,
                hidden_act: "quick_gelu".to_string(),
                projection_dim: Some(768),
                pad_token_id: 49407,
            },
            // CLIP-G config
            clip2: Some(Config {
                vocab_size: 49408,
                hidden_size: 1280,
                intermediate_size: 5120,
                num_hidden_layers: 32,
                num_attention_heads: 20,
                max_position_embeddings: 77,
                layer_norm_eps: 1e-5,
                hidden_act: "gelu".to_string(),
                projection_dim: Some(1280),
                pad_token_id: 49407,
            }),
            width: width.unwrap_or(1024),
            height: height.unwrap_or(1024),
            vocab_size: 49408,
            sliced_attention_size,
        }
    }

    pub fn build_vae(
        &self,
        vae_weights: &Path,
        device: &Device,
        dtype: DType,
    ) -> flame_core::Result<AutoEncoderKL> {
        use crate::loaders::WeightLoader;
        use crate::models::vae_complete::VAEConfig;

        // Load weights from safetensors
        let wl = WeightLoader::from_safetensors(vae_weights, device.clone())?;

        // Determine VAE type from config
        let config = if self.clip2.is_some() {
            // SDXL VAE
            VAEConfig::sdxl()
        } else {
            // SD 1.5/2.1 VAE (same config as SDXL)
            VAEConfig::sdxl()
        };

        // Create VAE with weights
        AutoEncoderKL::new(config, device, wl.weights)
    }

    pub fn build_unet(
        &self,
        unet_weights: &Path,
        device: &Device,
        in_channels: usize,
        use_flash_attn: bool,
        dtype: DType,
    ) -> flame_core::Result<UNet2DConditionModel> {
        use crate::loaders::WeightLoader;
        use crate::models::sdxl_unet_complete::UNet2DConditionModelConfig;

        // Load weights from safetensors
        let wl = WeightLoader::from_safetensors(unet_weights, device.clone())?;

        // Get the appropriate config based on SD version
        let config = if self.clip2.is_some() {
            // SDXL UNet config
            UNet2DConditionModelConfig::sdxl()
        } else {
            // SD 1.5/2.1 UNet config - similar to SDXL but with different dimensions
            let mut config = UNet2DConditionModelConfig::sdxl();
            config.cross_attention_dim = 768; // SD 1.5 uses 768
            config.transformer_layers_per_block = vec![1, 1, 1]; // Fewer transformer layers
            config.attention_head_dim = vec![8, 8, 8]; // Different attention head dimensions
            if self.clip.hidden_size == 1024 {
                // SD 2.1 config adjustments
                config.cross_attention_dim = 1024;
            }
            config
        };

        // Create UNet with weights
        UNet2DConditionModel::new(config, device, wl.weights)
    }
}

/// Helper function to build CLIP transformer
pub fn build_clip_transformer(
    config: &Config,
    clip_weights: &Path,
    device: &Device,
    dtype: DType,
) -> flame_core::Result<CLIPTextEncoder> {
    use crate::loaders::WeightLoader;
    use crate::models::text_encoder_complete::{CLIPConfig, CLIPTextEncoder};

    // Load weights from safetensors
    let wl = WeightLoader::from_safetensors(clip_weights, device.clone())?;

    // Convert Config to CLIPConfig - Config already has the right fields
    let clip_config = CLIPConfig {
        vocab_size: config.vocab_size,
        hidden_size: config.hidden_size,
        intermediate_size: config.intermediate_size,
        num_hidden_layers: config.num_hidden_layers,
        num_attention_heads: config.num_attention_heads,
        max_position_embeddings: config.max_position_embeddings,
        layer_norm_eps: config.layer_norm_eps,
        hidden_act: config.hidden_act.clone(),
        projection_dim: config.projection_dim,
        pad_token_id: config.pad_token_id,
    };

    // Create CLIPTextEncoder with weights
    CLIPTextEncoder::new(clip_config, device.clone(), wl.weights)
}
