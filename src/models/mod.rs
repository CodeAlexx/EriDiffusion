//! Model implementations for diffusion models
//! 
//! This module contains the actual model architectures

// pub mod flux_lora; // Temporarily disabled due to compilation issues
pub mod flux_vae;
pub mod flux_lora_adapter;
// pub mod flux_lora_wrapper;  // Temporarily disabled
pub mod flux_model_trait;
// pub mod flux_custom; // Temporarily disabled
pub mod flux_adaptive_loader;
pub mod direct_var_builder;
// pub mod flux_minimal; // Temporarily disabled - depends on flux_custom
pub mod sdxl_unet;
pub mod flash_attention;
pub mod sdxl_vae;
pub mod sdxl_time_ids;
// Commented out old SDXL LoRA implementations that are no longer used
// pub mod sdxl_lora_layer;
// pub mod sdxl_lora_unet;
// pub mod sdxl_lora_unet_wrapper;
// pub mod sdxl_hooked_unet;  // Disabled - depends on removed sdxl_lora_layer
// pub mod sdxl_lora_injected_unet;
// pub mod lora_attention;
// pub mod lora_transformer;  // Disabled - depends on removed lora_attention
// pub mod lora_unet_blocks;
// pub mod sdxl_lora_unet_v2;
pub mod with_tracing;
// pub mod sdxl_lora_layer_v3;
// pub mod lora_with_gradients;

// Re-export key types
// pub use flux_lora::{FluxModelWithLoRA, FluxConfig}; // Temporarily disabled
pub use flux_vae::{AutoencoderKL as FluxVAE, load_flux_vae};
pub use flux_lora_adapter::{FluxLoRAAdapter, FluxPatchLoRA};
pub use flux_model_trait::FluxModel;

// Re-export custom Flux implementation
// pub use flux_custom::{
//     FluxModelWithLoRA as FluxCustomModel,
//     FluxConfig as FluxCustomConfig,
//     create_flux_lora_model,
// }; // Temporarily disabled

// Re-export SDXL types
pub use sdxl_unet::{SDXLUNet2DConditionModel, SDXLConfig, load_sdxl_unet};
pub use sdxl_vae::{VAEModel, SDXLVAE, SD3VAE, sdxl_vae_vb_rename, sd3_vae_vb_rename};
pub use sdxl_time_ids::{TimeIdsConfig, TimeIdsGenerator, SDXLResolutions, SDXLConditioningHelper};
// Commented out re-exports for modules that were removed
// pub use sdxl_lora_layer::{LoRALinear, LoRAConv2d};
// pub use lora_attention::{CrossAttentionWithLoRA, BasicTransformerBlockWithLoRA, LoRAAttentionConfig};
// pub use lora_transformer::{SpatialTransformerWithLoRA, SpatialTransformerConfig};
// pub use lora_unet_blocks::{CrossAttnDownBlock2DWithLoRA, CrossAttnUpBlock2DWithLoRA, UNetMidBlock2DCrossAttnWithLoRA};
// pub use sdxl_lora_unet_v2::{SDXLUNetWithLoRA, SDXLUNetConfig};