// Imports moved to re-exports section below
use flame_core::device::Device;
use flame_core::{DType, Shape, Tensor};
// backward is a method on Tensor, not a standalone function
// GradientMap needs to be imported differently
use flame_core::optimizers::{Adam, SGD};

pub mod flame_clip;
pub mod flame_migration_helpers;
pub mod flame_unet;
pub mod flame_unet_lora;
pub mod flame_vae;

// FLAME model implementations for diffusion models

pub mod attention;
pub mod resnet;
pub mod sdxl_unet_complete;
pub mod text_encoder;
pub mod text_encoder_complete;
pub mod unet_2d;
pub mod vae;
pub mod vae_complete;
// pub mod sd35_mmdit_complete;  // Commented out - using mmdit_blocks instead
pub mod autoencoder_kl;
pub mod clip;
pub mod fast_vae;
pub mod flux_blocks;
pub mod flux_blocks_fixed;
pub mod flux_complete;
pub mod flux_lora_wrapper;
pub mod flux_model_complete;
pub mod flux_vae;
pub mod mmdit_blocks;
pub mod sdxl_time_ids;
pub mod sdxl_unet;
pub mod sdxl_unet_fixed;
pub mod sdxl_vae;
pub mod tensor_utils;
pub mod unified_vae;
// pub mod efficient_attention;
pub mod aligned_image_processor;
pub mod bucket_alignment;
pub mod bucket_aware_vae;
pub mod bucket_config;
pub mod cuda_alignment;
pub mod lora;
pub mod streaming_t5; // GPU-only streaming T5 encoder with cuDNN optimization
pub mod t5; // T5 text encoder model

pub use resnet::{
    Downsample2D, ResNetTensorExt, ResNetTensorExt as TensorExt, ResnetBlock2D, Upsample2D,
};
// Conv2d and GroupNorm come from flame_core, import them directly from there
pub use attention::{Attention, BasicTransformerBlock, SpatialTransformer};
pub use sdxl_unet_complete::{
    UNet2DConditionModel as SDXLUNet, UNet2DConditionModel, UNet2DConditionModelConfig,
};
pub use text_encoder_complete::{CLIPConfig, CLIPTextEncoder, T5Config, T5Encoder};
pub use unified_vae::{VAEConfig as UnifiedVAEConfig, VAE as UnifiedVAE, VAE as AutoencoderKL};
pub use vae_complete::{AutoEncoderKL, VAEConfig};
// pub use sd35_mmdit_complete::{SD35MMDiT, SD35Config};
pub use flux_complete::{FluxConfig, FluxModel as FluxComplete};
pub use flux_model_complete::{patchify_for_flux, FluxModel, FluxModelConfig};
pub use mmdit_blocks::{MMDiT as SD35MMDiT, MMDiTConfig as SD35Config};
// Re-export from sdxl_unet if needed
pub mod direct_var_builder;
pub use attention::{AttentionBlock, FeedForward};
pub use mmdit_blocks::{MMDiT, MMDiTConfig};
pub use sdxl_unet_complete::UNet2DConditionModelConfig as UNet2DConfig;
pub use text_encoder::{
    CLIPConfig as ClipConfig, T5Config as TextEncoderT5Config, T5Encoder as T5EncoderModel,
};

// BlockConfig for backwards compatibility
#[derive(Debug, Clone)]
pub struct BlockConfig {
    pub out_channels: usize,
    pub use_cross_attn: Option<usize>,
}
pub use text_encoder::CLIPTextEncoder as ClipTextTransformer;
pub use vae::{AutoEncoderKL as VAE, VAEConfig as BasicVAEConfig};
