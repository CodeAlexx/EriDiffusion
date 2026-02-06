use flame_core::device::Device;
use flame_core::{DType, Shape, Tensor};

// --- FLAME model implementations for diffusion models ---

// Core building blocks
pub mod attention;
pub mod resnet;
pub mod tensor_utils;
pub mod aligned_image_processor;

// Flux
pub mod flux_model_complete;
pub mod flux_blocks;
pub mod flux_blocks_fixed;
pub mod flux_vae;

// SDXL
pub mod sdxl_unet_complete;
pub mod sdxl_unet;
pub mod sdxl_vae;
pub mod sdxl_time_ids;

// SD3.5 / MMDiT
pub mod mmdit_blocks;
pub mod mmdit_cpu;

// Text encoders
pub mod text_encoder_complete;
pub mod text_encoders_cpu;

// VAE
pub mod vae_complete;

// LoRA
pub mod lora;
// pub mod lora_attention; // Disabled: uses candle-style API, only referenced by disabled lora_transformer/lora_unet_blocks
// pub mod lora_transformer; // Disabled: depends on lora_with_gradients (corrupted)
// pub mod lora_unet_blocks; // Disabled: corrupted/incomplete file with syntax errors
// pub mod lora_with_gradients; // Disabled: corrupted/incomplete file with syntax errors

// Flux LoRA wrapper (used by production trainers)
pub mod flux_lora_wrapper;

// Legacy FLAME wrappers (still used by production binaries)
pub mod flame_migration_helpers;
pub mod flame_vae;

// Other
pub mod unet_2d;

// --- Re-exports ---

pub use resnet::{
    Downsample2D, ResNetTensorExt, ResNetTensorExt as TensorExt, ResnetBlock2D, Upsample2D,
};
pub use attention::{Attention, AttentionBlock, BasicTransformerBlock, FeedForward, SpatialTransformer};
pub use sdxl_unet_complete::{
    UNet2DConditionModel as SDXLUNet, UNet2DConditionModel, UNet2DConditionModelConfig,
    UNet2DConditionModelConfig as UNet2DConfig,
};
pub use text_encoder_complete::{CLIPConfig, CLIPTextEncoder, T5Config, T5Encoder};
pub use vae_complete::{AutoEncoderKL, VAEConfig};
pub use flux_model_complete::{patchify_for_flux, FluxModel, FluxModelConfig};
pub use mmdit_blocks::{MMDiT, MMDiTConfig, MMDiT as SD35MMDiT, MMDiTConfig as SD35Config};

// Compatibility aliases for old import paths (clip.rs, t5.rs, text_encoder.rs, vae.rs were consolidated)
pub mod clip {
    pub use super::text_encoder_complete::{
        CLIPConfig as Config,
        CLIPTextEncoder as ClipTextTransformer,
        CLIPTextEncoderOutput,
        CLIPConfig,
        CLIPTextEncoder,
    };
}
pub mod t5 {
    pub use super::text_encoder_complete::{T5Config, T5Encoder as T5EncoderModel, T5Output};
}
pub mod text_encoder {
    pub use super::text_encoder_complete::*;
}
pub mod vae {
    pub use super::vae_complete::*;
}
pub mod flux_complete {
    pub use super::flux_model_complete::{FluxModel, FluxModelConfig as FluxConfig, FluxModelConfig};
    pub use super::flux_model_complete::*;
}

// Top-level alias so `use crate::models::T5EncoderModel` works
pub use text_encoder_complete::T5Encoder as T5EncoderModel;

// BlockConfig for backwards compatibility
#[derive(Debug, Clone)]
pub struct BlockConfig {
    pub out_channels: usize,
    pub use_cross_attn: Option<usize>,
}
