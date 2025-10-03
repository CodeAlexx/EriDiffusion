use anyhow::{anyhow, Context};
use flame_core::device::Device;
use flame_core::{DType, Result, Shape, Tensor};
use std::collections::HashMap;
use std::path::Path;

/// Model loading utilities
// Weight loader module
pub mod weight_loader;
pub use weight_loader::{PrefixedWeightLoader, WeightLoader};

// FLAME loaders

pub mod flame_loader;

// Legacy loaders (will be converted to FLAME)
pub mod lazy_safetensors;
pub mod tensor_remapper;
pub mod unified_loader;
// pub mod memory_efficient_loader; // Temporarily disabled - depends on flux_custom
pub mod sd_to_diffusers_converter;
pub mod sdxl_checkpoint_converter;
pub mod sdxl_checkpoint_loader;
pub mod sdxl_diffusers_loader;
pub mod sdxl_full_remapper;
pub mod sdxl_weight_mapper;
pub mod sdxl_weight_remapper;
pub mod unified_checkpoint_loader;

// FLAME exports
pub use flame_loader::{FlameCheckpointLoader, FlameWeightLoader};

// Legacy exports
pub use lazy_safetensors::{create_lazy_tensor_provider, LazySafetensorsLoader};
pub use tensor_remapper::{create_flux_remapper, TensorRemapper};
// pub use memory_efficient_loader::{MemoryEfficientFluxLoader, LazyHashMap<String, Tensor>, create_memory_efficient_flux_model}; // Temporarily disabled
pub use sd_to_diffusers_converter::{convert_sdxl_checkpoint_to_diffusers, SDToDiffusersConverter};
pub use sdxl_checkpoint_converter::{convert_sdxl_checkpoint, load_sdxl_unet_from_checkpoint};
pub use sdxl_checkpoint_loader::{load_sdxl_checkpoint, load_text_encoders_sdxl};
pub use sdxl_diffusers_loader::{
    is_diffusers_format, load_clip_text_encoder, load_clip_text_encoder_2,
    load_sdxl_unet_diffusers, load_sdxl_vae_diffusers, load_vae, SDXLDiffusersLoader,
};
pub use sdxl_weight_mapper::{load_sdxl_unet_with_remapping, SDXLWeightMapper};
pub use sdxl_weight_remapper::{check_and_strip_prefix, remap_sdxl_weights};

pub fn convert(tensor: Tensor) -> flame_core::Result<Tensor> {
    Ok(tensor)
}
pub use unified_loader::load_unet_unified;
