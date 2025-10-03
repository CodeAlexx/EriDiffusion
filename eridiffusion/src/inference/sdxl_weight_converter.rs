use flame_core::{DType, Result, Shape, Tensor};
use flame_core::device::Device;
use flame_core::optimizers::{Adam, SGD};
use std::{collections::HashMap, path::Path};

// FLAME uses flame_core::device::Device instead of Device

/// Convert ComfyUI SDXL weights to diffusers format
pub fn convert_comfyui_to_diffusers_format(
input_path: &Path,
device: &Device,
) -> flame_core::Result<Tensor> {
println!("Converting ComfyUI SDXL weights to diffusers format...");

// Load all tensors from the ComfyUI format file
let mut converted = HashMap::new();

for (name, tensor) in tensors {
// ComfyUI format: model.diffusion_model.* -> diffusers format: *
let new_name = if name.starts_with("model.diffusion_model.") {
name.strip_prefix("model.diffusion_model.").unwrap().to_string()
} else {
// Skip non-U-Net weights
continue;
};

converted.insert(new_name, tensor);
}

println!("Converted {} U-Net tensors", converted.len());
Ok(converted)
}

/// Remap weight names from ComfyUI to FLAME expected format
pub fn remap_unet_weights(weights: &mut std::collections::HashMap<String, Tensor>) -> flame_core::Result<()> {
let mut remapped = HashMap::new();

for (name, tensor) in weights.drain() {
// ComfyUI uses different naming conventions
let new_name = name
.replace("input_blocks", "down_blocks")
.replace("output_blocks", "up_blocks")
.replace("middle_block", "mid_block")
.replace("time_embed", "time_embedding")
.replace("label_emb", "add_embedding")
.replace("in_layers", "norm1")
.replace("out_layers", "norm2")
.replace("emb_layers", "time_emb_proj")
.replace("skip_connection", "conv_shortcut");

remapped.insert(new_name, tensor);
}

*weights = remapped;
}
