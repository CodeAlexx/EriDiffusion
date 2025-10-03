use flame_core::device::Device;
use flame_core::Result;
use flame_core::{DType, Tensor};
use std::{collections::HashMap, path::Path};

pub fn load_sdxl_checkpoint(
    checkpoint_path: &Path,
    device: &Device,
    dtype: DType,
) -> flame_core::Result<HashMap<String, Tensor>> {
    let loader = crate::loaders::WeightLoader::from_safetensors(checkpoint_path, device.clone())?;
    Ok(loader.weights)
}

pub fn load_text_encoders_sdxl(
    clip_g_path: &Path,
    clip_l_path: &Path,
    device: &Device,
    dtype: DType,
) -> flame_core::Result<(HashMap<String, Tensor>, HashMap<String, Tensor>)> {
    let clip_g_loader =
        crate::loaders::WeightLoader::from_safetensors(clip_g_path, device.clone())?;
    let clip_l_loader =
        crate::loaders::WeightLoader::from_safetensors(clip_l_path, device.clone())?;
    Ok((clip_g_loader.weights, clip_l_loader.weights))
}
