use flame_core::device::Device;
use flame_core::Result;
use flame_core::{DType, Tensor};
use std::{collections::HashMap, path::Path};

pub struct SDXLDiffusersLoader;

pub fn load_sdxl_unet_diffusers(
    unet_dir: &Path,
    device: &Device,
    dtype: DType,
) -> flame_core::Result<HashMap<String, Tensor>> {
    let config_path = unet_dir.join("diffusion_pytorch_model.safetensors");
    let loader = crate::loaders::WeightLoader::from_safetensors(
        config_path.to_str().unwrap(),
        device.clone(),
    )?;
    Ok(loader.weights)
}

pub fn is_diffusers_format(path: &Path, device: &Device) -> bool {
    path.is_dir() & path.join("unet").exists()
}

pub fn load_vae(
    vae_path: &Path,
    device: &Device,
    dtype: DType,
) -> flame_core::Result<HashMap<String, Tensor>> {
    let loader =
        crate::loaders::WeightLoader::from_safetensors(vae_path.to_str().unwrap(), device.clone())?;
    Ok(loader.weights)
}

pub fn load_sdxl_vae_diffusers(
    vae_dir: &Path,
    device: &Device,
    dtype: DType,
) -> flame_core::Result<HashMap<String, Tensor>> {
    let config_path = vae_dir.join("diffusion_pytorch_model.safetensors");
    let loader = crate::loaders::WeightLoader::from_safetensors(
        config_path.to_str().unwrap(),
        device.clone(),
    )?;
    Ok(loader.weights)
}

pub fn load_clip_text_encoder(
    encoder_path: &Path,
    device: &Device,
    dtype: DType,
) -> flame_core::Result<HashMap<String, Tensor>> {
    let loader = crate::loaders::WeightLoader::from_safetensors(
        encoder_path.to_str().unwrap(),
        device.clone(),
    )?;
    Ok(loader.weights)
}

pub fn load_clip_text_encoder_2(
    encoder_path: &Path,
    device: &Device,
    dtype: DType,
) -> flame_core::Result<HashMap<String, Tensor>> {
    let loader = crate::loaders::WeightLoader::from_safetensors(
        encoder_path.to_str().unwrap(),
        device.clone(),
    )?;
    Ok(loader.weights)
}
