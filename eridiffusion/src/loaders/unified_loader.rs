use anyhow::Context;
use flame_core::device::Device;
use flame_core::{DType, Error, Result, Shape, Tensor};
use std::{collections::HashMap, path::Path};

// Unified weight loader with automatic adaptation

#[derive(Debug, Clone, Copy)]
pub enum CheckpointFormat {
    SingleFile,
    Diffusers,
}

pub fn detect_checkpoint_format(path: &Path) -> flame_core::Result<CheckpointFormat> {
    if !path.exists() {
        return Err(flame_core::Error::InvalidOperation(format!("",)));
    }

    if path.extension().map_or(false, |ext| ext == "safetensors") {
        // Check if it's a single file or diffusers format
        if path.is_file() {
            Ok(CheckpointFormat::SingleFile)
        } else {
            Ok(CheckpointFormat::Diffusers)
        }
    } else if path.is_dir() {
        Ok(CheckpointFormat::Diffusers)
    } else {
        Ok(CheckpointFormat::SingleFile)
    }
}

pub fn load_unet_unified(
    path: &Path,
    device: &Device,
    dtype: DType,
) -> flame_core::Result<HashMap<String, Tensor>> {
    let weight_loader = crate::loaders::WeightLoader::from_safetensors(path, device.clone())?;
    Ok(weight_loader.weights)
}
