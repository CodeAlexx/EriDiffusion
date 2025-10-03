use flame_core::device::Device;
use flame_core::Result;
use flame_core::{DType, Tensor};
use std::{collections::HashMap, path::Path};

// Unified checkpoint loader that handles different formats

pub fn load_checkpoint_unified(
    path: &Path,
    device: &Device,
    dtype: DType,
) -> flame_core::Result<HashMap<String, Tensor>> {
    let loader =
        crate::loaders::WeightLoader::from_safetensors(path.to_str().unwrap(), device.clone())?;
    Ok(loader.weights)
}
