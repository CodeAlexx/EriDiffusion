use crate::loaders::WeightLoader;
use flame_core::device::Device;
use flame_core::Result;
use flame_core::{DType, Tensor};
use std::collections::HashMap;

/// Migration helpers for converting existing models to FLAME

/// Convert existing checkpoint to FLAME format
pub fn convert_checkpoint_to_flame(
    checkpoint_path: &str,
    output_path: &str,
    device: &Device,
) -> Result<()> {
    println!("Converting checkpoint from {} to {}", checkpoint_path, output_path);

    // Load weights
    let weights = WeightLoader::from_safetensors(checkpoint_path, device.clone())?;

    // TODO: Implement FLAME tensor to safetensors conversion
    // The safetensors crate expects tensors that implement its View trait
    // but FLAME tensors don't implement this trait yet
    return Err(flame_core::Error::InvalidOperation(
        "Tensor serialization not yet implemented for FLAME tensors".to_string(),
    ));

    Ok(())
}

/// Load model with automatic format detection
pub fn load_model_auto<T>(model_path: &str, device: &Device) -> Result<T>
where
    T: LoadableModel,
{
    // Try to load as FLAME format first
    if let Ok(weights) = WeightLoader::from_safetensors(model_path, device.clone()) {
        return T::load(&weights);
    }

    // Fall back to conversion
    println!("Model not in FLAME format, converting...");
    let temp_path = format!("{}.flame", model_path);
    convert_checkpoint_to_flame(model_path, &temp_path, device)?;

    let weights = WeightLoader::from_safetensors(&temp_path, device.clone())?;
    T::load(&weights)
}

/// Trait for loadable models
pub trait LoadableModel: Sized {
    fn load(weights: &WeightLoader) -> Result<Self>;
}

impl LoadableModel for super::flame_vae::VAE {
    fn load(weights: &WeightLoader) -> Result<Self> {
        super::flame_vae::VAE::load(weights)
    }
}

impl LoadableModel for super::flame_unet::UNet2DConditionModel {
    fn load(weights: &WeightLoader) -> Result<Self> {
        super::flame_unet::UNet2DConditionModel::load(weights)
    }
}

impl LoadableModel for super::flame_clip::CLIPTextModel {
    fn load(weights: &WeightLoader) -> Result<Self> {
        super::flame_clip::CLIPTextModel::load(weights)
    }
}

/// Trait for models that can collect their parameters
pub trait CollectParameters {
    /// Collect all parameters into a HashMap
    fn collect_parameters(&self) -> HashMap<String, Tensor>;

    /// Save parameters to a safetensors file
    fn save_parameters(&self, path: &str) -> Result<()> {
        use safetensors::{tensor::TensorView, Dtype as SafeDtype};

        let params = self.collect_parameters();
        let mut bytes_storage = Vec::new(); // Store all bytes data
        let mut tensor_info = Vec::new(); // Store tensor metadata

        // First pass: convert all tensors to bytes
        for (name, tensor) in params {
            let data = tensor.to_vec1::<f32>()?;
            let shape_vec = tensor.shape().dims().to_vec();

            // Convert f32 data to bytes
            let mut bytes = Vec::with_capacity(data.len() * 4);
            for &val in &data {
                bytes.extend_from_slice(&val.to_le_bytes());
            }

            tensor_info.push((name, shape_vec));
            bytes_storage.push(bytes);
        }

        // Second pass: create TensorViews
        let mut tensors = HashMap::new();
        for (idx, (name, shape_vec)) in tensor_info.into_iter().enumerate() {
            tensors.insert(
                name,
                TensorView::new(SafeDtype::F32, shape_vec, &bytes_storage[idx]).map_err(|e| {
                    flame_core::Error::InvalidOperation(format!(
                        "Failed to create tensor view: {}",
                        e
                    ))
                })?,
            );
        }

        let serialized = safetensors::serialize(tensors, &None).map_err(|e| {
            flame_core::Error::InvalidOperation(format!("Failed to serialize: {}", e))
        })?;
        std::fs::write(path, serialized).map_err(|e| {
            flame_core::Error::InvalidOperation(format!("Failed to write file: {}", e))
        })?;
        Ok(())
    }
}

/// Helper function to collect parameters with a prefix
pub fn collect_with_prefix(
    params: &mut HashMap<String, Tensor>,
    prefix: &str,
    tensors: Vec<(&str, &Tensor)>,
) {
    for (name, tensor) in tensors {
        let full_name =
            if prefix.is_empty() { name.to_string() } else { format!("{}.{}", prefix, name) };
        let cloned = tensor.clone();
        params.insert(full_name, cloned);
    }
}
