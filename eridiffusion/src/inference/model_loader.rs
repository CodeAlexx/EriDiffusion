use crate::loaders::WeightLoader;
use flame_core::{DType, Shape, Tensor};
use flame_core::device::Device;
use std::{collections::HashMap, path::Path};
use flame_core::{Result};

/// Load SDXL model components from either single file or separate files
pub struct SDXLModelLoader;

impl SDXLModelLoader {
    /// Load SDXL model from a single safetensors file (ComfyUI format)
    pub fn from_single_file(model_path: &Path, device: Device) -> flame_core::Result<HashMap<String, HashMap<String, Tensor>>> {
        println!("Loading SDXL model from single file: {:?}", model_path);

        // Load all tensors
        let tensors = WeightLoader::from_safetensors(model_path.to_str().unwrap(), device.clone())?;

        // Separate tensors by component
        let mut unet_weights = HashMap::new();
        let mut text_encoder_weights = HashMap::new();
        let mut text_encoder2_weights = HashMap::new();

        for (name, tensor) in tensors.weights {
            if name.starts_with("model.diffusion_model.") {
                // U-Net weights - remove prefix
                let new_name = name.strip_prefix("model.diffusion_model.").unwrap().to_string();
                unet_weights.insert(new_name, tensor);
            } else if name.starts_with("conditioner.embedders.0.") {
                // Text encoder 1 (CLIP-L) - remove prefix
                let new_name = name.strip_prefix("conditioner.embedders.0.").unwrap().to_string();
                text_encoder_weights.insert(new_name, tensor);
            } else if name.starts_with("conditioner.embedders.1.") {
                // Text encoder 2 (CLIP-G) - remove prefix
                let new_name = name.strip_prefix("conditioner.embedders.1.").unwrap().to_string();
                text_encoder2_weights.insert(new_name, tensor);
            }
        }

        let mut components = HashMap::new();
        components.insert("unet".to_string(), unet_weights);
        components.insert("text_encoder".to_string(), text_encoder_weights);
        components.insert("text_encoder2".to_string(), text_encoder2_weights);

        Ok(components)
    }

    /// Load SDXL model from separate component files (diffusers format)
    pub fn from_components(
        unet_path: &Path,
        text_encoder_path: &Path,
        text_encoder2_path: &Path,
        device: Device,
    ) -> flame_core::Result<HashMap<String, HashMap<String, Tensor>>> {
        println!("Loading SDXL model from component files");

        let unet_weights = WeightLoader::from_safetensors(unet_path.to_str().unwrap(), device.clone())?;
        let text_encoder_weights = WeightLoader::from_safetensors(text_encoder_path.to_str().unwrap(), device.clone())?;
        let text_encoder2_weights = WeightLoader::from_safetensors(text_encoder2_path.to_str().unwrap(), device.clone())?;

        let mut components = HashMap::new();
        components.insert("unet".to_string(), unet_weights.weights);
        components.insert("text_encoder".to_string(), text_encoder_weights.weights);
        components.insert("text_encoder2".to_string(), text_encoder2_weights.weights);

        Ok(components)
    }
}

/// Load SD 3.5 model components
pub struct SD35ModelLoader;

impl SD35ModelLoader {
    /// Load SD 3.5 model from a single safetensors file
    pub fn from_single_file(model_path: &Path, device: Device) -> flame_core::Result<HashMap<String, HashMap<String, Tensor>>> {
        println!("Loading SD 3.5 model from single file: {:?}", model_path);

        // Load all tensors
        let tensors = WeightLoader::from_safetensors(model_path.to_str().unwrap(), device.clone())?;

        // Separate tensors by component
        let mut mmdit_weights = HashMap::new();
        let mut text_encoder_weights = HashMap::new();
        let mut text_encoder2_weights = HashMap::new();
        let mut text_encoder3_weights = HashMap::new();

        for (name, tensor) in tensors.weights {
            if name.starts_with("model.diffusion_model.") {
                // MMDiT weights - remove prefix
                let new_name = name.strip_prefix("model.diffusion_model.").unwrap().to_string();
                mmdit_weights.insert(new_name, tensor);
            } else if name.starts_with("conditioner.embedders.0.") {
                // Text encoder 1 (CLIP-L) - remove prefix
                let new_name = name.strip_prefix("conditioner.embedders.0.").unwrap().to_string();
                text_encoder_weights.insert(new_name, tensor);
            } else if name.starts_with("conditioner.embedders.1.") {
                // Text encoder 2 (CLIP-G) - remove prefix
                let new_name = name.strip_prefix("conditioner.embedders.1.").unwrap().to_string();
                text_encoder2_weights.insert(new_name, tensor);
            } else if name.starts_with("conditioner.embedders.2.") || name.starts_with("text_encoders.t5xxl.") {
                // Text encoder 3 (T5-XXL) - remove prefix
                let prefix = if name.starts_with("conditioner.embedders.2.") {
                    "conditioner.embedders.2."
                } else {
                    "text_encoders.t5xxl."
                };
                let new_name = name.strip_prefix(prefix).unwrap().to_string();
                text_encoder3_weights.insert(new_name, tensor);
            }
        }

        let mut components = HashMap::new();
        components.insert("mmdit".to_string(), mmdit_weights);
        components.insert("text_encoder".to_string(), text_encoder_weights);
        components.insert("text_encoder2".to_string(), text_encoder2_weights);
        components.insert("text_encoder3".to_string(), text_encoder3_weights);

        Ok(components)
    }
}

/// Load Flux model components
pub struct FluxModelLoader;

impl FluxModelLoader {
    /// Load Flux model from a single safetensors file
    pub fn from_single_file(model_path: &Path, device: Device) -> flame_core::Result<HashMap<String, HashMap<String, Tensor>>> {
        println!("Loading Flux model from single file: {:?}", model_path);

        // Load all tensors
        let tensors = WeightLoader::from_safetensors(model_path.to_str().unwrap(), device.clone())?;

        // Separate tensors by component
        let mut flux_weights = HashMap::new();
        let mut text_encoder_weights = HashMap::new();
        let mut text_encoder2_weights = HashMap::new();

        for (name, tensor) in tensors.weights {
            if name.starts_with("model.") || name.starts_with("double_blocks.") || name.starts_with("single_blocks.") {
                // Flux model weights
                flux_weights.insert(name.clone(), tensor);
            } else if name.starts_with("conditioner.embedders.0.") || name.starts_with("text_encoders.clip_l.") {
                // Text encoder 1 (CLIP-L) - remove prefix
                let prefix = if name.starts_with("conditioner.embedders.0.") {
                    "conditioner.embedders.0."
                } else {
                    "text_encoders.clip_l."
                };
                let new_name = name.strip_prefix(prefix).unwrap().to_string();
                text_encoder_weights.insert(new_name, tensor);
            } else if name.starts_with("conditioner.embedders.1.") || name.starts_with("text_encoders.t5xxl.") {
                // Text encoder 2 (T5-XXL) - remove prefix
                let prefix = if name.starts_with("conditioner.embedders.1.") {
                    "conditioner.embedders.1."
                } else {
                    "text_encoders.t5xxl."
                };
                let new_name = name.strip_prefix(prefix).unwrap().to_string();
                text_encoder2_weights.insert(new_name, tensor);
            } else {
                // Other Flux-specific weights
                flux_weights.insert(name.clone(), tensor);
            }
        }

        let mut components = HashMap::new();
        components.insert("flux".to_string(), flux_weights);
        components.insert("text_encoder".to_string(), text_encoder_weights);
        components.insert("text_encoder2".to_string(), text_encoder2_weights);

        Ok(components)
    }
}

/// Universal model loader that detects model type
pub struct UniversalModelLoader;

impl UniversalModelLoader {
    /// Load model automatically detecting type from weights
    pub fn load_auto(model_path: &Path, device: Device) -> flame_core::Result<(String, HashMap<String, HashMap<String, Tensor>>)> {
        println!("Auto-detecting model type from: {:?}", model_path);

        // First, peek at the tensor names to determine model type
        let tensors = WeightLoader::from_safetensors(model_path.to_str().unwrap(), device.clone())?;
        let tensor_names: Vec<&String> = tensors.weights.keys().collect();

        // Detect model type based on key patterns
        let model_type = if tensor_names.iter().any(|name| name.contains("double_blocks") || name.contains("single_blocks")) {
            "flux"
        } else if tensor_names.iter().any(|name| name.contains("joint_blocks")) {
            "sd35"
        } else if tensor_names.iter().any(|name| name.contains("down_blocks") & !name.contains("diffusion_model")) {
            "sdxl"
        } else {
            "unknown"
        };

        println!("Detected model type: {}", model_type);

        // Load based on detected type
        let components = match model_type {
            "flux" => FluxModelLoader::from_single_file(model_path, device)?,
            "sd35" => SD35ModelLoader::from_single_file(model_path, device)?,
            "sdxl" => SDXLModelLoader::from_single_file(model_path, device)?,
            _ => {
                return Err(flame_core::Error::InvalidOperation(
                    format!("Unknown model type for file: {:?}", model_path)
                ));
            }
        };

        Ok((model_type.to_string(), components))
    }
}