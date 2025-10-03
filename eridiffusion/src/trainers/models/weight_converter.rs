use anyhow::Context;
use flame_core::device::Device;
use flame_core::Result;
use flame_core::{DType, Error, Tensor};
use log::{debug, info};
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone, Copy)]
pub enum WeightFormat {
    SafeTensors,
    PyTorch,
    ONNX,
    TensorRT,
}

/// Convert weights between different formats
pub fn convert_weights(
    input_path: &Path,
    output_path: &Path,
    from_format: WeightFormat,
    to_format: WeightFormat,
    device: &Device,
) -> flame_core::Result<()> {
    info!("Converting weights from {:?} to {:?}", from_format, to_format);

    match (from_format, to_format) {
        (WeightFormat::SafeTensors, WeightFormat::PyTorch) => {
            convert_safetensors_to_pytorch(input_path, output_path, device)
        }
        (WeightFormat::PyTorch, WeightFormat::SafeTensors) => {
            convert_pytorch_to_safetensors(input_path, output_path, device)
        }
        _ => {
            return Err(flame_core::Error::InvalidOperation(format!(
                "Conversion from {:?} to {:?} not supported",
                from_format, to_format
            )))
        }
    }
}

fn convert_safetensors_to_pytorch(
    input_path: &Path,
    output_path: &Path,
    device: &Device,
) -> flame_core::Result<()> {
    // Load safetensors
    let weights = crate::loaders::WeightLoader::from_safetensors(input_path, device.clone())?;

    // Convert to PyTorch format
    // This would require PyTorch bindings which we don't have in pure Rust
    return Err(flame_core::Error::InvalidOperation(
        "SafeTensors to PyTorch conversion requires Python interop".to_string(),
    ));
}

fn convert_pytorch_to_safetensors(
    input_path: &Path,
    output_path: &Path,
    device: &Device,
) -> flame_core::Result<()> {
    // This would require PyTorch bindings which we don't have in pure Rust
    return Err(flame_core::Error::InvalidOperation(
        "PyTorch to SafeTensors conversion requires Python interop".to_string(),
    ));
}

/// Convert between different model architectures
pub fn convert_model_architecture(
    weights: HashMap<String, Tensor>,
    from_arch: &str,
    to_arch: &str,
) -> flame_core::Result<HashMap<String, Tensor>> {
    match (from_arch, to_arch) {
        ("diffusers", "comfyui") => convert_diffusers_to_comfyui(weights),
        ("comfyui", "diffusers") => convert_comfyui_to_diffusers(weights),
        _ => {
            return Err(flame_core::Error::InvalidOperation(format!(
                "Architecture conversion from '{}' to '{}' not supported",
                from_arch, to_arch
            ))
            .into())
        }
    }
}

/// Convert Diffusers format to ComfyUI format
fn convert_diffusers_to_comfyui(
    mut weights: HashMap<String, Tensor>,
) -> flame_core::Result<HashMap<String, Tensor>> {
    let mut converted = HashMap::new();

    for (key, tensor) in weights {
        let new_key = convert_diffusers_key_to_comfyui(&key);
        converted.insert(new_key, tensor);
    }

    Ok(converted)
}

/// Convert ComfyUI format to Diffusers format
fn convert_comfyui_to_diffusers(
    mut weights: HashMap<String, Tensor>,
) -> flame_core::Result<HashMap<String, Tensor>> {
    let mut converted = HashMap::new();

    for (key, tensor) in weights {
        let new_key = convert_comfyui_key_to_diffusers(&key);
        converted.insert(new_key, tensor);
    }

    Ok(converted)
}

/// Convert Diffusers key naming to ComfyUI
fn convert_diffusers_key_to_comfyui(key: &str) -> String {
    // Example conversions
    key.replace("down_blocks", "input_blocks")
        .replace("up_blocks", "output_blocks")
        .replace("mid_block", "middle_block")
        .replace("conv_in", "input_blocks.0.0")
        .replace("conv_out", "out.2")
}

/// Convert ComfyUI key naming to Diffusers
fn convert_comfyui_key_to_diffusers(key: &str) -> String {
    // Example conversions
    key.replace("input_blocks", "down_blocks")
        .replace("output_blocks", "up_blocks")
        .replace("middle_block", "mid_block")
        .replace("input_blocks.0.0", "conv_in")
        .replace("out.2", "conv_out")
}

/// Merge LoRA weights into base model
pub fn merge_lora_into_model(
    base_weights: &mut HashMap<String, Tensor>,
    lora_weights: &HashMap<String, Tensor>,
    scale: f32,
) -> flame_core::Result<()> {
    info!("Merging LoRA weights with scale {}", scale);

    for (lora_key, lora_tensor) in lora_weights {
        // Extract base key from LoRA key
        // e.g., "unet.down_blocks.0.attentions.0.to_q.lora_down.weight" -> "unet.down_blocks.0.attentions.0.to_q.weight"
        if let Some(base_key) = extract_base_key_from_lora(lora_key) {
            if let Some(base_tensor) = base_weights.get_mut(&base_key) {
                // Find corresponding up weight
                let up_key = lora_key.replace("lora_down", "lora_up");
                if let Some(up_tensor) = lora_weights.get(&up_key) {
                    // Compute LoRA update: scale * up @ down
                    let update = up_tensor.matmul(lora_tensor)?.mul_scalar(scale as f64 as f32)?;

                    // Add to base weight
                    *base_tensor = base_tensor.add(&update)?;
                    debug!("Merged LoRA for {}", base_key);
                }
            }
        }
    }

    Ok(())
}

fn extract_base_key_from_lora(lora_key: &str) -> Option<String> {
    if lora_key.contains("lora_down") {
        Some(lora_key.replace(".lora_down", ""))
    } else {
        None
    }
}

/// Quantize model weights
pub fn quantize_weights(
    weights: &mut HashMap<String, Tensor>,
    target_dtype: DType,
    exclude_patterns: &[&str],
) -> flame_core::Result<()> {
    info!("Quantizing weights to {:?}", target_dtype);

    for (key, tensor) in weights.iter_mut() {
        // Skip if matches exclusion pattern
        if exclude_patterns.iter().any(|&pattern| key.contains(pattern)) {
            debug!("Skipping quantization for {}", key);
            continue;
        }

        // Convert dtype
        if tensor.dtype() != target_dtype {
            *tensor = tensor.to_dtype(target_dtype)?;
            debug!("Quantized {} to {:?}", key, target_dtype);
        }
    }

    Ok(())
}
