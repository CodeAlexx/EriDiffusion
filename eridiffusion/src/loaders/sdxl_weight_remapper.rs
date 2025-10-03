use flame_core::device::Device;
use flame_core::{DType, Tensor};
use std::{collections::HashMap, path::Path};

pub fn remap_sdxl_weights(weights: HashMap<String, Tensor>) -> HashMap<String, Tensor> {
    // For now, just return weights as-is
    weights
}

pub fn check_and_strip_prefix(weights: &mut HashMap<String, Tensor>, prefix: &str) {
    let keys_to_update: Vec<String> =
        weights.keys().filter(|k| k.starts_with(prefix)).cloned().collect();

    for key in keys_to_update {
        if let Some(tensor) = weights.remove(&key) {
            let new_key = key.strip_prefix(prefix).unwrap_or(&key).to_string();
            weights.insert(new_key, tensor);
        }
    }
}
