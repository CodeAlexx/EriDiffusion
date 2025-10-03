use flame_core::device::Device;
use flame_core::Result;
use flame_core::{DType, Shape, Tensor};
use std::collections::HashMap;

pub struct SDXLWeightMapper {
    mappings: HashMap<String, String>,
}

impl SDXLWeightMapper {
    pub fn new() -> Self {
        let mut mappings = HashMap::new();

        // Time embedding mappings
        mappings.insert("time_embed.0.weight".to_string(), "time_proj.weight".to_string());
        mappings.insert("time_embed.0.bias".to_string(), "time_proj.bias".to_string());
        mappings.insert(
            "time_embed.2.weight".to_string(),
            "time_embedding.linear_1.weight".to_string(),
        );
        mappings
            .insert("time_embed.2.bias".to_string(), "time_embedding.linear_1.bias".to_string());
        mappings.insert(
            "time_embed.4.weight".to_string(),
            "time_embedding.linear_2.weight".to_string(),
        );
        mappings
            .insert("time_embed.4.bias".to_string(), "time_embedding.linear_2.bias".to_string());

        Self { mappings }
    }

    pub fn map_weight_name(&self, name: &str, device: &Device) -> String {
        self.mappings.get(name).cloned().unwrap_or_else(|| name.to_string())
    }
}

pub fn load_sdxl_unet_with_remapping(
    path: &str,
    device: &Device,
    dtype: DType,
) -> flame_core::Result<HashMap<String, Tensor>> {
    let weight_loader = crate::loaders::WeightLoader::from_safetensors(path, device.clone())?;
    let mapper = SDXLWeightMapper::new();
    let mut remapped = HashMap::new();

    for (key, tensor) in weight_loader.weights {
        let new_key = mapper.map_weight_name(&key, device);
        remapped.insert(new_key, tensor);
    }

    Ok(remapped)
}
