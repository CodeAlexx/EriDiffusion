use flame_core::device::Device;
use flame_core::{DType, Shape, Tensor};
use std::{collections::HashMap, fs::File, path::Path, sync::Arc};

// Tensor remapper for handling checkpoint format mismatches

pub struct TensorRemapper {
    mappings: HashMap<String, String>,
    device: Device,
    dtype: DType,
}

impl TensorRemapper {
    pub fn new(device: Device, dtype: DType) -> Self {
        Self { mappings: HashMap::new(), device, dtype }
    }

    pub fn add_mapping(&mut self, from: String, to: String, device: &Device) {
        self.mappings.insert(from, to);
    }

    pub fn remap_tensor_name(&self, name: &str, device: &Device) -> String {
        self.mappings.get(name).cloned().unwrap_or_else(|| name.to_string())
    }
}

pub fn create_flux_remapper(device: Device, dtype: DType) -> TensorRemapper {
    let mut remapper = TensorRemapper::new(device.clone(), dtype);

    // Add common Flux mappings
    remapper.add_mapping("img_in.weight".to_string(), "image_proj.weight".to_string(), &device);
    remapper.add_mapping("txt_in.weight".to_string(), "text_proj.weight".to_string(), &device);

    remapper
}
