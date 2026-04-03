use std::path::Path;

use flame_core::device::Device;
use flame_core::Result;

use crate::models::flux_model_complete::{FluxModel, FluxModelConfig};

use super::flux_int8_loader::{load_flux_int8, FluxInt8Model};

pub struct FluxModelInt8 {
    quantized: FluxInt8Model,
    config: FluxModelConfig,
    device: Device,
}

impl FluxModelInt8 {
    pub fn from_safetensors<P: AsRef<Path>>(
        path: P,
        device: Device,
        config: FluxModelConfig,
    ) -> Result<Self> {
        let quantized = load_flux_int8(path, device.clone())?;
        Ok(Self { quantized, config, device })
    }

    pub fn to_flux_model(self) -> Result<FluxModel> {
        let weights = self.quantized.dequantize_all()?;
        FluxModel::new(self.config, self.device, weights)
    }
}
