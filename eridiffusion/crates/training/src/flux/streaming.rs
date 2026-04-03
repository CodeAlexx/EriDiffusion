//! Streaming façade around the strict safetensor loader.
//!
//! Trainers can use this helper to pull tensors lazily (one block at a time)
//! without materialising the whole shard on GPU. It wires through the
//! Phase-4 strict loader so we inherit dtype/shape validation for every key.

#![allow(dead_code)]

use std::path::Path;

use anyhow::Result;
use eridiffusion_common_weights::strict_loader::{tensor_from_bytes, StrictMmapLoader, TensorInfo};
use flame_core::{Device as FlameDevice, Tensor};

pub struct StreamProvider {
    loader: StrictMmapLoader,
    device: FlameDevice,
}

impl StreamProvider {
    pub fn open(path: impl AsRef<Path>, device: FlameDevice) -> Result<Self> {
        let loader = StrictMmapLoader::open(path.as_ref())?;
        Ok(Self { loader, device })
    }

    pub fn tensor(&mut self, key: &str) -> Result<Tensor> {
        let info: TensorInfo = self.loader.info(key)?;
        let bytes = self.loader.bytes(key)?;
        let tensor = tensor_from_bytes(self.device.clone(), &info, bytes)?;
        self.loader.mark_used(key);
        Ok(tensor)
    }

    pub fn validate(self) -> Result<()> {
        self.loader.validate_used_exactly()
    }
}
