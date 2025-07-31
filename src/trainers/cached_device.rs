use candle_core::Device;
use anyhow::Result;
use std::sync::OnceLock;

static CACHED_DEVICE: OnceLock<Device> = OnceLock::new();

/// Get a single cached CUDA device to avoid Candle's device ID bug
pub fn get_single_device() -> Result<Device> {
    Ok(CACHED_DEVICE.get_or_init(|| {
        Device::cuda_if_available(0).expect("Failed to create CUDA device")
    }).clone())
}