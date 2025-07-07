//! Device wrapper to ensure all operations use the same CUDA context
//! 
//! This is a workaround for Candle's device management issues where
//! different Device instances for the same GPU cause CUDA errors.

use candle_core::{Device, Result, Tensor, DType};
use std::sync::OnceLock;

// The ONE true device instance
static THE_DEVICE: OnceLock<Device> = OnceLock::new();

/// Initialize the device wrapper with a specific device
pub fn init_device(device: Device) -> Result<()> {
    THE_DEVICE.set(device).map_err(|_| {
        candle_core::Error::Msg("Device already initialized".to_string())
    })?;
    Ok(())
}

/// Get the wrapped device
pub fn device() -> Result<Device> {
    THE_DEVICE.get()
        .cloned()
        .ok_or_else(|| candle_core::Error::Msg("Device not initialized".to_string()))
}

/// Wrapper for Tensor::new that always uses the cached device
pub fn tensor_new<T: candle_core::WithDType>(data: &[T], shape: &[usize]) -> Result<Tensor> {
    Tensor::new(data, device()?)?.reshape(shape)
}

/// Wrapper for zeros
pub fn zeros(shape: &[usize], dtype: DType) -> Result<Tensor> {
    Tensor::zeros(shape, dtype, &device()?)
}

/// Wrapper for ones
pub fn ones(shape: &[usize], dtype: DType) -> Result<Tensor> {
    Tensor::ones(shape, dtype, &device()?)
}

/// Wrapper for randn
pub fn randn(mean: f32, std: f32, shape: &[usize]) -> Result<Tensor> {
    Tensor::randn(mean, std, shape, &device()?)
}

/// Ensure a tensor is on the correct device
pub fn ensure_device(tensor: &Tensor) -> Result<Tensor> {
    let dev = device()?;
    if !tensor.device().same_device(&dev) {
        tensor.to_device(&dev)
    } else {
        Ok(tensor.clone())
    }
}