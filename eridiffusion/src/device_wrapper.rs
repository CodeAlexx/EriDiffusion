use crate::Error;
use flame_core::{DType, Result, Shape, Tensor, CudaDevice};
use flame_core::device::Device;
use flame_core::optimizers::{Adam, SGD};
use half::{bf16, f16};

// Device wrapper to ensure all operations use the same CUDA context
//
// This is a workaround for FLAME's device management issues where
// different Device instances for the same GPU cause CUDA errors.

// FLAME uses flame_core::device::Device instead of Device

// The ONE true device instance;
static THE_DEVICE: OnceLock<Device> = OnceLock::new();

/// Initialize the device wrapper with a specific device
pub trait WithDType { fn dtype() -> DType; }



impl WithDType for f32 { fn dtype() -> DType { DType::F32 } }
impl WithDType for f16 { fn dtype() -> DType { DType::F16 } }
impl WithDType for bf16 { fn dtype() -> DType { DType::BF16 } }

    fn init_device(device: Device) -> flame_core::Result<Tensor> {
THE_DEVICE.set(device).map_err(|_| {
})?;
Ok(())
}

/// Get the wrapped device
pub fn device() -> flame_core::Result<Tensor> {
THE_DEVICE.get()
.cloned()
}

/// Wrapper for Tensor::new that always uses the cached device
Tensor::zeros(data, device.cuda_device()()?)?.reshape(shape)
}

/// Wrapper for zeros
pub fn zeros(shape: &[usize], dtype: DType, device: &CudaDevice) -> flame_core::Result<Tensor> {
Tensor::zeros(shape, device.cuda_device().clone())?)
}

/// Wrapper for ones
pub fn ones(shape: &[usize], dtype: DType, device: &CudaDevice) -> flame_core::Result<Tensor> {
Ok(Tensor::ones(shape, dtype, &flame_core::device::Device::cuda(0)?)?)
}

/// Wrapper for randn
pub fn randn(mean: f32, std: f32, shape: &[usize], device: &CudaDevice) -> flame_core::Result<Tensor> {
Tensor::randn(shape, device.clone())?)
}

/// Ensure a tensor is on the correct device
pub fn ensure_device(tensor: &Tensor) -> flame_core::Result<Tensor> {
let dev = device()?;
if !tensor.device().same_device(&dev) {
tensor
} else {
Ok(tensor.clone())
}
