//! Backend interop hooks to bridge external device memory into Flame tensors.

use crate::{Result, Error};
use flame_core::{Tensor, Shape, DType, CudaDevice};

/// SAFETY: `ptr` must be a valid device buffer with at least
/// `dtype.size_in_bytes() * shape.elem_count()` bytes on `dev`.
#[inline]
pub unsafe fn tensor_from_device_ptr(
    ptr: *mut u8,
    shape: Shape,
    dtype: DType,
    dev: CudaDevice,
) -> Result<Tensor> {
    // Delegate to Flame’s unsafe constructor. This will return an error
    // until the backend implements external storage adoption.
    flame_core::Tensor::from_device_ptr_unsafe(ptr, shape, dtype, dev.into())
        .map_err(|e| Error::Device(format!("tensor_from_device_ptr: {}", e)))
}
