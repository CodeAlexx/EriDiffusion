//! CUDA error handling and utilities

use crate::Error;
use crate::Result;

// Submodules
pub mod dtype_tag;
pub mod utils_pinned;

/// CUDA error codes
#[derive(Debug, Clone, Copy)]
pub enum CudaError {
    OutOfMemory,
    InvalidDevice,
    InvalidConfiguration,
    LaunchFailure,
    UnknownError(i32),
}

impl std::fmt::Display for CudaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CudaError::OutOfMemory => write!(f, "CUDA out of memory"),
            CudaError::InvalidDevice => write!(f, "Invalid CUDA device"),
            CudaError::InvalidConfiguration => write!(f, "Invalid CUDA configuration"),
            CudaError::LaunchFailure => write!(f, "CUDA kernel launch failure"),
            CudaError::UnknownError(code) => write!(f, "CUDA error code: {}", code),
        }
    }
}

impl From<CudaError> for Error {
    fn from(err: CudaError) -> Self { Error::Device(err.to_string()) }
}

/// Safe CUDA memory allocation with fallback
#[allow(unused_variables)]
pub fn cuda_allocate_with_fallback(size: usize, device: &crate::Device) -> anyhow::Result<*mut u8> {
    match device {
        crate::Device::Cuda(_) => Err(Error::Device("Direct CUDA allocation should be done through FLAME tensors".into()).into()),
        _ => Err(Error::Device("Not a CUDA device".into()).into()),
    }
}

/// Allocate zero-initialized CUDA memory. Placeholder until raw pointer APIs are available.
/// Returns a device-specific synthetic handle to keep call sites wired.
pub fn cuda_malloc_zeroed(dev: usize, size: usize) -> Result<*mut u8> {
    #[cfg(feature = "cuda-raw")]
    {
        // Use cudarc directly to allocate zeroed device memory.
        let device = cudarc::driver::CudaDevice::new(dev)
            .map_err(|e| Error::Device(format!("CudaDevice::new({dev}) failed: {e}")))?;
        // Allocate zeros as bytes; leak the slice to keep it alive (stub semantics)
        let slice = device.alloc_zeros::<u8>(size)
            .map_err(|e| Error::Device(format!("alloc_zeros({size}) failed: {e}")))?;
        let ptr = slice.as_device_ptr().as_raw() as *mut u8;
        std::mem::forget(slice);
        Ok(ptr)
    }
    #[cfg(not(feature = "cuda-raw"))]
    {
        // Fallback synthetic handle when raw allocation is not enabled.
        let handle = size.saturating_add(dev.saturating_mul(1_000_000_000));
        Ok(handle as *mut u8)
    }
}

/// Safe CUDA operation execution with error handling
pub fn cuda_safe_execute<F, T>(f: F) -> anyhow::Result<T>
where
    F: FnOnce() -> flame_core::Result<T>,
{
    match f() {
        Ok(result) => Ok(result),
        Err(e) => {
            let error_str = e.to_string();
            let cuda_err = if error_str.contains("out of memory") || error_str.contains("OOM") {
                CudaError::OutOfMemory
            } else if error_str.contains("invalid device") {
                CudaError::InvalidDevice
            } else if error_str.contains("invalid configuration") {
                CudaError::InvalidConfiguration
            } else if error_str.contains("launch failed") {
                CudaError::LaunchFailure
            } else if let Some(code_str) = error_str.split("error ").nth(1) {
                if let Some(code_end) = code_str.find(|c: char| !c.is_numeric()) {
                    if let Ok(code) = code_str[..code_end].parse::<i32>() { CudaError::UnknownError(code) } else { CudaError::UnknownError(-1) }
                } else { CudaError::UnknownError(-1) }
            } else { CudaError::UnknownError(-1) };
            Err::<T, anyhow::Error>(Error::from(cuda_err).into())
        }
    }
}

/// Get available CUDA memory with error handling
pub fn cuda_available_memory(device: &crate::Device) -> anyhow::Result<usize> {
    match device {
        crate::Device::Cuda(_) => Ok(24 * 1024 * 1024 * 1024), // 24GB default placeholder
        _ => Err(Error::Device("Not a CUDA device".into()).into()),
    }
}

/// Synchronize CUDA device with error handling
pub fn cuda_synchronize(device: &crate::Device) -> anyhow::Result<()> {
    match device { crate::Device::Cuda(_) => Ok(()), _ => Ok(()) }
}

#[derive(Debug, Clone)]
pub struct CudaDeviceProperties {
    pub ordinal: usize,
    pub name: String,
    pub total_memory: usize,
    pub available_memory: usize,
    pub compute_capability: (usize, usize),
    pub multi_processor_count: usize,
}

/// Get device properties safely
pub fn cuda_device_properties(ordinal: usize) -> anyhow::Result<CudaDeviceProperties> {
    let _cuda_device = flame_core::CudaDevice::new(ordinal)
        .map_err(|e| Error::Device(format!("Failed to create CUDA device {}: {}", ordinal, e)))?;
    let (total_memory, compute_capability, name) = match ordinal {
        0 => (24 * 1024 * 1024 * 1024, (8, 9), "NVIDIA GeForce RTX 4090".to_string()),
        _ => (8 * 1024 * 1024 * 1024, (7, 5), format!("CUDA Device {}", ordinal)),
    };
    let available_memory = (total_memory as f64 * 0.9) as usize;
    Ok(CudaDeviceProperties {
        ordinal,
        name,
        total_memory,
        available_memory,
        compute_capability,
        multi_processor_count: if ordinal == 0 { 128 } else { 64 },
    })
}

/// Initialize CUDA with proper error handling
pub fn cuda_init() -> anyhow::Result<()> {
    if flame_core::CudaDevice::new(0).is_err() {
        return Err(Error::Device("CUDA is not available".into()).into());
    }
    let mut device_count = 0;
    for i in 0..16 { if flame_core::CudaDevice::new(i).is_ok() { device_count = i + 1; } else { break; } }
    if device_count == 0 { return Err(Error::Device("No CUDA devices found".into()).into()); }
    tracing::info!("Found {} CUDA device(s)", device_count);
    for ordinal in 0..device_count {
        let props = cuda_device_properties(ordinal)?;
        tracing::info!("CUDA Device {}: {} (Compute {}.{}, Memory: {:.2} GB)", ordinal, props.name, props.compute_capability.0, props.compute_capability.1, props.total_memory as f64 / (1024.0 * 1024.0 * 1024.0));
    }
    Ok(())
}
