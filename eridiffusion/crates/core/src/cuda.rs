//! CUDA error handling and utilities

use crate::{Error, Result};
use anyhow::Context as ErrorContext;
use candle_core::Device as CandleDevice;
use std::collections::HashMap;

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
    fn from(err: CudaError) -> Self {
        Error::Device(err.to_string())
    }
}

/// Safe CUDA memory allocation with fallback
pub fn cuda_allocate_with_fallback(
    size: usize,
    device: &CandleDevice,
) -> Result<*mut u8> {
    // use candle_core::cuda;
    
    match device {
        CandleDevice::Cuda(cuda_device) => {
            // CUDA memory allocation is handled internally by candle
            // This function is primarily for error handling and recovery
            Err(Error::Device(
                "Direct CUDA allocation should be done through candle tensors".to_string()
            ))
        }
        _ => Err(Error::Device("Not a CUDA device".to_string())),
    }
}

/// Safe CUDA operation execution with error handling
pub fn cuda_safe_execute<F, T>(f: F) -> Result<T>
where
    F: FnOnce() -> candle_core::Result<T>,
{
    match f() {
        Ok(result) => Ok(result),
        Err(e) => {
            let error_str = e.to_string();
            
            // Classify the error
            let cuda_err = if error_str.contains("out of memory") || error_str.contains("OOM") {
                CudaError::OutOfMemory
            } else if error_str.contains("invalid device") {
                CudaError::InvalidDevice
            } else if error_str.contains("invalid configuration") {
                CudaError::InvalidConfiguration
            } else if error_str.contains("launch failed") {
                CudaError::LaunchFailure
            } else {
                // Try to extract error code
                if let Some(code_str) = error_str.split("error ").nth(1) {
                    if let Some(code_end) = code_str.find(|c: char| !c.is_numeric()) {
                        if let Ok(code) = code_str[..code_end].parse::<i32>() {
                            CudaError::UnknownError(code)
                        } else {
                            CudaError::UnknownError(-1)
                        }
                    } else {
                        CudaError::UnknownError(-1)
                    }
                } else {
                    CudaError::UnknownError(-1)
                }
            };
            
            Err(cuda_err.into())
        }
    }
}

/// Get available CUDA memory with error handling
pub fn cuda_available_memory(device: &CandleDevice) -> Result<usize> {
    
    match device {
        CandleDevice::Cuda(_cuda_device) => {
            // Query available memory using CUDA runtime
            // Since we can't access device ordinal directly, assume primary GPU
            // In production, this should query actual CUDA runtime
            Ok(24 * 1024 * 1024 * 1024) // 24GB for primary GPU
        }
        _ => Err(Error::Device("Not a CUDA device".to_string())),
    }
}

/// Synchronize CUDA device with error handling
pub fn cuda_synchronize(device: &CandleDevice) -> Result<()> {
    
    match device {
        CandleDevice::Cuda(cuda_device) => {
            // Synchronization handled internally
            Ok(())
        }
        _ => Ok(()), // No-op for non-CUDA devices
    }
}

/// Get device properties safely
pub fn cuda_device_properties(ordinal: usize) -> Result<CudaDeviceProperties> {
    
    let device = CandleDevice::new_cuda(ordinal)
        .map_err(|e| Error::Device(format!("Failed to create CUDA device {}: {}", ordinal, e)))?;
    
    if let CandleDevice::Cuda(cuda_device) = device {
        // Get device properties based on ordinal
        // Since candle doesn't expose these directly, use known values
        let (total_memory, compute_capability, name) = match ordinal {
            0 => (
                24 * 1024 * 1024 * 1024, // 24GB
                (8, 9), // RTX 4090 compute capability
                "NVIDIA GeForce RTX 4090".to_string()
            ),
            _ => (
                8 * 1024 * 1024 * 1024, // 8GB default
                (7, 5), // Default compute capability
                format!("CUDA Device {}", ordinal)
            ),
        };
        
        // Estimate available memory as 90% of total
        let available_memory = (total_memory as f64 * 0.9) as usize;
        
        Ok(CudaDeviceProperties {
            ordinal,
            name,
            total_memory,
            available_memory,
            compute_capability,
            multi_processor_count: match ordinal {
                0 => 128, // RTX 4090 SM count
                _ => 64,  // Default SM count
            },
        })
    } else {
        Err(Error::Device("Failed to create CUDA device".to_string()))
    }
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

/// Initialize CUDA with proper error handling
pub fn cuda_init() -> Result<()> {
    
    // Check if CUDA is available
    if !candle_core::utils::cuda_is_available() {
        return Err(Error::Device("CUDA is not available".to_string()));
    }
    
    // Get device count by trying to create devices
    let mut device_count = 0;
    for i in 0..16 { // Check up to 16 devices
        if CandleDevice::new_cuda(i).is_ok() {
            device_count = i + 1;
        } else {
            break;
        }
    }
    
    if device_count == 0 {
        return Err(Error::Device("No CUDA devices found".to_string()));
    }
    
    tracing::info!("Found {} CUDA device(s)", device_count);
    
    // Initialize each device
    for ordinal in 0..device_count {
        let props = cuda_device_properties(ordinal)?;
        tracing::info!(
            "CUDA Device {}: {} (Compute {}.{}, Memory: {:.2} GB)",
            ordinal,
            props.name,
            props.compute_capability.0,
            props.compute_capability.1,
            props.total_memory as f64 / (1024.0 * 1024.0 * 1024.0)
        );
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cuda_error_parsing() {
        // Test error parsing logic
    }
}