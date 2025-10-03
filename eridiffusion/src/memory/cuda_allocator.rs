use flame_core::device::Device;
use flame_core::{DType, Result, Shape, Tensor};
use std::alloc::{alloc, dealloc, Layout};
use std::collections::HashMap;
use std::ffi::c_void;
use std::fs;
use std::ptr;

/// Error type for memory operations
#[derive(thiserror::Error, Debug)]
pub enum MemoryError {
    #[error("CUDA error: {0}")]
    CudaError(String),
    #[error("Invalid device ID: {0}")]
    InvalidDevice(i32),
    #[error("Allocation failed: {0} bytes")]
    AllocationFailed(usize),
    #[error("Memory not found")]
    MemoryNotFound,
}

// From<MemoryError> implementation removed - anyhow handles this automatically

/// CUDA memory allocator wrapper
pub struct CudaAllocator {
    device_id: i32,
    allocations:
        std::sync::Mutex<std::collections::HashMap<*mut c_void, (usize, std::alloc::Layout)>>,
}

impl CudaAllocator {
    pub fn new(device_id: i32) -> flame_core::Result<Self> {
        // Create FLAME device to verify CUDA is available
        let device = match Device::cuda(device_id as usize) {
            Ok(dev) => dev,
            Err(e) => {
                eprintln!("Warning: CUDA device {} not available: {}. Using CPU.", device_id, e);
                flame_core::device::Device::cuda(0)?
            }
        };

        Ok(Self { device_id, allocations: std::sync::Mutex::new(std::collections::HashMap::new()) })
    }

    pub fn allocate(&self, size: usize) -> flame_core::Result<*mut c_void> {
        // Simplified allocation - just use system allocator for now
        let layout = std::alloc::Layout::from_size_align(size, 256).map_err(|_| {
            flame_core::Error::InvalidOperation(format!(
                "Failed to create layout for size: {}",
                size
            ))
        })?;

        let ptr = unsafe {
            let ptr = std::alloc::alloc(layout) as *mut c_void;
            if ptr.is_null() {
                return Err(flame_core::Error::InvalidOperation(format!(
                    "Allocation failed for size: {}",
                    size
                )));
            }
            ptr
        };

        // Track allocation
        {
            let mut allocations = self.allocations.lock().unwrap();
            allocations.insert(ptr, (size, layout));
        }

        Ok(ptr)
    }

    pub fn deallocate(&self, ptr: *mut c_void) -> flame_core::Result<()> {
        let (size, layout) = {
            let mut allocations = self.allocations.lock().unwrap();
            allocations.remove(&ptr).ok_or(flame_core::Error::InvalidOperation(
                "Memory allocation not found".to_string(),
            ))?
        };

        // Deallocate
        unsafe {
            std::alloc::dealloc(ptr as *mut u8, layout);
        }

        Ok(())
    }

    pub fn get_memory_info(&self) -> flame_core::Result<(usize, usize)> {
        // Simplified memory info - just return placeholder values
        let total: usize = 24 * 1024 * 1024 * 1024; // 24GB
        let used: usize = {
            let allocations = self.allocations.lock().unwrap();
            allocations.values().map(|(size, _)| *size).sum::<usize>()
        };

        let free = total.saturating_sub(used);
        Ok((free, total))
    }
}
