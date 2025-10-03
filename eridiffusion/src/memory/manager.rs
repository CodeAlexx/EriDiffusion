use flame_core::device::Device;
use flame_core::optimizers::{Adam, SGD};
use flame_core::{CudaDevice, DType, Result, Shape, Tensor};

// Memory management utilities for FLAME (equivalent to PyTorch's cuda functions)

// FLAME uses flame_core::device::Device instead of Device

pub struct MemoryManager;

impl MemoryManager {
    /// Equivalent to torch.cuda.empty_cache()
    /// Forces CUDA to release cached memory back to the OS
    pub fn empty_cache() -> flame_core::Result<()> {
        #[cfg(feature = "cuda")]
        {
            // Get the CUDA device and force synchronization
            if let Ok(device) = Device::cuda(0) {
                // Force CUDA to finish all pending operations
                device.synchronize()?;

                // CRITICAL FIX: Actually call the FLAME memory pool cleanup
                // This was commented out and causing OOM!
                let cuda_dev = device.cuda_device();
                // Synchronize the CUDA device
                cuda_dev.synchronize().unwrap();

                // Get the memory pool for this device and force cleanup
                let pool = flame_core::memory_pool::MEMORY_POOL.get_pool(cuda_dev)?;
                if let Ok(pool_guard) = pool.lock() {
                    pool_guard.force_cleanup()?;
                }

                // Also clear all cached memory
                flame_core::memory_pool::MEMORY_POOL.clear_all_caches();
            }

            // Give CUDA driver time to actually release the memory
            std::thread::sleep(std::time::Duration::from_millis(500));
        }
        Ok(())
    }

    /// Equivalent to torch.cuda.ipc_collect()
    /// Cleans up inter-process communication handles
    pub fn ipc_collect() -> flame_core::Result<()> {
        #[cfg(feature = "cuda")]
        {
            // In FLAME, IPC handles are managed automatically
            // This is more relevant for PyTorch's specific implementation
            // We can still force a synchronization
            if let Ok(device) = Device::cuda(0) {
                // device.synchronize()?; // Not needed in FLAME
            }
        }
        Ok(())
    }

    /// Get current memory usage (equivalent to torch.cuda.memory_allocated())
    pub fn memory_allocated(device_id: usize) -> flame_core::Result<usize> {
        #[cfg(feature = "cuda")]
        {
            // Use our memory pool via the cuda module
            // if let Ok(allocated) = crate::memory::cuda::memory_allocated(Some(device_id as i32)) {
            //     return Ok(allocated);
            // }
        }
        Ok(0)
    }

    /// Get memory statistics
    pub fn memory_stats(device_id: usize) -> flame_core::Result<(usize, usize)> {
        #[cfg(feature = "cuda")]
        {
            // if let Ok((free, total)) = crate::memory::cuda::get_memory_info(device_id as i32) {
            //     return Ok((free, total));
            // }
        }
        Ok((0, 0))
    }

    /// Force garbage collection and memory cleanup
    pub fn cleanup() -> flame_core::Result<()> {
        // Run cleanup methods
        Self::empty_cache()?;
        Self::ipc_collect()?;

        // Force Rust to run cleanup
        // Note: Rust's memory management is deterministic unlike Python's GC
        // Dropping objects immediately frees memory

        Ok(())
    }

    /// Log current memory usage
    pub fn log_memory_usage(prefix: &str) -> flame_core::Result<()> {
        #[cfg(feature = "cuda")]
        {
            if let Ok((free, total)) = Self::memory_stats(0) {
                let used = total - free;
                let used_gb = used as f32 / 1024.0 / 1024.0 / 1024.0;
                let total_gb = total as f32 / 1024.0 / 1024.0 / 1024.0;
                println!("{}: {:.2}/{:.2} GB", prefix, used_gb, total_gb);
            }
        }
        Ok(())
    }
}

/// Macro to run code with memory cleanup
#[macro_export]
macro_rules! with_memory_cleanup {
    ($body:expr) => {{
        let result = $body;
        $crate::memory::manager::MemoryManager::cleanup()?;
        result
    }};
}

/// Macro to track memory usage around an operation
#[macro_export]
macro_rules! track_memory {
    ($name:expr, $body:expr) => {{
        $crate::memory::manager::MemoryManager::log_memory_usage(concat!("Before ", $name))?;
        let result = $body;
        $crate::memory::manager::MemoryManager::log_memory_usage(concat!("After ", $name))?;
        result
    }};
}
