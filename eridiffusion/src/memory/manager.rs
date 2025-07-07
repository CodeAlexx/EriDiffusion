//! Memory management utilities for Candle (equivalent to PyTorch's cuda functions)

use candle_core::{Device, Result};

pub struct MemoryManager;

impl MemoryManager {
    /// Equivalent to torch.cuda.empty_cache()
    /// Forces CUDA to release cached memory back to the OS
    pub fn empty_cache() -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            // Candle uses cuDNN and CUDA directly
            // This will force synchronization and memory cleanup
            if let Ok(device) = Device::cuda_if_available(0) {
                device.synchronize()?;
            }
            
            // Use our CUDA memory pool's empty cache function
            if let Err(e) = crate::memory::cuda::empty_cache() {
                log::warn!("Failed to empty CUDA cache: {}", e);
            }
        }
        Ok(())
    }
    
    /// Equivalent to torch.cuda.ipc_collect()
    /// Cleans up inter-process communication handles
    pub fn ipc_collect() -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            // In Candle, IPC handles are managed automatically
            // This is more relevant for PyTorch's specific implementation
            // We can still force a synchronization
            if let Ok(device) = Device::cuda_if_available(0) {
                device.synchronize()?;
            }
        }
        Ok(())
    }
    
    /// Get current memory usage (equivalent to torch.cuda.memory_allocated())
    pub fn memory_allocated(device_id: usize) -> Result<usize> {
        #[cfg(feature = "cuda")]
        {
            // Use our memory pool via the cuda module
            if let Ok(allocated) = crate::memory::cuda::memory_allocated(Some(device_id as i32)) {
                return Ok(allocated);
            }
        }
        
        Ok(0)
    }
    
    /// Get memory statistics
    pub fn memory_stats(device_id: usize) -> Result<(usize, usize)> {
        #[cfg(feature = "cuda")]
        {
            use crate::memory::cuda::get_memory_info;
            if let Ok((free, total)) = get_memory_info(device_id as i32) {
                return Ok((free, total));
            }
        }
        
        Ok((0, 0))
    }
    
    /// Force garbage collection and memory cleanup
    pub fn cleanup() -> Result<()> {
        Self::empty_cache()?;
        Self::ipc_collect()?;
        
        // Force Rust to run cleanup
        // Note: Rust's memory management is deterministic unlike Python's GC
        // Dropping objects immediately frees memory
        
        Ok(())
    }
    
    /// Log current memory usage
    pub fn log_memory_usage(prefix: &str) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            if let Ok((free, total)) = Self::memory_stats(0) {
                let used = total - free;
                let used_gb = used as f32 / 1024.0 / 1024.0 / 1024.0;
                let total_gb = total as f32 / 1024.0 / 1024.0 / 1024.0;
                log::info!("{}: {:.2} GB / {:.2} GB", prefix, used_gb, total_gb);
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
        use $crate::memory::manager::MemoryManager;
        
        MemoryManager::log_memory_usage(&format!("Before {}", $name))?;
        let result = $body;
        MemoryManager::log_memory_usage(&format!("After {}", $name))?;
        
        result
    }};
}