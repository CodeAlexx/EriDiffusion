//! GPU memory management for efficient training
//! 
//! This module provides optimized memory allocation and pooling for
//! diffusion model training, especially for Flux on 24GB GPUs.

pub mod pool;
pub mod cuda_allocator;
pub mod profiler;
pub mod config;
pub mod block_swapping;
pub mod model_blocks;
pub mod manager;
pub mod quanto;

pub use pool::{MemoryPool, MemoryStats};
pub use cuda_allocator::CudaAllocator;
pub use profiler::{MemoryProfiler, MemoryEvent, MemoryEventType};
pub use config::{MemoryPoolConfig, DiffusionConfig, PrecisionMode, AttentionStrategy, MemoryFormat, QuantizationMode};
pub use block_swapping::{BlockSwapManager, BlockSwapConfig, SwappableBlock, BlockType, SwapStats};
pub use model_blocks::{ModelType, MemoryRequirements, FluxMemoryBlock, MMDiTMemoryBlock, WAN21VideoMemoryBlock, ShardingStrategy};
pub use manager::MemoryManager;
pub use quanto::{QuantoManager, QuantoConfig, QuantizedTensor};

use std::sync::Arc;
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::{Mutex, RwLock};

/// Global GPU memory manager
pub struct GpuMemoryManager {
    pools: RwLock<HashMap<i32, Arc<RwLock<MemoryPool>>>>,
    current_device: Mutex<i32>,
}

impl GpuMemoryManager {
    pub fn new() -> Self {
        Self {
            pools: RwLock::new(HashMap::new()),
            current_device: Mutex::new(0),
        }
    }

    pub fn set_device(&self, device_id: i32) -> Result<(), cuda_allocator::MemoryError> {
        *self.current_device.lock().unwrap() = device_id;
        Ok(())
    }

    pub fn current_device(&self) -> i32 {
        *self.current_device.lock().unwrap()
    }

    pub fn get_or_create_pool(&self, device_id: i32, config: Option<MemoryPoolConfig>) -> Result<Arc<RwLock<MemoryPool>>, cuda_allocator::MemoryError> {
        let pools = self.pools.read().unwrap();
        
        if let Some(pool) = pools.get(&device_id) {
            return Ok(pool.clone());
        }
        
        drop(pools);
        
        let config = config.unwrap_or_default();
        let pool = Arc::new(RwLock::new(MemoryPool::new(device_id, config)?));
        
        {
            let mut pools = self.pools.write().unwrap();
            pools.insert(device_id, pool.clone());
        }
        
        Ok(pool)
    }

    pub fn empty_cache_all_devices(&self) -> Result<(), cuda_allocator::MemoryError> {
        let pools = self.pools.read().unwrap();
        
        for pool in pools.values() {
            pool.read().unwrap().empty_cache()?;
        }
        
        Ok(())
    }
}

// Global instance
static GLOBAL_MEMORY_MANAGER: Lazy<GpuMemoryManager> = Lazy::new(|| {
    GpuMemoryManager::new()
});

/// Public API for CUDA memory management
pub mod cuda {
    use super::*;
    use crate::memory::config::PrecisionMode;

    pub fn current_device() -> i32 {
        GLOBAL_MEMORY_MANAGER.current_device()
    }

    pub fn set_device(device: i32) -> Result<(), cuda_allocator::MemoryError> {
        GLOBAL_MEMORY_MANAGER.set_device(device)
    }

    pub fn empty_cache() -> Result<(), cuda_allocator::MemoryError> {
        let device_id = GLOBAL_MEMORY_MANAGER.current_device();
        let pool = GLOBAL_MEMORY_MANAGER.get_or_create_pool(device_id, None)?;
        let result = pool.write().unwrap().empty_cache();
        result
    }

    pub fn memory_allocated(device: Option<i32>) -> Result<usize, cuda_allocator::MemoryError> {
        let device = device.unwrap_or_else(|| GLOBAL_MEMORY_MANAGER.current_device());
        let pool = GLOBAL_MEMORY_MANAGER.get_or_create_pool(device, None)?;
        let stats = pool.read().unwrap().get_stats();
        Ok(stats.allocated_bytes)
    }

    pub fn memory_reserved(device: Option<i32>) -> Result<usize, cuda_allocator::MemoryError> {
        let device = device.unwrap_or_else(|| GLOBAL_MEMORY_MANAGER.current_device());
        let pool = GLOBAL_MEMORY_MANAGER.get_or_create_pool(device, None)?;
        let stats = pool.read().unwrap().get_stats();
        Ok(stats.reserved_bytes)
    }

    pub fn get_recommended_batch_size(per_sample_memory: usize) -> Result<usize, cuda_allocator::MemoryError> {
        let device_id = GLOBAL_MEMORY_MANAGER.current_device();
        let pool = GLOBAL_MEMORY_MANAGER.get_or_create_pool(device_id, None)?;
        let batch_size = pool.read().unwrap().get_recommended_batch_size(per_sample_memory);
        Ok(batch_size)
    }

    pub fn allocate_for_attention(seq_len: usize, batch_size: usize, head_dim: usize) -> Result<*mut std::ffi::c_void, cuda_allocator::MemoryError> {
        let device_id = GLOBAL_MEMORY_MANAGER.current_device();
        let pool = GLOBAL_MEMORY_MANAGER.get_or_create_pool(device_id, None)?;
        let result = pool.read().unwrap().allocate_for_attention(seq_len, batch_size, head_dim);
        result
    }

    pub fn clear_gradients() -> Result<(), cuda_allocator::MemoryError> {
        let device_id = GLOBAL_MEMORY_MANAGER.current_device();
        let pool = GLOBAL_MEMORY_MANAGER.get_or_create_pool(device_id, None)?;
        let result = pool.read().unwrap().clear_gradients();
        result
    }
    
    pub fn get_memory_pool(device_id: i32) -> Result<Arc<RwLock<MemoryPool>>, cuda_allocator::MemoryError> {
        GLOBAL_MEMORY_MANAGER.get_or_create_pool(device_id, None)
    }
    
    pub fn get_memory_info(device_id: i32) -> Result<(usize, usize), cuda_allocator::MemoryError> {
        let pool = GLOBAL_MEMORY_MANAGER.get_or_create_pool(device_id, None)?;
        let stats = pool.read().unwrap().get_stats();
        // This is a simplified version - ideally we'd call CUDA APIs directly
        // For now, return reserved memory as "used" and a fixed total
        let total: usize = 24 * 1024 * 1024 * 1024; // 24GB
        let free = total.saturating_sub(stats.reserved_bytes);
        Ok((free, total))
    }
    
    pub fn device_count() -> Result<usize, cuda_allocator::MemoryError> {
        // Now actually tries to count CUDA devices
        let mut count = 0;
        
        // Try to create devices until we fail
        for i in 0..128 { // Reasonable upper limit
            match candle_core::Device::new_cuda(i) {
                Ok(_) => count += 1,
                Err(_) => break,
            }
        }
        
        // If CUDA_VISIBLE_DEVICES is set, respect it
        if let Ok(visible) = std::env::var("CUDA_VISIBLE_DEVICES") {
            if !visible.is_empty() && visible != "-1" {
                // Count comma-separated device IDs
                let visible_count = visible.split(',').count();
                count = count.min(visible_count);
            }
        }
        
        if count == 0 {
            Err(cuda_allocator::MemoryError::CudaError("No CUDA devices found".to_string()))
        } else {
            Ok(count)
        }
    }
}