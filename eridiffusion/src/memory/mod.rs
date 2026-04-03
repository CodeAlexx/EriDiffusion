use flame_core::device::Device;
use flame_core::optimizers::{Adam, SGD};
use flame_core::{CudaDevice, DType, Result, Shape, Tensor};
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

// GPU memory management for efficient training
//
// This module provides optimized memory allocation and pooling for
// diffusion model training, especially for Flux on 24GB GPUs.

pub mod block_swapping;
pub mod config;
pub mod cuda_allocator;
pub mod manager;
pub mod model_blocks;
pub mod pool;
pub mod profiler;
pub mod quanto;

pub use block_swapping::{BlockSwapConfig, BlockSwapManager, BlockType, SwapStats, SwappableBlock};
pub use config::{
    AttentionStrategy, DiffusionConfig, MemoryFormat, MemoryPoolConfig, PrecisionMode,
    QuantizationMode,
};
pub use cuda_allocator::CudaAllocator;
pub use manager::MemoryManager;
pub use model_blocks::{
    FluxMemoryBlock, MMDiTMemoryBlock, MemoryRequirements, ModelType, ShardingStrategy,
    WAN21VideoMemoryBlock,
};
pub use pool::{MemoryPool, MemoryStats};
pub use profiler::{MemoryEvent, MemoryEventType, MemoryProfiler};
pub use quanto::{QuantizedTensor, QuantoConfig, QuantoManager};

/// Global GPU memory manager
pub struct GpuMemoryManager {
    pools: RwLock<std::collections::HashMap<i32, Arc<RwLock<MemoryPool>>>>,
}

impl GpuMemoryManager {
    pub fn new() -> Self {
        Self { pools: RwLock::new(HashMap::new()) }
    }

    pub fn new_with_device(device: &CudaDevice) -> Self {
        Self { pools: RwLock::new(HashMap::new()) }
    }

    pub fn set_device(&self, device_id: i32) -> flame_core::Result<()> {
        // Note: current_device field doesn't exist in the struct
        // This would need to be added if device switching is needed
        Ok(())
    }

    pub fn current_device(&self) -> i32 {
        // Default to device 0 for now
        0
    }

    pub fn get_or_create_pool(
        &self,
        device_id: i32,
        config: Option<MemoryPoolConfig>,
    ) -> flame_core::Result<Arc<RwLock<MemoryPool>>> {
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

    pub fn empty_cache_all_devices(&self) -> flame_core::Result<()> {
        let pools = self.pools.read().unwrap();

        for pool in pools.values() {
            pool.read().unwrap().empty_cache()?;
        }

        Ok(())
    }
}

// Global instance
static GLOBAL_MEMORY_MANAGER: Lazy<GpuMemoryManager> = Lazy::new(|| GpuMemoryManager::new());

/// Public API for CUDA memory management
pub fn global_memory_manager() -> &'static GpuMemoryManager {
    &GLOBAL_MEMORY_MANAGER
}
