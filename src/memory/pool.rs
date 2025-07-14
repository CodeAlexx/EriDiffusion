//! Memory pool for efficient allocation and caching

use super::cuda_allocator::{CudaAllocator, MemoryError, Result};
use super::config::{MemoryPoolConfig, DiffusionConfig, PrecisionMode, MemoryFormat};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Instant;
use std::ffi::c_void;

/// Memory allocation statistics
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    pub allocated_bytes: usize,
    pub reserved_bytes: usize,
    pub cached_bytes: usize,
    pub peak_allocated: usize,
    pub peak_reserved: usize,
    pub allocation_count: u64,
    pub deallocation_count: u64,
    pub cache_hit_count: u64,
    pub cache_miss_count: u64,
    pub fragmentation_ratio: f64,
}

/// Memory block metadata
#[derive(Debug, Clone)]
struct MemoryBlock {
    ptr: *mut c_void,
    size: usize,
    allocated_time: Instant,
    last_accessed: Instant,
    ref_count: u32,
    is_cached: bool,
    device_id: i32,
    precision: PrecisionMode,
    memory_format: MemoryFormat,
    is_gradient: bool,
    is_activation: bool,
    checkpoint_eligible: bool,
    attention_block: bool,
}

unsafe impl Send for MemoryBlock {}
unsafe impl Sync for MemoryBlock {}

/// Memory pool for efficient allocation and caching
pub struct MemoryPool {
    device_id: i32,
    config: MemoryPoolConfig,
    diffusion_config: Option<DiffusionConfig>,
    allocator: CudaAllocator,
    allocated_blocks: RwLock<HashMap<*mut c_void, MemoryBlock>>,
    cached_blocks: RwLock<Vec<MemoryBlock>>,
    gradient_blocks: RwLock<HashMap<*mut c_void, MemoryBlock>>,
    activation_blocks: RwLock<HashMap<*mut c_void, MemoryBlock>>,
    attention_cache: RwLock<HashMap<(usize, usize), *mut c_void>>,
    stats: RwLock<MemoryStats>,
}

unsafe impl Send for MemoryPool {}
unsafe impl Sync for MemoryPool {}

impl MemoryPool {
    pub fn new(device_id: i32, config: MemoryPoolConfig) -> Result<Self> {
        let allocator = CudaAllocator::new(device_id)?;
        
        Ok(Self {
            device_id,
            config,
            diffusion_config: None,
            allocator,
            allocated_blocks: RwLock::new(HashMap::new()),
            cached_blocks: RwLock::new(Vec::new()),
            gradient_blocks: RwLock::new(HashMap::new()),
            activation_blocks: RwLock::new(HashMap::new()),
            attention_cache: RwLock::new(HashMap::new()),
            stats: RwLock::new(MemoryStats::default()),
        })
    }

    pub fn configure_for_diffusion(&mut self, diffusion_config: DiffusionConfig) -> Result<()> {
        self.diffusion_config = Some(diffusion_config.clone());
        
        // Pre-warm attention cache based on expected sequence lengths
        if diffusion_config.enable_flash_attention {
            self.prewarm_attention_cache(
                diffusion_config.max_sequence_length, 
                diffusion_config.batch_size
            )?;
        }
        
        Ok(())
    }

    fn prewarm_attention_cache(&self, max_seq_len: usize, batch_size: usize) -> Result<()> {
        let mut cache = self.attention_cache.write().unwrap();
        
        // Common sequence lengths for diffusion models
        let common_seq_lens = [64, 256, 1024, 4096, max_seq_len];
        
        for &seq_len in &common_seq_lens {
            if seq_len <= max_seq_len {
                let key = (seq_len, batch_size);
                let size = seq_len * batch_size * 16 * 2; // 2 bytes for f16
                
                match self.allocator.allocate(size) {
                    Ok(ptr) => {
                        cache.insert(key, ptr);
                    }
                    Err(_) => break,
                }
            }
        }
        
        Ok(())
    }

    pub fn allocate(&self, size: usize) -> Result<*mut c_void> {
        // Try to find a suitable cached block first
        if let Some(block) = self.find_cached_block(size) {
            self.update_stats_cache_hit(size);
            return Ok(block.ptr);
        }

        // Allocate new block
        let ptr = self.allocator.allocate(size)?;
        let block = MemoryBlock {
            ptr,
            size,
            allocated_time: Instant::now(),
            last_accessed: Instant::now(),
            ref_count: 1,
            is_cached: false,
            device_id: self.device_id,
            precision: PrecisionMode::Float32,
            memory_format: MemoryFormat::Contiguous,
            is_gradient: false,
            is_activation: false,
            checkpoint_eligible: false,
            attention_block: false,
        };

        {
            let mut allocated = self.allocated_blocks.write().unwrap();
            allocated.insert(ptr, block);
        }

        self.update_stats_allocation(size);
        
        // Check if we need garbage collection
        self.maybe_garbage_collect()?;
        
        Ok(ptr)
    }

    pub fn allocate_for_attention(&self, seq_len: usize, batch_size: usize, head_dim: usize) -> Result<*mut c_void> {
        let key = (seq_len, batch_size);
        
        // Check attention cache first
        {
            let cache = self.attention_cache.read().unwrap();
            if let Some(&ptr) = cache.get(&key) {
                self.update_stats_cache_hit(seq_len * batch_size * head_dim * 2);
                return Ok(ptr);
            }
        }
        
        // Allocate optimized memory for attention
        let size = seq_len * batch_size * head_dim * 2; // FP16 for memory efficiency
        let ptr = self.allocator.allocate(size)?;
        
        let block = MemoryBlock {
            ptr,
            size,
            allocated_time: Instant::now(),
            last_accessed: Instant::now(),
            ref_count: 1,
            is_cached: false,
            device_id: self.device_id,
            precision: PrecisionMode::Float16,
            memory_format: MemoryFormat::ChannelsLast,
            is_gradient: false,
            is_activation: true,
            checkpoint_eligible: false,
            attention_block: true,
        };

        {
            let mut allocated = self.allocated_blocks.write().unwrap();
            allocated.insert(ptr, block);
        }

        self.update_stats_allocation(size);
        
        // Cache for future use if it's a common size
        if seq_len <= 4096 {
            let mut cache = self.attention_cache.write().unwrap();
            cache.insert(key, ptr);
        }
        
        Ok(ptr)
    }

    pub fn deallocate(&self, ptr: *mut c_void) -> Result<()> {
        let block = {
            let mut allocated = self.allocated_blocks.write().unwrap();
            allocated.remove(&ptr)
        };

        if let Some(mut block) = block {
            block.ref_count -= 1;
            
            if block.ref_count == 0 {
                // Move to cache instead of immediate deallocation
                block.is_cached = true;
                block.last_accessed = Instant::now();
                
                {
                    let mut cached = self.cached_blocks.write().unwrap();
                    cached.push(block.clone());
                }
                
                self.update_stats_deallocation(block.size);
            }
        }
        
        Ok(())
    }

    pub fn clear_gradients(&self) -> Result<()> {
        let gradient_blocks = {
            let mut gradients = self.gradient_blocks.write().unwrap();
            let blocks: Vec<_> = gradients.drain().collect();
            blocks
        };

        for (ptr, _block) in gradient_blocks {
            self.allocator.deallocate(ptr)?;
        }

        Ok(())
    }

    pub fn empty_cache(&self) -> Result<()> {
        let mut cached = self.cached_blocks.write().unwrap();
        let mut total_freed = 0;
        
        for block in cached.drain(..) {
            self.allocator.deallocate(block.ptr)?;
            total_freed += block.size;
        }
        
        {
            let mut stats = self.stats.write().unwrap();
            stats.cached_bytes = 0;
        }
        
        log::info!("Emptied cache, freed {} bytes", total_freed);
        Ok(())
    }

    pub fn get_recommended_batch_size(&self, per_sample_memory: usize) -> usize {
        let (free, _total) = self.allocator.get_memory_info().unwrap_or((0, 1));
        let available = (free as f64 * 0.8) as usize;
        
        let recommended = available / per_sample_memory;
        std::cmp::max(1, recommended)
    }

    pub fn get_memory_pressure(&self) -> f64 {
        let (free, total) = self.allocator.get_memory_info().unwrap_or((1, 1));
        let used = total - free;
        used as f64 / total as f64
    }

    pub fn get_memory_info(&self) -> Result<(usize, usize)> {
        self.allocator.get_memory_info()
    }
    
    pub fn get_stats(&self) -> MemoryStats {
        let mut stats = self.stats.read().unwrap().clone();
        
        let (free, total) = self.allocator.get_memory_info().unwrap_or((0, 1));
        stats.fragmentation_ratio = 1.0 - (free as f64 / total as f64);
        
        stats
    }

    fn find_cached_block(&self, size: usize) -> Option<MemoryBlock> {
        let mut cached = self.cached_blocks.write().unwrap();
        
        let mut best_idx = None;
        let mut best_size = usize::MAX;
        
        for (idx, block) in cached.iter().enumerate() {
            if block.size >= size && block.size < best_size {
                best_idx = Some(idx);
                best_size = block.size;
            }
        }
        
        if let Some(idx) = best_idx {
            let mut block = cached.swap_remove(idx);
            block.last_accessed = Instant::now();
            block.is_cached = false;
            
            {
                let mut allocated = self.allocated_blocks.write().unwrap();
                allocated.insert(block.ptr, block.clone());
            }
            
            Some(block)
        } else {
            None
        }
    }

    fn update_stats_allocation(&self, size: usize) {
        let mut stats = self.stats.write().unwrap();
        stats.allocated_bytes += size;
        stats.allocation_count += 1;
        stats.peak_allocated = stats.peak_allocated.max(stats.allocated_bytes);
        stats.cache_miss_count += 1;
    }

    fn update_stats_cache_hit(&self, size: usize) {
        let mut stats = self.stats.write().unwrap();
        stats.cache_hit_count += 1;
        stats.cached_bytes = stats.cached_bytes.saturating_sub(size);
        stats.allocated_bytes += size;
    }

    fn update_stats_deallocation(&self, size: usize) {
        let mut stats = self.stats.write().unwrap();
        stats.allocated_bytes = stats.allocated_bytes.saturating_sub(size);
        stats.cached_bytes += size;
        stats.deallocation_count += 1;
    }

    fn maybe_garbage_collect(&self) -> Result<()> {
        let (allocated, threshold) = {
            let stats = self.stats.read().unwrap();
            (stats.allocated_bytes + stats.cached_bytes, 
             (self.config.max_size as f64 * self.config.garbage_collection_threshold) as usize)
        };

        if allocated > threshold {
            self.garbage_collect()?;
        }
        
        Ok(())
    }

    pub fn garbage_collect(&self) -> Result<()> {
        let mut cached = self.cached_blocks.write().unwrap();
        let mut to_remove = Vec::new();
        
        cached.sort_by_key(|block| block.last_accessed);
        
        let target_size = (self.config.max_size as f64 * self.config.garbage_collection_threshold * 0.8) as usize;
        let mut current_size = {
            let stats = self.stats.read().unwrap();
            stats.allocated_bytes + stats.cached_bytes
        };
        
        for (idx, block) in cached.iter().enumerate() {
            if current_size <= target_size {
                break;
            }
            
            to_remove.push(idx);
            current_size -= block.size;
        }
        
        for &idx in to_remove.iter().rev() {
            let block = cached.swap_remove(idx);
            self.allocator.deallocate(block.ptr)?;
            
            let mut stats = self.stats.write().unwrap();
            stats.cached_bytes -= block.size;
        }
        
        log::info!("Garbage collected {} blocks", to_remove.len());
        
        Ok(())
    }
}