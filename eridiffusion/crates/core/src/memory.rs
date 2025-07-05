//! Memory management system for efficient tensor allocation

use crate::{Result, Error, Device};
use candle_core::{Tensor, DType as CandleDType, Shape};
use dashmap::DashMap;
use parking_lot::{Mutex, RwLock};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Memory block information
#[derive(Debug, Clone)]
pub struct MemoryBlock {
    pub ptr: usize,
    pub size: usize,
    pub device: Device,
    pub dtype: CandleDType,
    pub allocated_at: Instant,
    pub last_used: Instant,
    pub allocation_id: u64,
}

/// Memory pool for efficient tensor allocation
pub struct MemoryPool {
    device: Device,
    pools: RwLock<HashMap<CandleDType, DTypePool>>,
    allocation_counter: Mutex<u64>,
    config: MemoryPoolConfig,
    stats: RwLock<MemoryStats>,
}

/// Pool for a specific data type
struct DTypePool {
    /// Free blocks organized by size
    free_blocks: BTreeMap<usize, Vec<MemoryBlock>>,
    /// Allocated blocks
    allocated_blocks: HashMap<u64, MemoryBlock>,
    /// Total allocated memory
    total_allocated: usize,
    /// High water mark
    high_water_mark: usize,
}

/// Memory pool configuration
#[derive(Debug, Clone)]
pub struct MemoryPoolConfig {
    /// Maximum memory to allocate (0 = unlimited)
    pub max_memory: usize,
    /// Minimum block size
    pub min_block_size: usize,
    /// Block size alignment
    pub alignment: usize,
    /// Enable memory defragmentation
    pub enable_defrag: bool,
    /// Defragmentation threshold (percentage)
    pub defrag_threshold: f32,
    /// Block reuse delay
    pub reuse_delay: Duration,
}

impl Default for MemoryPoolConfig {
    fn default() -> Self {
        Self {
            max_memory: 0, // Unlimited
            min_block_size: 1024, // 1KB minimum
            alignment: 256, // 256-byte alignment
            enable_defrag: true,
            defrag_threshold: 0.3, // Defrag when 30% fragmented
            reuse_delay: Duration::from_millis(100),
        }
    }
}

/// Memory statistics
#[derive(Debug, Default, Clone)]
pub struct MemoryStats {
    pub allocations: u64,
    pub deallocations: u64,
    pub reuses: u64,
    pub defragmentations: u64,
    pub peak_memory: usize,
    pub current_memory: usize,
    pub fragmentation_ratio: f32,
}

impl MemoryPool {
    /// Create a new memory pool
    pub fn new(device: Device, config: MemoryPoolConfig) -> Self {
        Self {
            device,
            pools: RwLock::new(HashMap::new()),
            allocation_counter: Mutex::new(0),
            config,
            stats: RwLock::new(MemoryStats::default()),
        }
    }
    
    /// Allocate a tensor from the pool
    pub fn allocate_tensor(&self, shape: &Shape, dtype: CandleDType) -> Result<Tensor> {
        let size = shape.elem_count() * dtype.size_in_bytes();
        let aligned_size = self.align_size(size);
        
        // Get or create dtype pool
        let mut pools = self.pools.write();
        let pool = pools.entry(dtype).or_insert_with(|| DTypePool::new());
        
        // Try to reuse existing block
        if let Some(block) = self.find_reusable_block(pool, aligned_size) {
            self.stats.write().reuses += 1;
            return self.create_tensor_from_block(block, shape, dtype);
        }
        
        // Allocate new block
        let block = self.allocate_new_block(pool, aligned_size, dtype)?;
        self.create_tensor_from_block(block, shape, dtype)
    }
    
    /// Release a tensor back to the pool
    pub fn release_tensor(&self, tensor: &Tensor) -> Result<()> {
        let dtype = tensor.dtype();
        let allocation_id = self.get_tensor_allocation_id(tensor)?;
        
        let mut pools = self.pools.write();
        if let Some(pool) = pools.get_mut(&dtype) {
            if let Some(block) = pool.allocated_blocks.remove(&allocation_id) {
                // Update stats
                pool.free_blocks
                    .entry(block.size)
                    .or_insert_with(Vec::new)
                    .push(block);
                
                self.stats.write().deallocations += 1;
                
                // Check if defragmentation needed
                if self.config.enable_defrag && self.should_defragment(pool) {
                    self.defragment_pool(pool)?;
                }
            }
        }
        
        Ok(())
    }
    
    /// Find a reusable block using best-fit strategy
    fn find_reusable_block(&self, pool: &mut DTypePool, size: usize) -> Option<MemoryBlock> {
        let now = Instant::now();
        
        // Best-fit strategy: find the smallest block that fits
        let mut best_fit: Option<(usize, usize, MemoryBlock)> = None;
        
        // Look for blocks of exact size or slightly larger
        for (&block_size, blocks) in pool.free_blocks.range(size..) {
            // Stop if blocks are too large (more than 50% waste)
            if block_size > size + (size / 2) {
                break;
            }
            
            // Find block that has been free long enough
            for (idx, block) in blocks.iter().enumerate() {
                if now.duration_since(block.last_used) >= self.config.reuse_delay {
                    let waste = block_size - size;
                    
                    match &best_fit {
                        None => best_fit = Some((block_size, idx, block.clone())),
                        Some((_, _, best_block)) if waste < best_block.size - size => {
                            best_fit = Some((block_size, idx, block.clone()));
                        }
                        _ => {}
                    }
                }
            }
        }
        
        // Use the best fitting block if found
        if let Some((block_size, idx, mut block)) = best_fit {
            if let Some(blocks) = pool.free_blocks.get_mut(&block_size) {
                blocks.remove(idx);
                if blocks.is_empty() {
                    pool.free_blocks.remove(&block_size);
                }
            }
            
            // Update block info
            block.last_used = now;
            let allocation_id = self.next_allocation_id();
            block.allocation_id = allocation_id;
            
            pool.allocated_blocks.insert(allocation_id, block.clone());
            return Some(block);
        }
        
        None
    }
    
    /// Allocate a new memory block
    fn allocate_new_block(
        &self,
        pool: &mut DTypePool,
        size: usize,
        dtype: CandleDType,
    ) -> Result<MemoryBlock> {
        // Check memory limits
        if self.config.max_memory > 0 && pool.total_allocated + size > self.config.max_memory {
            return Err(Error::Runtime(format!(
                "Memory limit exceeded: requested {} bytes, limit {} bytes",
                size, self.config.max_memory
            )));
        }
        
        let allocation_id = self.next_allocation_id();
        let now = Instant::now();
        
        // Create block
        let block = MemoryBlock {
            ptr: self.allocate_raw_memory(size)?,
            size,
            device: self.device.clone(),
            dtype,
            allocated_at: now,
            last_used: now,
            allocation_id,
        };
        
        // Update pool stats
        pool.allocated_blocks.insert(allocation_id, block.clone());
        pool.total_allocated += size;
        if pool.total_allocated > pool.high_water_mark {
            pool.high_water_mark = pool.total_allocated;
        }
        
        // Update global stats
        let mut stats = self.stats.write();
        stats.allocations += 1;
        stats.current_memory = pool.total_allocated;
        if stats.current_memory > stats.peak_memory {
            stats.peak_memory = stats.current_memory;
        }
        
        Ok(block)
    }
    
    /// Allocate raw memory
    fn allocate_raw_memory(&self, size: usize) -> Result<usize> {
        match &self.device {
            Device::Cpu => {
                // For CPU, we'll use a simple allocation
                // In practice, this would use aligned allocation
                Ok(size) // Placeholder
            }
            Device::Cuda(_) => {
                // CUDA allocation would go here
                Ok(size) // Placeholder
            }
        }
    }
    
    /// Create tensor from memory block
    fn create_tensor_from_block(
        &self,
        block: MemoryBlock,
        shape: &Shape,
        dtype: CandleDType,
    ) -> Result<Tensor> {
        // This is a simplified version - actual implementation would
        // create tensor from the allocated memory
        Tensor::zeros(shape, dtype, &self.device.to_candle()?)
            .map_err(|e| Error::TensorOp(e))
    }
    
    /// Get tensor allocation ID
    fn get_tensor_allocation_id(&self, _tensor: &Tensor) -> Result<u64> {
        // In practice, this would extract the allocation ID from tensor metadata
        Ok(0)
    }
    
    /// Check if defragmentation is needed
    fn should_defragment(&self, pool: &DTypePool) -> bool {
        let free_memory: usize = pool.free_blocks.values()
            .map(|blocks| blocks.iter().map(|b| b.size).sum::<usize>())
            .sum();
        
        let fragmentation_ratio = if pool.total_allocated > 0 {
            free_memory as f32 / pool.total_allocated as f32
        } else {
            0.0
        };
        
        fragmentation_ratio > self.config.defrag_threshold
    }
    
    /// Defragment memory pool
    fn defragment_pool(&self, pool: &mut DTypePool) -> Result<()> {
        // Collect all free blocks
        let mut all_blocks: Vec<MemoryBlock> = pool.free_blocks
            .values()
            .flat_map(|blocks| blocks.iter().cloned())
            .collect();
        
        // Sort by memory address
        all_blocks.sort_by_key(|b| b.ptr);
        
        // Clear free blocks
        pool.free_blocks.clear();
        
        // Merge adjacent blocks
        let mut merged_blocks = Vec::new();
        let mut current_block: Option<MemoryBlock> = None;
        
        for block in all_blocks {
            if let Some(mut current) = current_block.take() {
                if current.ptr + current.size == block.ptr {
                    // Merge blocks
                    current.size += block.size;
                    current_block = Some(current);
                } else {
                    // Can't merge, save current and start new
                    merged_blocks.push(current);
                    current_block = Some(block);
                }
            } else {
                current_block = Some(block);
            }
        }
        
        if let Some(block) = current_block {
            merged_blocks.push(block);
        }
        
        // Re-add merged blocks to pool
        for block in merged_blocks {
            pool.free_blocks
                .entry(block.size)
                .or_insert_with(Vec::new)
                .push(block);
        }
        
        self.stats.write().defragmentations += 1;
        
        Ok(())
    }
    
    /// Get next allocation ID
    fn next_allocation_id(&self) -> u64 {
        let mut counter = self.allocation_counter.lock();
        let id = *counter;
        *counter += 1;
        id
    }
    
    /// Align size to configured alignment
    fn align_size(&self, size: usize) -> usize {
        let alignment = self.config.alignment.max(1);
        ((size + alignment - 1) / alignment) * alignment
    }
    
    /// Get memory statistics
    pub fn stats(&self) -> MemoryStats {
        self.stats.read().clone()
    }
    
    /// Clear all memory pools
    pub fn clear(&self) {
        let mut pools = self.pools.write();
        pools.clear();
        *self.stats.write() = MemoryStats::default();
    }
}

impl DTypePool {
    fn new() -> Self {
        Self {
            free_blocks: BTreeMap::new(),
            allocated_blocks: HashMap::new(),
            total_allocated: 0,
            high_water_mark: 0,
        }
    }
}

/// Global memory pool manager
pub struct MemoryPoolManager {
    pools: DashMap<Device, Arc<MemoryPool>>,
    default_config: MemoryPoolConfig,
}

impl MemoryPoolManager {
    /// Create a new memory pool manager
    pub fn new(default_config: MemoryPoolConfig) -> Self {
        Self {
            pools: DashMap::new(),
            default_config,
        }
    }
    
    /// Get or create pool for device
    pub fn get_pool(&self, device: &Device) -> Arc<MemoryPool> {
        self.pools
            .entry(device.clone())
            .or_insert_with(|| Arc::new(MemoryPool::new(device.clone(), self.default_config.clone())))
            .clone()
    }
    
    /// Get all pools
    pub fn all_pools(&self) -> Vec<(Device, Arc<MemoryPool>)> {
        self.pools
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect()
    }
    
    /// Clear all pools
    pub fn clear_all(&self) {
        for pool in self.pools.iter() {
            pool.value().clear();
        }
    }
}

/// Global memory pool manager instance
static MEMORY_POOL_MANAGER: once_cell::sync::Lazy<MemoryPoolManager> = 
    once_cell::sync::Lazy::new(|| MemoryPoolManager::new(MemoryPoolConfig::default()));

/// Get the global memory pool manager
pub fn memory_pools() -> &'static MemoryPoolManager {
    &MEMORY_POOL_MANAGER
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_memory_pool_allocation() {
        let device = Device::Cpu;
        let config = MemoryPoolConfig::default();
        let pool = MemoryPool::new(device, config);
        
        // Test allocation
        let shape = Shape::from_dims(&[10, 10]);
        let tensor = pool.allocate_tensor(&shape, CandleDType::F32).unwrap();
        
        // Check stats
        let stats = pool.stats();
        assert_eq!(stats.allocations, 1);
        assert_eq!(stats.reuses, 0);
        
        // Release tensor
        pool.release_tensor(&tensor).unwrap();
        
        // Allocate again - should reuse
        let _tensor2 = pool.allocate_tensor(&shape, CandleDType::F32).unwrap();
        
        let stats = pool.stats();
        assert_eq!(stats.allocations, 1);
        assert_eq!(stats.reuses, 1);
    }
}