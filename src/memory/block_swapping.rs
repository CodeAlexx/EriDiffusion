// block_swapping.rs - Dynamic block swapping for training large models on limited VRAM
// Similar to Kohya's SD-Scripts gradient checkpointing and block swapping

use std::collections::{HashMap, VecDeque, HashSet};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use std::path::{Path, PathBuf};
use std::fs::{File, create_dir_all};
use std::io::{Write, Read, BufWriter, BufReader};
use std::ffi::c_void;

use anyhow::{Result, Context};
use candle_core::{Tensor, Device, DType};

use super::{
    cuda_allocator::MemoryError, cuda, PrecisionMode,
    MemoryFormat, DiffusionConfig,
};

/// Block swapping configuration
#[derive(Debug, Clone)]
pub struct BlockSwapConfig {
    /// Maximum GPU memory to use (bytes)
    pub max_gpu_memory: usize,
    /// Directory for disk swapping
    pub swap_dir: PathBuf,
    /// Use memory-mapped files for faster disk access
    pub use_mmap: bool,
    /// Number of blocks to keep in GPU (active set)
    pub active_blocks: usize,
    /// Prefetch next N blocks
    pub prefetch_blocks: usize,
    /// Use pinned memory for faster transfers
    pub use_pinned_memory: bool,
    /// Compression for disk storage
    pub enable_compression: bool,
    /// Async transfers
    pub async_transfers: bool,
    /// Block granularity (e.g., per layer, per attention block)
    pub granularity: BlockGranularity,
}

impl Default for BlockSwapConfig {
    fn default() -> Self {
        Self {
            max_gpu_memory: 20 * 1024 * 1024 * 1024, // 20GB for 24GB cards
            swap_dir: PathBuf::from("/tmp/flux_block_swap"),
            use_mmap: true,
            active_blocks: 8,
            prefetch_blocks: 4,
            use_pinned_memory: true,
            enable_compression: false,
            async_transfers: true,
            granularity: BlockGranularity::Layer,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BlockGranularity {
    Layer,           // Swap entire layers
    AttentionBlock,  // Swap attention blocks (Q,K,V,O)
    MLPBlock,        // Swap MLP blocks
    SubLayer,        // Swap individual sub-layers
    Custom,          // User-defined blocks
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StorageLocation {
    GPU,
    CPUMemory,
    PinnedMemory,
    Disk,
}

/// Represents a swappable model block
#[derive(Debug, Clone)]
pub struct SwappableBlock {
    pub id: String,
    pub layer_idx: usize,
    pub block_type: BlockType,
    pub size_bytes: usize,
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub location: StorageLocation,
    pub last_access: Instant,
    pub access_count: usize,
    pub is_gradient: bool,
    pub dependencies: Vec<String>, // Other blocks this depends on
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BlockType {
    // Transformer blocks
    Attention,
    AttentionQKV,
    AttentionOutput,
    MLP,
    MLPGateProj,
    MLPUpProj,
    MLPDownProj,
    LayerNorm,
    AdaLN,
    
    // Flux-specific
    DoubleBlock,
    SingleBlock,
    
    // LoRA
    LoRAAdapter,
    
    // Other
    Embedding,
    Custom,
}

/// Data storage for swappable blocks
struct BlockData {
    gpu_tensor: Option<Tensor>,
    cpu_buffer: Option<Vec<u8>>,
    disk_path: Option<PathBuf>,
    pinned_ptr: Option<*mut c_void>,
}

/// Manages block swapping for memory-efficient training
pub struct BlockSwapManager {
    config: BlockSwapConfig,
    blocks: RwLock<HashMap<String, SwappableBlock>>,
    block_data: RwLock<HashMap<String, BlockData>>,
    gpu_blocks: RwLock<HashSet<String>>,
    access_queue: Mutex<VecDeque<String>>,
    prefetch_queue: Mutex<VecDeque<String>>,
    stats: RwLock<SwapStats>,
}

#[derive(Default)]
pub struct SwapStats {
    pub total_swaps: u64,
    pub gpu_to_cpu: u64,
    pub cpu_to_gpu: u64,
    pub disk_to_cpu: u64,
    pub cpu_to_disk: u64,
}

impl BlockSwapManager {
    pub fn new(config: BlockSwapConfig) -> Result<Self> {
        // Create swap directory
        if !config.swap_dir.exists() {
            create_dir_all(&config.swap_dir)
                .context("Failed to create swap directory")?;
        }

        Ok(Self {
            config,
            blocks: RwLock::new(HashMap::new()),
            block_data: RwLock::new(HashMap::new()),
            gpu_blocks: RwLock::new(HashSet::new()),
            access_queue: Mutex::new(VecDeque::new()),
            prefetch_queue: Mutex::new(VecDeque::new()),
            stats: RwLock::new(SwapStats::default()),
        })
    }

    /// Register a tensor as a swappable block
    pub fn register_tensor(&self, id: String, tensor: &Tensor, block_type: BlockType) -> Result<()> {
        let shape = tensor.shape().dims().to_vec();
        let dtype = tensor.dtype();
        let size_bytes = tensor.elem_count() * dtype.size_in_bytes();

        let block = SwappableBlock {
            id: id.clone(),
            layer_idx: 0, // Will be set by the model
            block_type,
            size_bytes,
            shape,
            dtype,
            location: StorageLocation::GPU,
            last_access: Instant::now(),
            access_count: 0,
            is_gradient: false,
            dependencies: vec![],
        };

        // Store the tensor data
        let block_data = BlockData {
            gpu_tensor: Some(tensor.clone()),
            cpu_buffer: None,
            disk_path: None,
            pinned_ptr: None,
        };

        self.blocks.write().unwrap().insert(id.clone(), block);
        self.block_data.write().unwrap().insert(id.clone(), block_data);
        self.gpu_blocks.write().unwrap().insert(id);

        Ok(())
    }

    /// Access a block, swapping it to GPU if necessary
    pub fn access_block(&self, block_id: &str) -> Result<Tensor> {
        // Update access info
        {
            let mut blocks = self.blocks.write().unwrap();
            if let Some(block) = blocks.get_mut(block_id) {
                block.last_access = Instant::now();
                block.access_count += 1;
            } else {
                anyhow::bail!("Block {} not found", block_id);
            }
        }

        // Check if block is in GPU
        if self.gpu_blocks.read().unwrap().contains(block_id) {
            let data = self.block_data.read().unwrap();
            if let Some(block_data) = data.get(block_id) {
                if let Some(tensor) = &block_data.gpu_tensor {
                    return Ok(tensor.clone());
                }
            }
        }

        // Need to swap in
        self.swap_in(block_id)?;

        // Return the tensor
        let data = self.block_data.read().unwrap();
        if let Some(block_data) = data.get(block_id) {
            if let Some(tensor) = &block_data.gpu_tensor {
                Ok(tensor.clone())
            } else {
                anyhow::bail!("Failed to load block {} to GPU", block_id);
            }
        } else {
            anyhow::bail!("Block data not found for {}", block_id);
        }
    }

    /// Swap a block into GPU memory
    fn swap_in(&self, block_id: &str) -> Result<()> {
        // Check if we need to evict blocks first
        self.maybe_evict_blocks()?;

        let block_info = {
            let blocks = self.blocks.read().unwrap();
            blocks.get(block_id).cloned()
                .ok_or_else(|| anyhow::anyhow!("Block {} not found", block_id))?
        };

        let mut block_data = self.block_data.write().unwrap();
        let data = block_data.get_mut(block_id)
            .ok_or_else(|| anyhow::anyhow!("Block data not found for {}", block_id))?;

        match block_info.location {
            StorageLocation::CPUMemory => {
                // Load from CPU buffer
                if let Some(cpu_data) = &data.cpu_buffer {
                    let tensor = self.buffer_to_tensor(cpu_data, &block_info)?;
                    data.gpu_tensor = Some(tensor);
                    self.stats.write().unwrap().cpu_to_gpu += 1;
                }
            }
            StorageLocation::Disk => {
                // Load from disk
                if let Some(path) = &data.disk_path {
                    let cpu_data = self.load_from_disk(path)?;
                    let tensor = self.buffer_to_tensor(&cpu_data, &block_info)?;
                    data.gpu_tensor = Some(tensor);
                    data.cpu_buffer = None; // Clear CPU buffer to save memory
                    self.stats.write().unwrap().disk_to_cpu += 1;
                    self.stats.write().unwrap().cpu_to_gpu += 1;
                }
            }
            _ => {}
        }

        // Update location
        {
            let mut blocks = self.blocks.write().unwrap();
            if let Some(block) = blocks.get_mut(block_id) {
                block.location = StorageLocation::GPU;
            }
        }

        self.gpu_blocks.write().unwrap().insert(block_id.to_string());
        self.stats.write().unwrap().total_swaps += 1;

        Ok(())
    }

    /// Evict least recently used blocks if needed
    fn maybe_evict_blocks(&self) -> Result<()> {
        let current_gpu_blocks = self.gpu_blocks.read().unwrap().len();
        
        if current_gpu_blocks >= self.config.active_blocks {
            // Find LRU block
            let lru_block = {
                let blocks = self.blocks.read().unwrap();
                let gpu_blocks = self.gpu_blocks.read().unwrap();
                
                gpu_blocks.iter()
                    .filter_map(|id| blocks.get(id).map(|b| (id.clone(), b.last_access)))
                    .min_by_key(|(_, access)| *access)
                    .map(|(id, _)| id)
            };

            if let Some(block_id) = lru_block {
                self.swap_out(&block_id)?;
            }
        }

        Ok(())
    }

    /// Swap a block out of GPU memory
    fn swap_out(&self, block_id: &str) -> Result<()> {
        let mut block_data = self.block_data.write().unwrap();
        let data = block_data.get_mut(block_id)
            .ok_or_else(|| anyhow::anyhow!("Block data not found for {}", block_id))?;

        if let Some(tensor) = &data.gpu_tensor {
            // Convert to CPU buffer
            let cpu_buffer = self.tensor_to_buffer(tensor)?;
            
            // Decide where to store based on access pattern
            let access_count = {
                let blocks = self.blocks.read().unwrap();
                blocks.get(block_id).map(|b| b.access_count).unwrap_or(0)
            };

            if access_count > 5 {
                // Keep frequently accessed blocks in CPU memory
                data.cpu_buffer = Some(cpu_buffer);
                self.stats.write().unwrap().gpu_to_cpu += 1;
                
                // Update location
                let mut blocks = self.blocks.write().unwrap();
                if let Some(block) = blocks.get_mut(block_id) {
                    block.location = StorageLocation::CPUMemory;
                }
            } else {
                // Save to disk for rarely accessed blocks
                let path = self.save_to_disk(block_id, &cpu_buffer)?;
                data.disk_path = Some(path);
                data.cpu_buffer = None;
                self.stats.write().unwrap().cpu_to_disk += 1;
                
                // Update location
                let mut blocks = self.blocks.write().unwrap();
                if let Some(block) = blocks.get_mut(block_id) {
                    block.location = StorageLocation::Disk;
                }
            }

            // Clear GPU tensor
            data.gpu_tensor = None;
        }

        self.gpu_blocks.write().unwrap().remove(block_id);
        self.stats.write().unwrap().total_swaps += 1;

        Ok(())
    }

    /// Convert tensor to byte buffer
    fn tensor_to_buffer(&self, tensor: &Tensor) -> Result<Vec<u8>> {
        // For now, just get the raw bytes
        // In a real implementation, we'd handle different dtypes properly
        let bytes = tensor.to_vec1::<f32>()
            .or_else(|_| tensor.to_vec1::<half::f16>().map(|v| {
                v.into_iter().map(|h| h.to_f32()).collect()
            }))
            .context("Failed to convert tensor to buffer")?;

        Ok(bytemuck::cast_slice(&bytes).to_vec())
    }

    /// Convert byte buffer to tensor
    fn buffer_to_tensor(&self, buffer: &[u8], block_info: &SwappableBlock) -> Result<Tensor> {
        let device = Device::cuda_if_available(0)?;
        
        // Convert bytes back to the appropriate type
        let float_slice = bytemuck::cast_slice::<u8, f32>(buffer);
        
        Tensor::from_slice(float_slice, block_info.shape.as_slice(), &device)
            .context("Failed to create tensor from buffer")
    }

    /// Save block to disk
    fn save_to_disk(&self, block_id: &str, data: &[u8]) -> Result<PathBuf> {
        let path = self.config.swap_dir.join(format!("{}.block", block_id));
        
        let mut file = BufWriter::new(File::create(&path)?);
        file.write_all(data)?;
        
        Ok(path)
    }

    /// Load block from disk
    fn load_from_disk(&self, path: &Path) -> Result<Vec<u8>> {
        let mut file = BufReader::new(File::open(path)?);
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        
        Ok(buffer)
    }

    /// Get memory usage statistics
    pub fn get_stats(&self) -> SwapStats {
        self.stats.read().unwrap().clone()
    }

    /// Build blocks for Flux model
    pub fn build_flux_blocks(num_double_blocks: usize, num_single_blocks: usize, hidden_dim: usize) -> Vec<SwappableBlock> {
        let mut blocks = Vec::new();
        
        // Double blocks (image and text processing)
        for i in 0..num_double_blocks {
            // Image attention
            blocks.push(SwappableBlock {
                id: format!("double_block_{}_img_attn", i),
                layer_idx: i,
                block_type: BlockType::Attention,
                size_bytes: 4 * hidden_dim * hidden_dim * 2, // QKVO in FP16
                shape: vec![4, hidden_dim, hidden_dim],
                dtype: DType::F16,
                location: StorageLocation::Disk,
                last_access: Instant::now(),
                access_count: 0,
                is_gradient: false,
                dependencies: if i > 0 {
                    vec![format!("double_block_{}_output", i - 1)]
                } else {
                    vec![]
                },
            });

            // Text attention
            blocks.push(SwappableBlock {
                id: format!("double_block_{}_txt_attn", i),
                layer_idx: i,
                block_type: BlockType::Attention,
                size_bytes: 4 * hidden_dim * hidden_dim * 2,
                shape: vec![4, hidden_dim, hidden_dim],
                dtype: DType::F16,
                location: StorageLocation::Disk,
                last_access: Instant::now(),
                access_count: 0,
                is_gradient: false,
                dependencies: vec![format!("double_block_{}_img_attn", i)],
            });

            // MLP
            blocks.push(SwappableBlock {
                id: format!("double_block_{}_mlp", i),
                layer_idx: i,
                block_type: BlockType::MLP,
                size_bytes: hidden_dim * hidden_dim * 8 * 2, // Larger MLP
                shape: vec![hidden_dim * 4, hidden_dim * 2],
                dtype: DType::F16,
                location: StorageLocation::Disk,
                last_access: Instant::now(),
                access_count: 0,
                is_gradient: false,
                dependencies: vec![format!("double_block_{}_txt_attn", i)],
            });
        }

        // Single blocks (combined processing)
        for i in 0..num_single_blocks {
            blocks.push(SwappableBlock {
                id: format!("single_block_{}_attn", i),
                layer_idx: num_double_blocks + i,
                block_type: BlockType::Attention,
                size_bytes: 4 * hidden_dim * hidden_dim * 2,
                shape: vec![4, hidden_dim, hidden_dim],
                dtype: DType::F16,
                location: StorageLocation::Disk,
                last_access: Instant::now(),
                access_count: 0,
                is_gradient: false,
                dependencies: if i > 0 {
                    vec![format!("single_block_{}_output", i - 1)]
                } else {
                    vec![format!("double_block_{}_output", num_double_blocks - 1)]
                },
            });

            blocks.push(SwappableBlock {
                id: format!("single_block_{}_mlp", i),
                layer_idx: num_double_blocks + i,
                block_type: BlockType::MLP,
                size_bytes: hidden_dim * hidden_dim * 8 * 2,
                shape: vec![hidden_dim * 4, hidden_dim * 2],
                dtype: DType::F16,
                location: StorageLocation::Disk,
                last_access: Instant::now(),
                access_count: 0,
                is_gradient: false,
                dependencies: vec![format!("single_block_{}_attn", i)],
            });
        }

        blocks
    }
}

impl Clone for SwapStats {
    fn clone(&self) -> Self {
        Self {
            total_swaps: self.total_swaps,
            gpu_to_cpu: self.gpu_to_cpu,
            cpu_to_gpu: self.cpu_to_gpu,
            disk_to_cpu: self.disk_to_cpu,
            cpu_to_disk: self.cpu_to_disk,
        }
    }
}