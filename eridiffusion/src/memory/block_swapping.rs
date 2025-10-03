use crate::memory::{cuda_allocator::MemoryError, PrecisionMode};
use anyhow::Context;
use flame_core::device::Device;
use flame_core::optimizers::{Adam, SGD};
use flame_core::{CudaDevice, DType, Error, Result, Shape, Tensor};
use half::{bf16, f16};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    ffi::c_void,
    fs::{create_dir_all, File},
    io::{BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
    sync::{Arc, Mutex, RwLock},
    time::{Duration, Instant},
};

// Trait for additional tensor operations

pub struct BlockSwapConfig {
    /// Maximum GPU memory to use (bytes)
    pub max_gpu_memory: usize,
    /// Directory for disk swapping
    pub swap_dir: std::path::PathBuf,
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
    pub dependencies: Vec<String>, // Other blocks this depends on,
}
struct BlockData {
    gpu_tensor: Option<Tensor>,
    cpu_buffer: Option<Vec<u8>>,
    disk_path: Option<std::path::PathBuf>,
}
pub struct BlockSwapManager {
    config: BlockSwapConfig,
    blocks: RwLock<std::collections::HashMap<String, SwappableBlock>>,
    block_data: RwLock<std::collections::HashMap<String, BlockData>>,
    gpu_blocks: RwLock<std::collections::HashSet<String>>,
    access_queue: Mutex<VecDeque<String>>,
    prefetch_queue: Mutex<VecDeque<String>>,
    stats: RwLock<SwapStats>,
}
pub struct SwapStats {
    pub total_swaps: u64,
    pub gpu_to_cpu: u64,
    pub cpu_to_gpu: u64,
    pub disk_to_cpu: u64,
    pub cpu_to_disk: u64,
}

// block_swapping.rs - Dynamic block swapping for training large models on limited VRAM
// Similar to Kohya's SD-Scripts gradient checkpointing and block swapping

// FLAME uses flame_core::device::Device instead of Device

/// Block swapping configuration
// Extension trait for Tensor to add missing methods
// bf16 and f16 are already imported from half crate above
trait TensorExt {
    fn sum_dim(&self, dim: usize) -> flame_core::Result<Tensor>;
    fn add_scalar(&self, scalar: f32) -> flame_core::Result<Tensor>;
    fn mul_scalar(&self, scalar: f32) -> flame_core::Result<Tensor>;
    fn square(&self) -> flame_core::Result<Tensor>;
}

impl TensorExt for Tensor {
    fn sum_dim(&self, dim: usize) -> flame_core::Result<Tensor> {
        // Sum along dimension - FLAME sum_keepdim takes isize
        self.sum_keepdim(dim as isize)
    }

    fn add_scalar(&self, scalar: f32) -> flame_core::Result<Tensor> {
        // Add scalar to all elements
        let scalar_tensor = Tensor::full(self.shape().clone(), scalar, self.device().clone())?;
        self.add(&scalar_tensor)
    }

    fn mul_scalar(&self, scalar: f32) -> flame_core::Result<Tensor> {
        // Multiply all elements by scalar
        let scalar_tensor = Tensor::full(self.shape().clone(), scalar, self.device().clone())?;
        self.mul(&scalar_tensor)
    }

    fn square(&self) -> flame_core::Result<Tensor> {
        // Element-wise square
        self.mul(self)
    }
}

impl Default for BlockSwapConfig {
    fn default() -> Self {
        Self {
            max_gpu_memory: 20 * 1024 * 1024 * 1024, // 20GB for 24GB cards;
            swap_dir: std::path::PathBuf::from("/tmp/flux_block_swap"),
            use_mmap: true,
            active_blocks: 8,
            prefetch_blocks: 4,
            use_pinned_memory: true,
            enable_compression: false,
            async_transfers: true,
            granularity: BlockGranularity::AttentionBlock,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BlockGranularity {
    Layer,          // Swap entire layers
    AttentionBlock, // Swap attention blocks (Q,K,V,O)
    MLPBlock,       // Swap MLP blocks
    SubLayer,       // Swap individual sub-layers
    Custom,         // User-defined blocks
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StorageLocation {
    GPU,
    CPUMemory,
    PinnedMemory,
    Disk,
}

/// Represents a swappable model block
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

/// Manages block swapping for memory-efficient training
impl BlockSwapManager {
    pub fn new(config: BlockSwapConfig) -> flame_core::Result<Self> {
        // Create swap directory
        if !config.swap_dir.exists() {
            create_dir_all(&config.swap_dir).map_err(|e| {
                flame_core::Error::Io(format!("Failed to create swap directory: {}", e))
            })?;
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
    pub fn register_tensor(
        &self,
        id: String,
        tensor: &Tensor,
        block_type: BlockType,
    ) -> flame_core::Result<()> {
        let shape = tensor.shape().dims().to_vec();
        let dtype = tensor.dtype();
        let size_bytes = tensor.shape().dims().iter().product::<usize>() * dtype.size_in_bytes();

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
            dependencies: Vec::new(),
        };

        // Store the tensor data
        let block_data =
            BlockData { gpu_tensor: Some(tensor.clone()), cpu_buffer: None, disk_path: None };

        self.blocks.write().unwrap().insert(id.clone(), block);
        self.block_data.write().unwrap().insert(id.clone(), block_data);
        self.gpu_blocks.write().unwrap().insert(id);

        Ok(())
    }

    /// Access a block, swapping it to GPU if necessary
    pub fn access_block(&self, block_id: &str) -> flame_core::Result<Tensor> {
        // Update access info
        {
            let mut blocks = self.blocks.write().unwrap();
            if let Some(block) = blocks.get_mut(block_id) {
                block.last_access = Instant::now();
                block.access_count += 1;
            } else {
                return Err(flame_core::Error::InvalidOperation(format!("",)));
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
                return Err(flame_core::Error::InvalidOperation(format!("",)));
            }
        } else {
            return Err(flame_core::Error::InvalidOperation(format!("",)));
        }
    }

    /// Swap a block into GPU memory
    fn swap_in(&self, block_id: &str) -> flame_core::Result<()> {
        // Check if we need to evict blocks first
        self.maybe_evict_blocks()?;

        let block_info = {
            let blocks = self.blocks.read().unwrap();
            let block = blocks.get(block_id).ok_or_else(|| {
                flame_core::Error::InvalidOperation(format!("Block {} not found", block_id))
            })?;
            // Create a copy of just the needed info
            SwappableBlock {
                id: block.id.clone(),
                layer_idx: block.layer_idx,
                block_type: block.block_type.clone(),
                size_bytes: block.size_bytes,
                shape: block.shape.clone(),
                dtype: block.dtype,
                location: block.location.clone(),
                last_access: block.last_access,
                access_count: block.access_count,
                is_gradient: block.is_gradient,
                dependencies: block.dependencies.clone(),
            }
        };

        let mut block_data = self.block_data.write().unwrap();
        let data = block_data.get_mut(block_id).ok_or_else(|| {
            flame_core::Error::InvalidOperation(format!(
                "Block data not found for {}",
                block_id
            ))
        })?;

        match block_info.location {
            StorageLocation::CPUMemory => {
                // Load from CPU buffer
                if let Some(cpu_data) = &data.cpu_buffer {
                    let device = flame_core::device::Device::cuda(0)?;
                    let tensor =
                        self.buffer_to_tensor(cpu_data, &block_info, device.cuda_device())?;
                    data.gpu_tensor = Some(tensor);
                    self.stats.write().unwrap().cpu_to_gpu += 1;
                }
            }
            StorageLocation::Disk => {
                // Load from disk
                if let Some(path) = &data.disk_path {
                    let cpu_data = self.load_from_disk(path)?;
                    let device = flame_core::device::Device::cuda(0)?;
                    let tensor =
                        self.buffer_to_tensor(&cpu_data, &block_info, device.cuda_device())?;
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
    fn maybe_evict_blocks(&self) -> flame_core::Result<()> {
        let current_gpu_blocks = self.gpu_blocks.read().unwrap().len();

        if current_gpu_blocks >= self.config.active_blocks {
            // Find LRU block
            let lru_block = {
                let blocks = self.blocks.read().unwrap();
                let gpu_blocks = self.gpu_blocks.read().unwrap();

                gpu_blocks
                    .iter()
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
    fn swap_out(&self, block_id: &str) -> flame_core::Result<()> {
        let mut block_data = self.block_data.write().unwrap();
        let data = block_data.get_mut(block_id).ok_or_else(|| {
            flame_core::Error::InvalidOperation(format!(
                "Block data not found for {}",
                block_id
            ))
        })?;

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

        Ok(())
    }

    /// Convert tensor to byte buffer
    fn tensor_to_buffer(&self, tensor: &Tensor) -> flame_core::Result<Vec<u8>> {
        // For now, just get the raw bytes
        // Convert to f32 first then get bytes
        let tensor_f32 = if tensor.dtype() == DType::F32 {
            tensor.clone()
        } else {
            tensor.to_dtype(DType::F32)?
        };

        let bytes = tensor_f32.to_vec1::<f32>().map_err(|e| {
            flame_core::Error::InvalidOperation(format!(
                "Failed to convert tensor to buffer: {:?}",
                e
            ))
        })?;

        Ok(bytemuck::cast_slice(&bytes).to_vec())
    }

    /// Convert byte buffer to tensor
    fn buffer_to_tensor(
        &self,
        buffer: &[u8],
        block_info: &SwappableBlock,
        device: &Arc<CudaDevice>,
    ) -> flame_core::Result<Tensor> {
        // Use the provided device

        // Convert bytes back to the appropriate type
        let float_slice = bytemuck::cast_slice::<u8, f32>(buffer);

        Tensor::from_slice(float_slice, Shape::from_dims(&block_info.shape), device.clone())
            .map_err(|e| {
                flame_core::Error::InvalidOperation(format!(
                    "Failed to create tensor from buffer: {}",
                    e
                ))
            })
    }

    /// Save block to disk
    fn save_to_disk(&self, block_id: &str, data: &[u8]) -> flame_core::Result<PathBuf> {
        let path = self.config.swap_dir.join(format!("{}.block", block_id));

        let mut file =
            BufWriter::new(File::create(&path).map_err(|e| {
                flame_core::Error::Io(format!("Failed to create file: {}", e))
            })?);
        file.write_all(data).map_err(|e| {
            flame_core::Error::InvalidOperation(format!("Failed to write file: {}", e))
        })?;

        Ok(path)
    }

    /// Load block from disk
    fn load_from_disk(&self, path: &Path) -> flame_core::Result<Vec<u8>> {
        let mut file = BufReader::new(
            File::open(path)
                .map_err(|e| flame_core::Error::Io(format!("Failed to open file: {}", e)))?,
        );
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)
            .map_err(|e| flame_core::Error::Io(format!("Failed to read file: {}", e)))?;

        Ok(buffer)
    }

    /// Get memory usage statistics
    pub fn get_stats(&self) -> SwapStats {
        self.stats.read().unwrap().clone()
    }
}

/// Build blocks for Flux model
pub fn build_flux_blocks(
    num_double_blocks: usize,
    num_single_blocks: usize,
    hidden_dim: usize,
) -> Vec<SwappableBlock> {
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
            dependencies: vec![format!("single_block_{}_img_attn", i)],
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

impl Default for SwapStats {
    fn default() -> Self {
        Self { total_swaps: 0, gpu_to_cpu: 0, cpu_to_gpu: 0, disk_to_cpu: 0, cpu_to_disk: 0 }
    }
}
