//! Distributed training support

use eridiffusion_core::{Result, Error, Device};
use candle_core::{Tensor, DType};
use serde::{Serialize, Deserialize};
use std::sync::Arc;
use tokio::sync::{RwLock, Barrier, broadcast};

/// Distributed training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedConfig {
    pub backend: DistributedBackend,
    pub world_size: usize,
    pub rank: usize,
    pub local_rank: usize,
    pub master_addr: String,
    pub master_port: u16,
    pub find_unused_parameters: bool,
    pub bucket_cap_bytes: usize,
    pub gradient_as_bucket_view: bool,
    pub broadcast_buffers: bool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum DistributedBackend {
    Nccl,
    Gloo,
    Mpi,
}

impl Default for DistributedConfig {
    fn default() -> Self {
        Self {
            backend: DistributedBackend::Nccl,
            world_size: 1,
            rank: 0,
            local_rank: 0,
            master_addr: "localhost".to_string(),
            master_port: 29500,
            find_unused_parameters: false,
            bucket_cap_bytes: 25 * 1024 * 1024, // 25MB
            gradient_as_bucket_view: true,
            broadcast_buffers: true,
        }
    }
}

/// Process group for distributed training
pub struct ProcessGroup {
    config: DistributedConfig,
    communicator: Arc<RwLock<Box<dyn Communicator>>>,
    barrier: Arc<Barrier>,
}

impl ProcessGroup {
    /// Initialize process group
    pub async fn init(config: DistributedConfig) -> Result<Self> {
        let communicator: Box<dyn Communicator> = match config.backend {
            DistributedBackend::Nccl => Box::new(NcclCommunicator::new(&config).await?),
            DistributedBackend::Gloo => Box::new(GlooCommunicator::new(&config).await?),
            DistributedBackend::Mpi => Box::new(MpiCommunicator::new(&config).await?),
        };
        
        let barrier = Arc::new(Barrier::new(config.world_size));
        
        Ok(Self {
            config,
            communicator: Arc::new(RwLock::new(communicator)),
            barrier,
        })
    }
    
    /// Get world size
    pub fn world_size(&self) -> usize {
        self.config.world_size
    }
    
    /// Get rank
    pub fn rank(&self) -> usize {
        self.config.rank
    }
    
    /// Get local rank
    pub fn local_rank(&self) -> usize {
        self.config.local_rank
    }
    
    /// Check if this is the master process
    pub fn is_master(&self) -> bool {
        self.config.rank == 0
    }
    
    /// All-reduce operation
    pub async fn all_reduce(&self, tensor: &mut Tensor, op: ReduceOp) -> Result<()> {
        let comm = self.communicator.read().await;
        comm.all_reduce(tensor, op).await
    }
    
    /// Broadcast tensor from source rank
    pub async fn broadcast(&self, tensor: &mut Tensor, src: usize) -> Result<()> {
        let comm = self.communicator.read().await;
        comm.broadcast(tensor, src).await
    }
    
    /// All-gather operation
    pub async fn all_gather(&self, tensor: &Tensor) -> Result<Vec<Tensor>> {
        let comm = self.communicator.read().await;
        comm.all_gather(tensor).await
    }
    
    /// Reduce-scatter operation
    pub async fn reduce_scatter(&self, tensors: &[Tensor], op: ReduceOp) -> Result<Tensor> {
        let comm = self.communicator.read().await;
        comm.reduce_scatter(tensors, op).await
    }
    
    /// Synchronize all processes
    pub async fn barrier(&self) -> Result<()> {
        self.barrier.wait().await;
        Ok(())
    }
}

/// Reduce operation
#[derive(Debug, Clone, Copy)]
pub enum ReduceOp {
    Sum,
    Product,
    Min,
    Max,
    Average,
}

/// Communicator trait
#[async_trait::async_trait]
trait Communicator: Send + Sync {
    async fn all_reduce(&self, tensor: &mut Tensor, op: ReduceOp) -> Result<()>;
    async fn broadcast(&self, tensor: &mut Tensor, src: usize) -> Result<()>;
    async fn all_gather(&self, tensor: &Tensor) -> Result<Vec<Tensor>>;
    async fn reduce_scatter(&self, tensors: &[Tensor], op: ReduceOp) -> Result<Tensor>;
}

/// NCCL communicator
struct NcclCommunicator {
    rank: usize,
    world_size: usize,
}

impl NcclCommunicator {
    async fn new(config: &DistributedConfig) -> Result<Self> {
        // Would initialize NCCL
        Ok(Self {
            rank: config.rank,
            world_size: config.world_size,
        })
    }
}

#[async_trait::async_trait]
impl Communicator for NcclCommunicator {
    async fn all_reduce(&self, tensor: &mut Tensor, op: ReduceOp) -> Result<()> {
        // Simplified all-reduce
        match op {
            ReduceOp::Average => {
                *tensor = (tensor.as_ref() / self.world_size as f64)?;
            }
            _ => {}
        }
        Ok(())
    }
    
    async fn broadcast(&self, tensor: &mut Tensor, src: usize) -> Result<()> {
        // Would perform actual broadcast
        Ok(())
    }
    
    async fn all_gather(&self, tensor: &Tensor) -> Result<Vec<Tensor>> {
        // Would perform actual all-gather
        let mut results = Vec::new();
        for _ in 0..self.world_size {
            results.push(tensor.clone());
        }
        Ok(results)
    }
    
    async fn reduce_scatter(&self, tensors: &[Tensor], op: ReduceOp) -> Result<Tensor> {
        // Would perform actual reduce-scatter
        Ok(tensors[self.rank].clone())
    }
}

/// Gloo communicator
struct GlooCommunicator {
    rank: usize,
    world_size: usize,
}

impl GlooCommunicator {
    async fn new(config: &DistributedConfig) -> Result<Self> {
        Ok(Self {
            rank: config.rank,
            world_size: config.world_size,
        })
    }
}

#[async_trait::async_trait]
impl Communicator for GlooCommunicator {
    async fn all_reduce(&self, tensor: &mut Tensor, op: ReduceOp) -> Result<()> {
        Ok(())
    }
    
    async fn broadcast(&self, tensor: &mut Tensor, src: usize) -> Result<()> {
        Ok(())
    }
    
    async fn all_gather(&self, tensor: &Tensor) -> Result<Vec<Tensor>> {
        let mut results = Vec::new();
        for _ in 0..self.world_size {
            results.push(tensor.clone());
        }
        Ok(results)
    }
    
    async fn reduce_scatter(&self, tensors: &[Tensor], op: ReduceOp) -> Result<Tensor> {
        Ok(tensors[self.rank].clone())
    }
}

/// MPI communicator
struct MpiCommunicator {
    rank: usize,
    world_size: usize,
}

impl MpiCommunicator {
    async fn new(config: &DistributedConfig) -> Result<Self> {
        Ok(Self {
            rank: config.rank,
            world_size: config.world_size,
        })
    }
}

#[async_trait::async_trait]
impl Communicator for MpiCommunicator {
    async fn all_reduce(&self, tensor: &mut Tensor, op: ReduceOp) -> Result<()> {
        Ok(())
    }
    
    async fn broadcast(&self, tensor: &mut Tensor, src: usize) -> Result<()> {
        Ok(())
    }
    
    async fn all_gather(&self, tensor: &Tensor) -> Result<Vec<Tensor>> {
        let mut results = Vec::new();
        for _ in 0..self.world_size {
            results.push(tensor.clone());
        }
        Ok(results)
    }
    
    async fn reduce_scatter(&self, tensors: &[Tensor], op: ReduceOp) -> Result<Tensor> {
        Ok(tensors[self.rank].clone())
    }
}

/// Distributed data parallel wrapper
pub struct DistributedDataParallel<M> {
    module: Arc<RwLock<M>>,
    process_group: Arc<ProcessGroup>,
    gradient_buckets: Arc<RwLock<Vec<GradientBucket>>>,
    broadcast_tx: broadcast::Sender<()>,
}

struct GradientBucket {
    tensors: Vec<Tensor>,
    flat_tensor: Option<Tensor>,
    pending: bool,
}

impl<M> DistributedDataParallel<M> {
    /// Wrap module for distributed training
    pub fn new(module: M, process_group: Arc<ProcessGroup>) -> Result<Self> {
        let (broadcast_tx, _) = broadcast::channel(16);
        
        Ok(Self {
            module: Arc::new(RwLock::new(module)),
            process_group,
            gradient_buckets: Arc::new(RwLock::new(Vec::new())),
            broadcast_tx,
        })
    }
    
    /// Synchronize gradients across all processes
    pub async fn sync_gradients(&self) -> Result<()> {
        let mut buckets = self.gradient_buckets.write().await;
        
        for bucket in buckets.iter_mut() {
            if bucket.pending {
                // Flatten gradients
                let flat = if bucket.tensors.len() == 1 {
                    bucket.tensors[0].clone()
                } else {
                    let flattened: Vec<Tensor> = bucket.tensors.iter()
                        .map(|t| t.flatten_all().map_err(|e| Error::Training(e.to_string())))
                        .collect::<Result<Vec<_>>>()?;
                    Tensor::cat(&flattened, 0)?
                };
                
                bucket.flat_tensor = Some(flat);
                
                // All-reduce gradient
                if let Some(ref mut grad) = bucket.flat_tensor {
                    self.process_group.all_reduce(grad, ReduceOp::Average).await?;
                }
                
                // Unflatten gradients back
                if bucket.tensors.len() > 1 {
                    // Would split flat tensor back to original shapes
                }
                
                bucket.pending = false;
            }
        }
        
        Ok(())
    }
    
    /// Broadcast parameters from rank 0
    pub async fn broadcast_parameters(&self) -> Result<()> {
        // Would broadcast all model parameters
        self.process_group.barrier().await?;
        let _ = self.broadcast_tx.send(());
        Ok(())
    }
}

/// Data parallel utilities
pub mod utils {
    use super::*;
    
    /// Split batch across ranks
    pub fn split_batch(batch_size: usize, world_size: usize, rank: usize) -> (usize, usize) {
        let per_rank = batch_size / world_size;
        let remainder = batch_size % world_size;
        
        let start = rank * per_rank + rank.min(remainder);
        let size = per_rank + if rank < remainder { 1 } else { 0 };
        
        (start, size)
    }
    
    /// Get device for rank
    pub fn get_device_for_rank(rank: usize, local_rank: usize) -> Device {
        Device::Cuda(local_rank)
    }
    
    /// Initialize from environment
    pub fn init_from_env() -> Result<DistributedConfig> {
        use std::env;
        
        Ok(DistributedConfig {
            world_size: env::var("WORLD_SIZE")
                .unwrap_or_else(|_| "1".to_string())
                .parse()
                .map_err(|e: std::num::ParseIntError| Error::Training(e.to_string()))?,
            rank: env::var("RANK")
                .unwrap_or_else(|_| "0".to_string())
                .parse()
                .map_err(|e: std::num::ParseIntError| Error::Training(e.to_string()))?,
            local_rank: env::var("LOCAL_RANK")
                .unwrap_or_else(|_| "0".to_string())
                .parse()
                .map_err(|e: std::num::ParseIntError| Error::Training(e.to_string()))?,
            master_addr: env::var("MASTER_ADDR")
                .unwrap_or_else(|_| "localhost".to_string()),
            master_port: env::var("MASTER_PORT")
                .unwrap_or_else(|_| "29500".to_string())
                .parse()
                .map_err(|e: std::num::ParseIntError| Error::Training(e.to_string()))?,
            ..Default::default()
        })
    }
}