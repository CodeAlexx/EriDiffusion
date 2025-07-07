//! Advanced caching system for datasets

use eridiffusion_core::{Result, Error};
use candle_core::{Tensor, DType, Device};
use dashmap::DashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Serialize, Deserialize};
use bincode;
use std::fs;
use tracing::{info, debug, warn};

/// Cache entry containing tensor data
#[derive(Clone)]
pub struct CacheEntry {
    pub tensor: Tensor,
    pub metadata: CacheMetadata,
}

/// Metadata for cached items
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheMetadata {
    pub original_path: PathBuf,
    pub cache_version: u32,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub timestamp: u64,
}

/// Multi-level cache manager
pub struct CacheManager {
    /// In-memory cache (L1)
    memory_cache: Arc<DashMap<String, CacheEntry>>,
    
    /// Disk cache directory (L2)
    disk_cache_dir: PathBuf,
    
    /// Memory cache size limit in MB
    memory_limit_mb: usize,
    
    /// Current memory usage estimate
    current_memory_mb: Arc<RwLock<usize>>,
    
    /// Cache statistics
    stats: Arc<RwLock<CacheStats>>,
}

#[derive(Debug, Default, Clone)]
struct CacheStats {
    memory_hits: u64,
    memory_misses: u64,
    disk_hits: u64,
    disk_misses: u64,
    misses: u64,
    total_requests: u64,
    memory_size_mb: usize,
    disk_size_mb: usize,
}

impl CacheManager {
    pub fn new(disk_cache_dir: PathBuf, memory_limit_mb: usize) -> Result<Self> {
        // Create cache directory if it doesn't exist
        fs::create_dir_all(&disk_cache_dir)?;
        
        Ok(Self {
            memory_cache: Arc::new(DashMap::new()),
            disk_cache_dir,
            memory_limit_mb,
            current_memory_mb: Arc::new(RwLock::new(0)),
            stats: Arc::new(RwLock::new(CacheStats::default())),
        })
    }
    
    /// Generate cache key from path
    fn cache_key(path: &Path) -> String {
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        hasher.update(path.to_string_lossy().as_bytes());
        format!("{:x}", hasher.finalize())
    }
    
    /// Get tensor from cache
    pub async fn get(&self, path: &Path) -> Option<CacheEntry> {
        let key = Self::cache_key(path);
        
        // Check memory cache first
        if let Some(entry) = self.memory_cache.get(&key) {
            self.stats.write().await.memory_hits += 1;
            debug!("Memory cache hit for {}", path.display());
            return Some(entry.clone());
        }
        
        self.stats.write().await.memory_misses += 1;
        
        // Check disk cache
        let disk_path = self.disk_cache_dir.join(&key).with_extension("cache");
        if disk_path.exists() {
            match self.load_from_disk(&disk_path).await {
                Ok(entry) => {
                    self.stats.write().await.disk_hits += 1;
                    debug!("Disk cache hit for {}", path.display());
                    
                    // Add to memory cache if space available
                    let _ = self.add_to_memory_cache(key, entry.clone()).await;
                    
                    return Some(entry);
                }
                Err(e) => {
                    warn!("Failed to load from disk cache: {}", e);
                }
            }
        }
        
        self.stats.write().await.disk_misses += 1;
        None
    }
    
    /// Put tensor into cache
    pub async fn put(&self, path: &Path, tensor: Tensor, metadata: CacheMetadata) -> Result<()> {
        let key = Self::cache_key(path);
        let entry = CacheEntry { tensor, metadata };
        
        // Add to memory cache
        self.add_to_memory_cache(key.clone(), entry.clone()).await?;
        
        // Save to disk cache
        let disk_path = self.disk_cache_dir.join(&key).with_extension("cache");
        self.save_to_disk(&disk_path, &entry).await?;
        
        Ok(())
    }
    
    /// Add entry to memory cache with LRU eviction
    async fn add_to_memory_cache(&self, key: String, entry: CacheEntry) -> Result<()> {
        // Estimate tensor size in MB
        let tensor_size_mb = Self::estimate_tensor_size_mb(&entry.tensor);
        
        let mut current_size = self.current_memory_mb.write().await;
        
        // Evict entries if needed
        while *current_size + tensor_size_mb > self.memory_limit_mb && !self.memory_cache.is_empty() {
            // Simple FIFO eviction for now
            if let Some(evicted) = self.memory_cache.iter().next() {
                let evicted_key = evicted.key().clone();
                let evicted_size = Self::estimate_tensor_size_mb(&evicted.value().tensor);
                drop(evicted);
                
                self.memory_cache.remove(&evicted_key);
                *current_size = current_size.saturating_sub(evicted_size);
            } else {
                break;
            }
        }
        
        // Add new entry
        self.memory_cache.insert(key, entry);
        *current_size += tensor_size_mb;
        
        Ok(())
    }
    
    /// Estimate tensor size in MB
    fn estimate_tensor_size_mb(tensor: &Tensor) -> usize {
        let num_elements = tensor.shape().dims().iter().product::<usize>();
        let bytes_per_element = match tensor.dtype() {
            DType::F32 => 4,
            DType::F16 | DType::BF16 => 2,
            DType::F64 => 8,
            _ => 4,
        };
        (num_elements * bytes_per_element) / (1024 * 1024)
    }
    
    /// Save cache entry to disk
    async fn save_to_disk(&self, path: &Path, entry: &CacheEntry) -> Result<()> {
        // Save tensor data
        let tensor_path = path.with_extension("tensor");
        let mut tensors = std::collections::HashMap::new();
        tensors.insert("data".to_string(), entry.tensor.clone());
        let data = safetensors::serialize(&tensors, &None)?;
        tokio::fs::write(&tensor_path, data).await?;
        
        // Save metadata
        let metadata_path = path.with_extension("meta");
        let metadata_data = bincode::serialize(&entry.metadata)
            .map_err(|e| Error::DataError(format!("Failed to serialize metadata: {}", e)))?;
        tokio::fs::write(&metadata_path, metadata_data).await?;
        
        Ok(())
    }
    
    /// Load cache entry from disk
    async fn load_from_disk(&self, path: &Path) -> Result<CacheEntry> {
        // Load metadata
        let metadata_path = path.with_extension("meta");
        let metadata_data = tokio::fs::read(&metadata_path).await
            .map_err(|e| Error::IoError(e.to_string()))?;
        let metadata: CacheMetadata = bincode::deserialize(&metadata_data)
            .map_err(|e| Error::DataError(format!("Failed to deserialize metadata: {}", e)))?;
        
        // Load tensor
        let tensor_path = path.with_extension("tensor");
        let tensor_data = tokio::fs::read(&tensor_path).await
            .map_err(|e| Error::IoError(e.to_string()))?;
        use safetensors::SafeTensors;
        let tensors = SafeTensors::deserialize(&tensor_data)
            .map_err(|e| Error::DataError(format!("Failed to load safetensors: {}", e)))?;
        let tensor_view = tensors.tensor("data")
            .map_err(|e| Error::DataError(format!("Failed to get tensor: {}", e)))?;
        
        // Convert to candle tensor
        let shape: Vec<usize> = tensor_view.shape().to_vec();
        let data = tensor_view.data();
        let tensor = Tensor::from_slice(
            bytemuck::cast_slice::<u8, f32>(data),
            shape.as_slice(),
            &Device::Cpu,
        )?;
        
        Ok(CacheEntry { tensor, metadata })
    }
    
    /// Get cache statistics
    pub async fn stats(&self) -> CacheStats {
        let stats = self.stats.read().await;
        CacheStats {
            memory_hits: stats.memory_hits,
            memory_misses: stats.memory_misses,
            disk_hits: stats.disk_hits,
            disk_misses: stats.disk_misses,
            misses: stats.misses,
            total_requests: stats.total_requests,
            memory_size_mb: stats.memory_size_mb,
            disk_size_mb: stats.disk_size_mb,
        }
    }
    
    /// Clear all caches
    pub async fn clear(&self) -> Result<()> {
        // Clear memory cache
        self.memory_cache.clear();
        *self.current_memory_mb.write().await = 0;
        
        // Clear disk cache
        let entries = fs::read_dir(&self.disk_cache_dir)?;
        for entry in entries {
            let entry = entry?;
            if entry.path().extension().and_then(|s| s.to_str()) == Some("cache") {
                fs::remove_file(entry.path())?;
            }
        }
        
        // Reset stats
        *self.stats.write().await = CacheStats::default();
        
        info!("Cache cleared");
        Ok(())
    }
}

/// Latent cache specifically for VAE outputs
pub struct LatentCache {
    cache_manager: Arc<CacheManager>,
    vae_scale_factor: f32,
}

impl LatentCache {
    pub fn new(cache_dir: PathBuf, memory_limit_mb: usize, vae_scale_factor: f32) -> Result<Self> {
        let cache_manager = Arc::new(CacheManager::new(cache_dir, memory_limit_mb)?);
        
        Ok(Self {
            cache_manager,
            vae_scale_factor,
        })
    }
    
    /// Get or compute latent for image
    pub async fn get_or_compute<F>(
        &self,
        image_path: &Path,
        compute_fn: F,
    ) -> Result<Tensor>
    where
        F: FnOnce() -> Result<Tensor>,
    {
        // Check cache first
        if let Some(entry) = self.cache_manager.get(image_path).await {
            return Ok(entry.tensor);
        }
        
        // Compute if not cached
        let latent = compute_fn()?;
        
        // Create metadata
        let metadata = CacheMetadata {
            original_path: image_path.to_path_buf(),
            cache_version: 1,
            dtype: format!("{:?}", latent.dtype()),
            shape: latent.shape().dims().to_vec(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };
        
        // Cache the result
        self.cache_manager.put(image_path, latent.clone(), metadata).await?;
        
        Ok(latent)
    }
}