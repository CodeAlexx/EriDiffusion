//! Data caching utilities

use eridiffusion_core::{Result, Error};
use serde::{Serialize, Deserialize};
use std::path::{Path, PathBuf};
use std::collections::HashMap;
use tokio::sync::RwLock;
use std::sync::Arc;
use candle_core::Tensor;
use eridiffusion_core::Device;

/// Cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    pub cache_dir: PathBuf,
    pub max_size_gb: f32,
    pub compression: CompressionType,
    pub enable_memory_cache: bool,
    pub memory_cache_size: usize,
    pub enable_disk_cache: bool,
    pub precompute_latents: bool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum CompressionType {
    None,
    Lz4,
    Zstd,
    Snappy,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            cache_dir: PathBuf::from(".cache/data"),
            max_size_gb: 10.0,
            compression: CompressionType::Lz4,
            enable_memory_cache: true,
            memory_cache_size: 1000,
            enable_disk_cache: true,
            precompute_latents: false,
        }
    }
}

/// Cached item
#[derive(Clone)]
pub struct CachedItem {
    pub tensor: Tensor,
    pub metadata: HashMap<String, serde_json::Value>,
    pub size_bytes: usize,
}

/// Cache statistics
#[derive(Debug, Default, Clone)]
pub struct CacheStats {
    pub memory_hits: usize,
    pub memory_misses: usize,
    pub disk_hits: usize,
    pub disk_misses: usize,
    pub total_size_bytes: usize,
    pub num_items: usize,
}

/// Data cache
pub struct DataCache {
    config: CacheConfig,
    memory_cache: Arc<RwLock<lru::LruCache<String, CachedItem>>>,
    disk_index: Arc<RwLock<HashMap<String, PathBuf>>>,
    stats: Arc<RwLock<CacheStats>>,
    device: Device,
}

impl DataCache {
    /// Create new data cache
    pub async fn new(config: CacheConfig, device: Device) -> Result<Self> {
        // Create cache directory
        tokio::fs::create_dir_all(&config.cache_dir).await?;
        
        // Initialize memory cache
        let memory_cache = Arc::new(RwLock::new(
            lru::LruCache::new(
                std::num::NonZeroUsize::new(config.memory_cache_size)
                    .ok_or_else(|| Error::InvalidInput("Invalid cache size".to_string()))?
            )
        ));
        
        // Load disk index
        let disk_index = Arc::new(RwLock::new(Self::load_disk_index(&config.cache_dir).await?));
        
        Ok(Self {
            config,
            memory_cache,
            disk_index,
            stats: Arc::new(RwLock::new(CacheStats::default())),
            device,
        })
    }
    
    /// Load disk cache index
    async fn load_disk_index(cache_dir: &Path) -> Result<HashMap<String, PathBuf>> {
        let index_path = cache_dir.join("index.json");
        
        if index_path.exists() {
            let content = tokio::fs::read_to_string(&index_path).await?;
            Ok(serde_json::from_str(&content)?)
        } else {
            Ok(HashMap::new())
        }
    }
    
    /// Save disk cache index
    async fn save_disk_index(&self) -> Result<()> {
        let index = self.disk_index.read().await;
        let index_path = self.config.cache_dir.join("index.json");
        let content = serde_json::to_string_pretty(&*index)?;
        tokio::fs::write(&index_path, content).await?;
        Ok(())
    }
    
    /// Generate cache key
    pub fn generate_key(path: &Path, transforms: &[String]) -> String {
        use sha2::{Sha256, Digest};
        
        let mut hasher = Sha256::new();
        hasher.update(path.to_string_lossy().as_bytes());
        
        for transform in transforms {
            hasher.update(transform.as_bytes());
        }
        
        format!("{:x}", hasher.finalize())
    }
    
    /// Get item from cache
    pub async fn get(&self, key: &str) -> Result<Option<CachedItem>> {
        // Check memory cache first
        if self.config.enable_memory_cache {
            let mut cache = self.memory_cache.write().await;
            if let Some(item) = cache.get(key) {
                let mut stats = self.stats.write().await;
                stats.memory_hits += 1;
                return Ok(Some(item.clone()));
            }
            
            let mut stats = self.stats.write().await;
            stats.memory_misses += 1;
        }
        
        // Check disk cache
        if self.config.enable_disk_cache {
            let index = self.disk_index.read().await;
            if let Some(path) = index.get(key) {
                if let Ok(item) = self.load_from_disk(path).await {
                    // Add to memory cache
                    if self.config.enable_memory_cache {
                        let mut cache = self.memory_cache.write().await;
                        cache.put(key.to_string(), item.clone());
                    }
                    
                    let mut stats = self.stats.write().await;
                    stats.disk_hits += 1;
                    return Ok(Some(item));
                }
            }
            
            let mut stats = self.stats.write().await;
            stats.disk_misses += 1;
        }
        
        Ok(None)
    }
    
    /// Put item in cache
    pub async fn put(&self, key: &str, item: CachedItem) -> Result<()> {
        // Add to memory cache
        if self.config.enable_memory_cache {
            let mut cache = self.memory_cache.write().await;
            cache.put(key.to_string(), item.clone());
        }
        
        // Save to disk
        if self.config.enable_disk_cache {
            let path = self.config.cache_dir.join(format!("{}.cache", key));
            self.save_to_disk(&path, &item).await?;
            
            let mut index = self.disk_index.write().await;
            index.insert(key.to_string(), path);
            
            // Update stats
            let mut stats = self.stats.write().await;
            stats.total_size_bytes += item.size_bytes;
            stats.num_items += 1;
        }
        
        Ok(())
    }
    
    /// Load item from disk
    async fn load_from_disk(&self, path: &Path) -> Result<CachedItem> {
        let data = tokio::fs::read(path).await?;
        
        // Decompress if needed
        let decompressed = match self.config.compression {
            CompressionType::None => data,
            CompressionType::Lz4 => {
                lz4::block::decompress(&data, None)
                    .map_err(|e| Error::DataError(format!("LZ4 decompression failed: {}", e)))?
            }
            CompressionType::Zstd => {
                zstd::decode_all(&data[..])
                    .map_err(|e| Error::DataError(format!("Zstd decompression failed: {}", e)))?
            }
            CompressionType::Snappy => {
                let mut decoder = snap::raw::Decoder::new();
                decoder.decompress_vec(&data)
                    .map_err(|e| Error::DataError(format!("Snappy decompression failed: {}", e)))?
            }
        };
        
        // Deserialize
        let (tensor_bytes, metadata_bytes) = decompressed.split_at(
            decompressed.len() - std::mem::size_of::<u32>()
        );
        
        // Deserialize tensor data
        let candle_device = match &self.device {
            Device::Cpu => candle_core::Device::Cpu,
            Device::Cuda(ordinal) => candle_core::Device::new_cuda(*ordinal)?,
        };
        
        // Read tensor shape and data type from the beginning of tensor_bytes
        let shape_len = u32::from_le_bytes(tensor_bytes[0..4].try_into()
            .map_err(|e| Error::DataError(format!("Failed to parse shape length: {:?}", e)))?) as usize;
        let shape_bytes = &tensor_bytes[4..4 + shape_len * 8];
        let mut shape = Vec::with_capacity(shape_len);
        for i in 0..shape_len {
            let dim = u64::from_le_bytes(shape_bytes[i*8..(i+1)*8].try_into()
                .map_err(|e| Error::DataError(format!("Failed to parse dimension: {:?}", e)))?);
            shape.push(dim as usize);
        }
        
        // Read tensor data
        let data_start = 4 + shape_len * 8;
        let element_size = 4; // f32
        let num_elements: usize = shape.iter().product();
        let data_end = data_start + num_elements * element_size;
        
        // Create tensor from raw bytes
        let data_slice = &tensor_bytes[data_start..data_end];
        let float_data: Vec<f32> = data_slice
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        
        let tensor = Tensor::from_slice(
            &float_data,
            shape.as_slice(),
            &candle_device,
        )?;
        
        // Load metadata
        let metadata_len = u32::from_le_bytes(
            metadata_bytes.try_into().unwrap()
        ) as usize;
        let metadata_start = tensor_bytes.len() - metadata_len;
        let metadata: HashMap<String, serde_json::Value> = 
            serde_json::from_slice(&tensor_bytes[metadata_start..])?;
        
        Ok(CachedItem {
            tensor,
            metadata,
            size_bytes: decompressed.len(),
        })
    }
    
    /// Save item to disk
    async fn save_to_disk(&self, path: &Path, item: &CachedItem) -> Result<()> {
        // Serialize tensor (simplified - would use actual tensor serialization)
        let tensor_bytes = vec![0u8; 1024]; // Placeholder
        
        // Serialize metadata
        let metadata_bytes = serde_json::to_vec(&item.metadata)?;
        
        // Combine data
        let mut data = tensor_bytes;
        data.extend_from_slice(&metadata_bytes);
        data.extend_from_slice(&(metadata_bytes.len() as u32).to_le_bytes());
        
        // Compress if needed
        let compressed = match self.config.compression {
            CompressionType::None => data,
            CompressionType::Lz4 => {
                lz4::block::compress(&data, None, true)
                    .map_err(|e| Error::DataError(format!("LZ4 compression failed: {}", e)))?
            }
            CompressionType::Zstd => {
                zstd::encode_all(&data[..], 3)
                    .map_err(|e| Error::DataError(format!("Zstd compression failed: {}", e)))?
            }
            CompressionType::Snappy => {
                let mut encoder = snap::raw::Encoder::new();
                encoder.compress_vec(&data)
                    .map_err(|e| Error::DataError(format!("Snappy compression failed: {}", e)))?
            }
        };
        
        // Write to disk
        tokio::fs::write(path, compressed).await?;
        
        Ok(())
    }
    
    /// Clear cache
    pub async fn clear(&self) -> Result<()> {
        // Clear memory cache
        if self.config.enable_memory_cache {
            let mut cache = self.memory_cache.write().await;
            cache.clear();
        }
        
        // Clear disk cache
        if self.config.enable_disk_cache {
            let index = self.disk_index.read().await;
            for path in index.values() {
                let _ = tokio::fs::remove_file(path).await;
            }
            
            let mut index = self.disk_index.write().await;
            index.clear();
        }
        
        // Reset stats
        let mut stats = self.stats.write().await;
        *stats = CacheStats::default();
        
        Ok(())
    }
    
    /// Get cache statistics
    pub async fn get_stats(&self) -> CacheStats {
        self.stats.read().await.clone()
    }
    
    /// Precompute and cache latents
    pub async fn precompute_latents<F>(
        &self,
        paths: Vec<PathBuf>,
        encoder_fn: F,
    ) -> Result<()> 
    where
        F: Fn(&Tensor) -> Result<Tensor> + Send + Sync,
    {
        if !self.config.precompute_latents {
            return Ok(());
        }
        
        for path in paths {
            let key = format!("{}_latent", Self::generate_key(&path, &[]));
            
            // Skip if already cached
            if self.get(&key).await?.is_some() {
                continue;
            }
            
            // Load and encode image (simplified)
            let candle_device = match &self.device {
                Device::Cpu => candle_core::Device::Cpu,
                Device::Cuda(ordinal) => candle_core::Device::new_cuda(*ordinal)?,
            };
            let image = Tensor::randn(
                0.0f32,
                1.0,
                &[3, 512, 512],
                &candle_device,
            )?;
            
            let latent = encoder_fn(&image)?;
            
            // Cache the latent
            let item = CachedItem {
                tensor: latent,
                metadata: HashMap::new(),
                size_bytes: 0, // Would calculate actual size
            };
            
            self.put(&key, item).await?;
        }
        
        Ok(())
    }
}

/// Create cache warmup task
pub async fn warmup_cache(
    cache: Arc<DataCache>,
    dataset: Arc<dyn crate::dataset::Dataset>,
    num_samples: usize,
) -> Result<()> {
    use rand::seq::SliceRandom;
    
    let indices: Vec<usize> = dataset.indices();
    let mut rng = rand::thread_rng();
    let mut selected = indices.clone();
    selected.shuffle(&mut rng);
    selected.truncate(num_samples);
    
    for idx in selected {
        let item = dataset.get_item(idx)?;
        let key = DataCache::generate_key(
            Path::new(&format!("item_{}", idx)),
            &[],
        );
        
        let cached_item = CachedItem {
            tensor: item.image,
            metadata: item.metadata,
            size_bytes: 0, // Would calculate actual size
        };
        
        cache.put(&key, cached_item).await?;
    }
    
    Ok(())
}