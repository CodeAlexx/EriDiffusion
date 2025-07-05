//! Inference caching strategies

use eridiffusion_core::{Result, Error};
use candle_core::{Tensor, DType};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use lru::LruCache;
use std::num::NonZeroUsize;

/// Cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    pub memory_cache_size: usize,
    pub disk_cache_enabled: bool,
    pub disk_cache_path: String,
    pub ttl_seconds: Option<u64>,
    pub compression: CompressionType,
    pub eviction_policy: EvictionPolicy,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum CompressionType {
    None,
    Lz4,
    Zstd,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum EvictionPolicy {
    LRU,
    LFU,
    FIFO,
    TTL,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            memory_cache_size: 100,
            disk_cache_enabled: false,
            disk_cache_path: ".cache/inference".to_string(),
            ttl_seconds: Some(3600),
            compression: CompressionType::None,
            eviction_policy: EvictionPolicy::LRU,
        }
    }
}

/// Cache key
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct CacheKey {
    prompt: String,
    negative_prompt: Option<String>,
    width: usize,
    height: usize,
    guidance_scale: u32, // Quantized to avoid float comparison
    seed: Option<u64>,
    additional_params: Vec<(String, String)>, // Use Vec for Hash
}

impl std::hash::Hash for CacheKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.prompt.hash(state);
        self.negative_prompt.hash(state);
        self.width.hash(state);
        self.height.hash(state);
        self.guidance_scale.hash(state);
        self.seed.hash(state);
        // Sort additional params for consistent hashing
        let mut sorted_params = self.additional_params.clone();
        sorted_params.sort();
        sorted_params.hash(state);
    }
}

impl CacheKey {
    pub fn new(
        prompt: &str,
        negative_prompt: Option<&str>,
        width: usize,
        height: usize,
        guidance_scale: f32,
        seed: Option<u64>,
    ) -> Self {
        Self {
            prompt: prompt.to_string(),
            negative_prompt: negative_prompt.map(|s| s.to_string()),
            width,
            height,
            guidance_scale: (guidance_scale * 100.0) as u32,
            seed,
            additional_params: Vec::new(),
        }
    }
    
    pub fn with_param(mut self, key: &str, value: &str) -> Self {
        self.additional_params.push((key.to_string(), value.to_string()));
        self
    }
    
    /// Generate hash for disk storage
    pub fn hash(&self) -> String {
        use sha2::{Sha256, Digest};
        
        let mut hasher = Sha256::new();
        hasher.update(&self.prompt);
        
        if let Some(ref neg) = self.negative_prompt {
            hasher.update(neg);
        }
        
        hasher.update(&self.width.to_le_bytes());
        hasher.update(&self.height.to_le_bytes());
        hasher.update(&self.guidance_scale.to_le_bytes());
        
        if let Some(seed) = self.seed {
            hasher.update(&seed.to_le_bytes());
        }
        
        for (k, v) in &self.additional_params {
            hasher.update(k);
            hasher.update(v);
        }
        
        format!("{:x}", hasher.finalize())
    }
}

/// Cached value
#[derive(Clone)]
pub struct CachedValue {
    pub tensor: Tensor,
    pub metadata: HashMap<String, serde_json::Value>,
    pub timestamp: std::time::SystemTime,
    pub access_count: usize,
}

/// Inference cache
pub struct InferenceCache {
    config: CacheConfig,
    memory_cache: Arc<RwLock<Box<dyn CacheBackend>>>,
    disk_cache: Option<Arc<RwLock<DiskCache>>>,
    stats: Arc<RwLock<CacheStats>>,
}

#[derive(Debug, Default, Clone)]
pub struct CacheStats {
    pub hits: usize,
    pub misses: usize,
    pub evictions: usize,
    pub memory_usage_bytes: usize,
    pub disk_usage_bytes: usize,
}

impl InferenceCache {
    /// Create new inference cache
    pub async fn new(config: CacheConfig) -> Result<Self> {
        let memory_cache: Box<dyn CacheBackend> = match config.eviction_policy {
            EvictionPolicy::LRU => Box::new(LRUCache::new(config.memory_cache_size)?),
            EvictionPolicy::LFU => Box::new(LFUCache::new(config.memory_cache_size)),
            EvictionPolicy::FIFO => Box::new(FIFOCache::new(config.memory_cache_size)),
            EvictionPolicy::TTL => Box::new(TTLCache::new(
                config.memory_cache_size,
                config.ttl_seconds.unwrap_or(3600),
            )),
        };
        
        let disk_cache = if config.disk_cache_enabled {
            Some(Arc::new(RwLock::new(DiskCache::new(&config).await?)))
        } else {
            None
        };
        
        Ok(Self {
            config,
            memory_cache: Arc::new(RwLock::new(memory_cache)),
            disk_cache,
            stats: Arc::new(RwLock::new(CacheStats::default())),
        })
    }
    
    /// Get from cache
    pub async fn get(&self, key: &CacheKey) -> Option<CachedValue> {
        // Try memory cache first
        {
            let mut cache = self.memory_cache.write().await;
            if let Some(value) = cache.get(key) {
                let mut stats = self.stats.write().await;
                stats.hits += 1;
                return Some(value);
            }
        }
        
        // Try disk cache
        if let Some(ref disk_cache) = self.disk_cache {
            let disk = disk_cache.read().await;
            if let Ok(Some(value)) = disk.get(key).await {
                // Promote to memory cache
                let mut mem_cache = self.memory_cache.write().await;
                mem_cache.put(key.clone(), value.clone());
                
                let mut stats = self.stats.write().await;
                stats.hits += 1;
                return Some(value);
            }
        }
        
        let mut stats = self.stats.write().await;
        stats.misses += 1;
        None
    }
    
    /// Put in cache
    pub async fn put(&self, key: CacheKey, value: CachedValue) -> Result<()> {
        // Add to memory cache
        {
            let mut cache = self.memory_cache.write().await;
            if let Some(evicted) = cache.put(key.clone(), value.clone()) {
                let mut stats = self.stats.write().await;
                stats.evictions += 1;
                
                // Move evicted item to disk if enabled
                if let Some(ref disk_cache) = self.disk_cache {
                    let mut disk = disk_cache.write().await;
                    disk.put(evicted.0, evicted.1).await?;
                }
            }
        }
        
        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.memory_usage_bytes = self.estimate_memory_usage().await;
        }
        
        Ok(())
    }
    
    /// Clear cache
    pub async fn clear(&self) -> Result<()> {
        {
            let mut cache = self.memory_cache.write().await;
            cache.clear();
        }
        
        if let Some(ref disk_cache) = self.disk_cache {
            let mut disk = disk_cache.write().await;
            disk.clear().await?;
        }
        
        let mut stats = self.stats.write().await;
        *stats = CacheStats::default();
        
        Ok(())
    }
    
    /// Get cache statistics
    pub async fn get_stats(&self) -> CacheStats {
        self.stats.read().await.clone()
    }
    
    /// Estimate memory usage
    async fn estimate_memory_usage(&self) -> usize {
        let cache = self.memory_cache.read().await;
        cache.size() * std::mem::size_of::<CachedValue>() // Rough estimate
    }
}

/// Cache backend trait
#[async_trait::async_trait]
trait CacheBackend: Send + Sync {
    fn get(&mut self, key: &CacheKey) -> Option<CachedValue>;
    fn put(&mut self, key: CacheKey, value: CachedValue) -> Option<(CacheKey, CachedValue)>;
    fn clear(&mut self);
    fn size(&self) -> usize;
}

/// LRU cache implementation
struct LRUCache {
    cache: LruCache<CacheKey, CachedValue>,
}

impl LRUCache {
    fn new(capacity: usize) -> Result<Self> {
        Ok(Self {
            cache: LruCache::new(
                NonZeroUsize::new(capacity)
                    .ok_or_else(|| Error::InvalidInput("Invalid cache capacity".to_string()))?
            ),
        })
    }
}

#[async_trait::async_trait]
impl CacheBackend for LRUCache {
    fn get(&mut self, key: &CacheKey) -> Option<CachedValue> {
        self.cache.get(key).cloned()
    }
    
    fn put(&mut self, key: CacheKey, value: CachedValue) -> Option<(CacheKey, CachedValue)> {
        self.cache.push(key, value)
    }
    
    fn clear(&mut self) {
        self.cache.clear();
    }
    
    fn size(&self) -> usize {
        self.cache.len()
    }
}

/// LFU cache implementation
struct LFUCache {
    cache: HashMap<CacheKey, CachedValue>,
    frequencies: HashMap<CacheKey, usize>,
    capacity: usize,
}

impl LFUCache {
    fn new(capacity: usize) -> Self {
        Self {
            cache: HashMap::new(),
            frequencies: HashMap::new(),
            capacity,
        }
    }
}

#[async_trait::async_trait]
impl CacheBackend for LFUCache {
    fn get(&mut self, key: &CacheKey) -> Option<CachedValue> {
        if let Some(value) = self.cache.get_mut(key) {
            if let Some(freq) = self.frequencies.get_mut(key) {
                *freq += 1;
            }
            value.access_count += 1;
            Some(value.clone())
        } else {
            None
        }
    }
    
    fn put(&mut self, key: CacheKey, mut value: CachedValue) -> Option<(CacheKey, CachedValue)> {
        value.access_count = 1;
        
        let evicted = if self.cache.len() >= self.capacity && !self.cache.contains_key(&key) {
            // Find least frequently used
            let lfu_key = self.frequencies.iter()
                .min_by_key(|(_, &freq)| freq)
                .map(|(k, _)| k.clone());
            
            if let Some(evict_key) = lfu_key {
                self.frequencies.remove(&evict_key);
                self.cache.remove(&evict_key)
                    .map(|v| (evict_key, v))
            } else {
                None
            }
        } else {
            None
        };
        
        self.cache.insert(key.clone(), value);
        self.frequencies.insert(key, 1);
        
        evicted
    }
    
    fn clear(&mut self) {
        self.cache.clear();
        self.frequencies.clear();
    }
    
    fn size(&self) -> usize {
        self.cache.len()
    }
}

/// FIFO cache implementation
struct FIFOCache {
    cache: HashMap<CacheKey, CachedValue>,
    order: std::collections::VecDeque<CacheKey>,
    capacity: usize,
}

impl FIFOCache {
    fn new(capacity: usize) -> Self {
        Self {
            cache: HashMap::new(),
            order: std::collections::VecDeque::new(),
            capacity,
        }
    }
}

#[async_trait::async_trait]
impl CacheBackend for FIFOCache {
    fn get(&mut self, key: &CacheKey) -> Option<CachedValue> {
        self.cache.get(key).cloned()
    }
    
    fn put(&mut self, key: CacheKey, value: CachedValue) -> Option<(CacheKey, CachedValue)> {
        let evicted = if self.cache.len() >= self.capacity && !self.cache.contains_key(&key) {
            if let Some(evict_key) = self.order.pop_front() {
                self.cache.remove(&evict_key)
                    .map(|v| (evict_key, v))
            } else {
                None
            }
        } else {
            None
        };
        
        if !self.cache.contains_key(&key) {
            self.order.push_back(key.clone());
        }
        self.cache.insert(key, value);
        
        evicted
    }
    
    fn clear(&mut self) {
        self.cache.clear();
        self.order.clear();
    }
    
    fn size(&self) -> usize {
        self.cache.len()
    }
}

/// TTL cache implementation
struct TTLCache {
    cache: HashMap<CacheKey, CachedValue>,
    capacity: usize,
    ttl: std::time::Duration,
}

impl TTLCache {
    fn new(capacity: usize, ttl_seconds: u64) -> Self {
        Self {
            cache: HashMap::new(),
            capacity,
            ttl: std::time::Duration::from_secs(ttl_seconds),
        }
    }
    
    fn is_expired(&self, value: &CachedValue) -> bool {
        if let Ok(elapsed) = value.timestamp.elapsed() {
            elapsed > self.ttl
        } else {
            true
        }
    }
    
    fn remove_expired(&mut self) {
        let expired_keys: Vec<CacheKey> = self.cache.iter()
            .filter(|(_, v)| self.is_expired(v))
            .map(|(k, _)| k.clone())
            .collect();
        
        for key in expired_keys {
            self.cache.remove(&key);
        }
    }
}

#[async_trait::async_trait]
impl CacheBackend for TTLCache {
    fn get(&mut self, key: &CacheKey) -> Option<CachedValue> {
        self.remove_expired();
        
        if let Some(value) = self.cache.get(key) {
            if !self.is_expired(value) {
                Some(value.clone())
            } else {
                self.cache.remove(key);
                None
            }
        } else {
            None
        }
    }
    
    fn put(&mut self, key: CacheKey, value: CachedValue) -> Option<(CacheKey, CachedValue)> {
        self.remove_expired();
        
        let evicted = if self.cache.len() >= self.capacity && !self.cache.contains_key(&key) {
            // Remove oldest entry
            let oldest = self.cache.iter()
                .min_by_key(|(_, v)| v.timestamp)
                .map(|(k, _)| k.clone());
            
            if let Some(evict_key) = oldest {
                self.cache.remove(&evict_key)
                    .map(|v| (evict_key, v))
            } else {
                None
            }
        } else {
            None
        };
        
        self.cache.insert(key, value);
        evicted
    }
    
    fn clear(&mut self) {
        self.cache.clear();
    }
    
    fn size(&self) -> usize {
        self.cache.len()
    }
}

/// Disk cache
struct DiskCache {
    config: CacheConfig,
    index: HashMap<String, CacheMetadata>,
}

#[derive(Serialize, Deserialize)]
struct CacheMetadata {
    key_hash: String,
    timestamp: std::time::SystemTime,
    size_bytes: usize,
}

impl DiskCache {
    async fn new(config: &CacheConfig) -> Result<Self> {
        tokio::fs::create_dir_all(&config.disk_cache_path).await?;
        
        let mut cache = Self {
            config: config.clone(),
            index: HashMap::new(),
        };
        
        cache.load_index().await?;
        Ok(cache)
    }
    
    async fn get(&self, key: &CacheKey) -> Result<Option<CachedValue>> {
        let hash = key.hash();
        
        if self.index.contains_key(&hash) {
            let path = std::path::Path::new(&self.config.disk_cache_path)
                .join(format!("{}.cache", hash));
            
            if path.exists() {
                let data = tokio::fs::read(&path).await?;
                
                // Decompress if needed
                let decompressed = match self.config.compression {
                    CompressionType::None => data,
                    CompressionType::Lz4 => {
                        lz4::block::decompress(&data, None)?
                    }
                    CompressionType::Zstd => {
                        zstd::decode_all(&data[..])?
                    }
                };
                
                // Deserialize (simplified - would use actual tensor serialization)
                let value = CachedValue {
                    tensor: Tensor::zeros(&[1], DType::F32, &candle_core::Device::Cpu)?,
                    metadata: HashMap::new(),
                    timestamp: std::time::SystemTime::now(),
                    access_count: 0,
                };
                
                Ok(Some(value))
            } else {
                Ok(None)
            }
        } else {
            Ok(None)
        }
    }
    
    async fn put(&mut self, key: CacheKey, value: CachedValue) -> Result<()> {
        let hash = key.hash();
        let path = std::path::Path::new(&self.config.disk_cache_path)
            .join(format!("{}.cache", hash));
        
        // Serialize (simplified)
        let data = vec![0u8; 1024];
        
        // Compress if needed
        let compressed = match self.config.compression {
            CompressionType::None => data,
            CompressionType::Lz4 => {
                lz4::block::compress(&data, None, true)?
            }
            CompressionType::Zstd => {
                zstd::encode_all(&data[..], 3)?
            }
        };
        
        tokio::fs::write(&path, &compressed).await?;
        
        self.index.insert(hash.clone(), CacheMetadata {
            key_hash: hash,
            timestamp: value.timestamp,
            size_bytes: compressed.len(),
        });
        
        self.save_index().await?;
        Ok(())
    }
    
    async fn clear(&mut self) -> Result<()> {
        let path = std::path::Path::new(&self.config.disk_cache_path);
        
        for hash in self.index.keys() {
            let cache_path = path.join(format!("{}.cache", hash));
            let _ = tokio::fs::remove_file(cache_path).await;
        }
        
        self.index.clear();
        self.save_index().await?;
        Ok(())
    }
    
    async fn load_index(&mut self) -> Result<()> {
        let index_path = std::path::Path::new(&self.config.disk_cache_path)
            .join("index.json");
        
        if index_path.exists() {
            let data = tokio::fs::read_to_string(&index_path).await?;
            self.index = serde_json::from_str(&data)?;
        }
        
        Ok(())
    }
    
    async fn save_index(&self) -> Result<()> {
        let index_path = std::path::Path::new(&self.config.disk_cache_path)
            .join("index.json");
        
        let data = serde_json::to_string_pretty(&self.index)?;
        tokio::fs::write(&index_path, data).await?;
        
        Ok(())
    }
}