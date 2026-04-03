//! Flame-native multi-level cache for tensors (L1 in-memory, L2 on-disk)

use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;

use eridiffusion_core::{Device, Error};
use flame_core::{DType, Shape, Tensor};
use safetensors::{tensor::TensorView, Dtype as SafeDtype, SafeTensors};

/// Cache entry containing tensor and metadata
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

#[derive(Debug, Default, Clone)]
struct CacheStats {
    memory_hits: u64,
    memory_misses: u64,
    disk_hits: u64,
    disk_misses: u64,
    total_requests: u64,
}

/// Multi-level cache manager (Flame-only)
pub struct CacheManager {
    memory_cache: Arc<DashMap<String, CacheEntry>>, // L1
    disk_cache_dir: PathBuf,                         // L2 root
    memory_limit_mb: usize,
    current_memory_mb: Arc<RwLock<usize>>,
    stats: Arc<RwLock<CacheStats>>,
    device: Device, // shared device used to materialize tensors
}

impl CacheManager {
    pub fn new(disk_cache_dir: PathBuf, memory_limit_mb: usize, device: Device) -> anyhow::Result<Self> {
        std::fs::create_dir_all(&disk_cache_dir)?;
        Ok(Self {
            memory_cache: Arc::new(DashMap::new()),
            disk_cache_dir,
            memory_limit_mb,
            current_memory_mb: Arc::new(RwLock::new(0)),
            stats: Arc::new(RwLock::new(CacheStats::default())),
            device,
        })
    }

    /// Compute a stable cache key from a path
    fn cache_key(path: &Path) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(path.to_string_lossy().as_bytes());
        format!("{:x}", hasher.finalize())
    }

    /// Try to get an entry from cache
    pub async fn get(&self, path: &Path) -> Option<CacheEntry> {
        let key = Self::cache_key(path);
        // L1
        if let Some(entry) = self.memory_cache.get(&key) {
            self.stats.write().await.memory_hits += 1;
            return Some(entry.clone());
        }
        self.stats.write().await.memory_misses += 1;

        // L2
        let disk_path = self.disk_cache_dir.join(&key).with_extension("cache");
        if disk_path.exists() {
            match self.load_from_disk(&disk_path).await {
                Ok(entry) => {
                    self.stats.write().await.disk_hits += 1;
                    let _ = self.add_to_memory_cache(key, entry.clone()).await;
                    return Some(entry);
                }
                Err(e) => {
                    tracing::warn!("disk cache load failed: {}", e);
                }
            }
        } else {
            self.stats.write().await.disk_misses += 1;
        }
        None
    }

    /// Put an entry into cache (L1 + L2)
    pub async fn put(&self, path: &Path, tensor: Tensor, metadata: CacheMetadata) -> anyhow::Result<()> {
        let key = Self::cache_key(path);
        let entry = CacheEntry { tensor, metadata };
        self.add_to_memory_cache(key.clone(), entry.clone()).await?;
        let disk_path = self.disk_cache_dir.join(&key).with_extension("cache");
        self.save_to_disk(&disk_path, &entry).await?;
        Ok(())
    }

    async fn add_to_memory_cache(&self, key: String, entry: CacheEntry) -> anyhow::Result<()> {
        let size_mb = Self::estimate_tensor_size_mb(&entry.tensor);
        let mut cur = self.current_memory_mb.write().await;
        while *cur + size_mb > self.memory_limit_mb && !self.memory_cache.is_empty() {
            // FIFO eviction
            if let Some(ev) = self.memory_cache.iter().next() {
                let k = ev.key().clone();
                let sz = Self::estimate_tensor_size_mb(&ev.value().tensor);
                drop(ev);
                self.memory_cache.remove(&k);
                *cur = cur.saturating_sub(sz);
            } else {
                break;
            }
        }
        self.memory_cache.insert(key, entry);
        *cur += size_mb;
        Ok(())
    }

    fn estimate_tensor_size_mb(t: &Tensor) -> usize {
        let numel: usize = t.shape().dims().iter().product();
        let bpe = match t.dtype() { DType::F32 => 4, DType::BF16 | DType::F16 => 2, DType::I32 => 4, DType::Bool => 1, _ => 4 };
        (numel * bpe) / (1024 * 1024)
    }

    async fn save_to_disk(&self, path: &Path, entry: &CacheEntry) -> anyhow::Result<()> {
        // Serialize tensor to safetensors with true dtype
        let (bytes, dtype, shape_dims) = tensor_to_bytes(&entry.tensor)?;
        let mut map = std::collections::BTreeMap::new();
        let view = TensorView::new(dtype, shape_dims.clone(), &bytes)
            .map_err(|e| Error::Training(format!("safetensors view error: {}", e)))?;
        map.insert("data".to_string(), view);
        let serialized = safetensors::serialize(map, &None)
            .map_err(|e| Error::Training(format!("safetensors serialize error: {}", e)))?;
        // Write files
        std::fs::write(path.with_extension("tensor"), serialized)?;
        let meta = bincode::serialize(&entry.metadata)
            .map_err(|e| Error::DataError(format!("meta serialize: {}", e)))?;
        std::fs::write(path.with_extension("meta"), meta)?;
        Ok(())
    }

    async fn load_from_disk(&self, stem: &Path) -> anyhow::Result<CacheEntry> {
        let tens_path = stem.with_extension("tensor");
        let meta_path = stem.with_extension("meta");
        let bytes = std::fs::read(&tens_path)?;
        let st = SafeTensors::deserialize(&bytes)
            .map_err(|e| Error::DataError(format!("safetensors parse: {}", e)))?;
        let tv = st.tensor("data").map_err(|e| Error::DataError(e.to_string()))?;
        let metadata_bytes = std::fs::read(&meta_path)?;
        let metadata: CacheMetadata = bincode::deserialize(&metadata_bytes)
            .map_err(|e| Error::DataError(format!("meta parse: {}", e)))?;
        // Strict shape check
        if tv.shape() != &metadata.shape[..] {
            return Err(Error::DataError(format!(
                "shape mismatch: stored={:?} meta={:?}",
                tv.shape(), metadata.shape
            )).into());
        }
        // Reconstruct on shared device as F32 then cast to metadata dtype
        let dev = self.device.to_flame_cuda()?;
        let (f32_data, _shape) = match tv.dtype() {
            SafeDtype::F32 => {
                let v: Vec<f32> = tv
                    .data()
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect();
                (v, tv.shape().to_vec())
            }
            SafeDtype::BF16 => {
                use half::bf16;
                let v: Vec<f32> = tv
                    .data()
                    .chunks_exact(2)
                    .map(|b| bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
                    .collect();
                (v, tv.shape().to_vec())
            }
            other => {
                return Err(Error::DataError(format!("unsupported dtype in cache: {:?}", other)).into());
            }
        };
        let mut tensor = Tensor::from_slice(&f32_data, Shape::from_dims(tv.shape()), dev)?;
        // Cast to requested dtype
        let target_dt = parse_dtype(&metadata.dtype)?;
        if tensor.dtype() != target_dt { tensor = tensor.to_dtype(target_dt)?; }
        Ok(CacheEntry { tensor, metadata })
    }
}

fn tensor_to_bytes(t: &Tensor) -> anyhow::Result<(Vec<u8>, SafeDtype, Vec<usize>)> {
    let shape = t.shape().dims().to_vec();
    match t.dtype() {
        DType::F32 => {
            let v: Vec<f32> = t.to_vec()?;
            let bytes: Vec<u8> = bytemuck::cast_vec(v);
            Ok((bytes, SafeDtype::F32, shape))
        }
        DType::BF16 => {
            use half::bf16;
            let vf32: Vec<f32> = t.to_vec()?; // download as f32
            let vbits: Vec<u16> = vf32.into_iter().map(|x| bf16::from_f32(x).to_bits()).collect();
            let bytes: Vec<u8> = bytemuck::cast_vec(vbits);
            Ok((bytes, SafeDtype::BF16, shape))
        }
        other => Err(Error::DataError(format!("unsupported dtype for cache save: {:?}", other)).into()),
    }
}

fn parse_dtype(s: &str) -> anyhow::Result<DType> {
    Ok(match s {
        "F32" => DType::F32,
        "BF16" => DType::BF16,
        "F16" => DType::F16,
        "I32" => DType::I32,
        "Bool" | "BOOL" => DType::Bool,
        other => return Err(Error::DataError(format!("unknown dtype: {}", other)).into()),
    })
}

