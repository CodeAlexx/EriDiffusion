//! Persistent text embedding cache that saves to disk
//!
//! This module provides disk-based caching for text embeddings to avoid
//! re-encoding across training sessions.

use flame_core::device::Device;
use flame_core::{DType, Error, Shape, Tensor};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
// Removed safetensors dependency - using simple binary format instead
use flame_core::Result;

/// Persistent cache for text embeddings
pub struct PersistentEmbeddingCache {
    cache_dir: PathBuf,
    memory_cache: HashMap<String, (Tensor, Option<Tensor>)>,
    device: Device,
}

impl PersistentEmbeddingCache {
    /// Create a new persistent cache
    pub fn new(cache_dir: PathBuf, device: Device) -> Result<Self> {
        // Create cache directory if it doesn't exist
        fs::create_dir_all(&cache_dir)
            .map_err(|e| Error::Io(format!("Failed to create cache dir: {}", e)))?;

        println!("📁 Persistent embedding cache at: {:?}", cache_dir);

        Ok(Self { cache_dir, memory_cache: HashMap::new(), device })
    }

    /// Generate a safe filename from prompt text
    fn prompt_to_filename(prompt: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        prompt.hash(&mut hasher);
        let hash = hasher.finish();

        // Use first 20 chars of prompt (sanitized) + hash for readable filenames
        let safe_prompt: String = prompt
            .chars()
            .take(20)
            .map(|c| if c.is_alphanumeric() || c == ' ' { c } else { '_' })
            .collect::<String>()
            .replace(' ', "_");

        format!("{}_{:016x}.cache", safe_prompt, hash)
    }

    /// Check if embeddings exist in cache (memory or disk)
    pub fn get(&mut self, prompt: &str) -> Result<Option<(Tensor, Option<Tensor>)>> {
        // Check memory cache first
        if let Some(cached) = self.memory_cache.get(prompt) {
            println!("  💾 Memory cache hit: \"{}\"", prompt);
            return Ok(Some(cached.clone()));
        }

        // Check disk cache
        let filename = Self::prompt_to_filename(prompt);
        let filepath = self.cache_dir.join(&filename);

        if filepath.exists() {
            println!("  💿 Disk cache hit: \"{}\" from {}", prompt, filename);

            // Load from disk using our simple binary format
            let data = fs::read(&filepath)
                .map_err(|e| Error::Io(format!("Failed to read cache file: {}", e)))?;

            // Parse our simple binary format
            if data.len() < 24 {
                // At least 3 u64 lengths
                return Ok(None);
            }

            let mut offset = 0;
            let clip_len =
                u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap()) as usize;
            offset += 8;

            if offset + clip_len > data.len() {
                return Ok(None);
            }
            let clip_data = &data[offset..offset + clip_len];
            offset += clip_len;

            let t5_len = u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap()) as usize;
            offset += 8;

            let t5_embed = if t5_len > 0 {
                if offset + t5_len > data.len() {
                    return Ok(None);
                }
                let t5_data = &data[offset..offset + t5_len];
                offset += t5_len;
                Some(self.load_tensor_from_bytes(t5_data, &[256, 4096])?) // T5 shape
            } else {
                None
            };

            let clip_embed = self.load_tensor_from_bytes(clip_data, &[77, 768])?; // CLIP shape

            // Store in memory cache for faster subsequent access
            self.memory_cache.insert(prompt.to_string(), (clip_embed.clone(), t5_embed.clone()));

            return Ok(Some((clip_embed, t5_embed)));
        }

        Ok(None)
    }

    /// Save embeddings to cache (memory and disk)
    pub fn save(
        &mut self,
        prompt: &str,
        clip_embed: &Tensor,
        t5_embed: Option<&Tensor>,
    ) -> Result<()> {
        // Save to memory cache
        self.memory_cache.insert(prompt.to_string(), (clip_embed.clone(), t5_embed.cloned()));

        // Save to disk
        let filename = Self::prompt_to_filename(prompt);
        let filepath = self.cache_dir.join(&filename);

        // For now, use a simple binary format until safetensors is working
        // TODO: Replace with proper safetensors implementation
        let clip_data = self.tensor_to_bytes(clip_embed)?;
        let t5_data = if let Some(t5) = t5_embed { Some(self.tensor_to_bytes(t5)?) } else { None };

        // Create a simple binary format: [clip_len][clip_data][t5_len][t5_data][prompt_len][prompt]
        let mut serialized = Vec::new();
        serialized.extend_from_slice(&(clip_data.len() as u64).to_le_bytes());
        serialized.extend_from_slice(&clip_data);

        if let Some(ref t5_data) = t5_data {
            serialized.extend_from_slice(&(t5_data.len() as u64).to_le_bytes());
            serialized.extend_from_slice(t5_data);
        } else {
            serialized.extend_from_slice(&0u64.to_le_bytes());
        }

        let prompt_bytes = prompt.as_bytes();
        serialized.extend_from_slice(&(prompt_bytes.len() as u64).to_le_bytes());
        serialized.extend_from_slice(prompt_bytes);

        fs::write(&filepath, serialized)
            .map_err(|e| Error::Io(format!("Failed to write cache file: {}", e)))?;

        println!("  💾 Saved to disk: {}", filename);
        Ok(())
    }

    /// Convert Tensor to bytes for safetensors
    fn tensor_to_bytes(&self, tensor: &Tensor) -> Result<Vec<u8>> {
        // Get tensor data as f32 vector
        let data = tensor.to_vec1::<f32>()?;

        // Convert to bytes based on dtype
        let bytes = match tensor.dtype() {
            DType::F32 => data.iter().flat_map(|&f| f.to_le_bytes()).collect(),
            DType::F16 => data.iter().flat_map(|&f| half::f16::from_f32(f).to_le_bytes()).collect(),
            _ => {
                return Err(Error::InvalidOperation(format!(
                    "Unsupported dtype for caching: {:?}",
                    tensor.dtype()
                )));
            }
        };

        Ok(bytes)
    }

    /// Load tensor from raw bytes
    fn load_tensor_from_bytes(&self, data: &[u8], shape_dims: &[usize]) -> Result<Tensor> {
        let shape = Shape::from_dims(shape_dims);

        // Convert bytes back to f32 data
        let float_data: Vec<f32> = data
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        Tensor::from_vec(float_data, shape, self.device.cuda_device().clone())
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        let disk_files = fs::read_dir(&self.cache_dir).map(|entries| entries.count()).unwrap_or(0);

        CacheStats {
            memory_entries: self.memory_cache.len(),
            disk_entries: disk_files,
            cache_dir: self.cache_dir.clone(),
        }
    }

    /// Clear memory cache (disk cache remains)
    pub fn clear_memory(&mut self) {
        self.memory_cache.clear();
        println!("Memory cache cleared");
    }

    /// Clear disk cache
    pub fn clear_disk(&self) -> Result<()> {
        let entries = fs::read_dir(&self.cache_dir)
            .map_err(|e| Error::Io(format!("Failed to read cache dir: {}", e)))?;

        let mut removed = 0;
        for entry in entries {
            if let Ok(entry) = entry {
                if entry.path().extension().and_then(|s| s.to_str()) == Some("cache") {
                    fs::remove_file(entry.path()).ok();
                    removed += 1;
                }
            }
        }

        println!("Removed {} cached embeddings from disk", removed);
        Ok(())
    }

    /// Preload common prompts into memory
    pub fn preload_from_disk(&mut self) -> Result<()> {
        let entries = fs::read_dir(&self.cache_dir)
            .map_err(|e| Error::Io(format!("Failed to read cache dir: {}", e)))?;

        let mut loaded = 0;
        for entry in entries {
            if let Ok(entry) = entry {
                let path = entry.path();
                if path.extension().and_then(|s| s.to_str()) == Some("cache") {
                    // Try to load each cached file
                    if let Ok(data) = fs::read(&path) {
                        // Parse our simple binary format to get prompt
                        if data.len() >= 24 {
                            let mut offset = 8; // Skip clip length
                            let clip_len =
                                u64::from_le_bytes(data[0..8].try_into().unwrap()) as usize;
                            offset += clip_len;

                            let t5_len =
                                u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap())
                                    as usize;
                            offset += 8 + t5_len;

                            if offset + 8 <= data.len() {
                                let prompt_len = u64::from_le_bytes(
                                    data[offset..offset + 8].try_into().unwrap(),
                                ) as usize;
                                offset += 8;

                                if offset + prompt_len <= data.len() {
                                    let prompt =
                                        String::from_utf8_lossy(&data[offset..offset + prompt_len])
                                            .to_string();

                                    // Load the embeddings for this prompt
                                    if let Ok(Some((clip, t5))) = self.get(&prompt) {
                                        loaded += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        println!("Preloaded {} embeddings from disk into memory", loaded);
        Ok(())
    }
}

pub struct CacheStats {
    pub memory_entries: usize,
    pub disk_entries: usize,
    pub cache_dir: PathBuf,
}

impl std::fmt::Display for CacheStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Cache Stats: {} in memory, {} on disk at {:?}",
            self.memory_entries, self.disk_entries, self.cache_dir
        )
    }
}
