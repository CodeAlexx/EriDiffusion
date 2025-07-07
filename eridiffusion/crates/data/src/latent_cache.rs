//! Latent caching system for efficient training

use eridiffusion_core::{Result, Error, Device, ModelArchitecture, ErrorContext};
use eridiffusion_models::vae::{VAE, VAEFactory};
use candle_core::{Tensor, DType};
use candle_nn::{VarBuilder, VarMap};
use std::path::{Path, PathBuf};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use safetensors::{SafeTensors, serialize};

/// Latent cache for storing pre-computed VAE encodings
pub struct LatentCache {
    cache_dir: PathBuf,
    vae: Box<dyn VAE>,
    architecture: ModelArchitecture,
    device: Device,
    memory_cache: Arc<Mutex<HashMap<String, Tensor>>>,
    max_memory_items: usize,
}

impl LatentCache {
    /// Create new latent cache
    pub fn new(
        cache_dir: PathBuf,
        architecture: ModelArchitecture,
        device: Device,
        vae_path: Option<PathBuf>,
    ) -> Result<Self> {
        // Create cache directory
        std::fs::create_dir_all(&cache_dir)
            .context("Failed to create cache directory")?;
        
        // Create VAE
        let candle_device = match &device {
            Device::Cpu => candle_core::Device::Cpu,
            Device::Cuda(id) => candle_core::Device::new_cuda(*id)?,
        };
        
        let vae = if let Some(path) = vae_path {
            // Load VAE from checkpoint
            let mut varmap = VarMap::new();
            varmap.load(&path)?;
            let vb = VarBuilder::from_varmap(&varmap, DType::F32, &candle_device);
            VAEFactory::create(architecture, vb)?
        } else {
            // Create empty VAE (weights need to be loaded separately)
            let varmap = VarMap::new();
            let vb = VarBuilder::from_varmap(&varmap, DType::F32, &candle_device);
            VAEFactory::create(architecture, vb)?
        };
        
        Ok(Self {
            cache_dir,
            vae,
            architecture,
            device,
            memory_cache: Arc::new(Mutex::new(HashMap::new())),
            max_memory_items: 100, // Keep 100 items in memory
        })
    }
    
    /// Get cache key for an image
    pub fn get_cache_key(&self, image_path: &Path) -> String {
        let metadata = std::fs::metadata(image_path).ok();
        let modified = metadata
            .and_then(|m| m.modified().ok())
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs())
            .unwrap_or(0);
        
        format!(
            "{}_{}_{}_{}",
            image_path.file_stem().unwrap_or_default().to_string_lossy(),
            modified,
            self.architecture.to_string(),
            self.vae.latent_channels()
        )
    }
    
    /// Get cache file path
    pub fn get_cache_path(&self, cache_key: &str) -> PathBuf {
        self.cache_dir.join(format!("{}.safetensors", cache_key))
    }
    
    /// Load or compute latents for an image
    pub fn get_latents(&self, image_path: &Path, image_tensor: &Tensor) -> Result<Tensor> {
        let cache_key = self.get_cache_key(image_path);
        
        // Check memory cache first
        if let Ok(cache) = self.memory_cache.lock() {
            if let Some(latents) = cache.get(&cache_key) {
                return Ok(latents.clone());
            }
        }
        
        // Check disk cache
        let cache_path = self.get_cache_path(&cache_key);
        if cache_path.exists() {
            match self.load_from_disk(&cache_path) {
                Ok(latents) => {
                    // Add to memory cache
                    self.add_to_memory_cache(cache_key, latents.clone());
                    return Ok(latents);
                }
                Err(e) => {
                    eprintln!("Failed to load cached latents: {}", e);
                    // Continue to recompute
                }
            }
        }
        
        // Compute latents
        let latents = self.encode_image(image_tensor)?;
        
        // Save to disk
        if let Err(e) = self.save_to_disk(&cache_path, &latents) {
            eprintln!("Failed to save latents to cache: {}", e);
        }
        
        // Add to memory cache
        self.add_to_memory_cache(cache_key, latents.clone());
        
        Ok(latents)
    }
    
    /// Get or compute latents for an image
    pub async fn get_or_compute_latent(
        &self,
        image_tensor: &Tensor,
        image_path: &Path,
        _resolution: (usize, usize),
    ) -> Result<Tensor> {
        // This is just an async wrapper around get_latents
        self.get_latents(image_path, image_tensor)
    }
    
    /// Encode image to latents
    fn encode_image(&self, image: &Tensor) -> Result<Tensor> {
        // Ensure image is on correct device
        let candle_device = match &self.device {
            Device::Cpu => candle_core::Device::Cpu,
            Device::Cuda(id) => candle_core::Device::new_cuda(*id)?,
        };
        let image = image.to_device(&candle_device)?;
        
        // Normalize to [-1, 1] if needed
        let normalized = if image.dtype() == DType::U8 {
            image.to_dtype(DType::F32)?.affine(2.0 / 255.0, -1.0)?
        } else {
            // Assume already normalized
            image.clone()
        };
        
        // Encode deterministically for caching
        self.vae.encode_deterministic(&normalized)
    }
    
    /// Save latents to disk
    pub fn save_to_disk(&self, path: &Path, latents: &Tensor) -> Result<()> {
        // Convert to CPU for saving
        let latents_cpu = latents.to_device(&candle_core::Device::Cpu)?;
        
        // Create safetensors data
        let mut tensors = HashMap::new();
        tensors.insert("latents".to_string(), latents_cpu);
        
        let data = serialize(&tensors, &None)
            .map_err(|e| Error::Serialization(e.to_string()))?;
        
        std::fs::write(path, data)
            .context("Failed to write latents to disk")?;
        
        Ok(())
    }
    
    /// Load latents from disk
    fn load_from_disk(&self, path: &Path) -> Result<Tensor> {
        let data = std::fs::read(path)
            .context("Failed to read latent cache file")?;
        
        let safetensors = SafeTensors::deserialize(&data)
            .map_err(|e| Error::Serialization(e.to_string()))?;
        
        let latents = safetensors.tensor("latents")
            .map_err(|e| Error::Serialization(e.to_string()))?;
        
        // Convert to tensor and move to device
        let shape = latents.shape().to_vec();
        let data = latents.data();
        
        let candle_device = match &self.device {
            Device::Cpu => candle_core::Device::Cpu,
            Device::Cuda(id) => candle_core::Device::new_cuda(*id)?,
        };
        
        let tensor = Tensor::from_raw_buffer(
            data,
            latents.dtype().try_into()?,
            &shape,
            &candle_core::Device::Cpu,
        )?;
        
        Ok(tensor.to_device(&candle_device)?)
    }
    
    /// Add to memory cache with LRU eviction
    fn add_to_memory_cache(&self, key: String, tensor: Tensor) {
        if let Ok(mut cache) = self.memory_cache.lock() {
            // Simple eviction - remove oldest if at capacity
            if cache.len() >= self.max_memory_items {
                if let Some(first_key) = cache.keys().next().cloned() {
                    cache.remove(&first_key);
                }
            }
            
            cache.insert(key, tensor);
        }
    }
    
    /// Clear memory cache
    pub fn clear_memory_cache(&self) {
        if let Ok(mut cache) = self.memory_cache.lock() {
            cache.clear();
        }
    }
    
    /// Clear disk cache
    pub fn clear_disk_cache(&self) -> Result<()> {
        for entry in std::fs::read_dir(&self.cache_dir)? {
            let entry = entry?;
            if entry.path().extension().and_then(|s| s.to_str()) == Some("safetensors") {
                std::fs::remove_file(entry.path())?;
            }
        }
        Ok(())
    }
    
    /// Get cache statistics
    pub fn get_stats(&self) -> CacheStats {
        let memory_items = self.memory_cache.lock()
            .map(|c| c.len())
            .unwrap_or(0);
        
        let disk_items = std::fs::read_dir(&self.cache_dir)
            .ok()
            .map(|entries| {
                entries.filter_map(|e| e.ok())
                    .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("safetensors"))
                    .count()
            })
            .unwrap_or(0);
        
        CacheStats {
            memory_items,
            disk_items,
            cache_dir: self.cache_dir.clone(),
        }
    }
}

/// Cache statistics
#[derive(Debug)]
pub struct CacheStats {
    pub memory_items: usize,
    pub disk_items: usize,
    pub cache_dir: PathBuf,
}

/// Batch latent encoder for preprocessing entire datasets
pub struct BatchLatentEncoder {
    cache: LatentCache,
    batch_size: usize,
}

impl BatchLatentEncoder {
    pub fn new(cache: LatentCache, batch_size: usize) -> Self {
        Self { cache, batch_size }
    }
    
    /// Encode a batch of images
    pub async fn encode_batch(
        &self,
        image_paths: &[PathBuf],
        progress_callback: Option<Box<dyn Fn(usize, usize) + Send>>,
    ) -> Result<Vec<PathBuf>> {
        let total = image_paths.len();
        let mut encoded_paths = Vec::new();
        
        for (i, chunk) in image_paths.chunks(self.batch_size).enumerate() {
            // Load images
            let mut images = Vec::new();
            for path in chunk {
                let image = load_image(path)?;
                images.push(image);
            }
            
            // Stack into batch
            let batch = Tensor::cat(&images, 0)?;
            
            // Process each image
            for (j, path) in chunk.iter().enumerate() {
                let image = batch.narrow(0, j, 1)?;
                self.cache.get_latents(path, &image)?;
                encoded_paths.push(path.clone());
            }
            
            // Progress callback
            if let Some(ref callback) = progress_callback {
                let processed = (i + 1) * self.batch_size;
                callback(processed.min(total), total);
            }
        }
        
        Ok(encoded_paths)
    }
}

/// Load image from path
fn load_image(path: &Path) -> Result<Tensor> {
    use image::{ImageReader, DynamicImage};
    
    let img = ImageReader::open(path)
        .map_err(|e| Error::Io(e))?
        .decode()
        .map_err(|e| Error::DataError(format!("Failed to decode image: {}", e)))?;
    
    // Convert to RGB if needed
    let img = match img {
        DynamicImage::ImageRgb8(_) => img,
        _ => DynamicImage::ImageRgb8(img.to_rgb8()),
    };
    
    let (width, height) = (img.width() as usize, img.height() as usize);
    let pixels: Vec<u8> = img.to_rgb8().into_raw();
    
    // Create tensor [H, W, C]
    let tensor = Tensor::from_vec(pixels, &[height, width, 3], &candle_core::Device::Cpu)?;
    
    // Permute to [C, H, W] and add batch dimension
    Ok(tensor.permute((2, 0, 1))?.unsqueeze(0)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cache_key_generation() {
        let cache_dir = PathBuf::from("/tmp/test_cache");
        let cache = LatentCache::new(
            cache_dir,
            ModelArchitecture::SD3,
            Device::Cpu,
            None,
        ).unwrap();
        
        let path = PathBuf::from("/images/test.jpg");
        let key = cache.get_cache_key(&path);
        
        assert!(key.contains("test"));
        assert!(key.contains("SD3"));
        assert!(key.contains("16")); // SD3 has 16 latent channels
    }
}