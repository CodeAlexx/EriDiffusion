//! DataLoader with integrated latent encoding and caching

use crate::{Dataset, DatasetItem, DataLoaderBatch, LatentCache, VAEPreprocessor};
use eridiffusion_core::{Result, Error, Device, ModelArchitecture};
use eridiffusion_models::VAE;
use candle_core::{Tensor, DType};
use std::sync::Arc;
use std::path::PathBuf;
use tokio::sync::{mpsc, Mutex};
use rand::seq::SliceRandom;
use tracing::{debug, warn, info};

/// Configuration for latent dataloader
#[derive(Debug, Clone)]
pub struct LatentDataLoaderConfig {
    pub batch_size: usize,
    pub shuffle: bool,
    pub drop_last: bool,
    pub num_workers: usize,
    pub prefetch_factor: usize,
    pub cache_latents: bool,
    pub cache_dir: Option<PathBuf>,
}

impl Default for LatentDataLoaderConfig {
    fn default() -> Self {
        Self {
            batch_size: 4,
            shuffle: true,
            drop_last: false,
            num_workers: 4,
            prefetch_factor: 2,
            cache_latents: true,
            cache_dir: None,
        }
    }
}

/// Enhanced DataLoaderBatch with latents
#[derive(Debug, Clone)]
pub struct LatentDataLoaderBatch {
    /// Original images [batch_size, channels, height, width]
    pub images: Tensor,
    
    /// Encoded latents [batch_size, latent_channels, latent_height, latent_width]
    pub latents: Tensor,
    
    /// Text captions
    pub captions: Vec<String>,
    
    /// Optional masks for inpainting
    pub masks: Option<Tensor>,
    
    /// Loss weights per sample
    pub loss_weights: Vec<f32>,
    
    /// Additional metadata
    pub metadata: std::collections::HashMap<String, Vec<serde_json::Value>>,
}

/// DataLoader that produces latents
pub struct LatentDataLoader<D: Dataset> {
    dataset: Arc<D>,
    config: LatentDataLoaderConfig,
    device: Device,
    vae_preprocessor: Arc<VAEPreprocessor>,
    latent_cache: Option<Arc<LatentCache>>,
    indices: Vec<usize>,
    current_position: Arc<Mutex<usize>>,
    epoch: Arc<Mutex<usize>>,
}

impl<D: Dataset + 'static> LatentDataLoader<D> {
    /// Create new latent dataloader
    pub fn new(
        dataset: D,
        config: LatentDataLoaderConfig,
        device: Device,
        vae: Arc<dyn VAE>,
        architecture: ModelArchitecture,
    ) -> Result<Self> {
        let indices: Vec<usize> = (0..dataset.len()).collect();
        
        // Create VAE preprocessor
        let vae_preprocessor = Arc::new(VAEPreprocessor::new(vae.clone(), architecture)?);
        
        // Create latent cache if enabled
        let latent_cache = if config.cache_latents {
            let cache_dir = config.cache_dir.clone()
                .unwrap_or_else(|| PathBuf::from("./latent_cache"));
            Some(Arc::new(LatentCache::new(
                cache_dir,
                architecture,
                device.clone(),
                None, // VAE already provided via preprocessor
            )?))
        } else {
            None
        };
        
        Ok(Self {
            dataset: Arc::new(dataset),
            config,
            device,
            vae_preprocessor,
            latent_cache,
            indices,
            current_position: Arc::new(Mutex::new(0)),
            epoch: Arc::new(Mutex::new(0)),
        })
    }
    
    /// Pre-cache all latents
    pub async fn precache_all(&self) -> Result<()> {
        if let Some(cache) = &self.latent_cache {
            info!("Pre-caching latents for {} images", self.dataset.len());
            
            let mut cached_count = 0;
            let mut computed_count = 0;
            
            for i in 0..self.dataset.len() {
                let item = self.dataset.get_item(i)?;
                
                // Get image path from metadata
                let image_path = item.metadata.get("image_path")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| Error::DataError("Missing image_path in metadata".into()))?;
                let image_path = PathBuf::from(image_path);
                
                // Check if already cached
                let cache_key = cache.get_cache_key(&image_path);
                let cache_path = cache.get_cache_path(&cache_key);
                
                if cache_path.exists() {
                    cached_count += 1;
                    continue;
                }
                
                // Encode to latent
                let latent = self.vae_preprocessor.encode_image(&item.image)?;
                
                // Cache it
                cache.save_to_disk(&cache_path, &latent)?;
                computed_count += 1;
                
                // Progress report
                if (i + 1) % 10 == 0 {
                    info!("Pre-caching progress: {}/{} (cached: {}, computed: {})",
                        i + 1, self.dataset.len(), cached_count, computed_count);
                }
            }
            
            info!("Pre-caching complete: {} cached, {} computed",
                cached_count, computed_count);
        }
        
        Ok(())
    }
    
    /// Get number of batches
    pub fn len(&self) -> usize {
        if self.config.drop_last {
            self.dataset.len() / self.config.batch_size
        } else {
            (self.dataset.len() + self.config.batch_size - 1) / self.config.batch_size
        }
    }
    
    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.dataset.is_empty()
    }
    
    /// Create iterator
    pub async fn iter(&self) -> LatentDataLoaderIterator<D> {
        // Reset position
        *self.current_position.lock().await = 0;
        
        // Shuffle if needed
        let mut indices = self.indices.clone();
        if self.config.shuffle {
            let mut rng = rand::thread_rng();
            indices.shuffle(&mut rng);
        }
        
        LatentDataLoaderIterator {
            loader: self,
            indices,
            current: 0,
        }
    }
    
    /// Reset for new epoch
    pub async fn reset(&self) {
        *self.current_position.lock().await = 0;
        *self.epoch.lock().await += 1;
    }
}

/// Iterator over latent batches
pub struct LatentDataLoaderIterator<'a, D: Dataset> {
    loader: &'a LatentDataLoader<D>,
    indices: Vec<usize>,
    current: usize,
}

impl<'a, D: Dataset> LatentDataLoaderIterator<'a, D> {
    /// Get next batch
    pub async fn next(&mut self) -> Option<Result<LatentDataLoaderBatch>> {
        if self.current >= self.indices.len() {
            return None;
        }
        
        // Get batch indices
        let end = (self.current + self.loader.config.batch_size).min(self.indices.len());
        let batch_indices = &self.indices[self.current..end];
        
        // Check if we should drop last incomplete batch
        if self.loader.config.drop_last && batch_indices.len() < self.loader.config.batch_size {
            return None;
        }
        
        self.current = end;
        
        // Load items and encode to latents
        let mut images = Vec::new();
        let mut latents = Vec::new();
        let mut captions = Vec::new();
        let mut metadata = std::collections::HashMap::new();
        
        for &idx in batch_indices {
            let item = match self.loader.dataset.get_item(idx) {
                Ok(item) => item,
                Err(e) => return Some(Err(e)),
            };
            
            // Get latent
            let latent = if let Some(cache) = &self.loader.latent_cache {
                // Try to get from cache
                let image_path = item.metadata.get("image_path")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown");
                let image_path = PathBuf::from(image_path);
                
                match cache.get_latents(&image_path, &item.image) {
                    Ok(l) => l,
                    Err(e) => {
                        warn!("Failed to get cached latent, computing: {}", e);
                        match self.loader.vae_preprocessor.encode_image(&item.image) {
                            Ok(l) => l,
                            Err(e) => return Some(Err(e)),
                        }
                    }
                }
            } else {
                // Encode directly
                match self.loader.vae_preprocessor.encode_image(&item.image) {
                    Ok(l) => l,
                    Err(e) => return Some(Err(e)),
                }
            };
            
            images.push(item.image);
            latents.push(latent);
            captions.push(item.caption);
            
            // Collect metadata
            for (key, value) in item.metadata {
                metadata.entry(key).or_insert_with(Vec::new).push(value);
            }
        }
        
        // Stack into batches
        let candle_device = match &self.loader.device {
            Device::Cpu => candle_core::Device::Cpu,
            Device::Cuda(id) => match candle_core::Device::new_cuda(*id) {
                Ok(d) => d,
                Err(e) => return Some(Err(Error::from(e))),
            },
        };
        
        let images_batch = match Tensor::stack(&images, 0) {
            Ok(t) => match t.to_device(&candle_device) {
                Ok(t) => t,
                Err(e) => return Some(Err(Error::from(e))),
            },
            Err(e) => return Some(Err(Error::from(e))),
        };
        
        let latents_batch = match Tensor::stack(&latents, 0) {
            Ok(t) => match t.to_device(&candle_device) {
                Ok(t) => t,
                Err(e) => return Some(Err(Error::from(e))),
            },
            Err(e) => return Some(Err(Error::from(e))),
        };
        
        let loss_weights = vec![1.0; batch_indices.len()];
        
        Some(Ok(LatentDataLoaderBatch {
            images: images_batch,
            latents: latents_batch,
            captions,
            masks: None,
            loss_weights,
            metadata,
        }))
    }
}

/// Convert LatentDataLoaderBatch to regular DataLoaderBatch
impl From<LatentDataLoaderBatch> for DataLoaderBatch {
    fn from(batch: LatentDataLoaderBatch) -> Self {
        DataLoaderBatch {
            images: batch.latents, // Use latents as images
            captions: batch.captions,
            masks: batch.masks,
            loss_weights: batch.loss_weights,
            metadata: batch.metadata,
        }
    }
}