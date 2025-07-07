// dataset_manager.rs - Comprehensive dataset management for all model architectures

use eridiffusion_core::{Result, Error, Device, ModelArchitecture};
use eridiffusion_models::VAE;
use candle_core::{Tensor, DType};
use std::path::{Path, PathBuf};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, debug, warn};

use crate::{
    Dataset, ImageDataset, DatasetConfig,
    LatentCache, BucketSampler,
};

/// Dataset manager that handles all aspects of data preparation
pub struct DatasetManager {
    /// Model architecture
    architecture: ModelArchitecture,
    
    /// Base dataset
    dataset: Box<dyn Dataset>,
    
    /// VAE for encoding
    vae: Option<Arc<dyn VAE>>,
    
    /// Latent cache
    latent_cache: Option<Arc<LatentCache>>,
    
    /// Resolution configurations per architecture
    resolution_config: ResolutionConfig,
    
    /// Preprocessing pipeline
    preprocessor: Box<dyn DataPreprocessor>,
    
    /// Device
    device: Device,
    
    /// Statistics
    stats: Arc<RwLock<DatasetStats>>,
}

impl DatasetManager {
    /// Create new dataset manager
    pub fn new(
        architecture: ModelArchitecture,
        dataset_path: PathBuf,
        vae: Option<Arc<dyn VAE>>,
    ) -> Result<Self> {
        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
        
        // Get resolution config for architecture
        let resolution_config = ResolutionConfig::for_architecture(&architecture);
        
        // Create base dataset
        let dataset_config = DatasetConfig {
            root_dir: dataset_path.clone(),
            caption_ext: "txt".to_string(),
            resolution: resolution_config.base_resolution,
            center_crop: false,
            random_flip: true,
            cache_latents: vae.is_some(),
            cache_dir: Some(PathBuf::from(".cache/latents")),
        };
        
        let dataset = Box::new(ImageDataset::new(dataset_config)?);
        
        // Create latent cache if VAE provided
        let latent_cache = if vae.is_some() {
            let cache_dir = PathBuf::from(".cache/latents").join(architecture.to_string());
            Some(Arc::new(
                LatentCache::new(cache_dir, architecture.clone(), device.clone(), None)?
            ))
        } else {
            None
        };
        
        // Create preprocessor for architecture
        let preprocessor = create_preprocessor(&architecture)?;
        
        Ok(Self {
            architecture,
            dataset,
            vae,
            latent_cache,
            resolution_config,
            preprocessor,
            device,
            stats: Arc::new(RwLock::new(DatasetStats::default())),
        })
    }
    
    /// Prepare dataset for training
    pub async fn prepare(&mut self) -> Result<()> {
        info!("Preparing dataset for {} training", self.architecture);
        
        let dataset_len = self.dataset.len();
        info!("Found {} images in dataset", dataset_len);
        
        // Analyze dataset
        self.analyze_dataset().await?;
        
        // Precompute latents if enabled
        if self.latent_cache.is_some() {
            self.precompute_all_latents().await?;
        }
        
        // Print statistics
        let stats = self.stats.read().await;
        info!("Dataset statistics:");
        info!("  Total images: {}", stats.total_images);
        info!("  Unique resolutions: {}", stats.resolutions.len());
        info!("  Cached latents: {}", stats.cached_latents);
        info!("  Average caption length: {:.1} tokens", stats.avg_caption_length);
        
        Ok(())
    }
    
    /// Analyze dataset statistics
    async fn analyze_dataset(&mut self) -> Result<()> {
        let mut stats = self.stats.write().await;
        stats.total_images = self.dataset.len();
        
        let mut total_caption_length = 0;
        
        for i in 0..self.dataset.len() {
            let item = self.dataset.get_item(i)?;
            
            // Track resolutions
            let (_, h, w) = match item.image.dims() {
                [c, h, w] => (*c, *h, *w),
                _ => continue,
            };
            
            let resolution = (w, h);
            *stats.resolutions.entry(resolution).or_insert(0) += 1;
            
            // Track caption length
            total_caption_length += item.caption.split_whitespace().count();
        }
        
        stats.avg_caption_length = total_caption_length as f32 / stats.total_images as f32;
        
        Ok(())
    }
    
    /// Precompute all latents
    async fn precompute_all_latents(&mut self) -> Result<()> {
        let cache = self.latent_cache.as_ref()
            .ok_or_else(|| Error::Config("No latent cache configured".into()))?;
        
        let vae = self.vae.as_ref()
            .ok_or_else(|| Error::Config("No VAE configured".into()))?;
        
        info!("Precomputing latents for {} images", self.dataset.len());
        
        let mut cached_count = 0;
        let mut computed_count = 0;
        
        // Process in batches
        let batch_size = 4;
        let mut batch_items = Vec::new();
        
        for i in 0..self.dataset.len() {
            let item = self.dataset.get_item(i)?;
            batch_items.push((i, item));
            
            if batch_items.len() >= batch_size || i == self.dataset.len() - 1 {
                // Process batch
                for (idx, item) in &batch_items {
                    let image_path = item.metadata.get("image_path")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| Error::DataError("Missing image_path".into()))?;
                    
                    let image_path = PathBuf::from(image_path);
                    
                    // Check if already cached
                    let cache_stats = cache.get_stats();
                    if cache_stats.memory_items + cache_stats.disk_items > cached_count {
                        cached_count += 1;
                        continue;
                    }
                    
                    // Preprocess image
                    let processed = self.preprocessor.preprocess_image(&item.image)?;
                    
                    // Encode to latent
                    let base_res = self.resolution_config.base_resolution;
                    let latent = cache.get_or_compute_latent(
                        &processed,
                        &image_path,
                        (base_res, base_res),
                    ).await?;
                    
                    computed_count += 1;
                    
                    // Progress report
                    if (*idx + 1) % 10 == 0 {
                        info!("Progress: {}/{} (cached: {}, computed: {})",
                            idx + 1, self.dataset.len(), cached_count, computed_count);
                    }
                }
                
                batch_items.clear();
            }
        }
        
        // Update stats
        let mut stats = self.stats.write().await;
        stats.cached_latents = cached_count + computed_count;
        
        info!("Latent precomputation complete: {} cached, {} computed",
            cached_count, computed_count);
        
        Ok(())
    }
    
    /// Get preprocessed item
    pub async fn get_preprocessed_item(&self, index: usize) -> Result<PreprocessedItem> {
        let item = self.dataset.get_item(index)?;
        
        // Preprocess image
        let processed_image = self.preprocessor.preprocess_image(&item.image)?;
        
        // Get or compute latent if cache available
        let latent = if let Some(cache) = &self.latent_cache {
            let image_path = item.metadata.get("image_path")
                .and_then(|v| v.as_str())
                .ok_or_else(|| Error::DataError("Missing image_path".into()))?;
            
            let base_res = self.resolution_config.base_resolution;
            Some(cache.get_or_compute_latent(
                &processed_image,
                &PathBuf::from(image_path),
                (base_res, base_res),
            ).await?)
        } else {
            None
        };
        
        // Preprocess caption
        let processed_caption = self.preprocessor.preprocess_caption(&item.caption)?;
        
        Ok(PreprocessedItem {
            image: processed_image,
            latent,
            caption: processed_caption,
            original_size: extract_original_size(&item.metadata),
            crop_coords: extract_crop_coords(&item.metadata),
            metadata: item.metadata,
        })
    }
    
    /// Create bucket sampler for this dataset
    pub fn create_bucket_sampler(&self, batch_size: usize, shuffle: bool) -> Result<BucketSampler> {
        let buckets = self.resolution_config.get_buckets();
        let mut sampler = BucketSampler::new(buckets, batch_size, shuffle);
        
        // Add all indices to buckets
        for i in 0..self.dataset.len() {
            let item = self.dataset.get_item(i)?;
            let (_, h, w) = match item.image.dims() {
                [c, h, w] => (*c, *h, *w),
                _ => continue,
            };
            
            sampler.add_index(i, w, h);
        }
        
        sampler.reset();
        Ok(sampler)
    }
}

/// Resolution configuration for different architectures
#[derive(Debug, Clone)]
pub struct ResolutionConfig {
    pub base_resolution: usize,
    pub base_resolutions: Vec<usize>,
    pub min_resolution: usize,
    pub max_resolution: usize,
    pub divisor: usize,
    pub aspect_ratios: Vec<f32>,
}

impl ResolutionConfig {
    /// Get config for architecture
    pub fn for_architecture(arch: &ModelArchitecture) -> Self {
        match arch {
            ModelArchitecture::SD15 => Self {
                base_resolution: 512,
                base_resolutions: vec![512],
                min_resolution: 512,
                max_resolution: 768,
                divisor: 8,
                aspect_ratios: vec![1.0, 0.75, 1.33],
            },
            ModelArchitecture::SDXL => Self {
                base_resolution: 1024,
                base_resolutions: vec![1024],
                min_resolution: 768,
                max_resolution: 1536,
                divisor: 8,
                aspect_ratios: vec![1.0, 0.75, 1.33, 0.67, 1.5],
            },
            ModelArchitecture::SD3 | ModelArchitecture::SD35 => Self {
                base_resolution: 1024,
                base_resolutions: vec![512, 768, 1024, 1536, 2048],
                min_resolution: 512,
                max_resolution: 2048,
                divisor: 16, // SD3 uses 16x downsampling
                aspect_ratios: vec![1.0, 0.75, 1.33, 0.67, 1.5, 0.5, 2.0],
            },
            ModelArchitecture::Flux => Self {
                base_resolution: 1024,
                base_resolutions: vec![512, 768, 1024, 1536, 2048],
                min_resolution: 256,
                max_resolution: 2048,
                divisor: 16,
                aspect_ratios: vec![1.0, 0.75, 1.33, 0.67, 1.5, 0.5, 2.0, 0.25, 4.0],
            },
            _ => Self {
                base_resolution: 512,
                base_resolutions: vec![512],
                min_resolution: 512,
                max_resolution: 512,
                divisor: 8,
                aspect_ratios: vec![1.0],
            },
        }
    }
    
    /// Get bucket resolutions
    pub fn get_buckets(&self) -> Vec<(usize, usize)> {
        let mut buckets = Vec::new();
        
        for &ratio in &self.aspect_ratios {
            // Calculate dimensions maintaining aspect ratio
            let width = self.base_resolution;
            let height = (width as f32 / ratio).round() as usize;
            
            // Ensure divisibility
            let width = (width / self.divisor) * self.divisor;
            let height = (height / self.divisor) * self.divisor;
            
            // Check bounds
            if width >= self.min_resolution && width <= self.max_resolution &&
               height >= self.min_resolution && height <= self.max_resolution {
                buckets.push((width, height));
                
                // Also add rotated version if different
                if width != height {
                    buckets.push((height, width));
                }
            }
        }
        
        // Remove duplicates
        buckets.sort();
        buckets.dedup();
        
        buckets
    }
}

/// Data preprocessor trait
pub trait DataPreprocessor: Send + Sync {
    /// Preprocess image
    fn preprocess_image(&self, image: &Tensor) -> Result<Tensor>;
    
    /// Preprocess caption
    fn preprocess_caption(&self, caption: &str) -> Result<String>;
}

/// Create preprocessor for architecture
fn create_preprocessor(arch: &ModelArchitecture) -> Result<Box<dyn DataPreprocessor>> {
    match arch {
        ModelArchitecture::SD15 => {
            Ok(Box::new(SDPreprocessor::new()))
        }
        ModelArchitecture::SDXL => {
            Ok(Box::new(SDXLPreprocessor::new()))
        }
        ModelArchitecture::SD3 | ModelArchitecture::SD35 => {
            Ok(Box::new(SD3Preprocessor::new()))
        }
        ModelArchitecture::Flux => {
            Ok(Box::new(FluxPreprocessor::new()))
        }
        _ => {
            Ok(Box::new(DefaultPreprocessor::new()))
        }
    }
}

/// Default preprocessor implementation
struct DefaultPreprocessor;

impl DefaultPreprocessor {
    fn new() -> Self {
        Self
    }
}

impl DataPreprocessor for DefaultPreprocessor {
    fn preprocess_image(&self, image: &Tensor) -> Result<Tensor> {
        // Default [-1, 1] normalization
        Ok(if image.dtype() != DType::F32 {
            image.to_dtype(DType::F32)?.affine(2.0 / 255.0, -1.0)?
        } else {
            image.affine(2.0, -1.0)?
        })
    }
    
    fn preprocess_caption(&self, caption: &str) -> Result<String> {
        Ok(caption.trim().to_string())
    }
}

// Architecture-specific preprocessors
struct SDPreprocessor;
impl SDPreprocessor {
    fn new() -> Self { Self }
}
impl DataPreprocessor for SDPreprocessor {
    fn preprocess_image(&self, image: &Tensor) -> Result<Tensor> {
        DefaultPreprocessor.preprocess_image(image)
    }
    fn preprocess_caption(&self, caption: &str) -> Result<String> {
        Ok(caption.trim().to_string())
    }
}

struct SDXLPreprocessor;
impl SDXLPreprocessor {
    fn new() -> Self { Self }
}
impl DataPreprocessor for SDXLPreprocessor {
    fn preprocess_image(&self, image: &Tensor) -> Result<Tensor> {
        DefaultPreprocessor.preprocess_image(image)
    }
    fn preprocess_caption(&self, caption: &str) -> Result<String> {
        let caption = caption.trim();
        let words: Vec<&str> = caption.split_whitespace().collect();
        let truncated = words[..words.len().min(75)].join(" ");
        Ok(truncated)
    }
}

struct SD3Preprocessor;
impl SD3Preprocessor {
    fn new() -> Self { Self }
}
impl DataPreprocessor for SD3Preprocessor {
    fn preprocess_image(&self, image: &Tensor) -> Result<Tensor> {
        DefaultPreprocessor.preprocess_image(image)
    }
    fn preprocess_caption(&self, caption: &str) -> Result<String> {
        let caption = caption.trim();
        let caption = caption.split_whitespace().collect::<Vec<_>>().join(" ");
        let words: Vec<&str> = caption.split_whitespace().collect();
        let truncated = words[..words.len().min(256)].join(" ");
        Ok(truncated)
    }
}

struct FluxPreprocessor;
impl FluxPreprocessor {
    fn new() -> Self { Self }
}
impl DataPreprocessor for FluxPreprocessor {
    fn preprocess_image(&self, image: &Tensor) -> Result<Tensor> {
        Ok(if image.dtype() != DType::F32 {
            image.to_dtype(DType::F32)?.affine(1.0 / 127.5, -1.0)?
        } else {
            image.affine(2.0, -1.0)?
        })
    }
    fn preprocess_caption(&self, caption: &str) -> Result<String> {
        let caption = caption.trim();
        let caption = caption.split_whitespace().collect::<Vec<_>>().join(" ");
        let words: Vec<&str> = caption.split_whitespace().collect();
        let truncated = words[..words.len().min(512)].join(" ");
        Ok(truncated)
    }
}

/// Preprocessed dataset item
#[derive(Debug, Clone)]
pub struct PreprocessedItem {
    pub image: Tensor,
    pub latent: Option<Tensor>,
    pub caption: String,
    pub original_size: (u32, u32),
    pub crop_coords: (u32, u32),
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Dataset statistics
#[derive(Debug, Default)]
pub struct DatasetStats {
    pub total_images: usize,
    pub resolutions: HashMap<(usize, usize), usize>,
    pub cached_latents: usize,
    pub avg_caption_length: f32,
}

/// Extract original size from metadata
fn extract_original_size(metadata: &HashMap<String, serde_json::Value>) -> (u32, u32) {
    metadata.get("original_size")
        .and_then(|v| v.as_array())
        .and_then(|arr| {
            let w = arr.get(0)?.as_u64()? as u32;
            let h = arr.get(1)?.as_u64()? as u32;
            Some((w, h))
        })
        .unwrap_or((1024, 1024))
}

/// Extract crop coordinates from metadata
fn extract_crop_coords(metadata: &HashMap<String, serde_json::Value>) -> (u32, u32) {
    metadata.get("crop_coords")
        .and_then(|v| v.as_array())
        .and_then(|arr| {
            let x = arr.get(0)?.as_u64()? as u32;
            let y = arr.get(1)?.as_u64()? as u32;
            Some((x, y))
        })
        .unwrap_or((0, 0))
}