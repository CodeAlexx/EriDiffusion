//! Data loader for Flux LoRA training
//! 
//! Handles loading images and captions from a dataset folder with:
//! - Multi-resolution support with bucketing
//! - Caption file loading (.txt files)
//! - Latent caching for efficiency
//! - Proper batching

use anyhow::{Result, Context};
use candle_core::{Device, Tensor};
use image::{DynamicImage, ImageBuffer, Rgb};
use std::path::{Path, PathBuf};
use std::fs;
use std::collections::HashMap;
use rand::seq::SliceRandom;
use rand::thread_rng;

/// Dataset configuration
#[derive(Debug, Clone)]
pub struct DatasetConfig {
    pub folder_path: PathBuf,
    pub caption_ext: String,
    pub caption_dropout_rate: f32,
    pub shuffle_tokens: bool,
    pub cache_latents_to_disk: bool,
    pub resolutions: Vec<(usize, usize)>,
    pub center_crop: bool,
    pub random_flip: bool,
}

/// A single training sample
#[derive(Debug, Clone)]
pub struct TrainingSample {
    pub image_path: PathBuf,
    pub caption_path: PathBuf,
    pub resolution: (usize, usize),
    pub cached_latent_path: Option<PathBuf>,
}

/// Bucket for organizing images by resolution
#[derive(Debug)]
struct ResolutionBucket {
    resolution: (usize, usize),
    samples: Vec<TrainingSample>,
}

/// Main data loader for Flux training
pub struct FluxDataLoader {
    config: DatasetConfig,
    buckets: Vec<ResolutionBucket>,
    current_bucket_idx: usize,
    current_sample_idx: usize,
    device: Device,
    cache_dir: Option<PathBuf>,
}

impl FluxDataLoader {
    /// Create a new data loader
    pub fn new(config: DatasetConfig, device: Device) -> Result<Self> {
        // Create cache directory if latent caching is enabled
        let cache_dir = if config.cache_latents_to_disk {
            let cache_path = config.folder_path.join(".latent_cache");
            fs::create_dir_all(&cache_path)?;
            Some(cache_path)
        } else {
            None
        };
        
        // Scan dataset and organize into buckets
        let buckets = Self::scan_dataset(&config)?;
        
        if buckets.is_empty() {
            return Err(anyhow::anyhow!("No valid samples found in dataset"));
        }
        
        // Log dataset statistics
        let total_samples: usize = buckets.iter().map(|b| b.samples.len()).sum();
        println!("Dataset loaded:");
        println!("  Total samples: {}", total_samples);
        println!("  Resolution buckets:");
        for bucket in &buckets {
            println!("    {}x{}: {} samples", 
                bucket.resolution.0, bucket.resolution.1, bucket.samples.len());
        }
        
        Ok(Self {
            config,
            buckets,
            current_bucket_idx: 0,
            current_sample_idx: 0,
            device,
            cache_dir,
        })
    }
    
    /// Scan dataset folder and organize samples into resolution buckets
    fn scan_dataset(config: &DatasetConfig) -> Result<Vec<ResolutionBucket>> {
        let mut bucket_map: HashMap<(usize, usize), Vec<TrainingSample>> = HashMap::new();
        
        // Initialize buckets for each resolution
        for &resolution in &config.resolutions {
            bucket_map.insert(resolution, Vec::new());
        }
        
        // Scan for image files
        let entries = fs::read_dir(&config.folder_path)
            .with_context(|| format!("Failed to read dataset directory: {}", config.folder_path.display()))?;
        
        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            
            // Check if it's an image file
            if let Some(ext) = path.extension() {
                let ext_str = ext.to_string_lossy().to_lowercase();
                if matches!(ext_str.as_str(), "jpg" | "jpeg" | "png" | "webp") {
                    // Check for corresponding caption file
                    let caption_path = path.with_extension(&config.caption_ext);
                    if !caption_path.exists() {
                        println!("Warning: No caption file for image: {}", path.display());
                        continue;
                    }
                    
                    // Open image to get dimensions
                    let img = image::open(&path)
                        .with_context(|| format!("Failed to open image: {}", path.display()))?;
                    let (width, height) = (img.width() as usize, img.height() as usize);
                    
                    // Find best matching resolution bucket
                    let best_resolution = find_best_resolution(width, height, &config.resolutions);
                    
                    // Check for cached latent
                    let cached_latent_path = if let Some(cache_dir) = &config.folder_path.join(".latent_cache").to_str() {
                        let cache_name = format!("{}_{}x{}.pt", 
                            path.file_stem().unwrap().to_string_lossy(),
                            best_resolution.0, best_resolution.1);
                        let cache_path = PathBuf::from(cache_dir).join(cache_name);
                        if cache_path.exists() {
                            Some(cache_path)
                        } else {
                            None
                        }
                    } else {
                        None
                    };
                    
                    let sample = TrainingSample {
                        image_path: path,
                        caption_path,
                        resolution: best_resolution,
                        cached_latent_path,
                    };
                    
                    bucket_map.get_mut(&best_resolution).unwrap().push(sample);
                }
            }
        }
        
        // Convert to bucket vector and filter out empty buckets
        let buckets: Vec<ResolutionBucket> = bucket_map
            .into_iter()
            .filter(|(_, samples)| !samples.is_empty())
            .map(|(resolution, samples)| ResolutionBucket { resolution, samples })
            .collect();
        
        Ok(buckets)
    }
    
    /// Get next batch of samples
    pub fn get_batch(&mut self, batch_size: usize) -> Result<Vec<(DynamicImage, String)>> {
        let mut batch = Vec::new();
        
        // Shuffle buckets if we're starting a new epoch
        if self.current_bucket_idx == 0 && self.current_sample_idx == 0 {
            self.shuffle_all_buckets();
        }
        
        for _ in 0..batch_size {
            // Get current sample
            let (image_path, caption_path, resolution) = {
                let sample = self.get_next_sample()?;
                (sample.image_path.clone(), sample.caption_path.clone(), sample.resolution)
            };
            
            // Load image
            let img = image::open(&image_path)
                .with_context(|| format!("Failed to load image: {}", image_path.display()))?;
            
            // Resize/crop to target resolution
            let img = self.prepare_image(img, resolution)?;
            
            // Load caption
            let caption = fs::read_to_string(&caption_path)
                .with_context(|| format!("Failed to load caption: {}", caption_path.display()))?
                .trim()
                .to_string();
            
            // Apply caption dropout if configured
            let caption = if self.config.caption_dropout_rate > 0.0 && rand::random::<f32>() < self.config.caption_dropout_rate {
                String::new() // Empty caption for unconditional training
            } else {
                caption
            };
            
            batch.push((img, caption));
        }
        
        Ok(batch)
    }
    
    /// Get next sample, moving to next bucket if needed
    fn get_next_sample(&mut self) -> Result<&TrainingSample> {
        if self.buckets.is_empty() {
            return Err(anyhow::anyhow!("No samples available"));
        }
        
        // Get current bucket
        let bucket = &self.buckets[self.current_bucket_idx];
        
        if self.current_sample_idx >= bucket.samples.len() {
            // Move to next bucket
            self.current_bucket_idx = (self.current_bucket_idx + 1) % self.buckets.len();
            self.current_sample_idx = 0;
            
            // If we wrapped around, we completed an epoch
            if self.current_bucket_idx == 0 {
                self.shuffle_all_buckets();
            }
        }
        
        let sample = &self.buckets[self.current_bucket_idx].samples[self.current_sample_idx];
        self.current_sample_idx += 1;
        
        Ok(sample)
    }
    
    /// Shuffle all buckets
    fn shuffle_all_buckets(&mut self) {
        let mut rng = thread_rng();
        
        // Shuffle bucket order
        self.buckets.shuffle(&mut rng);
        
        // Shuffle samples within each bucket
        for bucket in &mut self.buckets {
            bucket.samples.shuffle(&mut rng);
        }
    }
    
    /// Prepare image for training (resize, crop, augment)
    fn prepare_image(&self, mut img: DynamicImage, target_resolution: (usize, usize)) -> Result<DynamicImage> {
        let (target_width, target_height) = target_resolution;
        let (img_width, img_height) = (img.width() as usize, img.height() as usize);
        
        // Calculate scaling to fit target resolution
        let scale = (target_width as f32 / img_width as f32).max(target_height as f32 / img_height as f32);
        let new_width = (img_width as f32 * scale) as u32;
        let new_height = (img_height as f32 * scale) as u32;
        
        // Resize image
        let mut img = img.resize_exact(new_width, new_height, image::imageops::FilterType::Lanczos3);
        
        // Center crop or random crop
        let crop_x = if self.config.center_crop {
            (new_width - target_width as u32) / 2
        } else {
            rand::random::<u32>() % (new_width - target_width as u32 + 1)
        };
        
        let crop_y = if self.config.center_crop {
            (new_height - target_height as u32) / 2
        } else {
            rand::random::<u32>() % (new_height - target_height as u32 + 1)
        };
        
        img = img.crop(crop_x, crop_y, target_width as u32, target_height as u32);
        
        // Random horizontal flip
        let img = if self.config.random_flip && rand::random::<bool>() {
            img.fliph()
        } else {
            img
        };
        
        Ok(img)
    }
    
    /// Get latent cache path for a sample
    pub fn get_latent_cache_path(&self, sample: &TrainingSample) -> Option<PathBuf> {
        self.cache_dir.as_ref().map(|dir| {
            let cache_name = format!("{}_{}x{}.safetensors", 
                sample.image_path.file_stem().unwrap().to_string_lossy(),
                sample.resolution.0, sample.resolution.1);
            dir.join(cache_name)
        })
    }
    
    /// Get total number of samples across all buckets
    pub fn total_samples(&self) -> usize {
        self.buckets.iter().map(|b| b.samples.len()).sum()
    }
    
    /// Check if we've completed an epoch
    pub fn is_epoch_complete(&self) -> bool {
        self.current_bucket_idx == 0 && self.current_sample_idx == 0
    }
}

/// Find best matching resolution from available buckets
fn find_best_resolution(width: usize, height: usize, resolutions: &[(usize, usize)]) -> (usize, usize) {
    let aspect_ratio = width as f32 / height as f32;
    
    resolutions.iter()
        .min_by_key(|&&(w, h)| {
            let bucket_ratio = w as f32 / h as f32;
            let ratio_diff = (aspect_ratio - bucket_ratio).abs();
            let area_diff = ((width * height) as i32 - (w * h) as i32).abs();
            
            // Prioritize aspect ratio match, then area
            ((ratio_diff * 1000.0) as i32, area_diff)
        })
        .copied()
        .unwrap_or((1024, 1024)) // Default resolution
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_find_best_resolution() {
        let resolutions = vec![
            (512, 512),
            (768, 512),
            (512, 768),
            (1024, 1024),
        ];
        
        // Square image should match square bucket
        assert_eq!(find_best_resolution(800, 800, &resolutions), (1024, 1024));
        
        // Wide image should match wide bucket
        assert_eq!(find_best_resolution(900, 600, &resolutions), (768, 512));
        
        // Tall image should match tall bucket
        assert_eq!(find_best_resolution(600, 900, &resolutions), (512, 768));
    }
}