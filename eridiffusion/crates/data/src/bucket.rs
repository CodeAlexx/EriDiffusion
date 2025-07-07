//! Aspect ratio bucketing for efficient training

use eridiffusion_core::{Result, Error};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Bucket configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BucketConfig {
    pub min_size: usize,
    pub max_size: usize,
    pub divisor: usize,
    pub max_aspect_ratio: f32,
    pub min_aspect_ratio: f32,
    pub square_only: bool,
}

impl Default for BucketConfig {
    fn default() -> Self {
        Self {
            min_size: 256,
            max_size: 2048,
            divisor: 64,
            max_aspect_ratio: 3.0,
            min_aspect_ratio: 0.33,
            square_only: false,
        }
    }
}

/// Image bucket
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Bucket {
    pub width: usize,
    pub height: usize,
}

impl Bucket {
    pub fn new(width: usize, height: usize) -> Self {
        Self { width, height }
    }
    
    pub fn aspect_ratio(&self) -> f32 {
        self.width as f32 / self.height as f32
    }
    
    pub fn pixel_count(&self) -> usize {
        self.width * self.height
    }
}

/// Bucket manager
pub struct BucketManager {
    config: BucketConfig,
    buckets: Vec<Bucket>,
    bucket_indices: HashMap<PathBuf, usize>,
    bucket_contents: HashMap<usize, Vec<PathBuf>>,
}

impl BucketManager {
    /// Create new bucket manager
    pub fn new(config: BucketConfig) -> Self {
        let buckets = if config.square_only {
            Self::generate_square_buckets(&config)
        } else {
            Self::generate_buckets(&config)
        };
        
        let mut bucket_contents = HashMap::new();
        for (i, _) in buckets.iter().enumerate() {
            bucket_contents.insert(i, Vec::new());
        }
        
        Self {
            config,
            buckets,
            bucket_indices: HashMap::new(),
            bucket_contents,
        }
    }
    
    /// Generate all valid buckets
    fn generate_buckets(config: &BucketConfig) -> Vec<Bucket> {
        let mut buckets = Vec::new();
        
        for width in (config.min_size..=config.max_size).step_by(config.divisor) {
            for height in (config.min_size..=config.max_size).step_by(config.divisor) {
                let aspect_ratio = width as f32 / height as f32;
                
                if aspect_ratio >= config.min_aspect_ratio && 
                   aspect_ratio <= config.max_aspect_ratio {
                    buckets.push(Bucket::new(width, height));
                }
            }
        }
        
        // Sort by pixel count for consistent ordering
        buckets.sort_by_key(|b| b.pixel_count());
        buckets
    }
    
    /// Generate square buckets only
    fn generate_square_buckets(config: &BucketConfig) -> Vec<Bucket> {
        let mut buckets = Vec::new();
        
        for size in (config.min_size..=config.max_size).step_by(config.divisor) {
            buckets.push(Bucket::new(size, size));
        }
        
        buckets
    }
    
    /// Find best bucket for image dimensions
    pub fn find_bucket(&self, width: usize, height: usize) -> Result<usize> {
        let aspect_ratio = width as f32 / height as f32;
        
        // Find bucket with closest aspect ratio and sufficient size
        let mut best_idx = None;
        let mut best_score = f32::MAX;
        
        for (idx, bucket) in self.buckets.iter().enumerate() {
            // Skip if bucket is too small
            if bucket.width < width && bucket.height < height {
                continue;
            }
            
            // Calculate resize scale
            let scale_w = bucket.width as f32 / width as f32;
            let scale_h = bucket.height as f32 / height as f32;
            let scale = scale_w.min(scale_h);
            
            // Calculate aspect ratio difference
            let ar_diff = (bucket.aspect_ratio() - aspect_ratio).abs();
            
            // Calculate wasted pixels after resize
            let resized_w = (width as f32 * scale) as usize;
            let resized_h = (height as f32 * scale) as usize;
            let wasted = bucket.pixel_count() - (resized_w * resized_h);
            
            // Combined score (lower is better)
            let score = ar_diff + (wasted as f32 / 1000000.0);
            
            if score < best_score {
                best_score = score;
                best_idx = Some(idx);
            }
        }
        
        best_idx.ok_or_else(|| Error::InvalidInput(
            format!("No suitable bucket found for {}x{}", width, height)
        ))
    }
    
    /// Add image to bucket
    pub fn add_image(
        &mut self,
        path: PathBuf,
        width: usize,
        height: usize,
    ) -> Result<usize> {
        let bucket_idx = self.find_bucket(width, height)?;
        
        self.bucket_indices.insert(path.clone(), bucket_idx);
        self.bucket_contents
            .get_mut(&bucket_idx)
            .unwrap()
            .push(path);
        
        Ok(bucket_idx)
    }
    
    /// Get bucket for image
    pub fn get_bucket_index(&self, path: &PathBuf) -> Option<usize> {
        self.bucket_indices.get(path).copied()
    }
    
    /// Get bucket dimensions
    pub fn get_bucket(&self, index: usize) -> Option<&Bucket> {
        self.buckets.get(index)
    }
    
    /// Get bucket contents
    pub fn get_bucket_contents(&self, index: usize) -> Option<&Vec<PathBuf>> {
        self.bucket_contents.get(&index)
    }
    
    /// Get all non-empty buckets
    pub fn get_active_buckets(&self) -> Vec<(usize, &Bucket, usize)> {
        self.bucket_contents
            .iter()
            .filter(|(_, contents)| !contents.is_empty())
            .map(|(&idx, contents)| (idx, &self.buckets[idx], contents.len()))
            .collect()
    }
    
    /// Get bucket statistics
    pub fn get_statistics(&self) -> BucketStatistics {
        let mut stats = BucketStatistics::default();
        
        for (bucket, contents) in self.buckets.iter().zip(self.bucket_contents.values()) {
            if !contents.is_empty() {
                stats.active_buckets += 1;
                stats.total_images += contents.len();
                
                let ar = bucket.aspect_ratio();
                if (ar - 1.0).abs() < 0.01 {
                    stats.square_images += contents.len();
                } else if ar > 1.0 {
                    stats.landscape_images += contents.len();
                } else {
                    stats.portrait_images += contents.len();
                }
            }
        }
        
        stats.total_buckets = self.buckets.len();
        stats
    }
    
    /// Balance buckets by redistributing images
    pub fn balance_buckets(&mut self, target_per_bucket: usize) -> Result<()> {
        // Find over-populated buckets
        let mut overflow_images = Vec::new();
        
        for (bucket_idx, contents) in self.bucket_contents.iter_mut() {
            if contents.len() > target_per_bucket {
                let overflow = contents.split_off(target_per_bucket);
                for path in overflow {
                    overflow_images.push((path, *bucket_idx));
                }
            }
        }
        
        // Try to redistribute to under-populated buckets
        for (path, original_bucket) in overflow_images {
            let original = &self.buckets[original_bucket];
            
            // Find similar bucket with space
            let mut best_bucket = None;
            let mut best_diff = f32::MAX;
            
            for (idx, bucket) in self.buckets.iter().enumerate() {
                if idx == original_bucket {
                    continue;
                }
                
                let contents = &self.bucket_contents[&idx];
                if contents.len() >= target_per_bucket {
                    continue;
                }
                
                let ar_diff = (bucket.aspect_ratio() - original.aspect_ratio()).abs();
                if ar_diff < best_diff {
                    best_diff = ar_diff;
                    best_bucket = Some(idx);
                }
            }
            
            if let Some(new_bucket) = best_bucket {
                self.bucket_indices.insert(path.clone(), new_bucket);
                if let Some(bucket) = self.bucket_contents.get_mut(&new_bucket) {
                    bucket.push(path);
                }
            } else {
                // Put back in original if no space found
                if let Some(bucket) = self.bucket_contents.get_mut(&original_bucket) {
                    bucket.push(path);
                }
            }
        }
        
        Ok(())
    }
}

/// Bucket statistics
#[derive(Debug, Default)]
pub struct BucketStatistics {
    pub total_buckets: usize,
    pub active_buckets: usize,
    pub total_images: usize,
    pub square_images: usize,
    pub landscape_images: usize,
    pub portrait_images: usize,
}

impl BucketStatistics {
    pub fn print_summary(&self) {
        println!("Bucket Statistics:");
        println!("  Total buckets: {}", self.total_buckets);
        println!("  Active buckets: {}", self.active_buckets);
        println!("  Total images: {}", self.total_images);
        println!("  Square images: {}", self.square_images);
        println!("  Landscape images: {}", self.landscape_images);
        println!("  Portrait images: {}", self.portrait_images);
        
        if self.active_buckets > 0 {
            println!("  Average images per bucket: {:.1}", 
                self.total_images as f32 / self.active_buckets as f32);
        }
    }
}

/// Create resolution buckets for common aspect ratios
pub fn create_resolution_buckets(base_resolution: usize) -> Vec<Bucket> {
    let aspect_ratios = [
        (1, 1),    // Square
        (3, 2),    // 3:2
        (2, 3),    // 2:3  
        (4, 3),    // 4:3
        (3, 4),    // 3:4
        (16, 9),   // 16:9
        (9, 16),   // 9:16
    ];
    
    let mut buckets = Vec::new();
    
    for (w_ratio, h_ratio) in aspect_ratios {
        let base_pixels = base_resolution * base_resolution;
        let ratio = w_ratio as f32 / h_ratio as f32;
        
        // Calculate dimensions maintaining pixel count
        let height = ((base_pixels as f32 / ratio).sqrt()) as usize;
        let width = (height as f32 * ratio) as usize;
        
        // Round to divisor
        let divisor = 64;
        let width = (width / divisor) * divisor;
        let height = (height / divisor) * divisor;
        
        buckets.push(Bucket::new(width, height));
    }
    
    buckets
}