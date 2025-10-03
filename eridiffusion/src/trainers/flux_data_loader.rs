use flame_core::device::Device;
use flame_core::optimizers::{Adam, SGD};
use flame_core::{DType, Error, Result, Shape, Tensor};
use image::{DynamicImage, ImageBuffer, Rgb};
use rand::{seq::SliceRandom, thread_rng};
use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
};

// Data loader for Flux LoRA training
//
// Handles loading images and captions from a dataset folder with:
// - Multi-resolution support with bucketing
// - Caption file loading (.txt files)
// - Latent caching for efficiency
// - Proper batching

// FLAME uses flame_core::device::Device instead of Device

/// Dataset configuration
#[derive(Debug, Clone)]
pub struct DatasetConfig {
    pub folder_path: std::path::PathBuf,
    pub caption_ext: String,
    pub caption_dropout_rate: f32,
    pub shuffle_tokens: bool,
    pub cache_latents_to_disk: bool,
    pub resolutions: Vec<(usize, usize)>,
    pub center_crop: bool,
    pub random_flip: bool,
    // Note: force_recache is read from higher-level config; not part of DatasetConfig here
}

/// A single training sample
#[derive(Debug, Clone)]
pub struct TrainingSample {
    pub image_path: std::path::PathBuf,
    pub caption_path: std::path::PathBuf,
    pub resolution: (usize, usize),
    pub cached_latent_path: Option<std::path::PathBuf>,
    pub token_len: usize,
}

/// Bucket for organizing images by resolution
#[derive(Debug)]
pub struct ResolutionBucket {
    pub resolution: (usize, usize),
    pub samples: Vec<TrainingSample>,
}

/// Main data loader for Flux training
pub struct FluxDataLoader {
    pub config: DatasetConfig,
    pub buckets: Vec<ResolutionBucket>,
    pub current_bucket_idx: usize,
    pub current_sample_idx: usize,
    pub device: Device,
    pub cache_dir: Option<std::path::PathBuf>,
    // Instance-scoped batch counter to avoid mutable statics during progress output
    pub batch_count: usize,
}

impl FluxDataLoader {
    /// Create a new data loader
    pub fn new(config: DatasetConfig, device: Device) -> flame_core::Result<Self> {
        // Create cache directory if latent caching is enabled
        let cache_dir = if config.cache_latents_to_disk {
            let cache_path = config.folder_path.join(".latent_cache");
            fs::create_dir_all(&cache_path).map_err(|e| {
                flame_core::Error::Io(format!("Failed to create cache directory: {}", e))
            })?;
            Some(cache_path)
        } else {
            None
        };

        // Scan dataset and organize into buckets
        let buckets = Self::scan_dataset(&config)?;

        if buckets.is_empty() {
            return Err(flame_core::Error::InvalidOperation(
                "No valid samples found in dataset".to_string(),
            ));
        }

        // Log dataset statistics
        let total_samples: usize = buckets.iter().map(|b| b.samples.len()).sum();
        println!("Dataset loaded:");
        println!("  Total samples: {}", total_samples);
        println!("  Resolution buckets:");
        for bucket in &buckets {
            println!(
                "    {}x{}: {} samples",
                bucket.resolution.0,
                bucket.resolution.1,
                bucket.samples.len()
            );
        }

        Ok(Self {
            config,
            buckets,
            current_bucket_idx: 0,
            current_sample_idx: 0,
            device,
            cache_dir,
            batch_count: 0,
        })
    }

    /// Scan dataset folder and organize samples into resolution buckets
    fn scan_dataset(config: &DatasetConfig) -> flame_core::Result<Vec<ResolutionBucket>> {
        let mut bucket_map: std::collections::HashMap<(usize, usize), Vec<TrainingSample>> =
            std::collections::HashMap::new();

        println!("Available resolutions from config: {:?}", config.resolutions);

        // Initialize buckets for each resolution
        for &resolution in &config.resolutions {
            bucket_map.insert(resolution, Vec::new());
        }

        // Scan for image files
        let entries = fs::read_dir(&config.folder_path).map_err(|e| {
            flame_core::Error::InvalidOperation(format!(
                "Failed to read dataset directory: {}",
                config.folder_path.display()
            ))
        })?;

        for entry in entries {
            let entry = entry.map_err(|e| {
                flame_core::Error::Io(format!("Failed to read directory entry: {}", e))
            })?;
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
                    let img = image::open(&path).map_err(|e| {
                        flame_core::Error::InvalidOperation(format!(
                            "Failed to open image: {}",
                            path.display()
                        ))
                    })?;
                    let (width, height) = (img.width() as usize, img.height() as usize);

                    // Find best matching resolution bucket
                    let best_resolution = find_best_resolution(width, height, &config.resolutions);
                    println!(
                        "Image {:?}: {}x{} -> bucket {}x{}",
                        path.file_name().unwrap(),
                        width,
                        height,
                        best_resolution.0,
                        best_resolution.1
                    );

                    // Check for cached latent
                    let cached_latent_path = if let Some(cache_dir) =
                        &config.folder_path.join(".latent_cache").to_str()
                    {
                        let cache_name = format!(
                            "{}_{}x{}.pt",
                            path.file_stem().unwrap().to_string_lossy(),
                            best_resolution.0,
                            best_resolution.1
                        );
                        let cache_path = std::path::PathBuf::from(cache_dir).join(cache_name);
                        if cache_path.exists() {
                            Some(cache_path)
                        } else {
                            None
                        }
                    } else {
                        None
                    };

                    // Read caption once here to compute approximate token length for bucketing
                    let cap_len = std::fs::read_to_string(&caption_path)
                        .unwrap_or_else(|_| String::new())
                        .split_whitespace()
                        .count();
                    let sample = TrainingSample {
                        image_path: path,
                        caption_path,
                        resolution: best_resolution,
                        cached_latent_path,
                        token_len: cap_len,
                    };
                    bucket_map.get_mut(&best_resolution).unwrap().push(sample);
                }
            }
        }

        // Convert to bucket vector and filter out empty buckets
        let mut buckets: Vec<ResolutionBucket> = bucket_map
            .into_iter()
            .filter(|(_, samples)| !samples.is_empty())
            .map(|(resolution, mut samples)| {
                // Dynamic bucketing: within each resolution bucket, sort by token length to form padless batches
                samples.sort_by_key(|s| s.token_len);
                ResolutionBucket { resolution, samples }
            })
            .collect();

        Ok(buckets)
    }

    /// Get next batch of samples
    pub fn get_batch(&mut self, batch_size: usize) -> flame_core::Result<FluxBatch> {
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
            let img = image::open(&image_path).map_err(|e| {
                flame_core::Error::InvalidOperation(format!(
                    "Failed to load image: {}",
                    image_path.display()
                ))
            })?;

            // Resize/crop to target resolution
            let img = self.prepare_image(img, resolution)?;

            // Load caption
            let caption = fs::read_to_string(&caption_path)
                .map_err(|e| {
                    flame_core::Error::InvalidOperation(format!(
                        "Failed to load caption: {}",
                        caption_path.display()
                    ))
                })?
                .trim()
                .to_string();

            // Apply caption dropout if configured
            let caption = if self.config.caption_dropout_rate > 0.0
                && rand::random::<f32>() < self.config.caption_dropout_rate
            {
                String::new() // Empty caption for unconditional training
            } else {
                caption
            };

            batch.push((img, caption, image_path));
        }

        Ok(FluxBatch { batch })
    }

    /// Get next sample, moving to next bucket if needed
    fn get_next_sample(&mut self) -> flame_core::Result<&TrainingSample> {
        if self.buckets.is_empty() {
            return Err(flame_core::Error::InvalidOperation(
                "No samples available".to_string(),
            ));
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
        use rand::seq::SliceRandom;
        use rand::thread_rng;

        let mut rng = thread_rng();

        // Shuffle bucket order
        self.buckets.shuffle(&mut rng);

        // Shuffle samples within each bucket
        for bucket in &mut self.buckets {
            bucket.samples.shuffle(&mut rng);
        }
    }

    /// Prepare image for training (resize, crop, augment)
    pub fn prepare_image(
        &self,
        img: DynamicImage,
        target_resolution: (usize, usize),
    ) -> flame_core::Result<Tensor> {
        let (target_width, target_height) = target_resolution;
        let (img_width, img_height) = (img.width() as usize, img.height() as usize);

        // Calculate scaling to fit target resolution
        let scale =
            (target_width as f32 / img_width as f32).max(target_height as f32 / img_height as f32);
        let new_width = (img_width as f32 * scale) as u32;
        let new_height = (img_height as f32 * scale) as u32;

        // Resize image
        let mut img =
            img.resize_exact(new_width, new_height, image::imageops::FilterType::Lanczos3);

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
        let img = if self.config.random_flip && rand::random::<bool>() { img.fliph() } else { img };

        // Debug: prepare_image (commented out for performance)
        // println!("    prepare_image: target_resolution = {:?}, actual image after crop = {}x{}",
        //          target_resolution, img.width(), img.height());

        // Convert to tensor
        let img_rgb = img.to_rgb8();
        let pixels = img_rgb.as_raw();

        // Convert from HWC to CHW format and normalize to [0, 1] in one pass
        // Note: height comes before width in tensor dimensions
        let mut pixels_chw = vec![0.0f32; 3 * target_height * target_width];
        for c in 0..3 {
            for h in 0..target_height {
                for w in 0..target_width {
                    let src_idx = (h * target_width + w) * 3 + c;
                    let dst_idx = c * target_height * target_width + h * target_width + w;
                    // Normalize to [0, 1] directly
                    pixels_chw[dst_idx] = pixels[src_idx] as f32 / 255.0;
                }
            }
        }

        // Create tensor with correct dimension order: [C, H, W]
        // CRITICAL: Use BF16 for FLUX training!
        let tensor = Tensor::from_vec_dtype(
            pixels_chw,
            Shape::from_dims(&[3, target_height, target_width]),
            self.device.cuda_device_arc(),
            DType::BF16,
        )?;

        Ok(tensor)
    }

    /// Get latent cache path for a sample
    pub fn get_latent_cache_path(&self, sample: &TrainingSample) -> Option<std::path::PathBuf> {
        self.cache_dir.as_ref().map(|dir| {
            let cache_name = format!(
                "{}_{}x{}.safetensors",
                sample.image_path.file_stem().unwrap().to_string_lossy(),
                sample.resolution.0,
                sample.resolution.1
            );
            dir.join(cache_name)
        })
    }

    /// Get total number of samples across all buckets
    pub fn total_samples(&self) -> usize {
        self.buckets.iter().map(|b| b.samples.len()).sum()
    }

    /// Get all samples from all buckets
    pub fn get_all_samples(&self) -> Vec<&TrainingSample> {
        self.buckets.iter().flat_map(|bucket| &bucket.samples).collect()
    }

    /// Get a sample at a specific global index
    pub fn get_sample_at(&self, index: usize) -> flame_core::Result<Option<&TrainingSample>> {
        let mut current_idx = 0;
        for bucket in &self.buckets {
            if index < current_idx + bucket.samples.len() {
                let local_idx = index - current_idx;
                return Ok(bucket.samples.get(local_idx));
            }
            current_idx += bucket.samples.len();
        }
        Ok(None)
    }

    /// Check if we've completed an epoch
    pub fn is_epoch_complete(&self) -> bool {
        self.current_bucket_idx == 0 && self.current_sample_idx == 0
    }

    /// Shuffle dataset samples - calls internal shuffle_all_buckets
    pub fn shuffle_dataset(&mut self) -> flame_core::Result<()> {
        self.shuffle_all_buckets();

        // Reset indices
        self.current_bucket_idx = 0;
        self.current_sample_idx = 0;

        Ok(())
    }

    /// Get the next batch of samples with specified batch size for pipeline_flux_lora_1024
    pub fn next_batch(&mut self, batch_size: usize) -> flame_core::Result<Vec<TrainingSample>> {
        let mut batch = Vec::new();

        // Shuffle buckets if we're starting a new epoch
        if self.current_bucket_idx == 0 && self.current_sample_idx == 0 {
            self.shuffle_all_buckets();
        }

        for _ in 0..batch_size {
            // Check if we have samples
            if self.buckets.is_empty() || self.total_samples() == 0 {
                break;
            }

            // Get next sample
            let sample = self.get_next_sample()?.clone();
            batch.push(sample);
        }

        Ok(batch)
    }

    /// Get the next batch (original method for compatibility)
    pub fn next_batch_old(
        &mut self,
    ) -> flame_core::Result<Option<crate::trainers::pipeline_flux_lora::TrainingBatch>> {
        // Check if we have samples
        if self.buckets.is_empty() || self.total_samples() == 0 {
            return Ok(None);
        }

        // Show progress for batch loading using instance counter
        self.batch_count += 1;
        if self.batch_count == 1 {
            println!("\n    🔄 Loading batch 1 (first batch takes longer)...");
        } else if self.batch_count % 10 == 0 {
            print!("\r    🔄 Loading batch {}... ", self.batch_count);
            std::io::Write::flush(&mut std::io::stdout()).ok();
        }

        // Get next sample directly to have access to its path
        let sample = self.get_next_sample()?.clone();
        let image_path = sample.image_path.clone();
        let caption_path = sample.caption_path.clone();
        let resolution = sample.resolution;

        // Load image
        let img = image::open(&image_path).map_err(|e| {
            flame_core::Error::InvalidOperation(format!("Failed to load image: {}", e))
        })?;

        // Resize/crop to target resolution
        let img_tensor = self.prepare_image(img, resolution)?;

        // Load caption
        let caption = fs::read_to_string(&caption_path)
            .map_err(|e| {
                flame_core::Error::InvalidOperation(format!(
                    "Failed to load caption: {}",
                    caption_path.display()
                ))
            })?
            .trim()
            .to_string();

        // Apply caption dropout if configured
        let caption = if self.config.caption_dropout_rate > 0.0
            && rand::random::<f32>() < self.config.caption_dropout_rate
        {
            String::new() // Empty caption for unconditional training
        } else {
            caption
        };

        // Add batch dimension
        let images = img_tensor.unsqueeze(0)?;

        // Create batch with image path
        let batch = crate::trainers::pipeline_flux_lora::TrainingBatch {
            images,
            prompts: vec![caption],
            timesteps: None,
            image_paths: vec![image_path],
            pixel_values: None,
            encoder_hidden_states: None,
            pooled_encoder_hidden_states: None,
        };

        Ok(Some(batch))
    }

    /// Get number of batches per epoch
    pub fn len(&self) -> usize {
        self.total_samples()
    }

    /// Get all captions from all buckets
    pub fn all_captions(&self) -> Result<Vec<String>> {
        let mut captions = Vec::new();
        for bucket in &self.buckets {
            for sample in &bucket.samples {
                // Load caption from file
                let caption = fs::read_to_string(&sample.caption_path)
                    .map_err(|e| {
                        flame_core::Error::InvalidOperation(format!(
                            "Failed to load caption: {}",
                            sample.caption_path.display()
                        ))
                    })?
                    .trim()
                    .to_string();
                captions.push(caption);
            }
        }
        Ok(captions)
    }
}

/// Find best matching resolution from available buckets
fn find_best_resolution(
    width: usize,
    height: usize,
    resolutions: &[(usize, usize)],
) -> (usize, usize) {
    let aspect_ratio = width as f32 / height as f32;

    let best = resolutions
        .iter()
        .min_by_key(|&(w, h)| {
            let bucket_ratio = *w as f32 / *h as f32;
            let ratio_diff = (aspect_ratio - bucket_ratio).abs();
            let area_diff = ((width * height) as i32 - (w * h) as i32).abs();

            // Prioritize aspect ratio match, then area
            ((ratio_diff * 1000.0) as i32, area_diff)
        })
        .copied()
        .unwrap_or((1024, 1024)); // Default resolution

    // Ensure the resolution is aligned to 64 for CUDA
    let aligned_width = ((best.0 + 63) / 64) * 64;
    let aligned_height = ((best.1 + 63) / 64) * 64;

    if aligned_width != best.0 || aligned_height != best.1 {
        println!(
            "  Aligning resolution from {}x{} to {}x{} for CUDA",
            best.0, best.1, aligned_width, aligned_height
        );
    }

    (aligned_width, aligned_height)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolution_bucketing() {
        // Add tests here
    }
}

/// Batch of data for training
pub struct FluxBatch {
    pub batch: Vec<(Tensor, String, PathBuf)>, // (image_tensor, caption, image_path)
}

// Implement Iterator for FluxDataLoader to satisfy the DataLoader trait
impl Iterator for FluxDataLoader {
    type Item = Result<crate::trainers::pipeline_flux_lora::TrainingBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        // Check if we have samples
        if self.buckets.is_empty() || self.total_samples() == 0 {
            return None;
        }

        // Get a batch of size 1 (can be adjusted later)
        match self.get_batch(1) {
            Ok(flux_batch) => {
                if flux_batch.batch.is_empty() {
                    return None;
                }

                // Convert FluxBatch to TrainingBatch
                let (img, caption, image_path) = &flux_batch.batch[0];

                // img is already a tensor from get_batch, prepared and normalized
                // Just add batch dimension
                match img.unsqueeze(0) {
                    Ok(images) => {
                        let batch = crate::trainers::pipeline_flux_lora::TrainingBatch {
                            images,
                            prompts: vec![caption.clone()],
                            timesteps: None,
                            image_paths: vec![image_path.clone()],
                            pixel_values: None,
                            encoder_hidden_states: None,
                            pooled_encoder_hidden_states: None,
                        };
                        Some(Ok(batch))
                    }
                    Err(e) => Some(Err(e)),
                }
            }
            Err(e) => Some(Err(e)),
        }
    }
}

// Implement required methods for DataLoader trait
impl crate::trainers::pipeline_flux_lora::DataLoader for FluxDataLoader {
    fn len(&self) -> usize {
        self.total_samples()
    }

    fn total_samples(&self) -> usize {
        self.buckets.iter().map(|b| b.samples.len()).sum()
    }
}
