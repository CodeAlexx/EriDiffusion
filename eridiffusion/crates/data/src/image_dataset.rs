// image_dataset.rs - Image folder dataset implementation

use crate::{Dataset, DatasetItem, DatasetMetadata};
use eridiffusion_core::{Result, Error, Device};
use candle_core::{Tensor, DType};
use image::{ImageBuffer, Rgb, DynamicImage};
use serde::{Serialize, Deserialize};
use std::path::{Path, PathBuf};
use std::collections::HashMap;
use std::fs;
use tracing::{info, debug, warn};

/// Configuration for image dataset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetConfig {
    pub root_dir: PathBuf,
    pub caption_ext: String,
    pub resolution: usize,
    pub center_crop: bool,
    pub random_flip: bool,
    pub cache_latents: bool,
    pub cache_dir: Option<PathBuf>,
}

/// Image dataset for training
pub struct ImageDataset {
    config: DatasetConfig,
    image_paths: Vec<PathBuf>,
    captions: HashMap<PathBuf, String>,
    transform: Option<Box<dyn Transform>>,
    metadata: DatasetMetadata,
    device: Device,
}

impl ImageDataset {
    /// Create new image dataset
    pub fn new(config: DatasetConfig) -> Result<Self> {
        // Validate path exists
        if !config.root_dir.exists() {
            return Err(Error::DataError(format!(
                "Dataset directory does not exist: {}",
                config.root_dir.display()
            )));
        }
        
        // Scan for images
        let image_paths = Self::scan_directory(&config.root_dir)?;
        
        if image_paths.is_empty() {
            return Err(Error::DataError(format!(
                "No images found in directory: {}",
                config.root_dir.display()
            )));
        }
        
        // Load captions
        let captions = Self::load_captions(&image_paths, &config.caption_ext)?;
        
        // Create transform pipeline
        let transform = Some(Self::create_transform(&config));
        
        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
        
        let metadata = DatasetMetadata {
            name: format!("ImageDataset: {}", config.root_dir.display()),
            size: image_paths.len(),
            ..Default::default()
        };
        
        Ok(Self {
            config,
            image_paths,
            captions,
            transform,
            metadata,
            device,
        })
    }
    
    /// Scan directory for images recursively
    fn scan_directory(root: &Path) -> Result<Vec<PathBuf>> {
        let mut images = Vec::new();
        let valid_extensions = ["jpg", "jpeg", "png", "webp", "bmp"];
        
        Self::scan_directory_recursive(root, &mut images, &valid_extensions)?;
        
        images.sort();
        info!("Found {} images in {}", images.len(), root.display());
        Ok(images)
    }
    
    /// Recursive directory scanner
    fn scan_directory_recursive(
        dir: &Path, 
        images: &mut Vec<PathBuf>, 
        valid_extensions: &[&str]
    ) -> Result<()> {
        let entries = fs::read_dir(dir)
            .map_err(|e| Error::Io(e))?;
        
        for entry in entries {
            let entry = entry.map_err(|e| Error::Io(e))?;
            let path = entry.path();
            
            if path.is_dir() {
                // Skip hidden directories
                if let Some(name) = path.file_name() {
                    if !name.to_str().unwrap_or("").starts_with('.') {
                        Self::scan_directory_recursive(&path, images, valid_extensions)?;
                    }
                }
            } else if path.is_file() {
                if let Some(ext) = path.extension() {
                    if valid_extensions.contains(&ext.to_str().unwrap_or("").to_lowercase().as_str()) {
                        images.push(path);
                    }
                }
            }
        }
        Ok(())
    }
    
    /// Load caption files
    fn load_captions(image_paths: &[PathBuf], caption_ext: &str) -> Result<HashMap<PathBuf, String>> {
        let mut captions = HashMap::new();
        let mut found_captions = 0;
        
        for image_path in image_paths {
            // Try multiple caption extensions
            let extensions = [caption_ext, "txt", "caption"];
            let mut caption_found = false;
            
            for ext in &extensions {
                let caption_path = image_path.with_extension(ext);
                
                if caption_path.exists() {
                    match fs::read_to_string(&caption_path) {
                        Ok(caption) => {
                            let caption = caption.trim().to_string();
                            
                            // Skip empty captions
                            if !caption.is_empty() {
                                captions.insert(image_path.clone(), caption);
                                caption_found = true;
                                found_captions += 1;
                                break;
                            }
                        }
                        Err(e) => {
                            warn!("Failed to read caption file {}: {}", caption_path.display(), e);
                        }
                    }
                }
            }
            
            if !caption_found {
                // Use filename as caption if no caption file
                let filename = image_path
                    .file_stem()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_default()
                    .replace('_', " ")
                    .replace('-', " ");
                captions.insert(image_path.clone(), filename);
            }
        }
        
        info!("Loaded {} captions for {} images", found_captions, image_paths.len());
        Ok(captions)
    }
    
    /// Create transform pipeline
    fn create_transform(config: &DatasetConfig) -> Box<dyn Transform> {
        Box::new(ImageTransform {
            resolution: config.resolution,
            center_crop: config.center_crop,
            random_flip: config.random_flip,
        })
    }
    
    /// Load and preprocess image
    fn load_image(&self, path: &Path) -> Result<Tensor> {
        // Load image
        let img = image::open(path)
            .map_err(|e| Error::DataError(format!("Failed to open image {}: {}", path.display(), e)))?;
        
        // Convert to RGB
        let img = img.to_rgb8();
        let (width, height) = img.dimensions();
        
        // Convert device
        let candle_device = match &self.device {
            Device::Cpu => candle_core::Device::Cpu,
            Device::Cuda(id) => candle_core::Device::new_cuda(*id)?,
        };
        
        // Convert to tensor [3, H, W] with proper memory layout
        let raw_pixels: Vec<u8> = img.into_raw();
        
        // Reorganize from HWC to CHW
        let mut data = vec![0.0f32; 3 * (width * height) as usize];
        for c in 0..3 {
            for y in 0..height {
                for x in 0..width {
                    let src_idx = ((y * width + x) * 3 + c) as usize;
                    let dst_idx = (c * height * width + y * width + x) as usize;
                    data[dst_idx] = raw_pixels[src_idx] as f32 / 255.0;
                }
            }
        }
        
        let tensor = Tensor::from_vec(
            data,
            &[3, height as usize, width as usize],
            &candle_device,
        )?;
        
        // Apply transforms
        if let Some(ref transform) = self.transform {
            transform.transform(&tensor)
        } else {
            Ok(tensor)
        }
    }
}

impl Dataset for ImageDataset {
    fn len(&self) -> usize {
        self.image_paths.len()
    }
    
    fn get_item(&self, index: usize) -> Result<DatasetItem> {
        if index >= self.len() {
            return Err(Error::InvalidInput(format!(
                "Index {} out of bounds for dataset of size {}",
                index,
                self.len()
            )));
        }
        
        let image_path = &self.image_paths[index];
        let image = self.load_image(image_path)?;
        let caption = self.captions
            .get(image_path)
            .cloned()
            .unwrap_or_default();
        
        let mut metadata = HashMap::new();
        metadata.insert(
            "image_path".to_string(),
            serde_json::Value::String(image_path.to_string_lossy().to_string()),
        );
        
        // Get actual image dimensions after transform
        let (_, h, w) = match image.dims() {
            [c, h, w] => (*c, *h, *w),
            _ => return Err(Error::InvalidShape("Expected CHW format".into())),
        };
        
        metadata.insert(
            "original_size".to_string(),
            serde_json::Value::Array(vec![
                serde_json::Value::Number(w.into()),
                serde_json::Value::Number(h.into()),
            ]),
        );
        
        Ok(DatasetItem {
            image,
            caption,
            metadata,
        })
    }
    
    fn metadata(&self) -> &DatasetMetadata {
        &self.metadata
    }
}

/// Transform trait
pub trait Transform: Send + Sync {
    fn transform(&self, tensor: &Tensor) -> Result<Tensor>;
}

/// Image transform implementation
pub struct ImageTransform {
    resolution: usize,
    center_crop: bool,
    random_flip: bool,
}

impl Transform for ImageTransform {
    fn transform(&self, tensor: &Tensor) -> Result<Tensor> {
        let mut result = tensor.clone();
        
        // Get dimensions
        let (channels, height, width) = match result.dims() {
            [c, h, w] => (*c, *h, *w),
            _ => return Err(Error::InvalidShape("Expected CHW format".into())),
        };
        
        // Resize to resolution
        if height != self.resolution || width != self.resolution {
            result = if self.center_crop {
                // Center crop to square
                let min_dim = height.min(width);
                let crop_h = (height - min_dim) / 2;
                let crop_w = (width - min_dim) / 2;
                
                // Crop
                let cropped = result.narrow(1, crop_h, min_dim)?
                    .narrow(2, crop_w, min_dim)?;
                
                // Resize to target resolution
                resize_tensor(&cropped, self.resolution, self.resolution)?
            } else {
                // Resize directly
                resize_tensor(&result, self.resolution, self.resolution)?
            };
        }
        
        // Random horizontal flip
        if self.random_flip && rand::random::<f32>() > 0.5 {
            result = flip_horizontal(&result)?;
        }
        
        // Normalize to [-1, 1]
        result = result.affine(2.0, -1.0)?;
        
        Ok(result)
    }
}

/// Resize tensor using bilinear interpolation
fn resize_tensor(tensor: &Tensor, target_h: usize, target_w: usize) -> Result<Tensor> {
    let (channels, height, width) = match tensor.dims() {
        [c, h, w] => (*c, *h, *w),
        _ => return Err(Error::InvalidShape("Expected CHW format".into())),
    };
    
    // Bilinear interpolation
    let h_scale = height as f32 / target_h as f32;
    let w_scale = width as f32 / target_w as f32;
    
    let input_data = tensor.flatten_all()?.to_vec1::<f32>()?;
    let mut output_data = vec![0.0f32; channels * target_h * target_w];
    
    for c in 0..channels {
        for y in 0..target_h {
            for x in 0..target_w {
                // Calculate source coordinates
                let src_y = y as f32 * h_scale;
                let src_x = x as f32 * w_scale;
                
                // Get integer and fractional parts
                let y0 = src_y.floor() as usize;
                let x0 = src_x.floor() as usize;
                let y1 = (y0 + 1).min(height - 1);
                let x1 = (x0 + 1).min(width - 1);
                
                let fy = src_y - y0 as f32;
                let fx = src_x - x0 as f32;
                
                // Bilinear interpolation
                let p00 = input_data[c * height * width + y0 * width + x0];
                let p01 = input_data[c * height * width + y0 * width + x1];
                let p10 = input_data[c * height * width + y1 * width + x0];
                let p11 = input_data[c * height * width + y1 * width + x1];
                
                let value = (1.0 - fy) * ((1.0 - fx) * p00 + fx * p01) +
                           fy * ((1.0 - fx) * p10 + fx * p11);
                
                output_data[c * target_h * target_w + y * target_w + x] = value;
            }
        }
    }
    
    Ok(Tensor::from_vec(
        output_data,
        &[channels, target_h, target_w],
        tensor.device(),
    )?)
}

/// Flip tensor horizontally
fn flip_horizontal(tensor: &Tensor) -> Result<Tensor> {
    let (channels, height, width) = match tensor.dims() {
        [c, h, w] => (*c, *h, *w),
        _ => return Err(Error::InvalidShape("Expected CHW format".into())),
    };
    
    let input_data = tensor.flatten_all()?.to_vec1::<f32>()?;
    let mut output_data = vec![0.0f32; input_data.len()];
    
    for c in 0..channels {
        for y in 0..height {
            for x in 0..width {
                let src_idx = c * height * width + y * width + x;
                let dst_idx = c * height * width + y * width + (width - 1 - x);
                output_data[dst_idx] = input_data[src_idx];
            }
        }
    }
    
    Ok(Tensor::from_vec(
        output_data,
        &[channels, height, width],
        tensor.device(),
    )?)
}

/// Bucket sampler for aspect ratio bucketing
pub struct BucketSampler {
    buckets: Vec<(usize, usize)>,
    bucket_indices: HashMap<usize, Vec<usize>>, // bucket index -> dataset indices
    batch_size: usize,
    pub shuffle: bool,
    current_bucket: usize,
    current_indices: Vec<usize>,
}

impl BucketSampler {
    pub fn new(buckets: Vec<(usize, usize)>, batch_size: usize, shuffle: bool) -> Self {
        Self {
            buckets,
            bucket_indices: HashMap::new(),
            batch_size,
            shuffle,
            current_bucket: 0,
            current_indices: Vec::new(),
        }
    }
    
    /// Get bucket for given aspect ratio
    pub fn get_bucket(&self, width: usize, height: usize) -> (usize, usize) {
        let aspect = width as f32 / height as f32;
        
        // Find closest bucket
        let mut best_bucket = self.buckets[0];
        let mut best_diff = f32::MAX;
        
        for &bucket in &self.buckets {
            let bucket_aspect = bucket.0 as f32 / bucket.1 as f32;
            let diff = (aspect - bucket_aspect).abs();
            
            if diff < best_diff {
                best_diff = diff;
                best_bucket = bucket;
            }
        }
        
        best_bucket
    }
    
    /// Add dataset index to appropriate bucket
    pub fn add_index(&mut self, index: usize, width: usize, height: usize) {
        let bucket = self.get_bucket(width, height);
        
        // Find bucket index
        let bucket_idx = self.buckets.iter()
            .position(|&b| b == bucket)
            .unwrap_or(0);
        
        self.bucket_indices
            .entry(bucket_idx)
            .or_insert_with(Vec::new)
            .push(index);
    }
    
    /// Get next batch of indices
    pub fn next_batch(&mut self) -> Option<Vec<usize>> {
        if self.current_indices.is_empty() {
            // Find next non-empty bucket
            let mut found = false;
            let start_bucket = self.current_bucket;
            
            loop {
                if let Some(indices) = self.bucket_indices.get(&self.current_bucket) {
                    if !indices.is_empty() {
                        self.current_indices = indices.clone();
                        if self.shuffle {
                            use rand::seq::SliceRandom;
                            self.current_indices.shuffle(&mut rand::thread_rng());
                        }
                        found = true;
                        break;
                    }
                }
                
                self.current_bucket = (self.current_bucket + 1) % self.buckets.len();
                
                // Wrapped around, no more data
                if self.current_bucket == start_bucket {
                    break;
                }
            }
            
            if !found {
                return None;
            }
        }
        
        // Take batch_size items
        let batch_size = self.batch_size.min(self.current_indices.len());
        let batch: Vec<usize> = self.current_indices.drain(0..batch_size).collect();
        
        // Move to next bucket if current is empty
        if self.current_indices.is_empty() {
            self.current_bucket = (self.current_bucket + 1) % self.buckets.len();
        }
        
        Some(batch)
    }
    
    /// Reset sampler for new epoch
    pub fn reset(&mut self) {
        self.current_bucket = 0;
        self.current_indices.clear();
        
        // Reshuffle if needed
        if self.shuffle {
            for indices in self.bucket_indices.values_mut() {
                use rand::seq::SliceRandom;
                indices.shuffle(&mut rand::thread_rng());
            }
        }
    }
}