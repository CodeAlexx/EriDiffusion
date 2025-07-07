//! Multi-resolution and aspect ratio handling for different models

use eridiffusion_core::{Result, Error, ModelArchitecture};
use candle_core::{Tensor, Device, DType};
use std::collections::HashMap;
use tracing::{info, debug};

/// Resolution manager for handling multiple resolutions and aspect ratios
pub struct ResolutionManager {
    architecture: ModelArchitecture,
    config: ResolutionConfig,
    buckets: Vec<ResolutionBucket>,
    bucket_indices: HashMap<usize, Vec<usize>>, // bucket_id -> dataset indices
}

impl ResolutionManager {
    /// Create new resolution manager
    pub fn new(architecture: ModelArchitecture) -> Result<Self> {
        let config = ResolutionConfig::for_architecture(&architecture);
        let buckets = Self::generate_buckets(&config)?;
        
        Ok(Self {
            architecture,
            config,
            buckets,
            bucket_indices: HashMap::new(),
        })
    }
    
    /// Analyze dataset and assign images to buckets
    pub fn analyze_dataset(&mut self, dataset: &dyn crate::Dataset) -> Result<()> {
        info!("Analyzing dataset for resolution bucketing");
        
        self.bucket_indices.clear();
        
        for i in 0..dataset.len() {
            let item = dataset.get_item(i)?;
            let (width, height) = self.get_image_dimensions(&item.image)?;
            
            // Find best bucket
            let bucket_id = self.find_best_bucket(width, height)?;
            
            // Add to bucket
            self.bucket_indices
                .entry(bucket_id)
                .or_insert_with(Vec::new)
                .push(i);
        }
        
        // Print statistics
        self.print_bucket_statistics();
        
        Ok(())
    }
    
    /// Get image dimensions
    fn get_image_dimensions(&self, image: &Tensor) -> Result<(usize, usize)> {
        match image.dims() {
            [_, h, w] => Ok((*w, *h)),
            [_, _, h, w] => Ok((*w, *h)),
            _ => Err(Error::InvalidShape("Expected CHW or BCHW format".into())),
        }
    }
    
    /// Find best bucket for given dimensions
    pub fn find_best_bucket(&self, width: usize, height: usize) -> Result<usize> {
        let aspect_ratio = width as f32 / height as f32;
        let pixel_count = width * height;
        
        let mut best_bucket = 0;
        let mut best_score = f32::MAX;
        
        for (i, bucket) in self.buckets.iter().enumerate() {
            // Calculate score based on aspect ratio difference and pixel count difference
            let ar_diff = (bucket.aspect_ratio - aspect_ratio).abs();
            let pixel_diff = (bucket.pixel_count as i32 - pixel_count as i32).abs() as f32;
            
            // Normalize scores
            let ar_score = ar_diff * 100.0;
            let pixel_score = pixel_diff / 1000.0;
            
            let score = ar_score + pixel_score;
            
            if score < best_score {
                best_score = score;
                best_bucket = i;
            }
        }
        
        Ok(best_bucket)
    }
    
    /// Get bucket by ID
    pub fn get_bucket(&self, bucket_id: usize) -> Result<&ResolutionBucket> {
        self.buckets.get(bucket_id)
            .ok_or_else(|| Error::InvalidInput(format!("Invalid bucket ID: {}", bucket_id)))
    }
    
    /// Get indices for bucket
    pub fn get_bucket_indices(&self, bucket_id: usize) -> Vec<usize> {
        self.bucket_indices
            .get(&bucket_id)
            .cloned()
            .unwrap_or_default()
    }
    
    /// Get all non-empty buckets
    pub fn get_active_buckets(&self) -> Vec<usize> {
        let mut active = Vec::new();
        
        for (bucket_id, indices) in &self.bucket_indices {
            if !indices.is_empty() {
                active.push(*bucket_id);
            }
        }
        
        active.sort();
        active
    }
    
    /// Resize image to bucket dimensions
    pub fn resize_to_bucket(
        &self,
        image: &Tensor,
        bucket_id: usize,
    ) -> Result<ResizedImage> {
        let bucket = self.get_bucket(bucket_id)?;
        let (orig_width, orig_height) = self.get_image_dimensions(image)?;
        
        // Calculate resize parameters
        let resize_params = calculate_resize_params(
            orig_width,
            orig_height,
            bucket.width,
            bucket.height,
            &self.config,
        )?;
        
        // Apply resize
        let resized = apply_resize(image, &resize_params)?;
        
        Ok(ResizedImage {
            tensor: resized,
            original_size: (orig_width as u32, orig_height as u32),
            crop_coords: (resize_params.crop_x as u32, resize_params.crop_y as u32),
            bucket_id,
        })
    }
    
    /// Generate buckets based on config
    fn generate_buckets(config: &ResolutionConfig) -> Result<Vec<ResolutionBucket>> {
        let mut buckets = Vec::new();
        let mut id_counter = 0;
        
        // Generate buckets for each base resolution
        for &base_res in &config.base_resolutions {
            // Generate for each aspect ratio
            for &ar in &config.aspect_ratios {
                // Calculate dimensions
                let (width, height) = if ar >= 1.0 {
                    let w = base_res;
                    let h = (base_res as f32 / ar).round() as usize;
                    (w, h)
                } else {
                    let h = base_res;
                    let w = (base_res as f32 * ar).round() as usize;
                    (w, h)
                };
                
                // Ensure divisibility
                let width = (width / config.divisor) * config.divisor;
                let height = (height / config.divisor) * config.divisor;
                
                // Check bounds
                if width >= config.min_resolution && width <= config.max_resolution &&
                   height >= config.min_resolution && height <= config.max_resolution {
                    buckets.push(ResolutionBucket {
                        id: id_counter,
                        width,
                        height,
                        aspect_ratio: width as f32 / height as f32,
                        pixel_count: width * height,
                    });
                    id_counter += 1;
                }
            }
        }
        
        // Sort by pixel count
        buckets.sort_by_key(|b| b.pixel_count);
        
        // Remove duplicates
        buckets.dedup_by_key(|b| (b.width, b.height));
        
        // Reassign IDs
        for (i, bucket) in buckets.iter_mut().enumerate() {
            bucket.id = i;
        }
        
        info!("Generated {} resolution buckets", buckets.len());
        
        Ok(buckets)
    }
    
    /// Print bucket statistics
    fn print_bucket_statistics(&self) {
        info!("Resolution bucket statistics:");
        
        for bucket in &self.buckets {
            if let Some(indices) = self.bucket_indices.get(&bucket.id) {
                if !indices.is_empty() {
                    info!(
                        "  Bucket {}: {}x{} (AR: {:.2}) - {} images",
                        bucket.id,
                        bucket.width,
                        bucket.height,
                        bucket.aspect_ratio,
                        indices.len()
                    );
                }
            }
        }
        
        let total_images: usize = self.bucket_indices.values()
            .map(|v| v.len())
            .sum();
        
        info!("Total images assigned to buckets: {}", total_images);
    }
}

/// Resolution configuration
#[derive(Debug, Clone)]
pub struct ResolutionConfig {
    pub base_resolutions: Vec<usize>,
    pub min_resolution: usize,
    pub max_resolution: usize,
    pub divisor: usize,
    pub aspect_ratios: Vec<f32>,
    pub resize_mode: ResizeMode,
    pub random_crop: bool,
    pub center_crop: bool,
}

impl ResolutionConfig {
    /// Get config for architecture
    pub fn for_architecture(arch: &ModelArchitecture) -> Self {
        match arch {
            ModelArchitecture::SD15 | ModelArchitecture::SD15 => Self {
                base_resolutions: vec![512, 768],
                min_resolution: 512,
                max_resolution: 768,
                divisor: 64,
                aspect_ratios: vec![1.0, 4.0/3.0, 3.0/4.0, 16.0/9.0, 9.0/16.0],
                resize_mode: ResizeMode::Stretch,
                random_crop: true,
                center_crop: false,
            },
            ModelArchitecture::SDXL => Self {
                base_resolutions: vec![1024],
                min_resolution: 768,
                max_resolution: 1536,
                divisor: 64,
                aspect_ratios: vec![
                    1.0, 4.0/3.0, 3.0/4.0, 16.0/9.0, 9.0/16.0,
                    3.0/2.0, 2.0/3.0, 5.0/4.0, 4.0/5.0
                ],
                resize_mode: ResizeMode::AspectFit,
                random_crop: true,
                center_crop: false,
            },
            ModelArchitecture::SD3 | ModelArchitecture::SD35 => Self {
                base_resolutions: vec![512, 768, 1024, 1536],
                min_resolution: 512,
                max_resolution: 2048,
                divisor: 64,
                aspect_ratios: vec![
                    1.0, 4.0/3.0, 3.0/4.0, 16.0/9.0, 9.0/16.0,
                    3.0/2.0, 2.0/3.0, 5.0/4.0, 4.0/5.0,
                    2.0/1.0, 1.0/2.0
                ],
                resize_mode: ResizeMode::AspectFit,
                random_crop: true,
                center_crop: false,
            },
            ModelArchitecture::Flux => Self {
                base_resolutions: vec![256, 512, 768, 1024, 1536, 2048],
                min_resolution: 256,
                max_resolution: 2048,
                divisor: 16,
                aspect_ratios: vec![
                    1.0, 4.0/3.0, 3.0/4.0, 16.0/9.0, 9.0/16.0,
                    3.0/2.0, 2.0/3.0, 5.0/4.0, 4.0/5.0,
                    2.0/1.0, 1.0/2.0, 3.0/1.0, 1.0/3.0,
                    4.0/1.0, 1.0/4.0
                ],
                resize_mode: ResizeMode::AspectFit,
                random_crop: false,
                center_crop: true,
            },
            _ => Self {
                base_resolutions: vec![512],
                min_resolution: 512,
                max_resolution: 512,
                divisor: 64,
                aspect_ratios: vec![1.0],
                resize_mode: ResizeMode::Stretch,
                random_crop: false,
                center_crop: true,
            },
        }
    }
}

/// Resize mode
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ResizeMode {
    /// Stretch to fit (may distort)
    Stretch,
    /// Maintain aspect ratio, crop if needed
    AspectFit,
    /// Maintain aspect ratio, pad if needed
    AspectFill,
}

/// Resolution bucket
#[derive(Debug, Clone)]
pub struct ResolutionBucket {
    pub id: usize,
    pub width: usize,
    pub height: usize,
    pub aspect_ratio: f32,
    pub pixel_count: usize,
}

/// Resized image information
#[derive(Debug, Clone)]
pub struct ResizedImage {
    pub tensor: Tensor,
    pub original_size: (u32, u32),
    pub crop_coords: (u32, u32),
    pub bucket_id: usize,
}

/// Resize parameters
#[derive(Debug, Clone)]
struct ResizeParams {
    target_width: usize,
    target_height: usize,
    resize_width: usize,
    resize_height: usize,
    crop_x: usize,
    crop_y: usize,
    pad_x: usize,
    pad_y: usize,
}

/// Calculate resize parameters
fn calculate_resize_params(
    orig_width: usize,
    orig_height: usize,
    target_width: usize,
    target_height: usize,
    config: &ResolutionConfig,
) -> Result<ResizeParams> {
    match config.resize_mode {
        ResizeMode::Stretch => {
            // Simple stretch to target size
            Ok(ResizeParams {
                target_width,
                target_height,
                resize_width: target_width,
                resize_height: target_height,
                crop_x: 0,
                crop_y: 0,
                pad_x: 0,
                pad_y: 0,
            })
        }
        ResizeMode::AspectFit => {
            // Maintain aspect ratio, crop if needed
            let orig_aspect = orig_width as f32 / orig_height as f32;
            let target_aspect = target_width as f32 / target_height as f32;
            
            let (resize_width, resize_height) = if orig_aspect > target_aspect {
                // Original is wider, fit height
                let scale = target_height as f32 / orig_height as f32;
                let w = (orig_width as f32 * scale).round() as usize;
                (w, target_height)
            } else {
                // Original is taller, fit width
                let scale = target_width as f32 / orig_width as f32;
                let h = (orig_height as f32 * scale).round() as usize;
                (target_width, h)
            };
            
            // Calculate crop
            let crop_x = if resize_width > target_width {
                if config.random_crop && rand::random::<bool>() {
                    rand::random::<usize>() % (resize_width - target_width)
                } else if config.center_crop {
                    (resize_width - target_width) / 2
                } else {
                    0
                }
            } else {
                0
            };
            
            let crop_y = if resize_height > target_height {
                if config.random_crop && rand::random::<bool>() {
                    rand::random::<usize>() % (resize_height - target_height)
                } else if config.center_crop {
                    (resize_height - target_height) / 2
                } else {
                    0
                }
            } else {
                0
            };
            
            Ok(ResizeParams {
                target_width,
                target_height,
                resize_width,
                resize_height,
                crop_x,
                crop_y,
                pad_x: 0,
                pad_y: 0,
            })
        }
        ResizeMode::AspectFill => {
            // Maintain aspect ratio, pad if needed
            let orig_aspect = orig_width as f32 / orig_height as f32;
            let target_aspect = target_width as f32 / target_height as f32;
            
            let (resize_width, resize_height) = if orig_aspect > target_aspect {
                // Original is wider, fit width
                let scale = target_width as f32 / orig_width as f32;
                let h = (orig_height as f32 * scale).round() as usize;
                (target_width, h)
            } else {
                // Original is taller, fit height
                let scale = target_height as f32 / orig_height as f32;
                let w = (orig_width as f32 * scale).round() as usize;
                (w, target_height)
            };
            
            // Calculate padding
            let pad_x = if resize_width < target_width {
                (target_width - resize_width) / 2
            } else {
                0
            };
            
            let pad_y = if resize_height < target_height {
                (target_height - resize_height) / 2
            } else {
                0
            };
            
            Ok(ResizeParams {
                target_width,
                target_height,
                resize_width,
                resize_height,
                crop_x: 0,
                crop_y: 0,
                pad_x,
                pad_y,
            })
        }
    }
}

/// Apply resize to image
fn apply_resize(image: &Tensor, params: &ResizeParams) -> Result<Tensor> {
    let device = image.device();
    
    // First, resize to intermediate size
    let resized = if params.resize_width != image.dims()[2] || params.resize_height != image.dims()[1] {
        resize_bilinear(image, params.resize_width, params.resize_height)?
    } else {
        image.clone()
    };
    
    // Then crop if needed
    let cropped = if params.crop_x > 0 || params.crop_y > 0 {
        let end_x = params.crop_x + params.target_width;
        let end_y = params.crop_y + params.target_height;
        
        resized.narrow(2, params.crop_y, params.target_height)?
            .narrow(3, params.crop_x, params.target_width)?
    } else {
        resized
    };
    
    // Finally pad if needed
    let padded = if params.pad_x > 0 || params.pad_y > 0 {
        pad_image(&cropped, params.pad_x, params.pad_y, params.target_width, params.target_height)?
    } else {
        cropped
    };
    
    Ok(padded)
}

/// Bilinear resize
fn resize_bilinear(tensor: &Tensor, target_w: usize, target_h: usize) -> Result<Tensor> {
    // Simplified bilinear interpolation
    // In practice, you'd use a proper image processing library
    
    let (channels, height, width) = match tensor.dims() {
        [c, h, w] => (*c, *h, *w),
        [_, c, h, w] => (*c, *h, *w),
        _ => return Err(Error::InvalidShape("Expected CHW or BCHW format".into())),
    };
    
    let h_scale = height as f32 / target_h as f32;
    let w_scale = width as f32 / target_w as f32;
    
    let mut output_data = vec![0.0f32; channels * target_h * target_w];
    let input_data = tensor.flatten_all()?.to_vec1::<f32>()?;
    
    for c in 0..channels {
        for y in 0..target_h {
            for x in 0..target_w {
                // Calculate source coordinates
                let src_y = (y as f32 * h_scale).min((height - 1) as f32);
                let src_x = (x as f32 * w_scale).min((width - 1) as f32);
                
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
                
                let p0 = p00 * (1.0 - fx) + p01 * fx;
                let p1 = p10 * (1.0 - fx) + p11 * fx;
                let p = p0 * (1.0 - fy) + p1 * fy;
                
                output_data[c * target_h * target_w + y * target_w + x] = p;
            }
        }
    }
    
    let output_shape = if tensor.dims().len() == 4 {
        vec![tensor.dims()[0], channels, target_h, target_w]
    } else {
        vec![channels, target_h, target_w]
    };
    
    Ok(Tensor::from_vec(output_data, output_shape.as_slice(), tensor.device())?)
}

/// Pad image
fn pad_image(
    tensor: &Tensor,
    pad_x: usize,
    pad_y: usize,
    target_w: usize,
    target_h: usize,
) -> Result<Tensor> {
    // Create padded tensor filled with zeros (or reflection padding)
    let shape = if tensor.dims().len() == 4 {
        vec![tensor.dims()[0], tensor.dims()[1], target_h, target_w]
    } else {
        vec![tensor.dims()[0], target_h, target_w]
    };
    
    let mut padded = Tensor::zeros(shape.as_slice(), tensor.dtype(), tensor.device())?;
    
    // Copy original image to center
    // This would need proper tensor slicing operations
    
    Ok(padded)
}