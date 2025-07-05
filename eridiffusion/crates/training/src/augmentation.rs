//! Data augmentation techniques

use eridiffusion_core::{Result, Error};
use candle_core::{Tensor, DType, Device};
use rand::Rng;

/// Augmentation configuration
#[derive(Debug, Clone)]
pub struct AugmentationConfig {
    pub random_flip: bool,
    pub random_rotation: Option<f32>, // max rotation in degrees
    pub color_jitter: Option<ColorJitterConfig>,
    pub random_crop: Option<(usize, usize)>,
    pub random_resize: Option<(f32, f32)>, // min, max scale
    pub cutout: Option<CutoutConfig>,
    pub mixup: Option<f32>, // alpha parameter
}

impl Default for AugmentationConfig {
    fn default() -> Self {
        Self {
            random_flip: true,
            random_rotation: Some(10.0),
            color_jitter: Some(ColorJitterConfig::default()),
            random_crop: None,
            random_resize: Some((0.8, 1.2)),
            cutout: None,
            mixup: None,
        }
    }
}

/// Color jitter configuration
#[derive(Debug, Clone)]
pub struct ColorJitterConfig {
    pub brightness: f32,
    pub contrast: f32,
    pub saturation: f32,
    pub hue: f32,
}

impl Default for ColorJitterConfig {
    fn default() -> Self {
        Self {
            brightness: 0.2,
            contrast: 0.2,
            saturation: 0.2,
            hue: 0.1,
        }
    }
}

/// Cutout configuration
#[derive(Debug, Clone)]
pub struct CutoutConfig {
    pub num_patches: usize,
    pub patch_size: usize,
}

/// Augmentation pipeline
pub struct AugmentationPipeline {
    config: AugmentationConfig,
    rng: rand::rngs::ThreadRng,
}

impl AugmentationPipeline {
    /// Create new augmentation pipeline
    pub fn new(config: AugmentationConfig) -> Self {
        Self {
            config,
            rng: rand::thread_rng(),
        }
    }
    
    /// Apply augmentations to image
    pub fn apply(&mut self, image: &Tensor) -> Result<Tensor> {
        let mut result = image.clone();
        
        // Random horizontal flip
        if self.config.random_flip && self.rng.gen_bool(0.5) {
            result = self.random_flip(&result)?;
        }
        
        // Random rotation
        if let Some(max_rotation) = self.config.random_rotation {
            let angle = self.rng.gen_range(-max_rotation..max_rotation);
            result = self.rotate(&result, angle)?;
        }
        
        // Color jitter
        if let Some(color_config) = self.config.color_jitter.clone() {
            result = self.color_jitter(&result, &color_config)?;
        }
        
        // Random crop
        if let Some(crop_size) = self.config.random_crop {
            result = self.random_crop(&result, crop_size)?;
        }
        
        // Random resize
        if let Some((min_scale, max_scale)) = self.config.random_resize {
            let scale = self.rng.gen_range(min_scale..max_scale);
            result = self.random_resize(&result, scale)?;
        }
        
        // Cutout
        if let Some(cutout_config) = self.config.cutout.clone() {
            result = self.cutout(&result, &cutout_config)?;
        }
        
        Ok(result)
    }
    
    /// Apply mixup between two images
    pub fn mixup(&mut self, image1: &Tensor, image2: &Tensor, label1: &Tensor, label2: &Tensor) -> Result<(Tensor, Tensor)> {
        if let Some(alpha) = self.config.mixup {
            let lambda = self.sample_beta(alpha, alpha);
            
            // Mix images
            let img1_scaled = image1.affine(lambda as f64, 0.0)?;
            let img2_scaled = image2.affine((1.0 - lambda) as f64, 0.0)?;
            let mixed_image = img1_scaled.add(&img2_scaled)?;
            
            // Mix labels
            let label1_scaled = label1.affine(lambda as f64, 0.0)?;
            let label2_scaled = label2.affine((1.0 - lambda) as f64, 0.0)?;
            let mixed_label = label1_scaled.add(&label2_scaled)?;
            
            Ok((mixed_image, mixed_label))
        } else {
            Ok((image1.clone(), label1.clone()))
        }
    }
    
    /// Random horizontal flip
    fn random_flip(&self, image: &Tensor) -> Result<Tensor> {
        let dims = image.dims();
        if dims.len() < 3 {
            return Err(Error::Model("Image must have at least 3 dimensions".to_string()));
        }
        
        // Flip along width dimension (assuming CHW format)
        let width_dim = dims.len() - 1;
        let width = dims[width_dim];
        
        // Create indices for flipping
        let indices: Vec<i64> = (0..width as i64).rev().collect();
        let indices_tensor = Tensor::new(indices.as_slice(), image.device())?;
        
        // Gather along the width dimension to flip
        image.gather(&indices_tensor, width_dim)
            .map_err(|e| Error::Tensor(e.to_string()))
    }
    
    /// Rotate image
    fn rotate(&self, image: &Tensor, angle: f32) -> Result<Tensor> {
        // Simplified rotation - in practice would use proper image rotation
        // For now, just return the original image
        Ok(image.clone())
    }
    
    /// Apply color jitter
    fn color_jitter(&mut self, image: &Tensor, config: &ColorJitterConfig) -> Result<Tensor> {
        let mut result = image.clone();
        
        // Brightness adjustment
        if config.brightness > 0.0 {
            let factor = 1.0 + self.rng.gen_range(-config.brightness..config.brightness);
            result = result.affine(factor as f64, 0.0)?;
        }
        
        // Contrast adjustment
        if config.contrast > 0.0 {
            let factor = 1.0 + self.rng.gen_range(-config.contrast..config.contrast);
            let mean = result.mean_all()?;
            let mean_val = mean.to_scalar::<f32>().unwrap();
            result = result.affine(1.0, -mean_val as f64)?
                .affine(factor as f64, 0.0)?
                .affine(1.0, mean_val as f64)?;
        }
        
        // Clamp values
        result = result.clamp(0.0, 1.0)?;
        
        Ok(result)
    }
    
    /// Random crop
    fn random_crop(&mut self, image: &Tensor, crop_size: (usize, usize)) -> Result<Tensor> {
        let dims = image.dims();
        if dims.len() < 3 {
            return Err(Error::Model("Image must have at least 3 dimensions".to_string()));
        }
        
        let height = dims[dims.len() - 2];
        let width = dims[dims.len() - 1];
        
        if height < crop_size.0 || width < crop_size.1 {
            return Ok(image.clone());
        }
        
        let y = self.rng.gen_range(0..=height - crop_size.0);
        let x = self.rng.gen_range(0..=width - crop_size.1);
        
        // Crop image
        image.narrow(dims.len() - 2, y, crop_size.0)?
            .narrow(dims.len() - 1, x, crop_size.1)
            .map_err(|e| Error::Tensor(e.to_string()))
    }
    
    /// Random resize
    fn random_resize(&self, image: &Tensor, scale: f32) -> Result<Tensor> {
        // Simplified resize - in practice would use proper image resizing
        Ok(image.clone())
    }
    
    /// Apply cutout
    fn cutout(&mut self, image: &Tensor, config: &CutoutConfig) -> Result<Tensor> {
        let mut result = image.clone();
        let dims = image.dims();
        
        if dims.len() < 3 {
            return Ok(result);
        }
        
        let height = dims[dims.len() - 2];
        let width = dims[dims.len() - 1];
        
        for _ in 0..config.num_patches {
            let y = self.rng.gen_range(0..height.saturating_sub(config.patch_size));
            let x = self.rng.gen_range(0..width.saturating_sub(config.patch_size));
            
            // Create mask
            let mask = Tensor::zeros(
                &[dims[0], config.patch_size, config.patch_size],
                image.dtype(),
                image.device(),
            )?;
            
            // Apply mask (simplified - would properly mask the region)
            // In practice, would set the region to zero or mean pixel value
        }
        
        Ok(result)
    }
    
    /// Sample from beta distribution
    fn sample_beta(&mut self, alpha: f32, beta: f32) -> f64 {
        // Simplified beta sampling using uniform distribution
        // In practice, would use proper beta distribution
        self.rng.gen_range(0.0..1.0) as f64
    }
}

/// Caption augmentation
pub struct CaptionAugmentation {
    dropout_prob: f32,
    shuffle_prob: f32,
    synonym_prob: f32,
}

impl CaptionAugmentation {
    /// Create new caption augmentation
    pub fn new(dropout_prob: f32, shuffle_prob: f32, synonym_prob: f32) -> Self {
        Self {
            dropout_prob,
            shuffle_prob,
            synonym_prob,
        }
    }
    
    /// Apply augmentation to caption
    pub fn apply(&self, caption: &str) -> String {
        let mut rng = rand::thread_rng();
        let mut words: Vec<String> = caption.split_whitespace().map(String::from).collect();
        
        // Token dropout
        if rng.gen_bool(self.dropout_prob as f64) {
            let num_drops = rng.gen_range(1..=words.len().max(1) / 3);
            for _ in 0..num_drops {
                if !words.is_empty() {
                    let idx = rng.gen_range(0..words.len());
                    words.remove(idx);
                }
            }
        }
        
        // Token shuffle
        if rng.gen_bool(self.shuffle_prob as f64) && words.len() > 1 {
            use rand::seq::SliceRandom;
            words.shuffle(&mut rng);
        }
        
        // Synonym replacement (simplified - would use actual synonym dictionary)
        if rng.gen_bool(self.synonym_prob as f64) && !words.is_empty() {
            let idx = rng.gen_range(0..words.len());
            words[idx] = format!("[{}]", words[idx]); // Placeholder for synonym
        }
        
        words.join(" ")
    }
}

/// Latent augmentation for diffusion models
pub struct LatentAugmentation {
    noise_scale: f32,
    flip_prob: f32,
}

impl LatentAugmentation {
    /// Create new latent augmentation
    pub fn new(noise_scale: f32, flip_prob: f32) -> Self {
        Self {
            noise_scale,
            flip_prob,
        }
    }
    
    /// Apply augmentation to latents
    pub fn apply(&self, latents: &Tensor) -> Result<Tensor> {
        let mut rng = rand::thread_rng();
        let mut result = latents.clone();
        
        // Add noise
        if self.noise_scale > 0.0 {
            let noise = Tensor::randn(
                0.0f32,
                self.noise_scale,
                latents.shape(),
                latents.device(),
            )?;
            result = (result + noise)?;
        }
        
        // Random flip
        if rng.gen_bool(self.flip_prob as f64) {
            let dims = result.dims();
            let width_dim = dims.len() - 1;
            let width = dims[width_dim];
            
            // Create indices for flipping
            let indices: Vec<i64> = (0..width as i64).rev().collect();
            let indices_tensor = Tensor::new(indices.as_slice(), result.device())?;
            
            // Gather along the width dimension to flip
            result = result.gather(&indices_tensor, width_dim)?;
        }
        
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_augmentation_pipeline() -> Result<()> {
        let config = AugmentationConfig::default();
        let mut pipeline = AugmentationPipeline::new(config);
        
        let image = Tensor::randn(0.0f32, 1.0, (3, 256, 256), &Device::Cpu)?;
        let augmented = pipeline.apply(&image)?;
        
        assert_eq!(augmented.dims(), image.dims());
        Ok(())
    }
    
    #[test]
    fn test_caption_augmentation() {
        let aug = CaptionAugmentation::new(0.0, 0.0, 0.0);
        let caption = "a beautiful sunset over the ocean";
        let result = aug.apply(caption);
        assert_eq!(result, caption);
    }
}