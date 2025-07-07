//! Advanced data augmentation for diffusion model training

use eridiffusion_core::{Result, Error};
use candle_core::{Tensor, DType};
use rand::{Rng, thread_rng};
use std::f32::consts::PI;

/// Augmentation configuration
#[derive(Debug, Clone)]
pub struct AugmentationConfig {
    // Geometric transforms
    pub random_flip: bool,
    pub random_rotation: Option<f32>, // max rotation in degrees
    pub random_crop: Option<f32>, // crop ratio 0.8-1.0
    pub random_shear: Option<f32>, // max shear angle
    
    // Color transforms
    pub color_jitter: Option<ColorJitterConfig>,
    pub random_grayscale: Option<f32>, // probability
    pub random_blur: Option<f32>, // max sigma
    
    // Advanced transforms
    pub cutout: Option<CutoutConfig>,
    pub mixup: Option<f32>, // alpha parameter
    pub cutmix: Option<f32>, // beta parameter
}

#[derive(Debug, Clone)]
pub struct ColorJitterConfig {
    pub brightness: f32,
    pub contrast: f32,
    pub saturation: f32,
    pub hue: f32,
}

#[derive(Debug, Clone)]
pub struct CutoutConfig {
    pub n_holes: usize,
    pub max_size: usize,
}

/// Data augmentation pipeline
pub struct Augmenter {
    config: AugmentationConfig,
}

impl Augmenter {
    pub fn new(config: AugmentationConfig) -> Self {
        Self { config }
    }
    
    /// Apply all augmentations to a tensor
    pub fn augment(&self, tensor: &Tensor) -> Result<Tensor> {
        let mut result = tensor.clone();
        
        // Geometric transforms
        if self.config.random_flip && thread_rng().gen_bool(0.5) {
            result = self.horizontal_flip(&result)?;
        }
        
        if let Some(max_rotation) = self.config.random_rotation {
            let angle = thread_rng().gen_range(-max_rotation..max_rotation);
            result = self.rotate(&result, angle)?;
        }
        
        if let Some(crop_ratio) = self.config.random_crop {
            let ratio = thread_rng().gen_range(crop_ratio..1.0);
            result = self.random_crop(&result, ratio)?;
        }
        
        if let Some(max_shear) = self.config.random_shear {
            let shear_x = thread_rng().gen_range(-max_shear..max_shear);
            let shear_y = thread_rng().gen_range(-max_shear..max_shear);
            result = self.shear(&result, shear_x, shear_y)?;
        }
        
        // Color transforms
        if let Some(ref jitter_config) = self.config.color_jitter {
            result = self.color_jitter(&result, jitter_config)?;
        }
        
        if let Some(gray_prob) = self.config.random_grayscale {
            if thread_rng().gen_bool(gray_prob as f64) {
                result = self.to_grayscale(&result)?;
            }
        }
        
        if let Some(max_sigma) = self.config.random_blur {
            let sigma = thread_rng().gen_range(0.0..max_sigma);
            if sigma > 0.0 {
                result = self.gaussian_blur(&result, sigma)?;
            }
        }
        
        // Advanced transforms
        if let Some(ref cutout_config) = self.config.cutout {
            result = self.cutout(&result, cutout_config)?;
        }
        
        Ok(result)
    }
    
    /// Horizontal flip
    fn horizontal_flip(&self, tensor: &Tensor) -> Result<Tensor> {
        let (c, h, w) = tensor.dims3()?;
        
        // Create flipped indices
        let indices: Vec<i64> = (0..w as i64).rev().collect();
        let indices_tensor = Tensor::from_vec(indices, w, tensor.device())?;
        
        // Gather along width dimension
        tensor.gather(&indices_tensor, 2)
            .map_err(|e| Error::TensorError(e.to_string()))
    }
    
    /// Rotate image by angle (degrees)
    fn rotate(&self, tensor: &Tensor, angle: f32) -> Result<Tensor> {
        let (c, h, w) = tensor.dims3()?;
        let angle_rad = angle * PI / 180.0;
        let cos_a = angle_rad.cos();
        let sin_a = angle_rad.sin();
        
        // Create rotation matrix
        let cx = w as f32 / 2.0;
        let cy = h as f32 / 2.0;
        
        // Create coordinate grids
        let mut rotated = vec![0.0f32; c * h * w];
        let data = tensor.flatten_all()?.to_vec1::<f32>()?;
        
        for y in 0..h {
            for x in 0..w {
                // Translate to center
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                
                // Apply rotation
                let new_x = (dx * cos_a - dy * sin_a + cx) as i32;
                let new_y = (dx * sin_a + dy * cos_a + cy) as i32;
                
                // Check bounds and copy pixels
                if new_x >= 0 && new_x < w as i32 && new_y >= 0 && new_y < h as i32 {
                    for ch in 0..c {
                        let src_idx = ch * h * w + new_y as usize * w + new_x as usize;
                        let dst_idx = ch * h * w + y * w + x;
                        rotated[dst_idx] = data[src_idx];
                    }
                }
            }
        }
        
        Tensor::from_vec(rotated, &[c, h, w], tensor.device())
            .map_err(|e| Error::TensorError(e.to_string()))
    }
    
    /// Random crop with given ratio
    fn random_crop(&self, tensor: &Tensor, ratio: f32) -> Result<Tensor> {
        let (c, h, w) = tensor.dims3()?;
        
        let crop_h = (h as f32 * ratio) as usize;
        let crop_w = (w as f32 * ratio) as usize;
        
        let start_h = thread_rng().gen_range(0..=h - crop_h);
        let start_w = thread_rng().gen_range(0..=w - crop_w);
        
        tensor.narrow(1, start_h, crop_h)?
            .narrow(2, start_w, crop_w)
            .map_err(|e| Error::TensorError(e.to_string()))
    }
    
    /// Shear transform
    fn shear(&self, tensor: &Tensor, shear_x: f32, shear_y: f32) -> Result<Tensor> {
        let (c, h, w) = tensor.dims3()?;
        let shear_x_rad = shear_x * PI / 180.0;
        let shear_y_rad = shear_y * PI / 180.0;
        
        // Create sheared output
        let mut sheared = vec![0.0f32; c * h * w];
        let data = tensor.flatten_all()?.to_vec1::<f32>()?;
        
        let cx = w as f32 / 2.0;
        let cy = h as f32 / 2.0;
        
        for y in 0..h {
            for x in 0..w {
                // Translate to center
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                
                // Apply shear transform (inverse mapping)
                let src_x = dx - dy * shear_x_rad.tan() + cx;
                let src_y = dy - dx * shear_y_rad.tan() + cy;
                
                // Bilinear interpolation
                let x0 = src_x.floor() as i32;
                let y0 = src_y.floor() as i32;
                let x1 = x0 + 1;
                let y1 = y0 + 1;
                
                if x0 >= 0 && x1 < w as i32 && y0 >= 0 && y1 < h as i32 {
                    let fx = src_x - x0 as f32;
                    let fy = src_y - y0 as f32;
                    
                    for ch in 0..c {
                        let idx00 = ch * h * w + y0 as usize * w + x0 as usize;
                        let idx01 = ch * h * w + y0 as usize * w + x1 as usize;
                        let idx10 = ch * h * w + y1 as usize * w + x0 as usize;
                        let idx11 = ch * h * w + y1 as usize * w + x1 as usize;
                        
                        let v00 = data[idx00];
                        let v01 = data[idx01];
                        let v10 = data[idx10];
                        let v11 = data[idx11];
                        
                        let v0 = v00 * (1.0 - fx) + v01 * fx;
                        let v1 = v10 * (1.0 - fx) + v11 * fx;
                        let v = v0 * (1.0 - fy) + v1 * fy;
                        
                        let dst_idx = ch * h * w + y * w + x;
                        sheared[dst_idx] = v;
                    }
                }
            }
        }
        
        Tensor::from_vec(sheared, &[c, h, w], tensor.device())
            .map_err(|e| Error::TensorError(e.to_string()))
    }
    
    /// Color jittering
    fn color_jitter(&self, tensor: &Tensor, config: &ColorJitterConfig) -> Result<Tensor> {
        let mut result = tensor.clone();
        
        // Brightness
        if config.brightness > 0.0 {
            let factor = 1.0 + thread_rng().gen_range(-config.brightness..config.brightness);
            result = result.affine(factor as f64, 0.0)?;
        }
        
        // Contrast
        if config.contrast > 0.0 {
            let factor = 1.0 + thread_rng().gen_range(-config.contrast..config.contrast);
            let mean = result.mean_all()?.to_scalar::<f32>()?;
            result = result.affine(factor as f64, mean as f64 * (1.0 - factor as f64))?;
        }
        
        // Saturation (simplified - convert to grayscale and mix)
        if config.saturation > 0.0 {
            let factor = 1.0 + thread_rng().gen_range(-config.saturation..config.saturation);
            let gray = self.to_grayscale(&result)?;
            let gray_3ch = gray.repeat(&[3, 1, 1])?;
            result = result.affine(factor as f64, 0.0)?
                .add(&gray_3ch.affine((1.0 - factor) as f64, 0.0)?)?;
        }
        
        Ok(result)
    }
    
    /// Convert to grayscale
    fn to_grayscale(&self, tensor: &Tensor) -> Result<Tensor> {
        let (c, h, w) = tensor.dims3()?;
        if c != 3 {
            return Ok(tensor.clone());
        }
        
        // Weights for RGB to grayscale conversion
        let r_weight = 0.299;
        let g_weight = 0.587;
        let b_weight = 0.114;
        
        let r = tensor.narrow(0, 0, 1)?;
        let g = tensor.narrow(0, 1, 1)?;
        let b = tensor.narrow(0, 2, 1)?;
        
        let gray = r.affine(r_weight, 0.0)?
            .add(&g.affine(g_weight, 0.0)?)?
            .add(&b.affine(b_weight, 0.0)?)?;
        
        // Repeat to 3 channels
        gray.repeat(&[3, 1, 1])
            .map_err(|e| Error::TensorError(e.to_string()))
    }
    
    /// Gaussian blur
    fn gaussian_blur(&self, tensor: &Tensor, sigma: f32) -> Result<Tensor> {
        // Simplified box blur approximation
        let kernel_size = ((sigma * 3.0).ceil() as usize * 2 + 1).max(3);
        let padding = kernel_size / 2;
        
        // Create simple averaging kernel
        let kernel_val = 1.0 / (kernel_size * kernel_size) as f32;
        
        // Pad tensor
        let padded = self.pad_reflect(tensor, padding)?;
        
        // Apply convolution (simplified - just averaging)
        let (c, h, w) = tensor.dims3()?;
        let mut blurred = vec![0.0f32; c * h * w];
        let padded_data = padded.flatten_all()?.to_vec1::<f32>()?;
        
        for ch in 0..c {
            for y in 0..h {
                for x in 0..w {
                    let mut sum = 0.0;
                    for ky in 0..kernel_size {
                        for kx in 0..kernel_size {
                            let py = y + ky;
                            let px = x + kx;
                            let idx = ch * (h + 2 * padding) * (w + 2 * padding) + 
                                     py * (w + 2 * padding) + px;
                            sum += padded_data[idx] * kernel_val;
                        }
                    }
                    blurred[ch * h * w + y * w + x] = sum;
                }
            }
        }
        
        Tensor::from_vec(blurred, &[c, h, w], tensor.device())
            .map_err(|e| Error::TensorError(e.to_string()))
    }
    
    /// Cutout augmentation
    fn cutout(&self, tensor: &Tensor, config: &CutoutConfig) -> Result<Tensor> {
        let (c, h, w) = tensor.dims3()?;
        let mut data = tensor.flatten_all()?.to_vec1::<f32>()?;
        
        for _ in 0..config.n_holes {
            let hole_h = thread_rng().gen_range(1..=config.max_size.min(h));
            let hole_w = thread_rng().gen_range(1..=config.max_size.min(w));
            
            let start_h = thread_rng().gen_range(0..=h - hole_h);
            let start_w = thread_rng().gen_range(0..=w - hole_w);
            
            // Apply cutout by setting region to zero
            for ch in 0..c {
                for y in start_h..(start_h + hole_h) {
                    for x in start_w..(start_w + hole_w) {
                        let idx = ch * h * w + y * w + x;
                        data[idx] = 0.0;
                    }
                }
            }
        }
        
        Tensor::from_vec(data, &[c, h, w], tensor.device())
            .map_err(|e| Error::TensorError(e.to_string()))
    }
    
    /// Pad with reflection
    fn pad_reflect(&self, tensor: &Tensor, padding: usize) -> Result<Tensor> {
        if padding == 0 {
            return Ok(tensor.clone());
        }
        
        let (c, h, w) = tensor.dims3()?;
        let new_h = h + 2 * padding;
        let new_w = w + 2 * padding;
        
        // Get tensor data
        let data = tensor.flatten_all()?.to_vec1::<f32>()?;
        let mut padded_data = vec![0.0f32; c * new_h * new_w];
        
        // Copy original data to center
        for ch in 0..c {
            for y in 0..h {
                for x in 0..w {
                    let src_idx = ch * h * w + y * w + x;
                    let dst_idx = ch * new_h * new_w + (y + padding) * new_w + (x + padding);
                    padded_data[dst_idx] = data[src_idx];
                }
            }
        }
        
        // Reflect padding for each channel
        for ch in 0..c {
            // Top padding
            for y in 0..padding {
                let src_y = padding + padding - y - 1; // Reflect around edge
                for x in padding..(new_w - padding) {
                    let src_idx = ch * new_h * new_w + src_y * new_w + x;
                    let dst_idx = ch * new_h * new_w + y * new_w + x;
                    padded_data[dst_idx] = padded_data[src_idx];
                }
            }
            
            // Bottom padding
            for y in (new_h - padding)..new_h {
                let src_y = 2 * (new_h - padding) - y - 1; // Reflect around edge
                for x in padding..(new_w - padding) {
                    let src_idx = ch * new_h * new_w + src_y * new_w + x;
                    let dst_idx = ch * new_h * new_w + y * new_w + x;
                    padded_data[dst_idx] = padded_data[src_idx];
                }
            }
            
            // Left padding (including corners)
            for y in 0..new_h {
                for x in 0..padding {
                    let src_x = padding + padding - x - 1; // Reflect around edge
                    let src_idx = ch * new_h * new_w + y * new_w + src_x;
                    let dst_idx = ch * new_h * new_w + y * new_w + x;
                    padded_data[dst_idx] = padded_data[src_idx];
                }
            }
            
            // Right padding (including corners)
            for y in 0..new_h {
                for x in (new_w - padding)..new_w {
                    let src_x = 2 * (new_w - padding) - x - 1; // Reflect around edge
                    let src_idx = ch * new_h * new_w + y * new_w + src_x;
                    let dst_idx = ch * new_h * new_w + y * new_w + x;
                    padded_data[dst_idx] = padded_data[src_idx];
                }
            }
        }
        
        Tensor::from_vec(padded_data, &[c, new_h, new_w], tensor.device())
            .map_err(|e| Error::TensorError(e.to_string()))
    }
}

/// MixUp augmentation for batch processing
pub fn mixup_batch(
    batch1: &Tensor,
    batch2: &Tensor,
    labels1: &Tensor,
    labels2: &Tensor,
    alpha: f32,
) -> Result<(Tensor, Tensor)> {
    let lambda = thread_rng().sample(rand_distr::Beta::new(alpha, alpha).unwrap());
    
    let mixed_images = batch1.affine(lambda as f64, 0.0)?
        .add(&batch2.affine((1.0 - lambda) as f64, 0.0)?)?;
    
    let mixed_labels = labels1.affine(lambda as f64, 0.0)?
        .add(&labels2.affine((1.0 - lambda) as f64, 0.0)?)?;
    
    Ok((mixed_images, mixed_labels))
}

/// CutMix augmentation
pub fn cutmix_batch(
    batch1: &Tensor,
    batch2: &Tensor,
    labels1: &Tensor,
    labels2: &Tensor,
    beta: f32,
) -> Result<(Tensor, Tensor)> {
    let (b, c, h, w) = batch1.dims4()?;
    let lambda = thread_rng().sample(rand_distr::Beta::new(beta, beta).unwrap());
    
    // Generate random box
    let cut_ratio = (1.0 - lambda).sqrt();
    let cut_h = (h as f32 * cut_ratio) as usize;
    let cut_w = (w as f32 * cut_ratio) as usize;
    
    let cx = thread_rng().gen_range(0..w);
    let cy = thread_rng().gen_range(0..h);
    
    let x1 = cx.saturating_sub(cut_w / 2);
    let y1 = cy.saturating_sub(cut_h / 2);
    let x2 = (cx + cut_w / 2).min(w);
    let y2 = (cy + cut_h / 2).min(h);
    
    // Create binary mask for the box region
    let mut mask_data = vec![0.0f32; b * c * h * w];
    let box_h = y2 - y1;
    let box_w = x2 - x1;
    
    for batch in 0..b {
        for ch in 0..c {
            for y in y1..y2 {
                for x in x1..x2 {
                    let idx = batch * c * h * w + ch * h * w + y * w + x;
                    mask_data[idx] = 1.0;
                }
            }
        }
    }
    
    let mask = Tensor::from_vec(mask_data, &[b, c, h, w], batch1.device())?;
    let inv_mask = mask.affine(-1.0, 1.0)?; // 1 - mask
    
    // Apply CutMix: keep batch1 outside box, batch2 inside box
    let mixed_images = batch1.broadcast_mul(&inv_mask)?
        .add(&batch2.broadcast_mul(&mask)?)?;
    
    // Calculate actual lambda based on box area
    let actual_lambda = 1.0 - (box_h * box_w) as f32 / (h * w) as f32;
    let mixed_labels = labels1.affine(actual_lambda as f64, 0.0)?
        .add(&labels2.affine((1.0 - actual_lambda) as f64, 0.0)?)?;
    
    Ok((mixed_images, mixed_labels))
}