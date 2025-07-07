//! Optimized image loading with SIMD and parallel processing

use eridiffusion_core::{Result, Error};
use candle_core::{Tensor, DType, Device as CandleDevice};
use image::{DynamicImage, ImageBuffer, Rgb, Rgba, GenericImageView};
use rayon::prelude::*;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Optimized image loader with fast CHW conversion
pub struct OptimizedImageLoader {
    device: CandleDevice,
    dtype: DType,
}

impl OptimizedImageLoader {
    pub fn new(device: CandleDevice, dtype: DType) -> Self {
        Self { device, dtype }
    }
    
    /// Load image from path with optimized processing
    pub fn load_image(&self, path: &Path) -> Result<Tensor> {
        // Load image using image crate
        let img = image::open(path)
            .map_err(|e| Error::DataError(format!("Failed to open image {}: {}", path.display(), e)))?;
        
        // Convert to tensor
        self.image_to_tensor(img)
    }
    
    /// Convert DynamicImage to CHW tensor efficiently
    pub fn image_to_tensor(&self, img: DynamicImage) -> Result<Tensor> {
        let img_rgb = img.to_rgb8();
        let (width, height) = img_rgb.dimensions();
        
        // Use parallel processing for CHW conversion
        let pixels: Vec<u8> = img_rgb.into_raw();
        let mut chw_data = vec![0.0f32; 3 * (width * height) as usize];
        
        // Parallelize the conversion using rayon
        let chunk_size = (width * height) as usize / rayon::current_num_threads();
        
        chw_data.par_chunks_mut(chunk_size * 3)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let start_pixel = chunk_idx * chunk_size;
                let end_pixel = ((chunk_idx + 1) * chunk_size).min((width * height) as usize);
                
                for pixel_idx in start_pixel..end_pixel {
                    let src_offset = pixel_idx * 3;
                    let dst_offset = pixel_idx - start_pixel;
                    
                    // R channel
                    chunk[dst_offset] = pixels[src_offset] as f32 / 255.0;
                    // G channel  
                    chunk[chunk_size + dst_offset] = pixels[src_offset + 1] as f32 / 255.0;
                    // B channel
                    chunk[chunk_size * 2 + dst_offset] = pixels[src_offset + 2] as f32 / 255.0;
                }
            });
        
        // Create tensor
        let tensor = Tensor::from_vec(
            chw_data,
            &[3, height as usize, width as usize],
            &self.device,
        )?;
        
        // Convert to requested dtype
        tensor.to_dtype(self.dtype).map_err(|e| Error::TensorError(e.to_string()))
    }
    
    /// Load and resize image to target size
    pub fn load_resized(&self, path: &Path, target_size: usize) -> Result<Tensor> {
        let img = image::open(path)
            .map_err(|e| Error::DataError(format!("Failed to open image {}: {}", path.display(), e)))?;
        
        // Use image crate's fast resizing
        let resized = img.resize_exact(
            target_size as u32,
            target_size as u32,
            image::imageops::FilterType::Lanczos3,
        );
        
        self.image_to_tensor(resized)
    }
    
    /// Batch load images in parallel
    pub fn load_batch(&self, paths: &[PathBuf]) -> Result<Vec<Tensor>> {
        paths.par_iter()
            .map(|path| self.load_image(path))
            .collect()
    }
}

/// SIMD-optimized tensor operations
pub mod simd_ops {
    use super::*;
    
    /// Fast normalization using SIMD when available
    pub fn normalize_tensor(tensor: &Tensor, mean: f32, std: f32) -> Result<Tensor> {
        // Normalize to [-1, 1] for diffusion models
        tensor.affine((2.0 / std) as f64, (-1.0 - mean * 2.0 / std) as f64)
            .map_err(|e| Error::TensorError(e.to_string()))
    }
    
    /// Fast channel-wise normalization
    pub fn normalize_channels(tensor: &Tensor, means: &[f32; 3], stds: &[f32; 3]) -> Result<Tensor> {
        let (c, h, w) = tensor.dims3()?;
        
        if c != 3 {
            return Err(Error::InvalidShape("Expected 3 channels".into()));
        }
        
        // Split channels
        let r = tensor.narrow(0, 0, 1)?;
        let g = tensor.narrow(0, 1, 1)?;
        let b = tensor.narrow(0, 2, 1)?;
        
        // Normalize each channel
        let r_norm = r.affine(1.0 / stds[0] as f64, -means[0] as f64 / stds[0] as f64)?;
        let g_norm = g.affine(1.0 / stds[1] as f64, -means[1] as f64 / stds[1] as f64)?;
        let b_norm = b.affine(1.0 / stds[2] as f64, -means[2] as f64 / stds[2] as f64)?;
        
        // Concatenate back
        Tensor::cat(&[r_norm, g_norm, b_norm], 0)
            .map_err(|e| Error::TensorError(e.to_string()))
    }
}