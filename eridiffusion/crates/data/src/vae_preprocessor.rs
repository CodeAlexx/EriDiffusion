//! VAE preprocessing pipeline for different model architectures

use eridiffusion_core::{Result, Error, Device, ModelArchitecture};
use eridiffusion_models::VAE;
use candle_core::{Tensor, DType};
use std::sync::Arc;
use std::collections::HashMap;
use tracing::{info, debug};

/// VAE preprocessor for encoding images to latents
pub struct VAEPreprocessor {
    vae: Arc<dyn VAE>,
    architecture: ModelArchitecture,
    config: VAEConfig,
    device: Device,
}

impl VAEPreprocessor {
    /// Create new VAE preprocessor
    pub fn new(
        vae: Arc<dyn VAE>,
        architecture: ModelArchitecture,
    ) -> Result<Self> {
        let config = VAEConfig::for_architecture(&architecture);
        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
        
        Ok(Self {
            vae,
            architecture,
            config,
            device,
        })
    }
    
    /// Encode single image to latent
    pub fn encode_image(&self, image: &Tensor) -> Result<Tensor> {
        debug!("Encoding image with shape {:?}", image.shape());
        
        // Ensure correct device - always move to target device
        let candle_device = match &self.device {
            Device::Cpu => candle_core::Device::Cpu,
            Device::Cuda(id) => candle_core::Device::new_cuda(*id)?,
        };
        let image = image.to_device(&candle_device)?;
        
        // Add batch dimension if needed
        let image = if image.dims().len() == 3 {
            image.unsqueeze(0)?
        } else {
            image
        };
        
        // Validate shape
        self.validate_image_shape(&image)?;
        
        // Normalize for VAE
        let normalized = self.normalize_for_vae(&image)?;
        
        // Encode
        let latent = self.vae.encode(&normalized)?;
        
        // Apply scaling factor
        let scaled = latent.affine(self.config.scale_factor as f64, 0.0)?;
        
        // Remove batch dimension if we added it
        if image.dims()[0] == 1 {
            Ok(scaled.squeeze(0)?)
        } else {
            Ok(scaled)
        }
    }
    
    /// Encode batch of images
    pub fn encode_batch(&self, images: &Tensor) -> Result<Tensor> {
        debug!("Encoding batch of {} images", images.dims()[0]);
        
        // Ensure correct device
        // Ensure correct device - always move to target device
        let candle_device = match &self.device {
            Device::Cpu => candle_core::Device::Cpu,
            Device::Cuda(id) => candle_core::Device::new_cuda(*id)?,
        };
        let images = images.to_device(&candle_device)?;
        
        // Validate shape
        self.validate_batch_shape(&images)?;
        
        // Normalize for VAE
        let normalized = self.normalize_for_vae(&images)?;
        
        // Encode
        let latents = self.vae.encode(&normalized)?;
        
        // Apply scaling factor
        Ok(latents.affine(self.config.scale_factor as f64, 0.0)?)
    }
    
    /// Decode latent to image
    pub fn decode_latent(&self, latent: &Tensor) -> Result<Tensor> {
        debug!("Decoding latent with shape {:?}", latent.shape());
        
        // Ensure correct device - always move to target device
        let candle_device = match &self.device {
            Device::Cpu => candle_core::Device::Cpu,
            Device::Cuda(id) => candle_core::Device::new_cuda(*id)?,
        };
        let latent = latent.to_device(&candle_device)?;
        
        // Add batch dimension if needed
        let latent = if latent.dims().len() == 3 {
            latent.unsqueeze(0)?
        } else {
            latent
        };
        
        // Apply inverse scaling
        let unscaled = latent.affine(1.0 / self.config.scale_factor as f64, 0.0)?;
        
        // Decode
        let image = self.vae.decode(&unscaled)?;
        
        // Denormalize from VAE output
        let denormalized = self.denormalize_from_vae(&image)?;
        
        // Remove batch dimension if we added it
        if latent.dims()[0] == 1 {
            Ok(denormalized.squeeze(0)?)
        } else {
            Ok(denormalized)
        }
    }
    
    /// Normalize image for VAE input
    fn normalize_for_vae(&self, image: &Tensor) -> Result<Tensor> {
        // Most VAEs expect [-1, 1] range
        match self.architecture {
            ModelArchitecture::SD15 | ModelArchitecture::SD15 => {
                // Already in [-1, 1] from preprocessor
                Ok(image.clone())
            }
            ModelArchitecture::SDXL => {
                // Already in [-1, 1] from preprocessor
                Ok(image.clone())
            }
            ModelArchitecture::SD3 | ModelArchitecture::SD35 => {
                // SD3 VAE expects specific normalization
                // Some implementations use different ranges
                Ok(image.clone())
            }
            ModelArchitecture::Flux => {
                // Flux may have different normalization
                Ok(image.clone())
            }
            _ => Ok(image.clone()),
        }
    }
    
    /// Denormalize image from VAE output
    fn denormalize_from_vae(&self, image: &Tensor) -> Result<Tensor> {
        // Convert from [-1, 1] to [0, 1]
        Ok(image.affine(0.5, 0.5)?)
    }
    
    /// Validate image shape
    fn validate_image_shape(&self, image: &Tensor) -> Result<()> {
        let dims = image.dims();
        
        if dims.len() != 4 {
            return Err(Error::InvalidShape(format!(
                "Expected 4D tensor [B, C, H, W], got {}D",
                dims.len()
            )));
        }
        
        let (_, channels, height, width) = (dims[0], dims[1], dims[2], dims[3]);
        
        // Check channels
        if channels != 3 {
            return Err(Error::InvalidShape(format!(
                "Expected 3 channels, got {}",
                channels
            )));
        }
        
        // Check dimensions are divisible by downsampling factor
        let factor = self.config.downsampling_factor;
        if height % factor != 0 || width % factor != 0 {
            return Err(Error::InvalidShape(format!(
                "Image dimensions ({}, {}) must be divisible by {}",
                height, width, factor
            )));
        }
        
        Ok(())
    }
    
    /// Validate batch shape
    fn validate_batch_shape(&self, images: &Tensor) -> Result<()> {
        self.validate_image_shape(images)
    }
    
    /// Get latent shape for given image shape
    pub fn get_latent_shape(&self, image_shape: &[usize]) -> Result<Vec<usize>> {
        if image_shape.len() != 4 {
            return Err(Error::InvalidShape("Expected 4D image shape".into()));
        }
        
        let batch = image_shape[0];
        let height = image_shape[2];
        let width = image_shape[3];
        
        let latent_height = height / self.config.downsampling_factor;
        let latent_width = width / self.config.downsampling_factor;
        
        Ok(vec![
            batch,
            self.config.latent_channels,
            latent_height,
            latent_width,
        ])
    }
    
    /// Precompute latents for a batch with progress callback
    pub async fn precompute_batch_with_progress<F>(
        &self,
        images: Vec<Tensor>,
        batch_size: usize,
        progress_fn: F,
    ) -> Result<Vec<Tensor>>
    where
        F: Fn(usize, usize) + Send,
    {
        let total = images.len();
        let mut latents = Vec::with_capacity(total);
        
        for (i, chunk) in images.chunks(batch_size).enumerate() {
            // Stack into batch
            let batch = if chunk.len() == 1 {
                chunk[0].unsqueeze(0)?
            } else {
                Tensor::stack(chunk, 0)?
            };
            
            // Encode batch
            let batch_latents = self.encode_batch(&batch)?;
            
            // Split back into individual latents
            for j in 0..chunk.len() {
                let latent = batch_latents.narrow(0, j, 1)?;
                latents.push(latent.squeeze(0)?);
            }
            
            // Progress callback
            let processed = (i + 1) * batch_size;
            progress_fn(processed.min(total), total);
        }
        
        Ok(latents)
    }
}

/// VAE configuration for different architectures
#[derive(Debug, Clone)]
pub struct VAEConfig {
    /// Number of latent channels
    pub latent_channels: usize,
    
    /// Downsampling factor (image size / latent size)
    pub downsampling_factor: usize,
    
    /// Scale factor for latents
    pub scale_factor: f32,
    
    /// Whether to use tiling for large images
    pub use_tiling: bool,
    
    /// Tile size for tiling
    pub tile_size: usize,
    
    /// Tile overlap for tiling
    pub tile_overlap: usize,
}

impl VAEConfig {
    /// Get config for architecture
    pub fn for_architecture(arch: &ModelArchitecture) -> Self {
        match arch {
            ModelArchitecture::SD15 | ModelArchitecture::SD15 => Self {
                latent_channels: 4,
                downsampling_factor: 8,
                scale_factor: 0.18215,
                use_tiling: false,
                tile_size: 512,
                tile_overlap: 64,
            },
            ModelArchitecture::SDXL => Self {
                latent_channels: 4,
                downsampling_factor: 8,
                scale_factor: 0.13025,
                use_tiling: true,
                tile_size: 1024,
                tile_overlap: 128,
            },
            ModelArchitecture::SD3 | ModelArchitecture::SD35 => Self {
                latent_channels: 16,
                downsampling_factor: 8,
                scale_factor: 0.13025,
                use_tiling: true,
                tile_size: 1024,
                tile_overlap: 128,
            },
            ModelArchitecture::Flux => Self {
                latent_channels: 16,
                downsampling_factor: 8,
                scale_factor: 0.3611,
                use_tiling: true,
                tile_size: 1024,
                tile_overlap: 128,
            },
            _ => Self {
                latent_channels: 4,
                downsampling_factor: 8,
                scale_factor: 0.18215,
                use_tiling: false,
                tile_size: 512,
                tile_overlap: 64,
            },
        }
    }
}

/// Tiled VAE encoder for large images
pub struct TiledVAEEncoder {
    vae_preprocessor: VAEPreprocessor,
    tile_size: usize,
    tile_overlap: usize,
}

impl TiledVAEEncoder {
    /// Create new tiled encoder
    pub fn new(vae_preprocessor: VAEPreprocessor) -> Self {
        let tile_size = vae_preprocessor.config.tile_size;
        let tile_overlap = vae_preprocessor.config.tile_overlap;
        
        Self {
            vae_preprocessor,
            tile_size,
            tile_overlap,
        }
    }
    
    /// Encode large image using tiling
    pub fn encode_tiled(&self, image: &Tensor) -> Result<Tensor> {
        let (batch, channels, height, width) = match image.dims() {
            [b, c, h, w] => (*b, *c, *h, *w),
            _ => return Err(Error::InvalidShape("Expected BCHW format".into())),
        };
        
        if batch != 1 {
            return Err(Error::InvalidShape("Tiled encoding only supports batch size 1".into()));
        }
        
        // Calculate number of tiles
        let tiles_y = (height + self.tile_size - self.tile_overlap - 1) / (self.tile_size - self.tile_overlap);
        let tiles_x = (width + self.tile_size - self.tile_overlap - 1) / (self.tile_size - self.tile_overlap);
        
        info!("Encoding {}x{} image with {}x{} tiles", height, width, tiles_y, tiles_x);
        
        // Calculate latent dimensions
        let factor = self.vae_preprocessor.config.downsampling_factor;
        let latent_tile_size = self.tile_size / factor;
        let latent_overlap = self.tile_overlap / factor;
        let latent_height = height / factor;
        let latent_width = width / factor;
        let latent_channels = self.vae_preprocessor.config.latent_channels;
        
        // Create output tensor
        let candle_device = match &self.vae_preprocessor.device {
            Device::Cpu => candle_core::Device::Cpu,
            Device::Cuda(id) => candle_core::Device::new_cuda(*id)?,
        };
        let mut output = Tensor::zeros(
            &[1, latent_channels, latent_height, latent_width],
            DType::F32,
            &candle_device,
        )?;
        
        let mut weight_map = Tensor::zeros(
            &[1, 1, latent_height, latent_width],
            DType::F32,
            &candle_device,
        )?;
        
        // Process each tile
        for tile_y in 0..tiles_y {
            for tile_x in 0..tiles_x {
                // Calculate tile boundaries
                let y_start = tile_y * (self.tile_size - self.tile_overlap);
                let x_start = tile_x * (self.tile_size - self.tile_overlap);
                let y_end = (y_start + self.tile_size).min(height);
                let x_end = (x_start + self.tile_size).min(width);
                
                // Extract tile
                let tile = image.narrow(2, y_start, y_end - y_start)?
                    .narrow(3, x_start, x_end - x_start)?;
                
                // Pad if necessary
                let tile = if tile.dims()[2] < self.tile_size || tile.dims()[3] < self.tile_size {
                    pad_to_size(&tile, self.tile_size, self.tile_size)?
                } else {
                    tile
                };
                
                // Encode tile
                let latent_tile = self.vae_preprocessor.encode_image(&tile)?;
                
                // Calculate latent tile position
                let latent_y_start = y_start / factor;
                let latent_x_start = x_start / factor;
                let latent_y_end = y_end / factor;
                let latent_x_end = x_end / factor;
                
                // Create weight tensor for blending
                let tile_weight = create_tile_weight(
                    latent_y_end - latent_y_start,
                    latent_x_end - latent_x_start,
                    latent_overlap,
                    &candle_device,
                )?;
                
                // Add to output with blending
                let (new_output, new_weight_map) = add_tile_to_output(
                    &output,
                    &weight_map,
                    &latent_tile,
                    &tile_weight,
                    latent_y_start,
                    latent_x_start,
                )?;
                output = new_output;
                weight_map = new_weight_map;
            }
        }
        
        // Normalize by weight map
        output = output.broadcast_div(&weight_map.maximum(1e-8)?)?;
        
        Ok(output)
    }
}

/// Pad tensor to target size
fn pad_to_size(tensor: &Tensor, target_h: usize, target_w: usize) -> Result<Tensor> {
    let (b, c, h, w) = match tensor.dims() {
        [b, c, h, w] => (*b, *c, *h, *w),
        _ => return Err(Error::InvalidShape("Expected BCHW format".into())),
    };
    
    if h >= target_h && w >= target_w {
        return Ok(tensor.clone());
    }
    
    let pad_h = target_h.saturating_sub(h);
    let pad_w = target_w.saturating_sub(w);
    
    // Create padded tensor
    let device = tensor.device();
    let dtype = tensor.dtype();
    
    // Flatten input tensor
    let tensor_data = tensor.flatten_all()?.to_vec1::<f32>()?;
    
    // Create output with reflection padding
    let mut padded_data = vec![0.0f32; b * c * target_h * target_w];
    
    // Copy original data to center of padded tensor
    let pad_top = pad_h / 2;
    let pad_left = pad_w / 2;
    
    for batch in 0..b {
        for chan in 0..c {
            for y in 0..h {
                for x in 0..w {
                    let src_idx = batch * c * h * w + chan * h * w + y * w + x;
                    let dst_y = y + pad_top;
                    let dst_x = x + pad_left;
                    let dst_idx = batch * c * target_h * target_w 
                        + chan * target_h * target_w 
                        + dst_y * target_w 
                        + dst_x;
                    padded_data[dst_idx] = tensor_data[src_idx];
                }
            }
            
            // Fill padding with edge values (reflection padding)
            // Top padding
            for y in 0..pad_top {
                for x in pad_left..(pad_left + w) {
                    let src_x = x - pad_left;
                    let src_idx = batch * c * target_h * target_w 
                        + chan * target_h * target_w 
                        + pad_top * target_w 
                        + x;
                    let dst_idx = batch * c * target_h * target_w 
                        + chan * target_h * target_w 
                        + y * target_w 
                        + x;
                    padded_data[dst_idx] = padded_data[src_idx];
                }
            }
            
            // Bottom padding
            for y in (pad_top + h)..target_h {
                for x in pad_left..(pad_left + w) {
                    let src_idx = batch * c * target_h * target_w 
                        + chan * target_h * target_w 
                        + (pad_top + h - 1) * target_w 
                        + x;
                    let dst_idx = batch * c * target_h * target_w 
                        + chan * target_h * target_w 
                        + y * target_w 
                        + x;
                    padded_data[dst_idx] = padded_data[src_idx];
                }
            }
            
            // Left padding
            for y in 0..target_h {
                for x in 0..pad_left {
                    let src_idx = batch * c * target_h * target_w 
                        + chan * target_h * target_w 
                        + y * target_w 
                        + pad_left;
                    let dst_idx = batch * c * target_h * target_w 
                        + chan * target_h * target_w 
                        + y * target_w 
                        + x;
                    padded_data[dst_idx] = padded_data[src_idx];
                }
            }
            
            // Right padding
            for y in 0..target_h {
                for x in (pad_left + w)..target_w {
                    let src_idx = batch * c * target_h * target_w 
                        + chan * target_h * target_w 
                        + y * target_w 
                        + (pad_left + w - 1);
                    let dst_idx = batch * c * target_h * target_w 
                        + chan * target_h * target_w 
                        + y * target_w 
                        + x;
                    padded_data[dst_idx] = padded_data[src_idx];
                }
            }
        }
    }
    
    Ok(Tensor::from_vec(padded_data, &[b, c, target_h, target_w], device)?.to_dtype(dtype)?)
}

/// Create weight tensor for tile blending
fn create_tile_weight(
    height: usize,
    width: usize,
    overlap: usize,
    device: &candle_core::Device,
) -> Result<Tensor> {
    let mut weight_data = vec![1.0f32; height * width];
    
    // Apply linear blending in overlap regions
    if overlap > 0 {
        // Top edge
        for y in 0..overlap.min(height) {
            let blend = y as f32 / overlap as f32;
            for x in 0..width {
                weight_data[y * width + x] *= blend;
            }
        }
        
        // Bottom edge
        for y in (height.saturating_sub(overlap))..height {
            let blend = (height - y - 1) as f32 / overlap as f32;
            for x in 0..width {
                weight_data[y * width + x] *= blend;
            }
        }
        
        // Left edge
        for x in 0..overlap.min(width) {
            let blend = x as f32 / overlap as f32;
            for y in 0..height {
                weight_data[y * width + x] *= blend;
            }
        }
        
        // Right edge
        for x in (width.saturating_sub(overlap))..width {
            let blend = (width - x - 1) as f32 / overlap as f32;
            for y in 0..height {
                weight_data[y * width + x] *= blend;
            }
        }
    }
    
    Ok(Tensor::from_vec(weight_data, &[1, 1, height, width], device)?)
}

/// Add tile to output with blending
fn add_tile_to_output(
    output: &Tensor,
    weight_map: &Tensor,
    tile: &Tensor,
    tile_weight: &Tensor,
    y_start: usize,
    x_start: usize,
) -> Result<(Tensor, Tensor)> {
    let (_, channels, tile_h, tile_w) = match tile.dims() {
        [b, c, h, w] => (*b, *c, *h, *w),
        _ => return Err(Error::InvalidShape("Expected BCHW format".into())),
    };
    
    // Since Candle doesn't support in-place operations on tensor slices,
    // we need to reconstruct the full tensor with the updated region
    
    // Get full dimensions
    let (batch, out_channels, out_h, out_w) = match output.dims() {
        [b, c, h, w] => (*b, *c, *h, *w),
        _ => return Err(Error::InvalidShape("Expected BCHW output format".into())),
    };
    
    // Create masks for the region we're updating
    let device = output.device();
    
    // Create a mask tensor that's 1.0 in the tile region, 0.0 elsewhere
    let mut mask_data = vec![0.0f32; batch * out_channels * out_h * out_w];
    for b in 0..batch {
        for c in 0..out_channels {
            for y in 0..tile_h {
                for x in 0..tile_w {
                    let out_y = y_start + y;
                    let out_x = x_start + x;
                    if out_y < out_h && out_x < out_w {
                        let idx = b * out_channels * out_h * out_w
                            + c * out_h * out_w
                            + out_y * out_w
                            + out_x;
                        mask_data[idx] = 1.0;
                    }
                }
            }
        }
    }
    let mask = Tensor::from_vec(mask_data, &[batch, out_channels, out_h, out_w], device)?;
    
    // Create a full-size tensor with the tile data in the right position
    let mut tile_full_data = vec![0.0f32; batch * out_channels * out_h * out_w];
    let tile_data = tile.flatten_all()?.to_vec1::<f32>()?;
    
    for b in 0..batch {
        for c in 0..channels.min(out_channels) {
            for y in 0..tile_h {
                for x in 0..tile_w {
                    let out_y = y_start + y;
                    let out_x = x_start + x;
                    if out_y < out_h && out_x < out_w {
                        let tile_idx = b * channels * tile_h * tile_w
                            + c * tile_h * tile_w
                            + y * tile_w
                            + x;
                        let out_idx = b * out_channels * out_h * out_w
                            + c * out_h * out_w
                            + out_y * out_w
                            + out_x;
                        tile_full_data[out_idx] = tile_data[tile_idx];
                    }
                }
            }
        }
    }
    let tile_full = Tensor::from_vec(tile_full_data, &[batch, out_channels, out_h, out_w], device)?;
    
    // Apply tile weight
    let weighted_tile_full = tile_full.broadcast_mul(&tile_weight.broadcast_as(&[batch, out_channels, out_h, out_w])?)?;
    
    // Blend: output * (1 - mask) + weighted_tile * mask
    let inv_mask = mask.affine(-1.0, 1.0)?; // 1 - mask
    let kept_output = output.broadcast_mul(&inv_mask)?;
    let new_output = kept_output.add(&weighted_tile_full.broadcast_mul(&mask)?)?;
    
    // Update weight map similarly
    let weight_mask = Tensor::from_vec(
        mask_data.iter().take(out_h * out_w).copied().collect(),
        &[1, 1, out_h, out_w],
        device
    )?;
    let weight_full = tile_weight.broadcast_as(&[1, 1, out_h, out_w])?.broadcast_mul(&weight_mask)?;
    let new_weight_map = weight_map.add(&weight_full)?;
    
    Ok((new_output, new_weight_map))
}