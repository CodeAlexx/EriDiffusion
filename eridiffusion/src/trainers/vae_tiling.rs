use super::{sdxl_vae_native::SDXLVAENative, *};
use anyhow::Context;
use flame_core::device::Device;
use flame_core::optimizers::{Adam, SGD};
use flame_core::{DType, Result, Shape, Tensor};

pub struct TiledVAE {
    vae: SDXLVAENative,
    config: TilingConfig,
}

// VAE Tiling for Memory-Efficient Image Encoding/Decoding
// Processes images in overlapping tiles to reduce memory usage

// FLAME uses flame_core::device::Device instead of Device

/// Configuration for VAE tiling
pub struct TilingConfig {
    /// Size of each tile (must be divisible by 8)
    pub tile_size: usize,
    /// Overlap between tiles to avoid seams
    pub overlap: usize,
    /// Blend mode for overlapping regions
    pub blend_mode: BlendMode,
}

#[derive(Clone, Copy)]
pub enum BlendMode {
    /// Simple averaging in overlap regions
    Average,
    /// Linear blending based on distance from edge
    Linear,
}

impl Default for TilingConfig {
    fn default() -> Self {
        Self {
            tile_size: 512, // 512x512 tiles
            overlap: 64,    // 64 pixel overlap
            blend_mode: BlendMode::Linear,
        }
    }
}

/// VAE with tiling support for memory-efficient processing

impl TiledVAE {
    pub fn new(vae: SDXLVAENative, config: TilingConfig) -> Self {
        Self { vae, config }
    }

    /// Encode image to latents using tiling
    pub fn encode_tiled(&self, image: &Tensor) -> flame_core::Result<Tensor> {
        let shape = image.shape();
        let dims = shape.dims();
        let batch_size = dims[0];
        let channels = dims[1];
        let height = dims[2];
        let width = dims[3];

        // Ensure dimensions are divisible by 8 (VAE requirement)
        if height % 8 != 0 || width % 8 != 0 {
            return Err(flame_core::Error::InvalidOperation(format!(
                "Image dimensions must be divisible by 8, got {}x{}",
                height, width
            )));
        }

        // Calculate tile layout
        let tile_size = self.config.tile_size;
        let overlap = self.config.overlap;
        let stride = tile_size - overlap;

        let n_tiles_h = (height - overlap + stride.saturating_sub(1)) / stride;
        let n_tiles_w = (width - overlap + stride.saturating_sub(1)) / stride;

        // Calculate latent dimensions (VAE downscales by 8x)
        let latent_height = height / 8;
        let latent_width = width / 8;
        let latent_channels = 4; // SDXL VAE has 4 latent channels

        // Initialize output tensor and weight map for blending
        let device = Device::from(image.device().clone());
        let dtype = image.dtype();
        let mut output = Tensor::zeros_dtype(
            Shape::from_dims(&[batch_size, latent_channels, latent_height, latent_width]),
            dtype,
            device.cuda_device().clone(),
        )?;
        let mut weights = Tensor::zeros_dtype(
            Shape::from_dims(&[batch_size, 1, latent_height, latent_width]),
            dtype,
            device.cuda_device().clone(),
        )?;

        // Process each tile
        for tile_y in 0..n_tiles_h {
            for tile_x in 0..n_tiles_w {
                // Calculate tile boundaries
                let y_start = tile_y * stride;
                let x_start = tile_x * stride;
                let y_end = (y_start + tile_size).min(height);
                let x_end = (x_start + tile_size).min(width);

                // Extract tile from image
                let tile = Self::extract_tile(image, y_start, y_end, x_start, x_end)?;

                // Encode tile
                let latent_tile = self.vae.encode(&tile)?;

                // Calculate latent tile position
                let lat_y_start = y_start / 8;
                let lat_x_start = x_start / 8;
                let lat_y_end = y_end / 8;
                let lat_x_end = x_end / 8;

                // Create weight mask for blending
                let tile_weight = Self::create_weight_mask(
                    lat_y_end - lat_y_start,
                    lat_x_end - lat_x_start,
                    self.config.overlap / 8,
                    self.config.blend_mode,
                    &device,
                    dtype,
                )?;

                // Accumulate weighted latent
                let (new_output, new_weights) = Self::add_tile_to_output(
                    &output,
                    &weights,
                    &latent_tile,
                    &tile_weight,
                    lat_y_start,
                    lat_y_end,
                    lat_x_start,
                    lat_x_end,
                )?;
                output = new_output;
                weights = new_weights;
            }
        }

        // Normalize by weights
        let output = output.div(&weights)?;
        Ok(output)
    }

    /// Decode latents to image using tiling
    pub fn decode_tiled(&self, latents: &Tensor) -> flame_core::Result<Tensor> {
        let shape = latents.shape();
        let dims = shape.dims();
        let batch_size = dims[0];
        let latent_channels = dims[1];
        let latent_height = dims[2];
        let latent_width = dims[3];

        // Calculate image dimensions (VAE upscales by 8x)
        let height = latent_height * 8;
        let width = latent_width * 8;
        let channels = 3; // RGB output

        // Calculate tile layout in latent space
        let latent_tile_size = self.config.tile_size / 8;
        let latent_overlap = self.config.overlap / 8;
        let latent_stride = latent_tile_size - latent_overlap;

        let n_tiles_h =
            (latent_height - latent_overlap + latent_stride.saturating_sub(1)) / latent_stride;
        let n_tiles_w =
            (latent_width - latent_overlap + latent_stride.saturating_sub(1)) / latent_stride;

        // Initialize output tensor and weight map
        let device = Device::from(latents.device().clone());
        let dtype = latents.dtype();
        let mut output = Tensor::zeros_dtype(
            Shape::from_dims(&[batch_size, channels, height, width]),
            dtype,
            device.cuda_device().clone(),
        )?;
        let mut weights = Tensor::zeros_dtype(
            Shape::from_dims(&[batch_size, 1, height, width]),
            dtype,
            device.cuda_device().clone(),
        )?;

        // Process each tile
        for tile_y in 0..n_tiles_h {
            for tile_x in 0..n_tiles_w {
                // Calculate latent tile boundaries
                let lat_y_start = tile_y * latent_stride;
                let lat_x_start = tile_x * latent_stride;
                let lat_y_end = (lat_y_start + latent_tile_size).min(latent_height);
                let lat_x_end = (lat_x_start + latent_tile_size).min(latent_width);

                // Extract latent tile
                let latent_tile =
                    Self::extract_tile(latents, lat_y_start, lat_y_end, lat_x_start, lat_x_end)?;

                // Decode tile
                let image_tile = self.vae.decode(&latent_tile)?;

                // Calculate image tile position
                let y_start = lat_y_start * 8;
                let x_start = lat_x_start * 8;
                let y_end = lat_y_end * 8;
                let x_end = lat_x_end * 8;

                // Create weight mask for blending
                let tile_weight = Self::create_weight_mask(
                    y_end - y_start,
                    x_end - x_start,
                    self.config.overlap,
                    self.config.blend_mode,
                    &device,
                    dtype,
                )?;

                // Accumulate weighted image
                let (new_output, new_weights) = Self::add_tile_to_output(
                    &output,
                    &weights,
                    &image_tile,
                    &tile_weight,
                    y_start,
                    y_end,
                    x_start,
                    x_end,
                )?;
                output = new_output;
                weights = new_weights;
            }
        }

        // Normalize by weights
        let output = output.div(&weights)?;
        Ok(output)
    }

    /// Extract a tile from a tensor
    fn extract_tile(
        tensor: &Tensor,
        y_start: usize,
        y_end: usize,
        x_start: usize,
        x_end: usize,
    ) -> flame_core::Result<Tensor> {
        let shape = tensor.shape();
        let dims = shape.dims();
        let batch_size = dims[0];
        let channels = dims[1];
        let _height = dims[2];
        let _width = dims[3];

        // Use narrow to extract the tile
        // Get tensor dimensions
        let dims = tensor.shape().dims();
        let tile = tensor.slice(&[
            (0, dims[0]),
            (0, dims[1]),
            (y_start as usize, (y_start + y_end - y_start) as usize),
            (x_start as usize, (x_start + x_end - x_start) as usize),
        ])?;

        Ok(tile)
    }

    /// Create a weight mask for blending tiles
    fn create_weight_mask(
        height: usize,
        width: usize,
        overlap: usize,
        blend_mode: BlendMode,
        device: &Device,
        dtype: DType,
    ) -> flame_core::Result<Tensor> {
        match blend_mode {
            BlendMode::Average => {
                // Simple uniform weights
                Ok(Tensor::ones(
                    Shape::from_dims(&[1, 1, height, width]),
                    device.cuda_device().clone(),
                )?
                .to_dtype(dtype)?)
            }
            BlendMode::Linear => {
                // Create linear blend weights
                let mut weights = vec![1.0f32; height * width];

                // Apply linear fade at edges
                for y in 0..height {
                    for x in 0..width {
                        let mut weight = 1.0;

                        // Fade at top edge
                        if y < overlap {
                            weight *= y as f32 / overlap as f32;
                        }
                        // Fade at bottom edge
                        if y >= height - overlap {
                            weight *= (height - 1 - y) as f32 / overlap as f32;
                        }
                        // Fade at left edge
                        if x < overlap {
                            weight *= x as f32 / overlap as f32;
                        }
                        // Fade at right edge
                        if x >= width - overlap {
                            weight *= (width - 1 - x) as f32 / overlap as f32;
                        }

                        weights[y * width + x] = weight;
                    }
                }

                let weight_tensor = Tensor::from_slice(
                    &weights,
                    Shape::from_dims(&[1, 1, height, width]),
                    device.cuda_device().clone(),
                )?
                .to_dtype(dtype)?;
                Ok(weight_tensor)
            }
        }
    }

    /// Add a tile to the output tensor with blending
    fn add_tile_to_output(
        output: &Tensor,
        weights: &Tensor,
        tile: &Tensor,
        tile_weight: &Tensor,
        y_start: usize,
        y_end: usize,
        x_start: usize,
        x_end: usize,
    ) -> flame_core::Result<(Tensor, Tensor)> {
        let shape = output.shape();
        let dims = shape.dims();
        let batch_size = dims[0];
        let channels = dims[1];
        let height = dims[2];
        let width = dims[3];
        let device = Device::from(output.device().clone());
        let dtype = output.dtype();

        // Apply weights to tile
        // FLAME doesn't have broadcast_mul, so ensure shapes match and multiply
        let tile_weight_broadcast = tile_weight.broadcast_to(tile.shape())?;
        let weighted_tile = tile.mul(&tile_weight_broadcast)?;

        // Since FLAME doesn't support slice_assign, we need to use a masking approach
        // We'll split the output into regions and concatenate

        // For simplicity and efficiency, we'll use a scatter-like approach
        // by creating a full-sized tensor with the tile at the right position;
        let tile_height = y_end - y_start;
        let tile_width = x_end - x_start;

        // Create indices for where to place the tile
        let mut tile_full = vec![0f32; batch_size * channels * height * width];
        let mut weight_full = vec![0f32; batch_size * 1 * height * width];

        // Convert tile and weight to CPU for manipulation
        let weighted_tile_vec = weighted_tile.to_vec2::<f32>()?;
        let tile_weight_vec = tile_weight.to_vec2::<f32>()?;

        // Place tile values in the correct positions
        for b in 0..batch_size {
            for c in 0..channels {
                for y in 0..tile_height {
                    for x in 0..tile_width {
                        let src_idx = ((b * channels + c) * tile_height + y) * tile_width + x;
                        let dst_idx =
                            ((b * channels + c) * height + (y_start + y)) * width + (x_start + x);

                        if c < weighted_tile_vec[0].len() / (tile_height * tile_width) {
                            tile_full[dst_idx] = weighted_tile_vec[b]
                                [c * tile_height * tile_width + y * tile_width + x];
                        }
                    }
                }
            }
        }

        // Weight tensor (single channel)
        for b in 0..batch_size {
            for y in 0..tile_height {
                for x in 0..tile_width {
                    let src_idx = y * tile_width + x;
                    let dst_idx = (b * height + (y_start + y)) * width + (x_start + x);
                    weight_full[dst_idx] = tile_weight_vec[b][src_idx];
                }
            }
        }

        // Convert back to tensors
        let tile_tensor = Tensor::from_slice(
            &tile_full,
            Shape::from_dims(&[batch_size, channels, height, width]),
            device.cuda_device().clone(),
        )?
        .to_dtype(dtype)?;
        let weight_tensor = Tensor::from_slice(
            &weight_full,
            Shape::from_dims(&[batch_size, 1, height, width]),
            device.cuda_device().clone(),
        )?
        .to_dtype(dtype)?;

        // Add to existing output and weights
        let new_output = output.add(&tile_tensor)?;
        let new_weights = weights.add(&weight_tensor)?;

        Ok((new_output, new_weights))
    }

    /// Memory usage estimation for tiling
    pub fn estimate_memory_usage(
        &self,
        image_height: usize,
        image_width: usize,
        tile_size: usize,
        batch_size: usize,
        dtype: DType,
    ) -> usize {
        let bytes_per_element = match dtype {
            DType::F32 => 4,
            DType::F16 | DType::BF16 => 2,
            _ => 4,
        };

        // Memory for one tile (3 channels for image)
        let tile_memory = batch_size * 3 * tile_size * tile_size * bytes_per_element;

        // Memory for one latent tile (4 channels, 8x smaller)
        let latent_tile_memory =
            batch_size * 4 * (tile_size / 8) * (tile_size / 8) * bytes_per_element;

        // Total memory is roughly 2x the larger of the two (for input + output)
        2 * tile_memory.max(latent_tile_memory)
    }
} // Close impl TiledVAE
