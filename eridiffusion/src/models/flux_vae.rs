use crate::loaders::WeightLoader;
use crate::models::aligned_image_processor::AlignedImageProcessor;
use crate::models::fast_vae::FastVAE;
use crate::models::vae::{AutoEncoderKL as BaseVAE, VAEConfig};
use crate::trainers::cpu_offload_manager::CPUOffloadManager;
use flame_core::device::Device;
use flame_core::{DType, Result, Shape, Tensor};
use std::collections::HashMap;
use std::path::Path;

// flux_vae.rs
// Flux VAE wrapper with proper configuration and CPU offloading

/// Flux-specific VAE wrapper
pub struct AutoencoderKL {
    inner: BaseVAE, // Use BaseVAE to avoid alignment issues
    scale_factor: f64,
    shift_factor: f64,
    device: Device,
    offload_manager: Option<CPUOffloadManager>,
}

impl AutoencoderKL {
    /// Create a new Flux VAE from weights
    pub fn new(wl: &WeightLoader, device: Device, enable_offloading: bool) -> Result<Self> {
        // Flux VAE config with 16 latent channels instead of 4
        let config = VAEConfig {
            block_out_channels: vec![128, 256, 512, 512],
            layers_per_block: 2,
            latent_channels: 16, // Flux uses 16 channels
            norm_num_groups: 32,
            use_quant_conv: false,      // Flux VAE doesn't use quant conv
            use_post_quant_conv: false, // Flux VAE doesn't use post quant conv
        };

        // DON'T remap weights - pass them directly with original keys
        let weights = wl.weights.clone();

        // Debug output disabled for performance
        // println!("Available weight keys (showing first 20):");
        // for key in weights.keys().take(20) {
        //     println!("  {}", key);
        // }
        // println!("Total weight keys: {}", weights.len());

        let inner = BaseVAE::new(config, device.clone(), weights)?;

        let offload_manager =
            if enable_offloading { Some(CPUOffloadManager::new(device.clone(), 4)?) } else { None };

        Ok(Self {
            inner,
            scale_factor: 0.13025, // Flux-specific scaling
            shift_factor: 0.0,     // No shift for standard Flux
            device,
            offload_manager,
        })
    }

    /// Encode image to latents
    pub fn encode(&self, x: &Tensor) -> Result<Tensor> {
        let shape = x.shape();
        let dims = shape.dims();
        let height = dims[2];
        let width = dims[3];

        // With BF16 and cuDNN, we can handle larger images directly
        // Only use tiling for very large images or if direct encoding fails
        if height > 1024 || width > 1024 {
            eprintln!("VAE: Using tiled encoding for {}x{} image", height, width);
            self.encode_tiled(x)
        } else {
            // Try direct encoding first for 1024x1024 and smaller
            eprintln!("VAE: Attempting direct encoding for {}x{} image with BF16", height, width);
            match self.encode_direct(x) {
                Ok(result) => {
                    eprintln!("VAE: Direct encoding successful!");
                    Ok(result)
                }
                Err(e) => {
                    eprintln!(
                        "VAE: Direct encoding failed ({}), falling back to tiled encoding",
                        e
                    );
                    self.encode_tiled(x)
                }
            }
        }
    }

    /// Direct encoding without tiling (for small images)
    fn encode_direct(&self, x: &Tensor) -> Result<Tensor> {
        // Keep using BF16 to save memory
        let x_bf16 = if x.dtype() != DType::BF16 { x.to_dtype(DType::BF16)? } else { x.clone() };

        // x should be in range [0, 1], convert to [-1, 1]
        let x_normalized = x_bf16.mul_scalar(2.0 as f32)?.add_scalar(-1.0)?;

        // Encode using the inner autoencoder
        let dist = self.inner.encode(&x_normalized)?;
        // Use mode() instead of sample() to avoid random tensor creation issues
        // This returns the mean without adding noise, which is fine for training
        let latents = dist.mode()?;

        // Apply scaling
        let scaled =
            latents.mul_scalar(self.scale_factor as f32)?.add_scalar(self.shift_factor as f32)?;

        // Convert back to F32 for saving if needed
        let scaled_f32 =
            if scaled.dtype() != DType::F32 { scaled.to_dtype(DType::F32)? } else { scaled };

        Ok(scaled_f32)
    }

    /// Tiled encoding for large images to avoid memory issues
    fn encode_tiled(&self, x: &Tensor) -> Result<Tensor> {
        eprintln!("Using tiled VAE encoding to avoid memory issues");

        let shape = x.shape();
        let dims = shape.dims();
        let batch_size = dims[0];
        let channels = dims[1];
        let height = dims[2];
        let width = dims[3];

        // Tile configuration - NO OVERLAP for exact tiling!
        let (tile_size, overlap) = if height == 1024 && width == 1024 {
            // For exact 1024x1024, use 512x512 with NO overlap = exactly 4 tiles
            (512, 0)
        } else if height >= 1024 || width >= 1024 {
            // For other large sizes, use 512x512 with minimal overlap
            (512, 32)
        } else if height >= 768 || width >= 768 {
            // For 768x768, try to fit in one tile if possible
            (768, 0)
        } else {
            // Small images - encode directly without tiling
            (height.max(width), 0)
        };

        let stride = if overlap == 0 { tile_size } else { tile_size - overlap };

        eprintln!(
            "  Tile size: {}x{}, Overlap: {}, Stride: {}",
            tile_size, tile_size, overlap, stride
        );

        // Calculate number of tiles
        let n_tiles_h = (height - overlap + stride - 1) / stride;
        let n_tiles_w = (width - overlap + stride - 1) / stride;

        // Latent dimensions (VAE downscales by 8x)
        let latent_height = height / 8;
        let latent_width = width / 8;
        let latent_channels = 16; // Flux VAE has 16 latent channels

        // Initialize output tensor - accumulate on CPU to avoid GPU memory issues
        let mut output_cpu =
            vec![0.0f32; batch_size * latent_channels * latent_height * latent_width];
        let mut weights_cpu = vec![0.0f32; batch_size * latent_height * latent_width];

        // Process each tile
        for tile_y in 0..n_tiles_h {
            for tile_x in 0..n_tiles_w {
                eprintln!(
                    "  Processing tile {}/{} (y={}, x={})",
                    tile_y * n_tiles_w + tile_x + 1,
                    n_tiles_h * n_tiles_w,
                    tile_y,
                    tile_x
                );

                // Calculate tile boundaries
                let y_start = tile_y * stride;
                let x_start = tile_x * stride;
                let y_end = (y_start + tile_size).min(height);
                let x_end = (x_start + tile_size).min(width);

                let actual_tile_h = y_end - y_start;
                let actual_tile_w = x_end - x_start;

                // Extract tile using narrow operations
                let tile =
                    x.narrow(2, y_start, actual_tile_h)?.narrow(3, x_start, actual_tile_w)?;

                // Encode this tile
                eprintln!("    Encoding {}x{} tile...", actual_tile_h, actual_tile_w);
                let latent_tile = self.encode_direct(&tile)?;
                eprintln!("    Tile encoded successfully");

                // Calculate latent tile position
                let latent_y_start = y_start / 8;
                let latent_x_start = x_start / 8;
                let latent_tile_h = actual_tile_h / 8;
                let latent_tile_w = actual_tile_w / 8;

                // Get tile data to CPU
                let tile_data = latent_tile.to_vec()?;

                // Create blending weights (linear blend at edges)
                let mut tile_weights = vec![1.0f32; latent_tile_h * latent_tile_w];

                // Apply linear blending at overlap regions
                let overlap_latent = overlap / 8;
                for y in 0..latent_tile_h {
                    for x in 0..latent_tile_w {
                        let mut weight = 1.0f32;

                        // Blend at edges if this is an overlap region
                        if tile_y > 0 && y < overlap_latent {
                            weight *= y as f32 / overlap_latent as f32;
                        }
                        if tile_x > 0 && x < overlap_latent {
                            weight *= x as f32 / overlap_latent as f32;
                        }
                        if tile_y < n_tiles_h - 1 && y >= latent_tile_h - overlap_latent {
                            weight *= (latent_tile_h - y - 1) as f32 / overlap_latent as f32;
                        }
                        if tile_x < n_tiles_w - 1 && x >= latent_tile_w - overlap_latent {
                            weight *= (latent_tile_w - x - 1) as f32 / overlap_latent as f32;
                        }

                        tile_weights[y * latent_tile_w + x] = weight;
                    }
                }

                // Accumulate weighted tile into output
                for b in 0..batch_size {
                    for c in 0..latent_channels {
                        for y in 0..latent_tile_h {
                            for x in 0..latent_tile_w {
                                let out_y = latent_y_start + y;
                                let out_x = latent_x_start + x;

                                let tile_idx = ((b * latent_channels + c) * latent_tile_h + y)
                                    * latent_tile_w
                                    + x;
                                let out_idx = ((b * latent_channels + c) * latent_height + out_y)
                                    * latent_width
                                    + out_x;
                                let weight_idx = (b * latent_height + out_y) * latent_width + out_x;

                                let weight = tile_weights[y * latent_tile_w + x];
                                output_cpu[out_idx] += tile_data[tile_idx] * weight;

                                if c == 0 {
                                    // Only accumulate weights once per pixel
                                    weights_cpu[weight_idx] += weight;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Normalize by weights
        for b in 0..batch_size {
            for c in 0..latent_channels {
                for y in 0..latent_height {
                    for x in 0..latent_width {
                        let idx =
                            ((b * latent_channels + c) * latent_height + y) * latent_width + x;
                        let weight_idx = (b * latent_height + y) * latent_width + x;

                        if weights_cpu[weight_idx] > 0.0 {
                            output_cpu[idx] /= weights_cpu[weight_idx];
                        }
                    }
                }
            }
        }

        // Convert back to tensor on GPU
        let output = Tensor::from_vec(
            output_cpu,
            Shape::from_dims(&[batch_size, latent_channels, latent_height, latent_width]),
            self.device.cuda_device_arc(),
        )?;

        Ok(output)
    }

    /// Decode latents to image
    pub fn decode(&self, z: &Tensor) -> Result<Tensor> {
        // Remove scaling
        let z = z.add_scalar(-(self.shift_factor as f32))?.div_scalar(self.scale_factor as f32)?;

        // Decode
        let decoded = self.inner.decode(&z)?;

        // Convert from [-1, 1] to [0, 1]
        Ok(decoded.add_scalar(1.0 as f32)?.mul_scalar(0.5)?)
    }
}

#[derive(Debug, Clone)]
pub struct AutoencoderKLConfig {
    pub block_out_channels: Vec<usize>,
    pub layers_per_block: usize,
    pub latent_channels: usize,
    pub norm_num_groups: usize,
    pub scaling_factor: f64,
}

impl Default for AutoencoderKLConfig {
    fn default() -> Self {
        Self {
            block_out_channels: vec![128, 256, 512, 512],
            layers_per_block: 2,
            latent_channels: 16,
            norm_num_groups: 32,
            scaling_factor: 0.13025, // Flux scaling
        }
    }
}

/// Load Flux VAE from ae.safetensors
// Alias for compatibility
pub type AutoEncoder = AutoencoderKL;
pub type AutoEncoderConfig = AutoencoderKLConfig;

/// Load Flux VAE from ae.safetensors
pub fn load_flux_vae(
    vae_path: &Path,
    device: Device,
    enable_offloading: bool,
) -> Result<AutoencoderKL> {
    // println!("Loading Flux VAE from: {:?}", vae_path);

    // Use BF16 for VAE to save memory (like SimpleTuner)
    let wl = WeightLoader::from_safetensors_with_dtype(vae_path, device.clone(), DType::BF16)?;

    AutoencoderKL::new(&wl, device, enable_offloading)
}

// Example usage
pub fn decode_latents(vae: &AutoencoderKL, latents: &Tensor) -> Result<Tensor> {
    // Latents should be [batch, 16, height/8, width/8]
    let images = vae.decode(latents)?;

    // Output is [batch, 3, height, width] in range [0, 1]
    Ok(images)
}
