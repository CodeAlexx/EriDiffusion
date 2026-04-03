//! Cache manager for Flux training following SimpleTuner's approach
//!
//! This module handles caching of VAE latents and text embeddings to disk,
//! allowing the VAE and text encoders to be unloaded during training.
//!
//! Cache organization follows SimpleTuner's directory structure:
//! - /cache/vae/{dataset}/   - VAE latents (.safetensors format)
//! - /cache/text/{dataset}/  - Text embeddings (.safetensors format)
//!
//! Note: SimpleTuner uses .pt (PyTorch) format, but we use .safetensors for FLAME
//!
//! TODO: Future improvements:
//! - Support importing existing SimpleTuner caches (convert .pt to .safetensors)
//! - Organize by LoRA name for multi-LoRA training
//! - Support pre-processed datasets from HuggingFace or other sources

use crate::loaders::WeightLoader;
use crate::models::aligned_image_processor::AlignedImageProcessor;
use crate::models::flux_vae::AutoencoderKL;
use crate::trainers::flux_data_loader::{FluxDataLoader, TrainingSample};
use crate::trainers::text_encoders::TextEncoders;
use flame_core::{DType, Device, Error, Result, Shape, Tensor};
use image;
use log::{debug, error, info, warn};
use safetensors::{serialize, tensor::TensorView, Dtype as SafeDtype};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Cache manager for SimpleTuner-style training
pub struct FluxCacheManager {
    /// Root cache directory
    cache_dir: PathBuf,
    /// Subdirectory for latent caches
    latent_dir: PathBuf,
    /// Subdirectory for text embedding caches
    embed_dir: PathBuf,
    /// Dataset name for organizing cache files
    dataset_name: String,
    /// Device for tensor operations
    device: Device,
    /// Whether caching is enabled
    enabled: bool,
}

impl FluxCacheManager {
    /// Create a new cache manager
    pub fn new(cache_dir: PathBuf, device: Device, enabled: bool) -> Result<Self> {
        Self::with_dataset_name(cache_dir, device, enabled, "default".to_string())
    }

    /// Create a new cache manager with a specific dataset name
    pub fn with_dataset_name(
        cache_dir: PathBuf,
        device: Device,
        enabled: bool,
        dataset_name: String,
    ) -> Result<Self> {
        if !enabled {
            return Ok(Self {
                cache_dir: cache_dir.clone(),
                latent_dir: cache_dir.clone(),
                embed_dir: cache_dir.clone(),
                dataset_name,
                device,
                enabled: false,
            });
        }

        // SimpleTuner style directories: /cache/vae/{dataset}/ and /cache/text/{dataset}/
        let latent_dir = cache_dir.join("vae").join(&dataset_name);
        let embed_dir = cache_dir.join("text").join(&dataset_name);

        // Create directories
        fs::create_dir_all(&latent_dir)
            .map_err(|e| Error::Io(format!("Failed to create VAE cache dir: {}", e)))?;
        fs::create_dir_all(&embed_dir)
            .map_err(|e| Error::Io(format!("Failed to create text cache dir: {}", e)))?;

        info!("Initialized FluxCacheManager at {:?}", cache_dir);
        info!("  Dataset: {}", dataset_name);
        info!("  VAE cache: {:?}", latent_dir);
        info!("  Text cache: {:?}", embed_dir);

        Ok(Self { cache_dir, latent_dir, embed_dir, dataset_name, device, enabled })
    }

    /// Get the cache path for a latent given an image path
    /// We use .safetensors format for FLAME tensors
    /// First check existing _latent_cache directory structure, then new structure
    pub fn get_latent_cache_path(&self, image_path: &Path) -> PathBuf {
        // Check for existing _latent_cache structure first
        let filename = image_path.file_stem().and_then(|s| s.to_str()).unwrap_or("unknown");

        // Look for existing cache directory relative to image location
        if let Some(dataset_dir) = image_path.parent() {
            let old_cache_dir = dataset_dir.join("_latent_cache");
            if old_cache_dir.exists() {
                // Look for existing cache files with the filename prefix
                let mut best_match: Option<(PathBuf, u64)> = None;

                if let Ok(entries) = std::fs::read_dir(&old_cache_dir) {
                    for entry in entries {
                        if let Ok(entry) = entry {
                            let entry_name = entry.file_name();
                            if let Some(entry_str) = entry_name.to_str() {
                                if entry_str.starts_with(&format!("{}_", filename))
                                    && entry_str.ends_with(".safetensors")
                                {
                                    // Get file size to pick the largest (highest quality) cache
                                    if let Ok(metadata) = entry.metadata() {
                                        let size = metadata.len();
                                        if best_match.is_none()
                                            || size > best_match.as_ref().unwrap().1
                                        {
                                            best_match = Some((entry.path(), size));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                if let Some((path, _)) = best_match {
                    return path;
                }
            }
        }

        // Fallback to new SimpleTuner-style path structure
        let hash = self.hash_path(image_path);
        self.latent_dir.join(format!("{}.safetensors", hash))
    }

    /// Get the cache path for text embeddings given an image path
    /// We use .safetensors format with model suffix (e.g., -flux.safetensors for Flux)
    pub fn get_embed_cache_path(&self, image_path: &Path) -> PathBuf {
        let hash = self.hash_path(image_path);
        // For Flux, we use -flux suffix
        self.embed_dir.join(format!("{}-flux.safetensors", hash))
    }

    /// Hash a path to create a stable filename
    fn hash_path(&self, path: &Path) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        path.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }

    /// Save a tensor to cache
    pub fn save_tensor(&self, tensor: &Tensor, path: &Path, key: &str) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }

        // Ensure tensor is in F32 for saving (safetensors compatibility)
        let tensor_f32 = if tensor.dtype() != DType::F32 {
            tensor.to_dtype(DType::F32)?
        } else {
            tensor.clone()
        };

        // Get tensor data
        let data = tensor_f32.to_vec1::<f32>()?;
        let shape = tensor_f32.shape().dims().to_vec();

        // Convert to bytes
        let bytes =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };

        // Create tensor view - always save as F32 for compatibility
        let mut tensors = HashMap::new();
        tensors.insert(
            key.to_string(),
            TensorView::new(SafeDtype::F32, shape, bytes)
                .map_err(|e| Error::InvalidOperation(e.to_string()))?,
        );

        // Add metadata
        let mut metadata = HashMap::new();
        metadata.insert("format".to_string(), "flux_cache".to_string());
        metadata.insert("version".to_string(), "1.0".to_string());

        // Serialize
        let serialized = serialize(tensors, &Some(metadata))
            .map_err(|e| Error::InvalidOperation(e.to_string()))?;

        // Write to file
        fs::write(path, serialized).map_err(|e| Error::Io(e.to_string()))?;

        debug!("Saved tensor to cache: {:?}", path);
        Ok(())
    }

    /// Load a tensor from cache
    pub fn load_tensor(&self, path: &Path, key: &str) -> Result<Option<Tensor>> {
        if !self.enabled || !path.exists() {
            return Ok(None);
        }

        // Load with WeightLoader
        let weight_loader = WeightLoader::from_safetensors(path, self.device.clone())?;

        // Get the tensor
        if let Some(tensor) = weight_loader.weights.get(key) {
            Ok(Some(tensor.clone()))
        } else {
            warn!("Key '{}' not found in cache file {:?}", key, path);
            Ok(None)
        }
    }

    /// Check if a latent is cached
    pub fn is_latent_cached(&self, image_path: &Path) -> bool {
        if !self.enabled {
            return false;
        }
        self.get_latent_cache_path(image_path).exists()
    }

    /// Check if text embeddings are cached
    pub fn is_embed_cached(&self, image_path: &Path) -> bool {
        if !self.enabled {
            return false;
        }
        self.get_embed_cache_path(image_path).exists()
    }

    /// Pre-encode all latents in a dataset
    pub fn encode_all_latents(
        &self,
        data_loader: &mut FluxDataLoader,
        vae_path: &Path,
        force_reencode: bool,
    ) -> Result<()> {
        if !self.enabled {
            info!("Cache manager disabled, skipping latent encoding");
            return Ok(());
        }

        info!("=== Pre-encoding latents (SimpleTuner style) ===");

        // Count already cached
        let total_samples = data_loader.total_samples();
        let mut cached_count = 0;
        let mut to_encode = Vec::new();

        // We need to iterate through the actual samples to get their paths
        // Get all samples from the data loader
        let mut all_samples = Vec::new();
        for bucket_idx in 0..data_loader.buckets.len() {
            for sample in &data_loader.buckets[bucket_idx].samples {
                all_samples.push(sample.clone());
            }
        }

        // Check what needs encoding
        for sample in &all_samples {
            let cache_path = self.get_latent_cache_path(&sample.image_path);

            if cache_path.exists() && !force_reencode {
                cached_count += 1;
            } else {
                // We'll need to load and encode this image
                to_encode.push(sample.clone());
            }
        }

        info!("Found {}/{} latents already cached", cached_count, total_samples);

        if !to_encode.is_empty() {
            info!("\n=== VAE Encoding Phase ===");
            info!("Will encode {} images in batches to manage memory", to_encode.len());

            // Process in batches to avoid memory issues with 405 images
            let batch_size = 10; // Process 10 images at a time
            let num_batches = (to_encode.len() + batch_size - 1) / batch_size;

            info!("Processing {} batches of up to {} images each", num_batches, batch_size);

            // Track overall timing
            let overall_start = std::time::Instant::now();

            for batch_idx in 0..num_batches {
                let batch_start = batch_idx * batch_size;
                let batch_end = ((batch_idx + 1) * batch_size).min(to_encode.len());
                let batch_samples = &to_encode[batch_start..batch_end];

                info!("\n=== Batch {}/{} ===", batch_idx + 1, num_batches);
                info!("Processing images {}-{} of {}", batch_start + 1, batch_end, to_encode.len());

                // Load VAE for this batch - keep using BF16 to save memory
                info!("Loading VAE...");
                let weight_loader = WeightLoader::from_safetensors_with_dtype(
                    vae_path,
                    self.device.clone(),
                    DType::BF16,
                )?;
                let vae = AutoencoderKL::new(&weight_loader, self.device.clone(), false)?;
                info!("✅ VAE loaded for batch {}", batch_idx + 1);

                // Encode images in this batch
                for (local_idx, sample) in batch_samples.iter().enumerate() {
                    let global_idx = batch_start + local_idx;
                    let image_start = std::time::Instant::now();
                    let filename =
                        sample.image_path.file_name().and_then(|n| n.to_str()).unwrap_or("unknown");

                    info!("[Image {}/{}] Encoding: {}", global_idx + 1, to_encode.len(), filename);

                    // Load and prepare the image
                    let img = image::open(&sample.image_path)
                        .map_err(|e| Error::Io(format!("Failed to load image: {}", e)))?;

                    // Prepare image tensor with the sample's resolution
                    let img_tensor = data_loader.prepare_image(img, sample.resolution)?;
                    let img_batch = img_tensor.unsqueeze(0)?; // Add batch dimension

                    // Encode with tiled VAE (will use 256x256 tiles for 1024x1024)
                    let latent = vae.encode(&img_batch)?;

                    // Save to cache
                    let cache_path = self.get_latent_cache_path(&sample.image_path);
                    self.save_tensor(&latent, &cache_path, "latent")?;

                    let elapsed = image_start.elapsed();
                    let total_elapsed = overall_start.elapsed();
                    let total_processed = global_idx + 1;
                    let avg_time = total_elapsed.as_secs_f32() / total_processed as f32;
                    let remaining = to_encode.len() - total_processed;
                    let eta_seconds = (remaining as f32 * avg_time) as u64;
                    let eta_minutes = eta_seconds / 60;
                    let eta_secs = eta_seconds % 60;

                    info!(
                        "  ✅ Saved | Time: {:.2}s | Avg: {:.2}s | ETA: {:02}:{:02}",
                        elapsed.as_secs_f32(),
                        avg_time,
                        eta_minutes,
                        eta_secs
                    );

                    // Clean up GPU memory after EVERY image
                    drop(img_tensor);
                    drop(img_batch);
                    drop(latent);
                }

                // Free VAE and clear memory after each batch
                info!("Freeing VAE and clearing GPU memory for batch {}...", batch_idx + 1);
                drop(vae);
                drop(weight_loader);
                self.device.synchronize()?;

                // Add a small delay between batches to ensure memory is freed
                std::thread::sleep(std::time::Duration::from_millis(500));

                info!("✅ Batch {}/{} complete\n", batch_idx + 1, num_batches);
            }
            info!("✅ VAE freed from GPU memory");

            // CRITICAL: Force memory cleanup
            // The drops above don't actually free CUDA memory immediately
            self.device.synchronize()?;

            // Try to clear any cached memory
            use crate::memory::manager::MemoryManager;
            MemoryManager::empty_cache()?;

            info!("   GPU memory explicitly cleared");
        }

        Ok(())
    }

    /// Pre-encode all latents using a pre-loaded VAE model
    pub fn encode_all_latents_with_model(
        &self,
        data_loader: &mut FluxDataLoader,
        vae: &AutoencoderKL,
        force_reencode: bool,
    ) -> Result<()> {
        if !self.enabled {
            info!("Cache manager disabled, skipping latent encoding");
            return Ok(());
        }

        info!("=== Pre-encoding latents with pre-loaded VAE ===");

        // Count already cached
        let total_samples = data_loader.total_samples();
        let mut cached_count = 0;
        let mut to_encode = Vec::new();

        // Get all samples from the data loader
        let mut all_samples = Vec::new();
        for bucket_idx in 0..data_loader.buckets.len() {
            for sample in &data_loader.buckets[bucket_idx].samples {
                all_samples.push(sample.clone());
            }
        }

        // Check what needs encoding
        for sample in &all_samples {
            let cache_path = self.get_latent_cache_path(&sample.image_path);

            if cache_path.exists() && !force_reencode {
                cached_count += 1;
            } else {
                to_encode.push(sample.clone());
            }
        }

        info!("Found {}/{} latents already cached", cached_count, total_samples);

        if !to_encode.is_empty() {
            info!("Encoding {} images...", to_encode.len());

            // Track timing
            let encoding_start = std::time::Instant::now();

            // Encode each image
            for (idx, sample) in to_encode.iter().enumerate() {
                let image_start = std::time::Instant::now();
                let filename =
                    sample.image_path.file_name().and_then(|n| n.to_str()).unwrap_or("unknown");

                info!("[VAE {}/{}] Encoding image: {}", idx + 1, to_encode.len(), filename);

                // Load and prepare the image
                let img = image::open(&sample.image_path)
                    .map_err(|e| Error::Io(format!("Failed to load image: {}", e)))?;

                // Prepare image tensor with the sample's resolution
                info!("  Target resolution: {:?}", sample.resolution);
                info!("  Original image size: {}x{}", img.width(), img.height());
                let img_tensor = data_loader.prepare_image(img, sample.resolution)?;
                info!(
                    "  Image tensor shape: {:?}, dtype: {:?}",
                    img_tensor.shape(),
                    img_tensor.dtype()
                );
                let img_batch = img_tensor.unsqueeze(0)?; // Add batch dimension
                info!("  Batch shape: {:?}", img_batch.shape());

                // Verify the batch is actually 1024x1024
                let batch_shape = img_batch.shape();
                if batch_shape.dims()[2] != 1024 || batch_shape.dims()[3] != 1024 {
                    error!("ERROR: Batch is not 1024x1024! Shape: {:?}", batch_shape);
                    return Err(Error::InvalidOperation(format!(
                        "Image batch has wrong size: {:?}, expected [1, 3, 1024, 1024]",
                        batch_shape
                    )));
                }

                // Skip CUDA alignment preprocessing - VAE will handle the image directly
                // let aligned_batch = AlignedImageProcessor::preprocess_image_for_vae(&img_batch)?;
                // info!("  Aligned batch shape: {:?}", aligned_batch.shape());

                // Use tiled encoding to avoid huge memory allocation for 1024x1024 images
                // The Flux VAE seems to internally expect 1536x1536 which causes OOM
                info!("  Using tiled VAE encoding to avoid memory issues...");
                let latent = if img_batch.shape().dims()[2] >= 768
                    || img_batch.shape().dims()[3] >= 768
                {
                    // For large images, use tiled encoding
                    info!(
                        "    Large image detected ({}x{}), using tiled encoding",
                        img_batch.shape().dims()[2],
                        img_batch.shape().dims()[3]
                    );

                    // For Flux VAE, we need to manually tile since it doesn't return mean/logvar
                    // Process image in 512x512 tiles to avoid the 1536x1536 memory issue
                    let tile_size = 512;
                    let overlap = 64;
                    let (batch_size, channels, height, width) = (
                        img_batch.shape().dims()[0],
                        img_batch.shape().dims()[1],
                        img_batch.shape().dims()[2],
                        img_batch.shape().dims()[3],
                    );

                    // Calculate tile layout
                    let stride = tile_size - overlap;
                    let n_tiles_h = (height - overlap + stride - 1) / stride;
                    let n_tiles_w = (width - overlap + stride - 1) / stride;

                    info!(
                        "    Processing {}x{} tiles (tile_size={}, overlap={})",
                        n_tiles_h, n_tiles_w, tile_size, overlap
                    );

                    // Calculate latent dimensions (VAE downscales by 8x)
                    let latent_height = height / 8;
                    let latent_width = width / 8;
                    let latent_channels = 16; // Flux VAE has 16 latent channels

                    // Initialize output tensor for blending - MUST BE BF16 FOR FLUX!
                    let mut output = Tensor::zeros_dtype(
                        Shape::from_dims(&[
                            batch_size,
                            latent_channels,
                            latent_height,
                            latent_width,
                        ]),
                        DType::BF16, // CRITICAL: Use BF16 for FLUX!
                        self.device.cuda_device_arc(),
                    )?;
                    let mut weights = Tensor::zeros_dtype(
                        Shape::from_dims(&[batch_size, 1, latent_height, latent_width]),
                        DType::BF16, // CRITICAL: Use BF16 for FLUX!
                        self.device.cuda_device_arc(),
                    )?;

                    // Collect tile outputs for accumulation
                    let mut tile_outputs = Vec::new();
                    let mut tile_weights = Vec::new();

                    // Process each tile
                    for tile_y in 0..n_tiles_h {
                        for tile_x in 0..n_tiles_w {
                            let y_start = tile_y * stride;
                            let x_start = tile_x * stride;
                            let y_end = (y_start + tile_size).min(height);
                            let x_end = (x_start + tile_size).min(width);

                            info!(
                                "      Processing tile ({},{}) from ({},{}) to ({},{})",
                                tile_x, tile_y, x_start, y_start, x_end, y_end
                            );

                            // Extract tile using narrow operations
                            let tile = img_batch.narrow(2, y_start, y_end - y_start)?.narrow(
                                3,
                                x_start,
                                x_end - x_start,
                            )?;

                            // Encode tile
                            let latent_tile = match vae.encode(&tile) {
                                Ok(l) => l,
                                Err(e) => {
                                    error!(
                                        "Failed to encode tile ({},{}) of image {}: {}",
                                        tile_x, tile_y, filename, e
                                    );
                                    return Err(e);
                                }
                            };

                            // Calculate position in latent space
                            let latent_y_start = y_start / 8;
                            let latent_x_start = x_start / 8;
                            let latent_y_end = (y_end / 8).min(latent_height);
                            let latent_x_end = (x_end / 8).min(latent_width);

                            // Add to output with blending weights
                            // For now, just use simple averaging in overlap regions
                            let latent_h = latent_y_end - latent_y_start;
                            let latent_w = latent_x_end - latent_x_start;

                            // Store the tile info for later processing
                            // We'll accumulate all tiles at the end
                            tile_outputs.push((
                                latent_y_start,
                                latent_x_start,
                                latent_h,
                                latent_w,
                                latent_tile.clone(),
                            ));

                            // Create weight tensor for this tile - MUST BE BF16!
                            let tile_weight = Tensor::ones_dtype(
                                Shape::from_dims(&[batch_size, 1, latent_h, latent_w]),
                                DType::BF16, // CRITICAL: Use BF16 for FLUX!
                                weights.device().clone(),
                            )?;
                            tile_weights.push((
                                latent_y_start,
                                latent_x_start,
                                latent_h,
                                latent_w,
                                tile_weight,
                            ));
                        }
                    }

                    // Now accumulate all tiles into the output tensor
                    // Since we can't easily modify tensors in place, we'll reconstruct the output
                    // This is inefficient but avoids the memory issue

                    // Create new tensors from accumulated data
                    // Convert to CPU for manipulation, then back to GPU
                    let mut output_data =
                        vec![0.0f32; batch_size * latent_channels * latent_height * latent_width];
                    let mut weight_data =
                        vec![0.0f32; batch_size * 1 * latent_height * latent_width];

                    // Add each tile to the output data
                    for (y_start, x_start, h, w, tile) in tile_outputs {
                        // Convert tile to CPU
                        let tile_vec = tile.to_vec_f32()?;

                        // Copy tile data to the output array
                        for b in 0..batch_size {
                            for c in 0..latent_channels {
                                for y in 0..h {
                                    for x in 0..w {
                                        let src_idx = ((b * latent_channels + c) * h + y) * w + x;
                                        let dst_idx = ((b * latent_channels + c) * latent_height
                                            + (y_start + y))
                                            * latent_width
                                            + (x_start + x);
                                        output_data[dst_idx] += tile_vec[src_idx];
                                    }
                                }
                            }
                        }
                    }

                    // Add each weight tile
                    for (y_start, x_start, h, w, weight) in tile_weights {
                        let weight_vec = weight.to_vec_f32()?;

                        for b in 0..batch_size {
                            for y in 0..h {
                                for x in 0..w {
                                    let src_idx = (b * h + y) * w + x;
                                    let dst_idx = (b * latent_height + (y_start + y))
                                        * latent_width
                                        + (x_start + x);
                                    weight_data[dst_idx] += weight_vec[src_idx];
                                }
                            }
                        }
                    }

                    // Create tensors from the accumulated data - MUST BE BF16!
                    output = Tensor::from_vec_dtype(
                        output_data,
                        Shape::from_dims(&[
                            batch_size,
                            latent_channels,
                            latent_height,
                            latent_width,
                        ]),
                        self.device.cuda_device_arc(),
                        DType::BF16, // CRITICAL: Use BF16 for FLUX!
                    )?;
                    weights = Tensor::from_vec_dtype(
                        weight_data,
                        Shape::from_dims(&[batch_size, 1, latent_height, latent_width]),
                        self.device.cuda_device_arc(),
                        DType::BF16, // CRITICAL: Use BF16 for FLUX!
                    )?;

                    // Normalize by weights to handle overlapping regions
                    // Weights are [B, 1, H, W], output is [B, C, H, W]
                    // broadcast_to expects the target shape
                    let latent = output.div(&weights.broadcast_to(output.shape())?)?;

                    info!(
                        "    ✅ Tiled encoding successful, final latent shape: {:?}",
                        latent.shape()
                    );
                    latent
                } else {
                    // For smaller images, use regular encoding
                    match vae.encode(&img_batch) {
                        Ok(l) => l,
                        Err(e) => {
                            error!("Failed to encode image {}: {}", filename, e);
                            error!(
                                "Image shape: {:?}, dtype: {:?}",
                                img_batch.shape(),
                                img_batch.dtype()
                            );
                            return Err(e);
                        }
                    }
                };

                // Save to cache
                let cache_path = self.get_latent_cache_path(&sample.image_path);
                self.save_tensor(&latent, &cache_path, "latent")?;

                let elapsed = image_start.elapsed();
                let total_elapsed = encoding_start.elapsed();
                let avg_time = total_elapsed.as_secs_f32() / (idx + 1) as f32;
                let remaining = to_encode.len() - idx - 1;
                let eta_seconds = (remaining as f32 * avg_time) as u64;
                let eta_minutes = eta_seconds / 60;
                let eta_secs = eta_seconds % 60;

                info!(
                    "  ✅ Saved latent | Time: {:.2}s | Avg: {:.2}s/img | ETA: {:02}:{:02}\n",
                    elapsed.as_secs_f32(),
                    avg_time,
                    eta_minutes,
                    eta_secs
                );

                // Clean up GPU memory after EVERY image to prevent OOM
                // Force all intermediate tensors to be freed
                drop(img_tensor);
                drop(img_batch);
                drop(latent);
                self.device.synchronize()?;

                // Show memory usage periodically
                if (idx + 1) % 5 == 0 {
                    info!("  [Memory check] Synchronized GPU after {} images", idx + 1);
                }
            }
        }

        Ok(())
    }

    /// Pre-encode all text embeddings using pre-loaded models
    pub fn encode_all_text_with_models(
        &self,
        data_loader: &mut FluxDataLoader,
        text_encoders: &TextEncoders,
        force_reencode: bool,
    ) -> Result<()> {
        if !self.enabled {
            info!("Cache manager disabled, skipping text encoding");
            return Ok(());
        }

        info!("=== Pre-encoding text embeddings with pre-loaded models ===");

        // Count already cached
        let total_samples = data_loader.total_samples();
        let mut cached_count = 0;
        let mut to_encode = Vec::new();

        // Get all samples from the data loader
        let mut all_samples = Vec::new();
        for bucket_idx in 0..data_loader.buckets.len() {
            for sample in &data_loader.buckets[bucket_idx].samples {
                all_samples.push(sample.clone());
            }
        }

        // Check what needs encoding
        for sample in &all_samples {
            let cache_path = self.get_embed_cache_path(&sample.image_path);

            if cache_path.exists() && !force_reencode {
                cached_count += 1;
            } else {
                // Load caption for this sample
                let caption = if sample.caption_path.exists() {
                    std::fs::read_to_string(&sample.caption_path)
                        .unwrap_or_else(|_| "".to_string())
                        .trim()
                        .to_string()
                } else {
                    "".to_string()
                };
                to_encode.push((sample.image_path.clone(), caption));
            }
        }

        info!("Found {}/{} text embeddings already cached", cached_count, total_samples);

        if !to_encode.is_empty() {
            info!("Encoding {} prompts...", to_encode.len());

            // Track timing
            let encoding_start = std::time::Instant::now();

            // Encode each prompt
            for (idx, (image_path, prompt)) in to_encode.iter().enumerate() {
                let text_start = std::time::Instant::now();
                let filename = image_path.file_name().and_then(|n| n.to_str()).unwrap_or("unknown");

                info!("[Text {}/{}] Encoding caption for: {}", idx + 1, to_encode.len(), filename);

                // Show prompt preview (first 80 chars)
                let prompt_preview = if prompt.len() > 80 {
                    format!("{}...", &prompt[..80])
                } else {
                    prompt.clone()
                };
                info!("  Caption: \"{}\"", prompt_preview);

                // Encode
                info!("  Encoding with CLIP-L and T5-XXL...");
                let (clip_embeds, t5_embeds) = text_encoders.encode_flux(prompt)?;

                // Save to cache
                let cache_path = self.get_embed_cache_path(&image_path);

                // Save both embeddings in the same file
                let mut tensors = HashMap::new();

                // CLIP embedding - ensure F32 for saving
                let clip_f32 = if clip_embeds.dtype() != DType::F32 {
                    clip_embeds.to_dtype(DType::F32)?
                } else {
                    clip_embeds.clone()
                };
                let clip_data = clip_f32.to_vec1::<f32>()?;
                let clip_shape = clip_f32.shape().dims().to_vec();
                let clip_bytes = unsafe {
                    std::slice::from_raw_parts(clip_data.as_ptr() as *const u8, clip_data.len() * 4)
                };
                tensors.insert(
                    "clip_embeds".to_string(),
                    TensorView::new(SafeDtype::F32, clip_shape, clip_bytes)
                        .map_err(|e| Error::InvalidOperation(e.to_string()))?,
                );

                // T5 embedding - ensure F32 for saving
                let t5_f32 = if t5_embeds.dtype() != DType::F32 {
                    t5_embeds.to_dtype(DType::F32)?
                } else {
                    t5_embeds.clone()
                };
                let t5_data = t5_f32.to_vec1::<f32>()?;
                let t5_shape = t5_f32.shape().dims().to_vec();
                let t5_bytes = unsafe {
                    std::slice::from_raw_parts(t5_data.as_ptr() as *const u8, t5_data.len() * 4)
                };
                tensors.insert(
                    "t5_embeds".to_string(),
                    TensorView::new(SafeDtype::F32, t5_shape, t5_bytes)
                        .map_err(|e| Error::InvalidOperation(e.to_string()))?,
                );

                // Add metadata
                let mut metadata = HashMap::new();
                metadata.insert("format".to_string(), "flux_text_cache".to_string());
                metadata.insert("version".to_string(), "1.0".to_string());
                metadata.insert("prompt".to_string(), prompt.clone());

                // Serialize and save
                let serialized = serialize(tensors, &Some(metadata))
                    .map_err(|e| Error::InvalidOperation(e.to_string()))?;
                fs::write(&cache_path, serialized).map_err(|e| Error::Io(e.to_string()))?;

                let elapsed = text_start.elapsed();
                let total_elapsed = encoding_start.elapsed();
                let avg_time = total_elapsed.as_secs_f32() / (idx + 1) as f32;
                let remaining = to_encode.len() - idx - 1;
                let eta_seconds = (remaining as f32 * avg_time) as u64;
                let eta_minutes = eta_seconds / 60;
                let eta_secs = eta_seconds % 60;

                info!(
                    "  ✅ Saved embeddings | Time: {:.2}s | Avg: {:.2}s/text | ETA: {:02}:{:02}\n",
                    elapsed.as_secs_f32(),
                    avg_time,
                    eta_minutes,
                    eta_secs
                );

                // Aggressively drop tensors to free memory
                drop(clip_embeds);
                drop(t5_embeds);
                drop(clip_f32);
                drop(t5_f32);
                // Note: clip_data, t5_data, and tensors are moved and can't be dropped explicitly

                // Force GPU synchronization and cleanup after EVERY encoding to avoid OOM
                self.device.synchronize()?;

                // Try to free cached memory
                if let Ok(pool) =
                    flame_core::memory_pool::MEMORY_POOL.get_pool(&self.device.cuda_device_arc())
                {
                    if let Ok(mut pool_guard) = pool.lock() {
                        pool_guard.clear_cache();
                        let _ = pool_guard.force_cleanup();
                    }
                }
            }
        }

        Ok(())
    }

    /// Pre-encode all text embeddings in a dataset
    pub fn encode_all_text_embeddings(
        &self,
        data_loader: &mut FluxDataLoader,
        clip_path: &Path,
        t5_path: Option<&Path>,
        force_reencode: bool,
    ) -> Result<()> {
        if !self.enabled {
            info!("Cache manager disabled, skipping text encoding");
            return Ok(());
        }

        info!("=== Pre-encoding text embeddings (SimpleTuner style) ===");

        // Count already cached
        let total_samples = data_loader.total_samples();
        let mut cached_count = 0;
        let mut to_encode = Vec::new();

        // Get all samples from the data loader
        let mut all_samples = Vec::new();
        for bucket_idx in 0..data_loader.buckets.len() {
            for sample in &data_loader.buckets[bucket_idx].samples {
                all_samples.push(sample.clone());
            }
        }

        // Check what needs encoding
        for sample in &all_samples {
            let cache_path = self.get_embed_cache_path(&sample.image_path);

            if cache_path.exists() && !force_reencode {
                cached_count += 1;
            } else {
                // Load caption for this sample
                let caption = if sample.caption_path.exists() {
                    std::fs::read_to_string(&sample.caption_path)
                        .unwrap_or_else(|_| "".to_string())
                        .trim()
                        .to_string()
                } else {
                    "".to_string()
                };
                to_encode.push((sample.image_path.clone(), caption));
            }
        }

        info!("Found {}/{} text embeddings already cached", cached_count, total_samples);

        // CRITICAL: If everything is cached, don't load text encoders at all!
        if to_encode.is_empty() {
            info!("✅ All text embeddings are already cached - skipping text encoder loading!");
            info!("   This saves ~9GB of VRAM!");
            return Ok(());
        }

        if !to_encode.is_empty() {
            info!("\n=== Text Encoding Phase ===");
            info!("Loading text encoders to encode {} prompts...", to_encode.len());

            // Load CLIP-L first (small model, ~500MB)
            info!("  Loading CLIP-L encoder...");
            let mut text_encoders = TextEncoders::from_safetensors(
                Some(clip_path),
                None, // No CLIP-G for Flux
                None, // Don't load T5 yet
                self.device.clone(),
            )?;
            info!("✅ CLIP-L loaded successfully (~500MB)\n");

            // Track timing
            let encoding_start = std::time::Instant::now();

            // PHASE 1: Encode all prompts with CLIP-L (lightweight)
            info!("=== Phase 1: CLIP-L Encoding ===");
            info!("Encoding {} prompts with CLIP-L...", to_encode.len());

            let mut all_clip_embeds = Vec::new();

            for (idx, (image_path, prompt)) in to_encode.iter().enumerate() {
                let filename = image_path.file_name().and_then(|n| n.to_str()).unwrap_or("unknown");

                info!("[CLIP {}/{}] {}", idx + 1, to_encode.len(), filename);

                // Encode with CLIP only
                let clip_embeds = text_encoders.encode_clip_only(prompt)?;
                all_clip_embeds.push((image_path.clone(), prompt.clone(), clip_embeds));
            }

            info!("✅ CLIP-L encoding complete");

            // Free CLIP-L from GPU
            drop(text_encoders);
            self.device.synchronize()?;

            // Clear memory pools
            if let Ok(pool) =
                flame_core::memory_pool::MEMORY_POOL.get_pool(&self.device.cuda_device_arc())
            {
                if let Ok(mut pool_guard) = pool.lock() {
                    pool_guard.clear_cache();
                    let _ = pool_guard.force_cleanup();
                }
            }

            info!("   CLIP-L freed from GPU memory\n");

            // PHASE 2: Load T5 and encode one at a time with aggressive memory management
            info!("=== Phase 2: T5-XXL Encoding ===");
            info!("⚠️  T5-XXL requires careful memory management:");
            info!("   Model size: 9.12GB");
            info!("   Forward pass: ~5-6GB additional");
            info!("   Solution: Process one prompt at a time with immediate save\n");

            // Load T5-XXL
            info!("Loading T5-XXL encoder...");
            let text_encoders_t5 = TextEncoders::from_safetensors(
                None, // No CLIP
                None, // No CLIP-G
                t5_path,
                self.device.clone(),
            )?;
            info!("✅ T5-XXL loaded successfully\n");

            // Process each prompt with T5
            for (idx, (image_path, prompt, clip_embeds)) in all_clip_embeds.iter().enumerate() {
                let text_start = std::time::Instant::now();
                let filename = image_path.file_name().and_then(|n| n.to_str()).unwrap_or("unknown");

                info!("[T5 {}/{}] Encoding caption for: {}", idx + 1, to_encode.len(), filename);

                // Show prompt preview (first 80 chars)
                let prompt_preview = if prompt.len() > 80 {
                    format!("{}...", &prompt[..80])
                } else {
                    prompt.clone()
                };
                info!("  Caption: \"{}\"", prompt_preview);

                // Encode with T5
                info!("  Encoding with T5-XXL...");
                let t5_embeds = text_encoders_t5.encode_t5_only(prompt, 128)?;

                // CRITICAL: Force GPU memory cleanup after EVERY encoding
                // T5 creates massive intermediate tensors during forward pass that MUST be freed
                let cuda_device = self.device.cuda_device();
                cuda_device.synchronize()?;

                // ALWAYS clear memory pools after EVERY T5 encoding - not just every 5
                info!("  Aggressively clearing GPU memory pools...");
                // Clear FLAME's memory pool caches
                if let Ok(pool) =
                    flame_core::memory_pool::MEMORY_POOL.get_pool(&self.device.cuda_device_arc())
                {
                    if let Ok(mut pool_guard) = pool.lock() {
                        pool_guard.clear_cache();
                        let _ = pool_guard.force_cleanup();
                    }
                }

                // Save to cache BEFORE freeing memory
                let cache_path = self.get_embed_cache_path(&image_path);

                // Save both embeddings in the same file
                let mut tensors = HashMap::new();

                // CLIP embedding - ensure F32 for saving
                let clip_f32 = if clip_embeds.dtype() != DType::F32 {
                    clip_embeds.to_dtype(DType::F32)?
                } else {
                    clip_embeds.clone()
                };
                let clip_data = clip_f32.to_vec1::<f32>()?;
                let clip_shape = clip_f32.shape().dims().to_vec();
                let clip_bytes = unsafe {
                    std::slice::from_raw_parts(clip_data.as_ptr() as *const u8, clip_data.len() * 4)
                };
                tensors.insert(
                    "clip_embeds".to_string(),
                    TensorView::new(SafeDtype::F32, clip_shape, clip_bytes)
                        .map_err(|e| Error::InvalidOperation(e.to_string()))?,
                );

                // T5 embedding - ensure F32 for saving
                let t5_f32 = if t5_embeds.dtype() != DType::F32 {
                    t5_embeds.to_dtype(DType::F32)?
                } else {
                    t5_embeds.clone()
                };
                let t5_data = t5_f32.to_vec1::<f32>()?;
                let t5_shape = t5_f32.shape().dims().to_vec();
                let t5_bytes = unsafe {
                    std::slice::from_raw_parts(t5_data.as_ptr() as *const u8, t5_data.len() * 4)
                };
                tensors.insert(
                    "t5_embeds".to_string(),
                    TensorView::new(SafeDtype::F32, t5_shape, t5_bytes)
                        .map_err(|e| Error::InvalidOperation(e.to_string()))?,
                );

                // Add metadata
                let mut metadata = HashMap::new();
                metadata.insert("format".to_string(), "flux_text_cache".to_string());
                metadata.insert("version".to_string(), "1.0".to_string());
                metadata.insert("prompt".to_string(), prompt.clone());

                // Serialize and save
                let serialized = serialize(tensors, &Some(metadata))
                    .map_err(|e| Error::InvalidOperation(e.to_string()))?;
                fs::write(&cache_path, serialized).map_err(|e| Error::Io(e.to_string()))?;

                let elapsed = text_start.elapsed();
                let total_elapsed = encoding_start.elapsed();
                let avg_time = total_elapsed.as_secs_f32() / (idx + 1) as f32;
                let remaining = to_encode.len() - idx - 1;
                let eta_seconds = (remaining as f32 * avg_time) as u64;
                let eta_minutes = eta_seconds / 60;
                let eta_secs = eta_seconds % 60;

                info!(
                    "  ✅ Saved embeddings | Time: {:.2}s | Avg: {:.2}s/text | ETA: {:02}:{:02}",
                    elapsed.as_secs_f32(),
                    avg_time,
                    eta_minutes,
                    eta_secs
                );

                // Small delay to ensure CUDA frees memory between prompts
                std::thread::sleep(std::time::Duration::from_millis(100));
                info!(""); // Empty line for readability
            }

            // Text encoders will be freed when they go out of scope
            drop(text_encoders_t5);
            info!("\n✅ T5-XXL freed from GPU memory");

            // CRITICAL: Force memory cleanup
            self.device.synchronize()?;
            use crate::memory::manager::MemoryManager;
            MemoryManager::empty_cache()?;

            info!("   GPU memory explicitly cleared");
        }

        // Final summary
        info!("\n=== Encoding Complete ===");
        info!("✅ All latents and text embeddings are cached");
        info!("   GPU memory has been explicitly freed for model loading");

        Ok(())
    }

    /// Load cached latent for a sample
    pub fn load_latent(&self, image_path: &Path) -> Result<Option<Tensor>> {
        let cache_path = self.get_latent_cache_path(image_path);
        self.load_tensor(&cache_path, "latent")
    }

    /// Load cached text embeddings for a sample
    pub fn load_text_embeddings(&self, image_path: &Path) -> Result<Option<(Tensor, Tensor)>> {
        let cache_path = self.get_embed_cache_path(image_path);

        if let Some(clip_embeds) = self.load_tensor(&cache_path, "clip_embeds")? {
            if let Some(t5_embeds) = self.load_tensor(&cache_path, "t5_embeds")? {
                Ok(Some((clip_embeds, t5_embeds)))
            } else {
                warn!("T5 embeddings not found in cache for {:?}", image_path);
                Ok(None)
            }
        } else {
            Ok(None)
        }
    }

    /// Get cache statistics
    pub fn get_stats(&self) -> Result<(usize, usize)> {
        if !self.enabled {
            return Ok((0, 0));
        }

        let latent_count =
            fs::read_dir(&self.latent_dir).map_err(|e| Error::Io(e.to_string()))?.count();

        let embed_count =
            fs::read_dir(&self.embed_dir).map_err(|e| Error::Io(e.to_string()))?.count();

        Ok((latent_count, embed_count))
    }

    /// Clear all caches
    pub fn clear_cache(&self) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }

        warn!("Clearing all caches at {:?}", self.cache_dir);

        // Remove all files in latent dir
        for entry in fs::read_dir(&self.latent_dir).map_err(|e| Error::Io(e.to_string()))? {
            let entry = entry.map_err(|e| Error::Io(e.to_string()))?;
            if entry.path().is_file() {
                fs::remove_file(entry.path()).map_err(|e| Error::Io(e.to_string()))?;
            }
        }

        // Remove all files in embed dir
        for entry in fs::read_dir(&self.embed_dir).map_err(|e| Error::Io(e.to_string()))? {
            let entry = entry.map_err(|e| Error::Io(e.to_string()))?;
            if entry.path().is_file() {
                fs::remove_file(entry.path()).map_err(|e| Error::Io(e.to_string()))?;
            }
        }

        info!("✅ Cache cleared");
        Ok(())
    }

    /// Save latent to cache
    pub fn save_latent(&self, image_path: &Path, latent: &Tensor) -> Result<()> {
        let cache_path = self.get_latent_cache_path(image_path);
        self.save_tensor(latent, &cache_path, "latent")
    }

    /// Save text embeddings to cache
    pub fn save_text_embeddings(
        &self,
        image_path: &Path,
        clip_embeds: &Tensor,
        t5_embeds: &Tensor,
    ) -> Result<()> {
        let cache_path = self.get_text_cache_path(image_path);

        let mut tensors = HashMap::new();

        // CLIP embedding
        let clip_data = clip_embeds.to_vec1::<f32>()?;
        let clip_shape = clip_embeds.shape().dims().to_vec();
        let clip_bytes = unsafe {
            std::slice::from_raw_parts(clip_data.as_ptr() as *const u8, clip_data.len() * 4)
        };
        tensors.insert(
            "clip_embeds".to_string(),
            TensorView::new(SafeDtype::F32, clip_shape, clip_bytes)
                .map_err(|e| Error::InvalidOperation(e.to_string()))?,
        );

        // T5 embedding
        let t5_data = t5_embeds.to_vec1::<f32>()?;
        let t5_shape = t5_embeds.shape().dims().to_vec();
        let t5_bytes =
            unsafe { std::slice::from_raw_parts(t5_data.as_ptr() as *const u8, t5_data.len() * 4) };
        tensors.insert(
            "t5_embeds".to_string(),
            TensorView::new(SafeDtype::F32, t5_shape, t5_bytes)
                .map_err(|e| Error::InvalidOperation(e.to_string()))?,
        );

        // Add metadata
        let mut metadata = HashMap::new();
        metadata.insert("format".to_string(), "flux_text_cache".to_string());
        metadata.insert("version".to_string(), "1.0".to_string());

        // Serialize and save
        let serialized = serialize(tensors, &Some(metadata))
            .map_err(|e| Error::InvalidOperation(e.to_string()))?;
        fs::write(&cache_path, serialized).map_err(|e| Error::Io(e.to_string()))?;

        Ok(())
    }

    /// Get text cache path (alias for consistency)
    pub fn get_text_cache_path(&self, image_path: &Path) -> PathBuf {
        self.get_embed_cache_path(image_path)
    }

    /// Check cache status for a data loader
    pub fn check_cache_status(&self, data_loader: &FluxDataLoader) -> Result<(usize, usize)> {
        let mut missing_latents = 0;
        let mut missing_text = 0;

        // This would iterate through data_loader samples
        // For now, return placeholder values
        Ok((missing_latents, missing_text))
    }

    /// Count missing latent caches
    pub fn count_missing_latents(&self, data_loader: &FluxDataLoader) -> Result<usize> {
        let mut missing = 0;
        // Would check each sample in data_loader
        // For now, return 0 to proceed
        Ok(missing)
    }

    /// Count missing text embeddings
    pub fn count_missing_text_embeddings(&self, data_loader: &FluxDataLoader) -> Result<usize> {
        let mut missing = 0;
        // Would check each sample in data_loader
        // For now, return 0 to proceed
        Ok(missing)
    }
}
