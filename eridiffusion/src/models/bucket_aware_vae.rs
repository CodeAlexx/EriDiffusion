// Integration with your existing VAE and training pipeline

use crate::models::bucket_alignment::{BucketAlignmentManager, BucketAwareDataLoader};
use flame_core::{DType, Device, Result, Tensor};
use std::collections::HashMap;

/// Enhanced Flux VAE with bucket-aware alignment
pub struct BucketAwareFluxVAE {
    inner_vae: crate::models::flux_vae::AutoencoderKL,
    alignment_manager: BucketAlignmentManager,
    device: Device,
}

impl BucketAwareFluxVAE {
    pub fn new(
        vae_path: &std::path::Path,
        device: Device,
        enable_offloading: bool,
    ) -> Result<Self> {
        // Load the base VAE with BF16 to reduce memory
        let weight_loader = crate::loaders::WeightLoader::from_safetensors_with_dtype(
            vae_path,
            device.clone(),
            DType::BF16,
        )?;
        let inner_vae = crate::models::flux_vae::AutoencoderKL::new(
            &weight_loader,
            device.clone(),
            enable_offloading,
        )?;

        // Create alignment manager for BF16 (memory efficient)
        let mut alignment_manager = BucketAlignmentManager::new(DType::BF16);
        alignment_manager.precompute_common_buckets()?;

        Ok(Self { inner_vae, alignment_manager, device })
    }

    /// Encode with automatic alignment for any bucket size
    pub fn encode_aligned(&mut self, x: &Tensor) -> Result<Tensor> {
        // Apply alignment based on input dimensions
        let aligned_input = self.alignment_manager.align_tensor(x)?;

        // Encode using the inner VAE
        let latents = self.inner_vae.encode(&aligned_input)?;

        Ok(latents)
    }

    /// Decode with alignment awareness
    pub fn decode_aligned(&mut self, z: &Tensor) -> Result<Tensor> {
        let decoded = self.inner_vae.decode(z)?;

        // Note: Decoded output will be in aligned dimensions
        // You may need to crop back to original size if needed
        Ok(decoded)
    }

    /// Get alignment info for debugging
    pub fn get_alignment_info(&self, height: usize, width: usize) -> String {
        let strategy = self.alignment_manager.alignment_cache.get(&(height, width));
        match strategy {
            Some(s) => format!(
                "{}x{} -> {}x{} (aligned: {})",
                height, width, s.aligned_height, s.aligned_width, s.is_aligned
            ),
            None => format!("{}x{} (not computed)", height, width),
        }
    }
}

/// Enhanced dataset cache manager with bucket alignment
pub struct AlignedCacheManager {
    base_cache_dir: std::path::PathBuf,
    alignment_manager: BucketAlignmentManager,
    vae: BucketAwareFluxVAE,
}

impl AlignedCacheManager {
    pub fn new(
        cache_dir: std::path::PathBuf,
        vae_path: &std::path::Path,
        device: Device,
    ) -> Result<Self> {
        let vae = BucketAwareFluxVAE::new(vae_path, device, true)?;
        let alignment_manager = BucketAlignmentManager::new(DType::BF16);

        Ok(Self { base_cache_dir: cache_dir, alignment_manager, vae })
    }

    /// Generate cache key that includes alignment info
    fn get_cache_key(
        &mut self,
        image_path: &std::path::Path,
        height: usize,
        width: usize,
    ) -> String {
        let strategy = self.alignment_manager.get_alignment_strategy(height, width);

        // Include alignment dimensions in cache key
        format!(
            "{}_{}x{}_aligned_{}x{}.safetensors",
            image_path.file_stem().unwrap().to_string_lossy(),
            height,
            width,
            strategy.aligned_height,
            strategy.aligned_width
        )
    }

    /// Cache latents with alignment info
    pub fn cache_image_latents(
        &mut self,
        image_path: &std::path::Path,
        bucket_height: usize,
        bucket_width: usize,
        force_recache: bool,
    ) -> Result<std::path::PathBuf> {
        let cache_key = self.get_cache_key(image_path, bucket_height, bucket_width);
        let cache_path = self.base_cache_dir.join(&cache_key);

        if cache_path.exists() && !force_recache {
            println!("Using cached latents: {}", cache_key);
            return Ok(cache_path);
        }

        println!(
            "Encoding and caching: {} ({}x{}) -> {}",
            image_path.display(),
            bucket_height,
            bucket_width,
            cache_key
        );

        // Load image at bucket size
        let image_tensor = self.load_image_for_bucket(image_path, bucket_height, bucket_width)?;

        // Encode with alignment
        let latents = self.vae.encode_aligned(&image_tensor)?;

        // Save to cache
        self.save_tensor_to_cache(&latents, &cache_path)?;

        println!("Cached latents for {}x{} bucket: {}", bucket_height, bucket_width, cache_key);

        Ok(cache_path)
    }

    /// Load image and resize to bucket dimensions
    fn load_image_for_bucket(
        &self,
        image_path: &std::path::Path,
        height: usize,
        width: usize,
    ) -> Result<Tensor> {
        // Your existing image loading logic here
        // This is a placeholder
        let tensor = Tensor::randn(
            flame_core::Shape::from_dims(&[1, 3, height, width]),
            0.0,
            1.0,
            self.vae.device.cuda_device_arc(),
        )?;

        // Normalize to [0, 1] range
        let normalized = tensor.add_scalar(1.0)?.mul_scalar(0.5)?;

        Ok(normalized)
    }

    /// Save tensor to cache file
    fn save_tensor_to_cache(&self, tensor: &Tensor, cache_path: &std::path::Path) -> Result<()> {
        // Create parent directory if it doesn't exist
        if let Some(parent) = cache_path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                flame_core::Error::InvalidOperation(format!(
                    "Failed to create cache dir: {}",
                    e
                ))
            })?;
        }

        // Note: FLAME only supports CUDA devices, so we keep tensor on GPU
        // In a real implementation, you would save the tensor data here

        // Placeholder for actual safetensors saving
        println!("Saving tensor to: {}", cache_path.display());

        Ok(())
    }

    /// Process entire dataset with bucket-aware caching
    pub fn cache_dataset_buckets(
        &mut self,
        dataset_items: Vec<(std::path::PathBuf, usize, usize)>, // (path, height, width)
        force_recache: bool,
    ) -> Result<Vec<std::path::PathBuf>> {
        println!("Caching {} items with bucket-aware alignment...", dataset_items.len());

        let mut cached_paths = Vec::new();
        let mut bucket_stats: HashMap<(usize, usize), usize> = HashMap::new();

        for (i, (image_path, height, width)) in dataset_items.iter().enumerate() {
            // Track bucket usage
            *bucket_stats.entry((*height, *width)).or_insert(0) += 1;

            let cache_path =
                self.cache_image_latents(image_path, *height, *width, force_recache)?;
            cached_paths.push(cache_path);

            if (i + 1) % 100 == 0 {
                println!("Cached {}/{} items", i + 1, dataset_items.len());

                // Print alignment stats periodically
                println!(
                    "Alignment info for {}x{}: {}",
                    height,
                    width,
                    self.vae.get_alignment_info(*height, *width)
                );
            }
        }

        // Print final statistics
        println!("=== Bucket Usage Statistics ===");
        for ((h, w), count) in bucket_stats {
            println!("  {}x{}: {} images ({})", h, w, count, self.vae.get_alignment_info(h, w));
        }

        Ok(cached_paths)
    }
}

/// Training integration with bucket-aware alignment
pub struct AlignedFluxTrainer {
    cache_manager: AlignedCacheManager,
    current_batch_alignment: HashMap<(usize, usize), Vec<std::path::PathBuf>>,
}

impl AlignedFluxTrainer {
    pub fn new(
        cache_dir: std::path::PathBuf,
        vae_path: &std::path::Path,
        device: Device,
    ) -> Result<Self> {
        let cache_manager = AlignedCacheManager::new(cache_dir, vae_path, device)?;

        Ok(Self { cache_manager, current_batch_alignment: HashMap::new() })
    }

    /// Prepare training batch with proper alignment
    pub fn prepare_aligned_batch(
        &mut self,
        batch_items: Vec<(std::path::PathBuf, usize, usize)>,
        force_recache: bool,
    ) -> Result<Vec<Tensor>> {
        let mut aligned_latents = Vec::new();

        // Group by bucket size for efficient processing
        let mut buckets: HashMap<(usize, usize), Vec<std::path::PathBuf>> = HashMap::new();

        for (path, h, w) in batch_items {
            buckets.entry((h, w)).or_default().push(path);
        }

        // Process each bucket
        for ((height, width), paths) in buckets {
            println!("Processing {} items in {}x{} bucket", paths.len(), height, width);

            for path in paths {
                let cache_path =
                    self.cache_manager.cache_image_latents(&path, height, width, force_recache)?;
                let latents = self.load_cached_latents(&cache_path)?;
                aligned_latents.push(latents);
            }
        }

        Ok(aligned_latents)
    }

    /// Load cached latents
    fn load_cached_latents(&self, cache_path: &std::path::Path) -> Result<Tensor> {
        // Your existing cache loading logic
        // Placeholder for actual safetensors loading
        println!("Loading cached latents from: {}", cache_path.display());

        Ok(Tensor::zeros(
            flame_core::Shape::from_dims(&[1, 16, 128, 128]), // Flux latent dimensions
            self.cache_manager.vae.device.cuda_device_arc(),
        )?)
    }
}
