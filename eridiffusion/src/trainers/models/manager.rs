use flame_core::device::Device;
use flame_core::Result;
use flame_core::{DType, Tensor};
use log::{debug, info};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use super::loader::{LoadedModels, ModelLoader};

/// Cache for loaded models to avoid reloading
pub struct ModelCache {
    cache: Arc<Mutex<HashMap<String, Arc<LoadedModels>>>>,
    device: Device,
    dtype: DType,
}

impl ModelCache {
    pub fn new(device: Device, dtype: DType) -> Self {
        Self { cache: Arc::new(Mutex::new(HashMap::new())), device, dtype }
    }

    /// Get or load models
    pub fn get_or_load(
        &self,
        key: &str,
        load_fn: impl FnOnce() -> flame_core::Result<LoadedModels>,
    ) -> flame_core::Result<Arc<LoadedModels>> {
        let mut cache = self.cache.lock().unwrap();

        if let Some(models) = cache.get(key) {
            debug!("Using cached models for key: {}", key);
            return Ok(Arc::clone(models));
        }

        info!("Loading models for key: {}", key);
        let models = Arc::new(load_fn()?);
        cache.insert(key.to_string(), Arc::clone(&models));

        Ok(models)
    }

    /// Clear cache
    pub fn clear(&self) {
        let mut cache = self.cache.lock().unwrap();
        cache.clear();
        info!("Model cache cleared");
    }
}

/// Manages model lifecycle and memory
pub struct ModelManager {
    loader: ModelLoader,
    cache: ModelCache,
    device: Device,
    dtype: DType,
}

impl ModelManager {
    pub fn new(device: Device, dtype: DType) -> Self {
        let loader = ModelLoader::new(device.clone(), dtype);
        let cache = ModelCache::new(device.clone(), dtype);

        Self { loader, cache, device, dtype }
    }

    /// Load SDXL models with caching
    pub fn load_sdxl_cached(
        &self,
        unet_path: &std::path::Path,
        vae_path: Option<&std::path::Path>,
        text_encoder_paths: Option<(&std::path::Path, &std::path::Path)>,
    ) -> flame_core::Result<Arc<LoadedModels>> {
        let key = format!("sdxl_{:?}", unet_path);

        self.cache.get_or_load(&key, || {
            self.loader.load_sdxl_models(unet_path, vae_path, text_encoder_paths)
        })
    }

    /// Move models to device
    pub fn to_device(&self, models: &mut LoadedModels, device: &Device) -> flame_core::Result<()> {
        // Move UNet weights
        for (name, tensor) in models.unet_weights.iter_mut() {
            // In FLAME, tensors are already on their device, so this is a no-op
            debug!("Tensor {} already on device", name);
        }

        // Move VAE if present
        if let Some(ref mut vae) = models.vae_encoder {
            // VAE to_device would be implemented in the VAE struct
            debug!("Moving VAE to device");
        }

        // Move text encoders if present
        if let Some(ref mut encoders) = models.text_encoders {
            // Text encoders to_device would be implemented in the struct
            debug!("Moving text encoders to device");
        }

        Ok(())
    }

    /// Optimize models for inference
    pub fn optimize_for_inference(&self, models: &mut LoadedModels) -> flame_core::Result<()> {
        info!("Optimizing models for inference");

        // Convert to half precision if using F32
        if self.dtype == DType::F32 {
            for (name, tensor) in models.unet_weights.iter_mut() {
                if tensor.dtype() == DType::F32 {
                    *tensor = tensor.to_dtype(DType::F16)?;
                    debug!("Converted {} to F16", name);
                }
            }
        }

        // Additional optimizations could go here
        // - Fuse operations
        // - Optimize memory layout
        // - Enable cudnn benchmarking

        Ok(())
    }

    /// Estimate memory usage
    pub fn estimate_memory_usage(&self, models: &LoadedModels) -> usize {
        let mut total_bytes = 0;

        // UNet weights
        for (_, tensor) in &models.unet_weights {
            let shape = tensor.shape();
            let dtype_size = match tensor.dtype() {
                DType::F32 => 4,
                DType::F16 | DType::BF16 => 2,
                DType::U8 | DType::I8 => 1,
                _ => 4,
            };
            total_bytes += shape.dims().iter().product::<usize>() * dtype_size;
        }

        // Add estimates for VAE and text encoders
        if models.vae_encoder.is_some() {
            total_bytes += 500_000_000; // ~500MB for VAE
        }

        if models.text_encoders.is_some() {
            total_bytes += 1_000_000_000; // ~1GB for text encoders
        }

        total_bytes
    }

    /// Free unused models
    pub fn free_unused(&self) -> flame_core::Result<()> {
        self.cache.clear();
        Ok(())
    }
}
