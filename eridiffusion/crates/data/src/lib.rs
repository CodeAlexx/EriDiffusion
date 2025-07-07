//! Data loading and processing module

pub mod dataset;
pub mod dataloader;
pub mod transforms;
pub mod caption;
pub mod bucket;
pub mod cache;
pub mod validation;
pub mod latent_cache;
pub mod image_dataset;
pub mod dataloader_batch;
pub mod dataloader_impl;
pub mod woman_dataset;
pub mod dataset_manager;
pub mod vae_preprocessor;
pub mod caption_preprocessor;
pub mod resolution_manager;
pub mod batch_processor;
pub mod optimized_loader;
pub mod cache_manager;
pub mod augmentations;
pub mod vae_normalization;
pub mod webdataset;
pub mod latent_dataloader;

pub use dataset::{Dataset, DatasetItem, DatasetMetadata};
pub use image_dataset::{ImageDataset, DatasetConfig, BucketSampler, Transform, ImageTransform};
pub use dataloader::{DataLoader, DataLoaderConfig, DataLoaderBatch, DataLoaderIterator};
pub use transforms::{Compose, Resize, RandomCrop, Normalize};
pub use caption::{CaptionProcessor, CaptionConfig};
pub use bucket::{BucketManager, BucketConfig};
pub use cache::{DataCache, CacheConfig};
pub use validation::{DataValidator, ValidationConfig};
pub use latent_cache::{LatentCache, BatchLatentEncoder, CacheStats};
pub use woman_dataset::{WomanDataset, WomanDatasetConfig};
pub use dataset_manager::{DatasetManager, PreprocessedItem, DatasetStats, ResolutionConfig, DataPreprocessor};
pub use vae_preprocessor::{VAEPreprocessor, VAEConfig, TiledVAEEncoder};
pub use caption_preprocessor::{CaptionPreprocessor, ProcessedCaption, CaptionAugmenter};
pub use resolution_manager::{ResolutionManager, ResolutionBucket, ResizedImage};
pub use batch_processor::{BatchProcessor, ProcessedBatch, ModelInputs, TextEmbeddings};
pub use dataloader_batch::LatentBatch;
pub use optimized_loader::{OptimizedImageLoader, simd_ops};
pub use cache_manager::{CacheManager, CacheEntry, CacheMetadata, LatentCache as LatentCacheManager};
pub use augmentations::{Augmenter, AugmentationConfig, ColorJitterConfig, CutoutConfig, mixup_batch, cutmix_batch};
pub use vae_normalization::{VAENormalizer, VAENormalization, batch_ops};
pub use webdataset::{WebDatasetReader, WebDatasetLoader, WebDatasetWriter, WebDatasetSample, SampleMetadata, DataBatch};
pub use latent_dataloader::{LatentDataLoader, LatentDataLoaderConfig, LatentDataLoaderBatch, LatentDataLoaderIterator};

// Device conversion utility
fn to_candle_device(device: &eridiffusion_core::Device) -> eridiffusion_core::Result<candle_core::Device> {
    match device {
        eridiffusion_core::Device::Cpu => Ok(candle_core::Device::Cpu),
        eridiffusion_core::Device::Cuda(id) => candle_core::Device::new_cuda(*id)
            .map_err(|e| eridiffusion_core::Error::Device(format!("Failed to create CUDA device: {}", e))),
    }
}