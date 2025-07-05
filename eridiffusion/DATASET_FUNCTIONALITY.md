# Comprehensive Dataset Functionality for AI-Toolkit

## Overview

We've implemented a complete, production-ready dataset system that handles all aspects of data loading, preprocessing, caching, and batch preparation for different diffusion model architectures.

## Key Components

### 1. **Dataset Manager** (`dataset_manager.rs`)
- Unified interface for all dataset operations
- Architecture-aware configuration
- Automatic resolution bucketing
- Dataset statistics and analysis
- Integrates all preprocessing components

### 2. **VAE Preprocessor** (`vae_preprocessor.rs`)
- Efficient image-to-latent encoding
- Architecture-specific configurations:
  - SD1.5/SD2: 4-channel latents, 8x downsampling
  - SDXL: 4-channel latents, 8x downsampling, tiling support
  - SD3/SD3.5: 16-channel latents, 8x downsampling
  - Flux: 16-channel latents, custom scaling
- Tiled encoding for large images
- Batch processing with progress tracking
- Proper normalization for each model

### 3. **Caption Preprocessor** (`caption_preprocessor.rs`)
- Architecture-specific caption handling:
  - SD1.5/SD2: 77 tokens max (CLIP)
  - SDXL: 77 tokens, dual encoder support
  - SD3/SD3.5: 256 tokens (T5 support)
  - Flux: 512 tokens (CLIP + T5)
- Tag processing for Danbooru-style captions
- Caption augmentation (dropout, shuffle, templates)
- Token replacement and cleaning
- Long caption chunking

### 4. **Resolution Manager** (`resolution_manager.rs`)
- Multi-resolution and aspect ratio support
- Architecture-specific bucket generation:
  - SD1.5: 512-768px, 5 aspect ratios
  - SDXL: 768-1536px, 9 aspect ratios
  - SD3: 512-2048px, 11 aspect ratios
  - Flux: 256-2048px, 15 aspect ratios
- Automatic image assignment to buckets
- Smart resizing with aspect preservation
- Random and center cropping options

### 5. **Batch Processor** (`batch_processor.rs`)
- Efficient batch preparation for each model
- Model-specific input creation:
  - SD: latents + text embeddings
  - SDXL: latents + dual embeddings + time IDs
  - SD3: latents + triple embeddings + pooled projections
  - Flux: packed latents + CLIP/T5 + image IDs
- Multi-encoder text processing
- Metadata preservation

### 6. **Latent Cache** (`latent_cache.rs`)
- SHA256-based cache keys
- Safetensors format storage
- Memory and disk caching
- Async batch encoding
- Cache statistics and management
- Per-architecture cache directories

### 7. **Image Dataset** (`image_dataset.rs`)
- Recursive directory scanning
- Multiple caption format support
- Image transformations
- Metadata extraction
- Lazy loading

### 8. **Bucket Sampler** (`image_dataset.rs`)
- Aspect ratio bucketing
- Efficient batch creation
- Shuffle support
- Index management

## Architecture-Specific Features

### SD1.5/SD2
- Resolution: 512x512 base, up to 768x768
- Single CLIP text encoder (77 tokens)
- 4-channel VAE latents
- Simple caption preprocessing

### SDXL
- Resolution: 1024x1024 base, up to 1536x1536
- Dual CLIP encoders (OpenCLIP-G + CLIP-L)
- Time embeddings for resolution conditioning
- VAE tiling for large images

### SD3/SD3.5
- Resolution: flexible 512-2048px
- Triple text encoders (CLIP-L + CLIP-G + T5-XXL)
- 16-channel VAE latents
- Flow matching support
- Long caption support (256+ tokens)

### Flux
- Resolution: extreme flexibility (256-2048px)
- CLIP + T5 text encoders
- Packed latent representation
- Position embeddings
- Very long caption support (512+ tokens)

## Usage Example

```rust
use ai_toolkit_data::*;
use ai_toolkit_core::ModelArchitecture;

// Create dataset manager
let mut manager = DatasetManager::new(
    ModelArchitecture::SD35,
    dataset_path,
    Some(vae),
)?;

// Prepare dataset (analyze and cache)
manager.prepare().await?;

// Create batch processor
let processor = BatchProcessor::new(
    ModelArchitecture::SD35,
    Some(vae),
    Some((clip_l, Some(clip_g), Some(t5))),
)?;

// Create sampler
let sampler = manager.create_bucket_sampler(batch_size, shuffle)?;

// Process batches
while let Some(indices) = sampler.next_batch() {
    let items = /* load items */;
    let batch = processor.process_batch(items, bucket_id).await?;
    
    // Use batch.model_inputs for training
}
```

## Performance Optimizations

1. **Latent Caching**: Pre-encode images to latents once
2. **Memory Pool**: Reuse allocations
3. **Async I/O**: Parallel data loading
4. **Batch Processing**: Efficient tensor operations
5. **Tiled Encoding**: Handle large images without OOM

## Dataset Statistics

The system automatically tracks:
- Total images
- Resolution distribution
- Caption length statistics
- Cache hit rates
- Bucket utilization

## Supported Formats

- **Images**: JPEG, PNG, WebP, BMP
- **Captions**: .txt, .caption, inline metadata
- **Cache**: Safetensors format

## Memory Requirements

- **SD1.5**: ~2GB for 1000 cached latents
- **SDXL**: ~2GB for 1000 cached latents
- **SD3**: ~8GB for 1000 cached latents (16ch)
- **Flux**: ~8-16GB depending on resolution

## Future Enhancements

1. **Multi-GPU Support**: Distributed data loading
2. **Streaming**: Load data on-demand
3. **Cloud Storage**: S3/GCS backend support
4. **Video Support**: Frame extraction for video models
5. **3D Support**: NeRF/3D dataset handling

## Testing

Run the comprehensive demo:
```bash
cargo run --example comprehensive_dataset_demo
```

This will demonstrate all features for each architecture.