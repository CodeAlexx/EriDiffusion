# SD 3.5 Training Pipeline Summary

## Overview
The SD 3.5 training pipeline has been enhanced with:
1. Flow matching objective for improved training
2. MMDiT (Modulated DiT) architecture support
3. Multi-encoder text conditioning (CLIP-L + CLIP-G + T5-XXL)
4. Latent encoding and caching for efficient training
5. **IMPORTANT: T5 max token length must be limited to 154 tokens**

## Key Configuration Parameters (eri1024.yaml)

```yaml
model:
  name_or_path: "/home/alex/SwarmUI/Models/diffusion_models/sd3.5_large.safetensors"
  is_v3: true
  quantize: true  # 8bit mixed precision
  t5_max_length: 154  # CRITICAL: Must not exceed 154 tokens
  
train:
  batch_size: 4
  noise_scheduler: "flowmatch"  # SD3 uses flow matching
  optimizer: "adamw8bit"        # Memory efficient optimizer
  lr: 5e-05
  gradient_checkpointing: true
  dtype: bf16
  
datasets:
  - folder_path: "/home/alex/diffusers-rs/datasets/40_woman"
    cache_latents_to_disk: true  # Enable latent caching
    resolution: [1024]
```

## Critical T5 Token Length Fix

The T5-XXL encoder in SD 3.5 **MUST** be limited to 154 tokens maximum. The current implementation needs to be updated:

### Current Issue
In `sd3_candle.rs`, the T5 encoder is incorrectly using 77 tokens (same as CLIP):
```rust
// Line 151 - WRONG: This truncates T5 to 77 tokens
tokens.resize(self.max_position_embeddings, 0);
```

### Required Fix
```rust
// In T5WithTokenizer::new() and encode_text_to_embedding()
impl T5WithTokenizer {
    fn new(vb: VarBuilder, t5_max_length: usize) -> AnyhowResult<Self> {
        // ... existing code ...
        Ok(Self {
            t5: model,
            tokenizer,
            max_position_embeddings: t5_max_length, // Use 154 instead of 77
        })
    }
    
    fn encode_text_to_embedding(&mut self, prompt: &str, device: &Device) -> AnyhowResult<Tensor> {
        let mut tokens = self.tokenizer
            .encode(prompt, true)
            .map_err(anyhow::Error::msg)?
            .get_ids()
            .to_vec();
        
        // Resize to T5 max length (154)
        tokens.resize(self.max_position_embeddings.min(154), 0);
        // ... rest of encoding
    }
}
```

## Latent Encoding & Caching

### New LatentDataLoader Features
- Automatically encodes images to latents during data loading
- Caches latents to disk for faster subsequent epochs
- Reduces GPU memory usage during training
- Supports pre-caching all latents before training starts

### Usage Example
```rust
use eridiffusion_data::{LatentDataLoader, LatentDataLoaderConfig};

// Create config with caching enabled
let config = LatentDataLoaderConfig {
    batch_size: 4,
    cache_latents: true,
    cache_dir: Some(PathBuf::from("./latent_cache")),
    ..Default::default()
};

// Create latent dataloader
let dataloader = LatentDataLoader::new(
    dataset,
    config,
    device,
    vae.clone(),
    ModelArchitecture::SD3,
)?;

// Pre-cache all latents (optional but recommended)
dataloader.precache_all().await?;

// Iterate over batches - latents are automatically loaded/cached
let mut iter = dataloader.iter().await;
while let Some(batch) = iter.next().await {
    let batch = batch?;
    // batch.latents contains pre-encoded latents
    // batch.images contains original images (for logging/visualization)
}
```

## Training Pipeline Flow

1. **Text Encoding** (with 154 token limit for T5):
   - CLIP-L: 77 tokens → 768 dim embeddings
   - CLIP-G: 77 tokens → 1280 dim embeddings  
   - T5-XXL: **154 tokens** → 4096 dim embeddings
   - Combined: [batch, seq_len, 6144]

2. **Image → Latent Encoding**:
   - Images: [batch, 3, 1024, 1024]
   - VAE encode: → [batch, 16, 128, 128] (16 channels for SD3)
   - Cached to disk for reuse

3. **Flow Matching Training**:
   - Linear interpolation between latents and noise
   - Velocity prediction instead of noise prediction
   - Mean flow loss for improved convergence

## Memory Optimizations

1. **8-bit AdamW**: 75% memory reduction vs standard AdamW
2. **Gradient Checkpointing**: Reduces activation memory
3. **Latent Caching**: No VAE encoding during training
4. **BF16 Mixed Precision**: Half precision compute

## Important Notes

1. **T5 Token Length**: The 154 token limit is critical - exceeding it can cause crashes or poor results
2. **Latent Channels**: SD3/3.5 uses 16 latent channels (not 4 like SDXL)
3. **Flow Matching**: Uses velocity prediction, not epsilon prediction
4. **Text Encoders**: All three encoders (CLIP-L, CLIP-G, T5) must be loaded

## Next Steps

1. Fix T5 max token length in `sd3_candle.rs`
2. Test latent caching with the dataset
3. Verify memory usage with 8-bit optimizer
4. Run training with the eri1024.yaml config