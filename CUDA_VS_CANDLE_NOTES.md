# CUDA Training vs Candle Inference

## Key Differences

### Our Training (CUDA-focused)
- Pure CUDA kernels for performance
- Memory pooling for 24GB GPUs
- Custom CUDA implementations for:
  - RoPE (Rotary Position Embeddings)
  - GroupNorm
  - Fused attention operations
  - Block swapping for memory efficiency
- BF16/F32 mixed precision
- Optimized for A5000/A6000/3090/4090 GPUs

### Candle (CPU-friendly)
- Great CPU performance
- CUDA support exists but not the primary focus
- More portable across devices
- Better for inference on various hardware
- Simpler deployment (no CUDA dependency required)

## Why This Split Makes Sense

### Training = CUDA
- Need maximum performance
- Memory efficiency crucial for large models
- Custom kernels give significant speedup
- Users training Flux have GPUs

### Inference = Candle/CPU Options
- Broader hardware support
- Easier deployment
- Many users only need inference
- Can run on:
  - CPU-only systems
  - Mac Metal
  - Smaller GPUs
  - Cloud instances without GPUs

## Our Implementation Strategy

```rust
// Training path - CUDA optimized
#[cfg(feature = "cuda")]
pub fn train_flux_lora_cuda() {
    // Custom CUDA kernels
    // Memory pooling
    // Block swapping
    // Pure GPU pipeline
}

// Save in standard format
save_flux_lora(weights, "flux_lora.safetensors")?;

// Users can then use Candle for inference
// which handles CPU/GPU/Metal automatically
```

## Memory Requirements

### Training (CUDA)
- Minimum: 24GB VRAM
- Recommended: 40GB+ for larger batches
- Uses memory pooling and gradient checkpointing

### Inference (Candle)
- Can run on 8GB GPUs
- CPU inference possible (slower)
- Supports quantization for smaller memory footprint

## Performance Comparison

### Training Speed (relative)
- Our CUDA implementation: 1.0x (baseline)
- Candle GPU: ~0.6-0.8x (estimated)
- Candle CPU: ~0.01-0.05x (very slow)

### Inference Speed
- Candle GPU: Fast enough for most uses
- Candle CPU: Acceptable for single images
- Candle quantized: Good balance of speed/quality

## Recommendations

### For Training
- Use our CUDA implementation
- Requires NVIDIA GPU with 24GB+ VRAM
- Saves in AI-Toolkit format

### For Inference
- Use Candle or other inference engines
- Works on various hardware
- Supports our saved LoRA format

## Code Example

```rust
// Training (CUDA)
use eridiffusion::trainers::FluxLoRATrainer;

let trainer = FluxLoRATrainer::new_cuda(
    device,
    config,
)?;
trainer.train()?;
trainer.save_lora("flux_lora.safetensors")?;

// Inference (Candle - separate project)
// User loads flux_lora.safetensors
// Candle handles CPU/GPU automatically
```

## Why Not Use Candle for Training?

1. **Performance**: Our CUDA kernels are specifically optimized for training
2. **Memory**: Custom memory management for 24GB GPUs
3. **Features**: Block swapping, gradient checkpointing, etc.
4. **Focus**: Candle prioritizes portability over maximum GPU performance

## Summary

- **Training**: CUDA-only, maximum performance, 24GB+ VRAM
- **Inference**: Candle/others, portable, CPU/GPU/Metal
- **Bridge**: Standard safetensors format with AI-Toolkit naming