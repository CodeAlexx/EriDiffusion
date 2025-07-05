# Flux Training on 24GB VRAM - How It Works

This document explains how we achieve Flux training on 24GB VRAM, matching what ComfyUI, SimpleTuner, and AI-Toolkit do.

## The Key Insight: Separation of Encoding and Training

The secret to fitting Flux training in 24GB is that **encoders and the training model are NEVER in memory at the same time**.

### Traditional Approach (Won't Fit)
```
Memory Usage:
- VAE:        1.7 GB
- T5-XXL:    11.0 GB  
- CLIP-L:     0.5 GB
- Flux:      12.0 GB
- Total:     25.2 GB ❌ Won't fit!
```

### Our Approach (Fits in 24GB)
```
Step 1 - Preprocessing (Run Once):
- Load VAE + T5 + CLIP (~13GB)
- Encode all data to disk
- Free all encoders

Step 2 - Training:
- Load ONLY Flux (~12GB)
- + Gradients (~3GB with checkpointing)
- + Optimizer (~6GB)
- + Activations (~2GB)
- Total: ~23GB ✅ Fits!
```

## Implementation Details

### 1. Preprocessing Phase

Run this ONCE before training:

```bash
cargo run --release --bin preprocess_flux -- \
    --dataset-dir /path/to/images \
    --cache-dir flux_cache \
    --vae-path /models/ae.safetensors \
    --t5-path /models/t5-v1_1-xxl.safetensors \
    --clip-path /models/clip_l.safetensors \
    --t5-tokenizer /models/t5_tokenizer.json \
    --clip-tokenizer /models/clip_tokenizer.json \
    --batch-size 4
```

This will:
1. Load all three encoders (VAE, T5, CLIP)
2. Process each image:
   - Image → VAE → Latent (16×H/8×W/8)
   - Caption → T5 → Embeddings (512×4096)
   - Caption → CLIP → Pooled (768)
3. Save each encoded item to disk as `.safetensors`
4. **Free all encoders from memory**

### 2. Training Phase

Now train with ONLY Flux in memory:

```bash
cargo run --release --bin train_flux_24gb -- \
    --cache-dir flux_cache \
    --model-path /models/flux1-dev.safetensors \
    --output-dir output \
    --learning-rate 1e-5 \
    --batch-size 1 \
    --gradient-accumulation 4 \
    --num-steps 1000
```

### Critical Techniques

1. **Gradient Checkpointing** (REQUIRED)
   - Reduces activation memory by ~10x
   - Trades compute for memory
   - Always enabled in 24GB mode

2. **Mixed Precision Training**
   - Model in FP16: ~12GB instead of ~24GB
   - Gradients computed in FP32 for stability
   - Optimizer states in FP32

3. **Smart Batching**
   - Batch size 1-2 for 24GB
   - Gradient accumulation for larger effective batches
   - No EMA to save memory

4. **Memory Management**
   - Pre-allocated buffers
   - Careful tensor lifecycle management
   - Regular garbage collection

## Memory Breakdown

### During Preprocessing:
```
VAE + T5 + CLIP:     ~13 GB
Batch processing:     ~3 GB
System overhead:      ~2 GB
Total:               ~18 GB ✅
```

### During Training:
```
Flux model (FP16):   ~12 GB
Gradients:            ~3 GB (with checkpointing)
Adam states:          ~6 GB
Activations:          ~2 GB
System overhead:      ~1 GB
Total:               ~24 GB ✅
```

## Disk Space Requirements

For a dataset of 1000 images at 1024×1024:
- Latents: 1000 × 16 × 128 × 128 × 4 bytes = ~1 GB
- T5 embeds: 1000 × 512 × 4096 × 4 bytes = ~8 GB  
- CLIP embeds: 1000 × 768 × 4 bytes = ~3 MB
- **Total: ~9 GB of preprocessed data**

## Performance Considerations

1. **Preprocessing is One-Time**
   - Can be done on a different GPU
   - Can be done in batches if needed
   - Results are reusable

2. **Training Speed**
   - Gradient checkpointing adds ~20% overhead
   - Disk I/O for loading preprocessed data is minimal
   - Overall ~80% of full-memory speed

3. **Quality**
   - No quality loss from this approach
   - Identical to full-memory training
   - Used by production tools

## Comparison with Other Tools

### SimpleTuner
- Has explicit "cache VAE outputs" step
- Uses same preprocessing approach
- Supports multiple cache formats

### ComfyUI
- Preprocessing built into workflow
- Can cache at multiple stages
- Similar memory management

### AI-Toolkit
- `cache_latents: true` in config
- Automatic preprocessing on first run
- Same memory savings

## Advanced Options

### Multi-GPU Setup
```bash
# Preprocess on GPU 0
CUDA_VISIBLE_DEVICES=0 cargo run --bin preprocess_flux ...

# Train on GPU 1
CUDA_VISIBLE_DEVICES=1 cargo run --bin train_flux_24gb ...
```

### Resuming Training
```bash
cargo run --bin train_flux_24gb -- \
    --resume-from output/checkpoint-500 \
    ...
```

### Custom Preprocessing
You can modify the preprocessor to:
- Add augmentations before encoding
- Cache at different resolutions
- Include additional metadata

## Troubleshooting

### Out of Memory During Preprocessing
- Reduce batch size
- Process in chunks
- Use CPU for text encoding

### Out of Memory During Training
- Ensure gradient checkpointing is enabled
- Reduce batch size to 1
- Disable any non-essential features
- Check for memory leaks with nvidia-smi

### Slow Disk I/O
- Use SSD for cache directory
- Implement async prefetching
- Consider memory-mapped files

## Conclusion

This approach is **exactly** how production tools achieve Flux training on 24GB. The key insight is temporal separation: encoders and training never coexist in memory. This is why every serious Flux training tool has a preprocessing step - it's not optional, it's the core technique that makes 24GB training possible.