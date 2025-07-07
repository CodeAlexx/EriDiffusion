# Flux LoRA Training on 24GB GPU - Memory Solution

## Problem
- Flux-dev model: 22GB
- 24GB GPU cannot fit model + training overhead (gradients, optimizer states, activations)
- Total memory needed: ~30-35GB minimum

## Solution Implemented

### 1. Memory Infrastructure (`src/memory/`)
- **QuantizationMode**: INT8, INT4, NF4, FP8 support
- **BlockSwapManager**: Dynamic block swapping between GPU/CPU/Disk
- **MemoryPool**: Optimized allocation with Flux-specific config
- **QuantoManager**: Full quantization system based on production implementations

### 2. Quantized Model Loader (`src/trainers/flux_quantized_loader.rs`)
- Loads full model to CPU first (avoids GPU OOM)
- Quantizes to INT8 (22GB → 11GB)
- Moves quantized weights to GPU in batches
- Integrates with memory pool and block swapping

### 3. Updated Flux LoRA Trainer
- Detects pre-quantized models (FP8/INT8)
- Falls back to quantization for full models
- Memory-aware loading strategy

## Usage

### Option 1: Use Pre-Quantized Model (Recommended)
```yaml
model:
  name_or_path: "/path/to/flux1-dev-kontext_fp8_scaled.safetensors"
  arch: "fluxdev"
```

### Option 2: Automatic INT8 Quantization
```yaml
model:
  name_or_path: "/path/to/flux1-dev.safetensors"  # Full 22GB model
  arch: "fluxdev"
  quantize: true  # Enable INT8 quantization
```

## Memory Breakdown

### Full Precision (Cannot fit)
- Model: 22GB
- Gradients: 22GB  
- Optimizer: 44GB (Adam)
- Total: ~88GB needed

### INT8 Quantized (Fits with optimizations)
- Model: 11GB (INT8)
- Gradients: 2-4GB (LoRA only)
- Optimizer: 4-8GB (LoRA params)
- Activations: 2-3GB (with checkpointing)
- Total: ~20-26GB (fits in 24GB!)

## Key Features

1. **Quanto-style Quantization**
   - INT8/INT4/NF4 support
   - Per-layer configuration
   - Minimal accuracy loss

2. **Block Swapping**
   - Swap transformer blocks GPU↔CPU
   - Keep 8-12 blocks active
   - Prefetch next blocks

3. **Memory Pool**
   - Pre-allocated blocks
   - Garbage collection
   - Defragmentation

## Status

✅ Memory infrastructure complete
✅ Quantization system implemented
✅ Block swapping available
✅ Detection of pre-quantized models
⚠️ Full integration with training loop in progress

## Recommendations

For immediate use on 24GB GPU:
1. Use the FP8 model: `flux1-dev-kontext_fp8_scaled.safetensors`
2. Enable gradient checkpointing
3. Use batch size 1
4. Enable mixed precision (bf16)

The INT8 quantization path is implemented and will be fully integrated soon.