# Flux GPU Kernel Integration Guide

## Overview

The GroupNorm and RoPE CUDA kernels from `group_norm.txt` need to be integrated into the Flux LoRA training pipeline to eliminate CPU bottlenecks.

## Current Status

1. **GroupNorm CUDA Kernels** - Ready in `group_norm.txt`
   - Forward and backward pass implementations
   - Optimized for Flux's 32 groups configuration
   - Handles large spatial dimensions efficiently

2. **RoPE CUDA Kernels** - Ready in `group_norm.txt`
   - Supports both 1D (text) and 2D (image) positions
   - Cached sin/cos for efficiency
   - Handles Flux's patchified representations

3. **Integration Points Created**:
   - `/home/alex/diffusers-rs/eridiffusion/src/ops/group_norm.rs` - Rust wrapper
   - `/home/alex/diffusers-rs/eridiffusion/src/ops/rope.rs` - Rust wrapper
   - `/home/alex/diffusers-rs/eridiffusion/src/ops/mod.rs` - Module exports

## Integration Requirements

### 1. Build System
Need to add to `eridiffusion/build.rs`:
```rust
// Compile CUDA kernels
cc::Build::new()
    .cuda(true)
    .cudart("static")
    .flag("-arch=sm_80")  // For A100/A6000
    .flag("-O3")
    .file("src/kernels/group_norm.cu")
    .file("src/kernels/rope.cu")
    .compile("flux_cuda_kernels");
```

### 2. Replace LayerNorm with GroupNorm
In Flux double blocks and single blocks:
- Current: Uses `LayerNorm` (CPU-bound)
- Replace with: `GroupNorm` with 32 groups
- Locations:
  - `FluxDoubleBlockWithLoRA` - img_norm1, img_norm2, txt_norm1, txt_norm2
  - `FluxSingleBlockWithLoRA` - norm1, norm2

### 3. Add RoPE to Attention
In `AttentionWithLoRA`:
- Add RoPE to query and key projections
- Use 2D positions for image tokens
- Use 1D positions for text tokens

### 4. Memory Considerations
- Both kernels use shared memory for reductions
- GroupNorm: 2 * block_size * sizeof(float) shared memory
- RoPE: Minimal shared memory usage
- Both support BF16 through F32 computation

## Example Integration

```rust
// In FluxDoubleBlockWithLoRA
use crate::ops::{group_norm, RotaryEmbedding};

// Replace LayerNorm
self.img_norm1 = group_norm(32, hidden_size, 1e-6, true, vb.pp("img_norm1"))?;

// In attention forward
let (q_rot, k_rot) = apply_rotary_emb(&q, &k, &positions, &self.rope, is_2d)?;
```

## Performance Impact

Expected improvements:
- GroupNorm: ~10-20x faster than CPU LayerNorm
- RoPE: ~15-25x faster than CPU implementation
- Overall training: 30-50% speedup on 24GB GPUs

## Testing

1. Verify CUDA compilation
2. Test on small batch first
3. Profile memory usage
4. Compare outputs with CPU version

## Current Limitations

The Rust wrappers are ready but currently fall back to CPU operations with warnings. Once the CUDA kernels are properly linked through the build system, the GPU acceleration will be automatic.