# Flux Training Fix Strategy

## Root Cause Identified
1. Multiple device contexts created during:
   - Model weight loading (`safetensors::load`)
   - Quantization process
   - Model creation
   - Forward pass

2. Even with same DeviceId(1), CUDA contexts are incompatible

## The Fix: Single-Pass Loading

Instead of:
```
Load weights → Quantize → Create model → Forward pass
```

We need:
```
Create model with lazy loading → Load weights on-demand → Forward pass
```

## Implementation Plan

1. **Bypass Quantization** (temporary)
   - Too many device contexts
   - Load full model directly

2. **Use Lazy Loading**
   - Don't load all weights upfront
   - Load only when needed
   - Keep same device context

3. **Memory Management**
   - Use gradient checkpointing
   - CPU offloading for optimizer
   - Smaller batch size

## Next Steps
1. Disable quantization path
2. Implement lazy weight loading
3. Test with full dataset