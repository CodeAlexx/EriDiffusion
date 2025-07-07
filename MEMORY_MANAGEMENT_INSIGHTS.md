# Memory Management Insights for Diffusion Model Training

## Date: July 5, 2025

### Summary of Achievements

We successfully implemented SimpleTuner-style memory management in pure Rust using Candle, achieving the same memory efficiency without PyTorch's meta device.

### Key Insights

## 1. SimpleTuner's Memory Management Strategy

SimpleTuner avoids OOM by using a three-phase approach:
1. **Preprocessing Phase**: Load VAE/text encoders → encode data → cache results
2. **Unload Phase**: Move models to "meta" device, set to None, call torch.cuda.empty_cache()
3. **Training Phase**: Load only the main model with cached data

The "meta" device is PyTorch-specific - it holds model structure without allocating memory for weights.

## 2. Our Rust/Candle Implementation

Since Candle doesn't have a meta device, we achieved the same result through:

### Phase-Based Model Loading
```rust
pub struct FluxLoRATrainer {
    // Models as Options - only loaded when needed
    model: Option<FluxModelWithLoRA>,
    vae: Option<AutoencoderKL>,
    vae_cpu: Option<AutoencoderKL>,
    text_encoders: Option<TextEncoders>,
    
    // Cached data persists across phases
    latent_cache: HashMap<usize, Tensor>,
    text_embed_cache: HashMap<usize, (Tensor, Tensor)>,
}
```

### Memory Management Utilities
Created Candle equivalents of PyTorch's memory functions:
- `MemoryManager::empty_cache()` - Forces CUDA synchronization and cache cleanup
- `MemoryManager::cleanup()` - Comprehensive memory cleanup
- `MemoryManager::log_memory_usage()` - Track memory between operations

### Three-Phase Training
1. **Preprocessing**: 
   - Load VAE → Encode images → Drop VAE → Cleanup memory
   - Load text encoders → Encode text → Drop encoders → Cleanup memory
   - Cache all results to disk

2. **Memory Cleanup**:
   - Explicit drops (Rust immediately frees memory unlike Python GC)
   - CUDA synchronization
   - Memory pool cleanup

3. **Training**:
   - Load only the main model
   - Use cached embeddings (no VAE/text encoders in memory)

## 3. Quantization Requirements

SimpleTuner handles large models like Flux through quantization:
- **No quantization**: 30GB VRAM
- **int8 quantization**: 18GB VRAM (suitable for 24GB cards)
- **int4 quantization**: 13GB VRAM
- **NF4/int2 quantization**: 9GB VRAM

They use libraries like `optimum-quanto` and `torchao` with selective quantization, excluding:
- Normalization layers
- Embeddings
- Output projections

## 4. Key Differences: Rust vs Python Memory Management

### Python/PyTorch:
- Garbage collected (non-deterministic)
- Requires explicit torch.cuda.empty_cache()
- Meta device for zero-memory model structures
- Reference counting complexity

### Rust/Candle:
- Deterministic memory management
- Memory freed immediately on drop
- No meta device - use Option<Model>
- Simpler, more predictable behavior

## 5. Implementation Success

Our approach successfully:
1. ✅ Preprocessed all 55 images and text samples
2. ✅ Properly unloaded models between phases
3. ✅ Maintained memory efficiency throughout
4. ✅ Created reusable memory management utilities

## 6. Remaining Challenges

1. **Model Size**: Even with preprocessing complete, Flux model itself needs ~12GB
2. **Quantization**: Need to implement int8 quantization in Candle or use pre-quantized models
3. **Tensor Naming**: Minor issue with Flux tensor names (w1/w2 vs mlp naming)

## 7. Future Model Support

This memory management pattern can be applied to other large models:
- **SD 3.5**: Similar phase-based loading
- **Video models**: Even more critical due to larger memory requirements
- **Multi-modal models**: Can load/unload modality-specific encoders

## 8. Code Patterns for Future Models

### Pattern 1: Deferred Loading
```rust
impl Trainer {
    fn new() -> Self {
        Self {
            model: None,  // Don't load until needed
            preprocessors: None,
        }
    }
    
    fn load_for_phase(&mut self, phase: Phase) {
        match phase {
            Phase::Preprocessing => self.load_preprocessors(),
            Phase::Training => self.load_model(),
        }
    }
}
```

### Pattern 2: Explicit Cleanup
```rust
fn process_phase(&mut self) -> Result<()> {
    self.load_components()?;
    let result = self.do_work()?;
    self.unload_components()?;
    MemoryManager::cleanup()?;
    result
}
```

### Pattern 3: Memory Tracking
```rust
track_memory!("VAE encoding", {
    self.encode_images()?
});
```

## 9. Performance Optimizations

1. **Cache Everything**: Disk I/O is faster than re-encoding
2. **Process in Batches**: But drop each batch's intermediates
3. **Use CPU Loading**: Load to CPU first, then move to GPU when needed
4. **Block Swapping**: For training, swap attention blocks as needed

## 10. Lessons Learned

1. **SimpleTuner's approach works** - The phase-based loading is universally applicable
2. **Rust's memory model is advantageous** - More predictable than Python's GC
3. **Quantization is essential** - For 24GB cards, int8 is the sweet spot
4. **Memory logging is crucial** - Track usage at each phase
5. **Preprocessing can be separated** - This is key to fitting large models

## Next Steps

1. Implement int8 quantization for Candle
2. Add support for loading GGUF quantized models
3. Create a generic trait for phase-based model loading
4. Add memory profiling to identify bottlenecks

## Conclusion

We successfully demonstrated that SimpleTuner's memory management strategy can be implemented in pure Rust with Candle, achieving the same memory efficiency through different mechanisms. The key insight is that phase-based loading with aggressive cleanup between phases allows training of models that wouldn't otherwise fit in memory.