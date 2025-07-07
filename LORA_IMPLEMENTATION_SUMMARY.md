# LoRA Implementation Summary

## Executive Summary

This document summarizes the comprehensive analysis and implementation plans for enabling LoRA training in EriDiffusion for both Flux and SD 3.5 models.

## Core Problem

The fundamental issue preventing LoRA training in EriDiffusion is **Candle's static architecture**:
- Candle models are compiled with fixed architecture
- No support for dynamic layer modification or interception
- Cannot inject LoRA adapters into existing model layers
- No access to internal model parameters or modules

This is in stark contrast to PyTorch/SimpleTuner which allows:
- Dynamic module replacement at runtime
- Easy parameter access and modification
- PEFT library handling all LoRA complexity
- Automatic gradient flow through adapters

## Current Status

### SD 3.5 Pipeline
**Working:**
- ✅ LoKr training (0.6 it/s on 24GB VRAM)
- ✅ Memory management and VAE offloading
- ✅ Text encoding (CLIP-L, CLIP-G, T5-XXL)
- ✅ Flow matching training objective

**Not Working:**
- ❌ Standard LoRA implementation
- ❌ Native inference (uses external binary)
- ❌ LoRA/LoKr weight application during inference
- ❌ Integrated sampling pipeline

### Flux Pipeline
**Working:**
- ✅ Complete inference/sampling pipeline
- ✅ Model loading and text encoding
- ✅ Patchification and flow matching
- ✅ All core infrastructure

**Not Working:**
- ❌ LoRA training (completely non-functional)
- ❌ Parameter access from model
- ❌ Gradient flow through adapters
- ❌ Real data preprocessing (uses dummy tensors)

## Solution Strategy

Since we cannot modify existing Candle models, we must create new implementations with LoRA built-in from the ground up.

### Key Approach: LoRA-Aware Modules

Instead of trying to inject LoRA into existing layers, create new layer types that include both base weights and optional LoRA parameters:

```rust
pub struct LinearWithLoRA {
    // Base layer (frozen)
    weight: Tensor,
    bias: Option<Tensor>,
    
    // LoRA parameters (trainable)
    lora_a: Option<Var>,  // Down projection: [rank, in_features]
    lora_b: Option<Var>,  // Up projection: [out_features, rank]
    
    // Configuration
    rank: Option<usize>,
    alpha: f32,
}
```

### Implementation Pattern

1. **Load base model weights** into our custom architecture
2. **Initialize LoRA parameters** where needed
3. **Forward pass** combines base + LoRA: `output = Wx + BAx`
4. **Only LoRA parameters** are trainable
5. **Gradient flows** only through LoRA weights

## Implementation Timeline

### Flux LoRA (5 weeks)

**Week 1: Foundation**
- Create LinearWithLoRA and AttentionWithLoRA modules
- Implement proper forward pass with gradient support
- Test gradient flow through LoRA parameters

**Week 2: Model Architecture**
- Implement FluxDoubleBlockWithLoRA
- Implement FluxSingleBlockWithLoRA
- Create complete FluxModelWithLoRA

**Week 3: Training Integration**
- Fix preprocessing (real VAE/text encoding)
- Connect to optimizer (LoRA params only)
- Implement flow matching loss

**Week 4: Validation**
- Gradient flow tests
- Training convergence tests
- Inference with trained LoRA

**Week 5: Optimization**
- Memory profiling
- Implement checkpointing
- Create configs for 24GB VRAM

### SD 3.5 LoRA (5 weeks)

**Week 1: Standard LoRA**
- Create SD35LoRALayer
- Implement MMDiTBlockWithLoRA
- Add to existing infrastructure

**Week 2: Native Inference**
- Complete SD35NativeSampler
- Fix MMDiT forward pass
- Replace external binary dependency

**Week 3: Training Updates**
- Create SD35LoRATrainer
- Integrate with existing pipeline
- Add sampling during training

**Week 4: Testing**
- Verify against LoKr baseline
- Test weight loading/merging
- Validate generation quality

**Week 5: Polish**
- Multi-resolution support
- Memory optimizations
- Migration guide from LoKr

## Technical Challenges & Solutions

### Challenge 1: No Dynamic Model Modification
**Solution**: Create new model architectures with LoRA built-in

### Challenge 2: Gradient Flow
**Solution**: Use Candle's Var type for trainable parameters, ensure proper backward pass

### Challenge 3: Memory Constraints (24GB)
**Solution**: 
- Gradient checkpointing with configurable intervals
- VAE offloading after encoding
- LoRA-only parameter updates (much smaller than full model)

### Challenge 4: Compatibility
**Solution**: Load base model weights into our architecture, maintain same interface

## Success Criteria

1. **Functional Training**
   - LoRA weights update with non-zero gradients
   - Training loss decreases properly
   - Optimizer only updates LoRA parameters

2. **Working Inference**
   - Can generate images with trained LoRA
   - Native implementation (no external binaries)
   - Proper weight application

3. **Memory Efficiency**
   - Fits in 24GB VRAM
   - Comparable to SimpleTuner performance
   - Supports batch size 1 with accumulation

4. **Code Quality**
   - No dummy/fake implementations
   - All components use real computations
   - Comprehensive test coverage

## Key Differences from SimpleTuner

| Aspect | SimpleTuner | EriDiffusion |
|--------|-------------|--------------|
| Framework | PyTorch | Candle |
| LoRA Method | PEFT library | Custom implementation |
| Model Access | Full access via hooks | Must rebuild model |
| Memory | 9-18GB with quantization | ~22GB without quantization |
| Flexibility | High (dynamic) | Low (static) |

## Recommendations

1. **Start with Flux** - Has working inference to build upon
2. **Implement minimal version first** - Single layer LoRA for testing
3. **Verify each phase** - Don't proceed without working gradients
4. **Consider alternatives** - If Candle limitations prove insurmountable:
   - Fork Candle to add needed features
   - Use different Rust ML framework
   - Consider hybrid approach (inference only)

## Files Created

1. **COMPREHENSIVE_PIPELINE_ANALYSIS.md** - Detailed analysis of current state
2. **FLUX_LORA_IMPLEMENTATION_PLAN.md** - 5-week plan for Flux LoRA
3. **SD35_LORA_IMPLEMENTATION_PLAN.md** - 5-week plan for SD 3.5 LoRA
4. **FLUX_PIPELINE_STATUS.md** - Current Flux implementation status
5. **SD35_PIPELINE_STATUS.md** - Current SD 3.5 implementation status

## Conclusion

Enabling LoRA training in EriDiffusion requires significant architectural changes due to Candle's limitations. The proposed solution of creating new model implementations with built-in LoRA support is feasible but requires substantial development effort. The plans provide a clear path forward with realistic timelines and concrete implementation details.