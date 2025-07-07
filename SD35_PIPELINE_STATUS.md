# SD 3.5 Pipeline Status Report

## Overview
This document provides a comprehensive analysis of the SD 3.5 implementation in the EriDiffusion codebase, detailing what works, what doesn't, and what's missing.

## Working Components ✅

### 1. LoKr Training Pipeline
- **Full training loop** with flow matching objective
- **Memory-efficient training** with VAE CPU offloading (SimpleTuner-style)
- **Proper text encoding** integration:
  - CLIP-L encoder support
  - CLIP-G encoder support  
  - T5-XXL encoder support
- **SNR gamma weighting** for improved convergence
- **Checkpoint saving** in safetensors format
- **Performance**: ~0.6 it/s on 24GB VRAM GPUs

### 2. Memory Management
- **VAE offloading** to CPU during training to save VRAM
- **Latent caching** to disk for efficient data loading
- **Gradient checkpointing** support for memory optimization
- **Optimized for 24GB GPUs** with specific configurations

### 3. Text Encoding Pipeline
- **Triple encoder setup** properly implemented
- **Tokenizer support** for all three text encoders
- **Pooled and sequence embeddings** correctly generated
- **Caching system** for encoded text to speed up training

## Non-Functional Components ❌

### 1. Integrated Sampling/Inference
- **No native SD 3.5 inference pipeline** in the codebase
- **`mmdit_forward()` is a placeholder** - returns random noise instead of actual predictions
- **External dependency** on candle's standalone binary for image generation
- **Cannot apply LoKr weights** during inference within the same codebase

### 2. LoKr Weight Application
- **Sampler receives weights but cannot use them** - no implementation for applying LoKr
- **No MMDiT layer interception** - LoKr layers aren't integrated into the model forward pass
- **Weight merging attempted but non-functional** - merge utility exists but doesn't work properly

### 3. Model Architecture Issues
- **RMS norm CUDA workarounds** indicate framework limitations
- **MMDiT forward pass doesn't integrate LoKr** - adapters are trained but not used
- **Incomplete wrapper architecture** - `mmdit_lokr_wrapper.rs` is scaffolding without implementation

### 4. Output Format Limitations
- **PPM file output only** - no standard image format support (PNG/JPEG)
- **Hacky JPG to PPM conversion** when using external sampler
- **No built-in image processing** pipeline

## Missing Components 🚧

### 1. Native Inference Pipeline
- **No integrated diffusion sampling loop**
- **Missing timestep scheduling** for inference
- **No classifier-free guidance** implementation
- **No noise scheduling** for the reverse diffusion process

### 2. Proper LoKr Integration Architecture
- **Layer-wise interception not implemented** 
- **Cannot apply trained adapters** during generation
- **MMDiT wrapper incomplete** - needs full forward pass modification
- **No dynamic weight application** system

### 3. Network Adapter Support
- **Only LoKr implemented** for training
- **LoRA is a stub** - no actual implementation
- **DoRA is a stub** - no actual implementation  
- **LoCoN is a stub** - no actual implementation
- **No support for SD 3.5 variants** (Medium, Large-Turbo)

### 4. Quality of Life Features
- **No progress visualization** during training
- **Limited validation** - no periodic sample generation
- **No automatic mixed precision** handling
- **Missing training metrics** beyond basic loss reporting

## Architectural Problems

### Core Issue: Training-Inference Separation
The fundamental architectural problem is the complete separation between training and inference:

1. **Training produces LoKr weights** that cannot be used
2. **Inference relies on external binary** (candle SD3 example)
3. **No unified pipeline** connecting training outputs to inference inputs
4. **Weights are saved but unusable** within the same codebase

### Framework Limitations
The codebase shows clear signs of hitting Candle framework limitations:

1. **CUDA operation issues** - RMS norm requires workarounds
2. **Dynamic model modification** - Cannot properly inject LoKr layers
3. **Missing operators** - Some operations not available in Candle
4. **External binary dependency** - Suggests core functionality gaps

### Design Flaws
1. **Placeholder implementations** throughout sampling code
2. **External process spawning** for basic inference
3. **Temporary file juggling** between components
4. **No architectural consistency** between training and inference

## Current Workarounds

### 1. External Sampler Approach
```rust
// Current approach: spawn external process
Command::new("candle_sd3_binary")
    .args(&["--prompt", prompt])
    .spawn()
```

### 2. Memory Management Hacks
- Drop entire model before sampling
- Reload model after sampling
- Force garbage collection with sleep

### 3. File Format Conversions
- Generate with external tool (JPG)
- Convert to PPM for consistency
- Clean up temporary files

## Required Fixes

### Priority 1: Implement Native Inference
1. Implement proper `mmdit_forward()` function
2. Add diffusion sampling loop
3. Integrate LoKr weight application
4. Remove external binary dependency

### Priority 2: Complete LoKr Integration  
1. Implement MMDiT layer interception
2. Add dynamic weight application
3. Complete the wrapper architecture
4. Test weight merging functionality

### Priority 3: Add Missing Features
1. Implement other network adapters (LoRA, DoRA)
2. Add proper image format support
3. Implement validation sampling
4. Add training visualization

### Priority 4: Architecture Refactor
1. Unify training and inference pipelines
2. Remove external dependencies
3. Implement missing Candle operations
4. Create consistent API surface

## Conclusion

The SD 3.5 implementation has a **functional training pipeline** that can successfully train LoKr adapters, but it **cannot use those adapters for image generation**. The inference side is essentially non-existent, replaced by external binary calls that cannot leverage the trained weights. This makes the current implementation useful only for training experiments, not for practical image generation with trained models.

The path forward requires implementing a complete inference pipeline within the codebase, properly integrating LoKr weight application, and removing the dependency on external binaries. Until these issues are addressed, the SD 3.5 pipeline remains incomplete despite having working training capabilities.