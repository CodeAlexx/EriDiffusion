# Candle Diffusion Models Support Analysis

## Summary

Candle provides primarily **inference-only** support for diffusion models. While it has basic backward pass and optimizer support, there are **NO complete training implementations** for diffusion models in the official Candle repository.

## Supported Diffusion Models (Inference Only)

### 1. **Stable Diffusion 1.5, 2.1, SDXL** 
- **Location**: `candle-examples/examples/stable-diffusion/`
- **Features**:
  - Text-to-image generation
  - Image-to-image generation
  - Inpainting support
  - SDXL and SDXL-Turbo variants
  - Multiple schedulers (DDIM, DDPM, Euler Ancestral)
  - FP16 support for reduced memory usage
  - Flash attention support
- **Status**: Inference only, no training

### 2. **Stable Diffusion 3 & 3.5**
- **Location**: `candle-examples/examples/stable-diffusion-3/`
- **Variants**: 
  - SD 3 Medium
  - SD 3.5 Large
  - SD 3.5 Large-Turbo
  - SD 3.5 Medium
- **Features**:
  - MMDiT architecture support
  - Triple CLIP text encoder
  - Flow matching
  - Skip Layer Guidance (SLG) for SD 3.5 Medium
  - Flash attention support
- **Status**: Inference only, no training

### 3. **Flux**
- **Location**: `candle-examples/examples/flux/`
- **Variants**: Flux Schnell, Flux Dev
- **Features**:
  - T5 text encoder integration
  - Quantized model support
  - Custom attention mechanisms
- **Status**: Inference only, no training

### 4. **Wuerstchen**
- **Location**: `candle-examples/examples/wuerstchen/`
- **Features**:
  - Prior-decoder architecture
  - VQGAN integration
  - Separate CLIP models for prior
- **Status**: Inference only, no training

## Training Capabilities

### What Candle DOES Have:
1. **Basic Autograd Support**:
   - Backward pass via `tensor.backward()`
   - Gradient storage in `GradStore`
   - Variable tracking with `Var`

2. **Optimizers**:
   - SGD (without momentum)
   - AdamW
   - Basic optimizer interface

3. **Training Examples** (Non-Diffusion):
   - MNIST training example
   - Llama2-c training
   - Reinforcement learning examples

### What Candle LACKS for Diffusion Training:
1. **No LoRA/DoRA/LoKr implementations**
2. **No diffusion-specific training loops**
3. **No noise scheduling for training**
4. **No SNR weighting implementations**
5. **No flow matching training**
6. **No memory-efficient training techniques**
7. **No distributed training support for diffusion**
8. **No gradient checkpointing**

## Video Diffusion Models

**NOT SUPPORTED**: No implementations found for:
- LTX
- Hunyuan Video
- Wan Vace 2.1
- HiDream
- Any other video diffusion models

## Missing Features for Complete Training

To implement a diffusion model trainer in Candle, you would need to build:

1. **LoRA/Adapter Layers**: 
   - Low-rank decomposition modules
   - Proper merging/unmerging logic
   - Rank and alpha configuration

2. **Training Infrastructure**:
   - Noise scheduling for forward diffusion
   - Loss computation (MSE, flow matching, v-prediction)
   - SNR weighting
   - Gradient accumulation
   - Mixed precision training
   - Checkpoint saving/loading with LoRA weights

3. **Data Pipeline**:
   - Image loading and preprocessing
   - Caption/text handling
   - Batch generation with proper padding
   - Data augmentation

4. **Memory Optimization**:
   - Gradient checkpointing
   - CPU offloading
   - Attention slicing
   - VAE tiling

## Conclusion

Candle is primarily designed for **inference** of pre-trained diffusion models. While it has the basic building blocks for training (autograd, optimizers), it lacks all the diffusion-specific training infrastructure. 

For a complete SD 3.5/SDXL/Flux trainer, you would need to implement everything from scratch on top of Candle's basic tensor operations and autograd system. This is a significant undertaking compared to using existing training frameworks.