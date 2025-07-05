# Supported Models in AI-Toolkit Rust

This document lists all models supported by the AI-Toolkit Rust implementation for training and inference.

## Overview

AI-Toolkit Rust supports training LoRA/DoRA/LoCoN/LoKr/GLoRA adapters for **ALL** of the following diffusion models, not just SD 3.5. Each model architecture has its own implementation with full forward pass, training support, and optimized inference.

## Image Generation Models

### Stable Diffusion Family

#### SD 1.5
- **Architecture**: UNet2DConditionModel
- **Base Models**: CompVis/stable-diffusion-v1-5, runwayml/stable-diffusion-v1-5
- **VAE Channels**: 4
- **Text Encoder**: CLIP ViT-L/14
- **Training Features**: LoRA on UNet attention/convolution layers
- **Use Cases**: General image generation, fine art, photorealism

#### SDXL
- **Architecture**: UNet2DConditionModel (larger)
- **Base Models**: stabilityai/stable-diffusion-xl-base-1.0
- **VAE Channels**: 4
- **Text Encoders**: Dual CLIP (ViT-L/14 + OpenCLIP ViT-bigG/14)
- **Training Features**: LoRA on UNet + text encoders
- **Use Cases**: High-resolution generation, professional art

#### SD3
- **Architecture**: MMDiT (Multi-Modal Diffusion Transformer)
- **Base Models**: stabilityai/stable-diffusion-3-medium
- **VAE Channels**: 16
- **Text Encoders**: CLIP L + CLIP G + T5-XXL
- **Training Features**: LoRA on MMDiT blocks, flow matching objective
- **Use Cases**: Advanced prompt understanding, complex compositions

#### SD3.5
- **Architecture**: MMDiT (enhanced)
- **Base Models**: 
  - stabilityai/stable-diffusion-3.5-medium
  - stabilityai/stable-diffusion-3.5-large
  - stabilityai/stable-diffusion-3.5-large-turbo
- **VAE Channels**: 16
- **Text Encoders**: CLIP L + CLIP G + T5-XXL
- **Training Features**: LoRA/LoKr on MMDiT blocks, flow matching
- **Use Cases**: State-of-the-art generation, professional workflows

### Flux Family

#### Flux (Base)
- **Architecture**: Flow Transformer
- **Base Models**: black-forest-labs/FLUX.1-dev
- **VAE Channels**: 16
- **Text Encoder**: T5-XXL + CLIP
- **Training Features**: LoRA on double/single transformer blocks
- **Use Cases**: High-quality generation with flow matching

#### Flux Schnell
- **Architecture**: Flow Transformer (distilled)
- **Base Models**: black-forest-labs/FLUX.1-schnell
- **VAE Channels**: 16
- **Text Encoder**: T5-XXL + CLIP
- **Training Features**: LoRA adaptation, 4-step generation
- **Use Cases**: Fast inference, real-time applications

#### Flux Dev
- **Architecture**: Flow Transformer (development)
- **Base Models**: black-forest-labs/FLUX.1-dev
- **VAE Channels**: 16
- **Text Encoder**: T5-XXL + CLIP
- **Training Features**: Full LoRA support, 20-step generation
- **Use Cases**: Development, experimentation

### DiT-Based Models

#### PixArt-α
- **Architecture**: Diffusion Transformer (DiT)
- **Base Models**: PixArt-alpha/PixArt-XL-2-1024-MS
- **VAE Channels**: 8
- **Text Encoder**: T5-XXL
- **Training Features**: LoRA on transformer blocks
- **Use Cases**: Efficient high-res generation

#### PixArt-Σ
- **Architecture**: Enhanced DiT
- **Base Models**: PixArt-alpha/PixArt-Sigma
- **VAE Channels**: 8
- **Text Encoder**: T5-XXL
- **Training Features**: Improved LoRA support
- **Use Cases**: Higher quality than PixArt-α

#### AuraFlow
- **Architecture**: Flow-based DiT
- **Base Models**: fal/AuraFlow
- **VAE Channels**: 16
- **Text Encoder**: T5-XXL
- **Training Features**: Joint attention LoRA
- **Use Cases**: Artistic generation with flow matching

### Advanced Image Models

#### HiDream
- **Architecture**: High-Resolution Diffusion
- **Base Models**: HiDream/HiDream-v1
- **VAE Channels**: 4
- **Training Features**: Multi-scale LoRA
- **Use Cases**: Ultra high-resolution generation

#### OmniGen 2
- **Architecture**: Multi-Modal Transformer
- **Base Models**: OmniGen/OmniGen-v2
- **VAE Channels**: 8
- **Training Features**: Cross-modal LoRA
- **Use Cases**: Text + image conditioning

#### Flex 1 & 2
- **Architecture**: Flexible Diffusion Transformer
- **Base Models**: FlexGen/Flex-v1, FlexGen/Flex-v2
- **VAE Channels**: 8
- **Training Features**: Adaptive LoRA ranks
- **Use Cases**: Variable resolution generation

#### Chroma
- **Architecture**: Color-Focused Diffusion
- **Base Models**: ChromaAI/Chroma-v1
- **VAE Channels**: 4
- **Training Features**: Color-space LoRA
- **Use Cases**: Enhanced color generation

#### Lumina
- **Architecture**: Luminance-Aware Diffusion
- **Base Models**: Alpha-VLLM/Lumina-Next
- **VAE Channels**: 4
- **Training Features**: Brightness-aware LoRA
- **Use Cases**: HDR and lighting control

## Video Generation Models

### Wan 2.1
- **Architecture**: Video Flow Transformer
- **Base Models**: Wan/Wan-v2.1
- **VAE**: Flux-style VAE (16 channels)
- **Training Features**: Temporal LoRA layers
- **Use Cases**: Short video generation

### LTX (Latent Text-to-Video)
- **Architecture**: Latent Video Diffusion
- **Base Models**: Lightricks/LTX-Video
- **VAE Channels**: 8
- **Training Features**: 3D LoRA (spatial + temporal)
- **Use Cases**: Text-to-video generation

### Hunyuan Video
- **Architecture**: Video Diffusion Transformer
- **Base Models**: Tencent/HunyuanVideo
- **VAE Channels**: 8
- **Training Features**: Hierarchical LoRA
- **Use Cases**: High-quality video generation

## Control Models

### KonText
- **Architecture**: Contextual Control for Flux
- **Base Model**: Modifies Flux models
- **Features**: Spatial control without ControlNet
- **Training**: LoRA on control embeddings
- **Use Cases**: Flux with spatial guidance

### ControlNet
- **Architecture**: Conditioning Network
- **Compatible With**: SD1.5, SDXL, SD3/3.5
- **Training Features**: Full network or LoRA adaptation
- **Use Cases**: Pose, depth, canny edge control

### IP-Adapter
- **Architecture**: Image Prompt Adapter
- **Compatible With**: SD1.5, SDXL, SD3/3.5, Flux
- **Variants**: Base, Plus, Full, FaceID
- **Training Features**: Cross-attention LoRA
- **Use Cases**: Image-based conditioning

### T2I-Adapter
- **Architecture**: Lightweight Control
- **Compatible With**: SD1.5, SDXL
- **Training Features**: Adapter LoRA
- **Use Cases**: Efficient conditioning

## Training Capabilities

### Supported Network Types
All models support training with:
- **LoRA**: Low-Rank Adaptation (rank 1-256)
- **DoRA**: Weight-Decomposed LoRA
- **LoCoN**: LoRA for Convolutions
- **LoKr**: LoRA with Kronecker Product
- **GLoRA**: Generalized LoRA with multiple variants

### Training Features
- Multi-GPU distributed training
- Mixed precision (FP16/BF16)
- Gradient accumulation
- Advanced optimizers (AdamW, Lion, Prodigy)
- Learning rate scheduling
- Automatic model architecture detection

### Target Modules
Training can target:
- Attention layers (Q, K, V, O projections)
- Feed-forward networks
- Convolution layers
- Cross-attention layers
- Normalization layers (with GLoRA)
- Text encoders (when applicable)

## Usage Examples

### Training LoRA for Different Models

```bash
# Train LoRA for SD3.5
ai-toolkit train \
  --model stabilityai/stable-diffusion-3.5-large \
  --network lora \
  --rank 32

# Train LoKr for Flux
ai-toolkit train \
  --model black-forest-labs/FLUX.1-dev \
  --network lokr \
  --rank 16 \
  --alpha 16

# Train DoRA for SDXL
ai-toolkit train \
  --model stabilityai/stable-diffusion-xl-base-1.0 \
  --network dora \
  --rank 64

# Train LoRA for PixArt
ai-toolkit train \
  --model PixArt-alpha/PixArt-XL-2-1024-MS \
  --network lora \
  --rank 32
```

### Multi-Model Training
The same dataset can be used to train adapters for multiple models:

```bash
# Train adapters for multiple architectures
ai-toolkit train \
  --models sd15,sdxl,sd3.5,flux \
  --network lora \
  --rank 32 \
  --dataset ./my_dataset
```

## Performance Characteristics

| Model | Parameters | Memory (Training) | Memory (Inference) | Speed |
|-------|-----------|------------------|-------------------|--------|
| SD1.5 | 860M | 6GB | 2.5GB | Fast |
| SDXL | 3.5B | 16GB | 8GB | Medium |
| SD3 Medium | 2B | 12GB | 5GB | Medium |
| SD3.5 Large | 8B | 24GB | 10GB | Slow |
| Flux Dev | 12B | 32GB | 12GB | Slow |
| PixArt-α | 600M | 8GB | 3GB | Fast |
| AuraFlow | 6.8B | 20GB | 8GB | Medium |

## Model Selection Guide

Choose models based on your use case:

- **Fast Generation**: SD1.5, Flux Schnell, PixArt-α
- **High Quality**: SD3.5, Flux Dev, AuraFlow
- **High Resolution**: SDXL, HiDream, SD3.5 Large
- **Video**: Wan 2.1, LTX, Hunyuan Video
- **Artistic**: AuraFlow, Chroma, SD3
- **Photorealism**: SDXL, SD3.5, Flux
- **Resource Constrained**: SD1.5, PixArt-α, Flux Schnell

## Future Models

The architecture is designed to easily add support for new models:
- Mochi (video)
- CogVideoX (video)
- Stable Cascade
- Kandinsky 3
- Custom architectures via plugin system

All models benefit from the same training infrastructure, optimization techniques, and deployment options.