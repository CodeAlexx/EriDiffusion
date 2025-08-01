# EriDiffusion - Multi-Model Diffusion Trainer

Pure Rust implementation for training modern diffusion models with GPU acceleration.

## Supported Models

### Currently Implemented
- **SDXL** - Stable Diffusion XL 1.0
- **SD 3.5** - Stable Diffusion 3.5 (Medium/Large/Large-Turbo)  
- **Flux** - Black Forest Labs Flux (Dev/Schnell)

### Planned Models
**Image Models:**
- **Flex** - Next-gen architecture
- **OmniGen 2** - Multi-modal generation
- **HiDream** - High-resolution synthesis
- **Chroma** - Advanced color model
- **Sana** - Efficient transformer
- **Kolors** - Bilingual diffusion model

**Video Models:**
- **Wan Vace 2.1** - Video generation
- **LTX** - Long-form video synthesis
- **Hunyuan** - Multi-modal video model

## Features

### Current Features
- ✅ **LoRA Training**: Low-rank adaptation for all supported models
- ✅ **FLAME Framework**: Pure Rust tensor framework with automatic differentiation
- ✅ **GPU-Only**: Industry-standard GPU requirement (no CPU fallback)
- ✅ **ComfyUI Compatible**: Saves LoRA weights in ComfyUI format
- ✅ **Memory Optimized**: Designed for 24GB VRAM with gradient checkpointing
- ✅ **Integrated Sampling**: Generate samples during training to monitor progress
- ✅ **8-bit Adam**: Memory-efficient optimizer
- ✅ **Mixed Precision**: BF16/FP16 training support

### Planned Features
- 🚧 **Full Finetune**: Complete model fine-tuning (not just LoRA)
- 🚧 **DoRA**: Weight-Decomposed Low-Rank Adaptation
- 🚧 **LoKr**: Low-rank Kronecker product adaptation
- 🚧 **Multi-GPU**: Distributed training support
- 🚧 **FSDP**: Fully Sharded Data Parallel training
- 🚧 **Flash Attention 3**: Latest attention optimizations

## Requirements

- CUDA-capable GPU (required - no CPU training support)
- 24GB+ VRAM recommended
- CUDA 11.0 or higher
- Rust 1.70+
- **FLAME Framework** (included as local dependency)

### Important: FLAME Framework

This project uses FLAME (Flexible Learning and Adaptive Memory Engine), a pure Rust tensor framework with:
- GPU-accelerated automatic differentiation
- Conv2D forward and backward passes with CUDA kernels
- Advanced gradient modifications (clipping, normalization, noise injection)
- Native Rust implementation without Python dependencies

## Setup

1. Clone the repository:
```bash
git clone https://github.com/CodeAlexx/EriDiffusion.git
cd EriDiffusion
```

2. The project includes FLAME as a local dependency in the `flame/` directory.

3. Build the project:
```bash
cargo build --release
```

4. The executable will be at `target/release/trainer`. Copy it to your PATH or project root:
```bash
# Option 1: Copy to project root
cp target/release/trainer .

# Option 2: Install to system (requires sudo)
sudo cp target/release/trainer /usr/local/bin/
```

## Usage

### Training Different Models

EriDiffusion uses a single `trainer` binary that automatically detects the model type from your YAML configuration:

```bash
# After building, run from project root:
./trainer config/sdxl_lora_24gb_optimized.yaml
./trainer config/sd35_lora_training.yaml  
./trainer config/flux_lora_24gb.yaml

# Or with full path:
trainer /path/to/config/sdxl_lora_24gb_optimized.yaml
```

The trainer reads the model architecture from the YAML and automatically routes to the correct training pipeline.

### Configuration

Each model has its own config file with model-specific settings:
- Model paths (must be local .safetensors files)
- Dataset location
- Training parameters
- LoRA rank and alpha
- Sampling settings

### Example Config Structure

```yaml
model:
  name_or_path: "/path/to/sdxl_model.safetensors"
  is_sdxl: true

network:
  type: "lora"
  linear: 16  # LoRA rank
  linear_alpha: 16

train:
  batch_size: 1
  steps: 2000
  gradient_accumulation: 4
  lr: 1e-4
  optimizer: "adamw8bit"
  gradient_checkpointing: true
```

## Output

- LoRA weights saved to `output/[model_name]/checkpoints/`
- Sample images saved to `output/[model_name]/samples/`
- All outputs are ComfyUI-compatible

## GPU Memory Usage

With default settings on 24GB GPU:
- Batch size 1: ~18-20GB
- With gradient checkpointing: ~16-18GB
- Higher resolutions (1024x1024) may require VAE tiling

## Technical Details

### Model Architectures

- **SDXL**: U-Net based with dual text encoders (CLIP-L + CLIP-G)
- **SD 3.5**: MMDiT (Multimodal Diffusion Transformer) with triple text encoding
- **Flux**: Hybrid architecture with double/single stream blocks

### Training Approach

All models use the FLAME framework which provides:
1. Automatic differentiation with gradient tracking
2. GPU-accelerated tensor operations with CUDA kernels
3. Advanced gradient modifications (clipping, normalization, noise)
4. Direct safetensors loading and weight management

### FLAME Integration

FLAME (Flexible Learning and Adaptive Memory Engine) is our pure Rust tensor framework that replaces Candle entirely. It provides native automatic differentiation and GPU acceleration without any Python dependencies.

## Roadmap

### Phase 1 (Current)
- ✅ LoRA training for SDXL, SD 3.5, Flux
- ✅ Basic sampling during training
- ✅ Memory optimizations for 24GB GPUs

### Phase 2 (In Progress)
- 🚧 Full model fine-tuning support
- 🚧 Complete sampling for all models
- 🚧 Additional model architectures

### Phase 3 (Planned)
- 📋 Video model support (Wan Vace 2.1, LTX, Hunyuan)
- 📋 Multi-GPU distributed training
- 📋 Advanced adaptation methods (DoRA, LoKr)

See the documentation for detailed development guidelines and model specifications.

## License

MIT OR Apache-2.0