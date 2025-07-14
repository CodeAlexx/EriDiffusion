# EriDiffusion - Multi-Model Diffusion Trainer

Pure Rust implementation for training modern diffusion models with GPU acceleration.

## Supported Models

- **SDXL** - Stable Diffusion XL 1.0
- **SD 3.5** - Stable Diffusion 3.5 (Medium/Large/Large-Turbo)  
- **Flux** - Black Forest Labs Flux (Dev/Schnell)

## Features

- ✅ **Var-based Training**: Direct gradient tracking without VarBuilder limitations
- ✅ **GPU-Only**: Industry-standard GPU requirement (no CPU fallback)
- ✅ **ComfyUI Compatible**: Saves LoRA weights in ComfyUI format
- ✅ **Memory Optimized**: Designed for 24GB VRAM with gradient checkpointing
- ✅ **Integrated Sampling**: Generate samples during training to monitor progress
- ✅ **8-bit Adam**: Memory-efficient optimizer
- ✅ **Mixed Precision**: BF16/FP16 training support

## Requirements

- CUDA-capable GPU (required - no CPU training support)
- 24GB+ VRAM recommended
- CUDA 11.0 or higher
- Rust 1.70+
- **Trainable-Candle fork** (required for training support)

### Important: Trainable-Candle Fork

This project requires the Trainable-Candle fork from https://github.com/CodeAlexx/Trainable-Candle which provides:
- GPU-accelerated LoRA backward pass with cuBLAS
- Direct Var creation for training (bypasses VarBuilder limitations)
- Training-enabled Candle without the inference-only restrictions

## Setup

1. Clone both repositories:
```bash
# Clone Trainable-Candle fork (required)
git clone https://github.com/CodeAlexx/Trainable-Candle.git

# Clone EriDiffusion
git clone https://github.com/CodeAlexx/EriDiffusion.git
cd EriDiffusion
```

2. Update Cargo.toml to point to your local Trainable-Candle:
```toml
[dependencies]
candle-core = { path = "../Trainable-Candle/candle-core", features = ["cuda", "cuda-backward"] }
candle-nn = { path = "../Trainable-Candle/candle-nn" }
candle-transformers = { path = "../Trainable-Candle/candle-transformers" }
```

3. Build the project:
```bash
cargo build --release --features cuda-backward
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

All models use the Trainable-Candle fork which enables:
1. Direct `Var::from_tensor()` for trainable parameters (no VarBuilder)
2. GPU-accelerated LoRA backward pass with cuBLAS
3. Gradient tracking throughout the entire model
4. Direct safetensors loading without inference-only limitations

### Key Differences from Standard Candle

Standard Candle's VarBuilder returns immutable `Tensor` objects, making training impossible. The Trainable-Candle fork bypasses this entirely, allowing us to create trainable `Var` objects directly and implement proper backpropagation.

## License

MIT OR Apache-2.0