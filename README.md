# EriDiffusion - SDXL LoRA Trainer

Pure Rust implementation of SDXL LoRA training with GPU acceleration.

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

## Setup

1. Clone the repository:
```bash
git clone https://github.com/CodeAlexx/EriDiffusion.git
cd EriDiffusion
```

2. Build the project:
```bash
cargo build --release
```

## Usage

### Basic Training

```bash
cargo run --release --bin train_sdxl_lora -- config/sdxl_lora_24gb_optimized.yaml
```

### Configuration

Edit `config/sdxl_lora_24gb_optimized.yaml` to customize:
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

This implementation uses a custom Candle fork that enables training by:
1. Bypassing VarBuilder to create trainable Var parameters
2. Using GPU-accelerated LoRA backward pass with cuBLAS
3. Direct safetensors loading without inference-only limitations

## License

MIT OR Apache-2.0