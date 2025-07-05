# Working Diffusion Models in AI-Toolkit-RS

This document lists all the diffusion models that are currently working with real inference in ai-toolkit-rs.

## ✅ Fully Working Models

### 1. Stable Diffusion 1.5
- **Status**: Fully working
- **Example**: `cargo run --example sd15_generate`
- **Features**:
  - Text-to-image generation
  - 512x512 default resolution
  - DDIM/DDPM schedulers
  - Classifier-free guidance
- **Model Path**: `/home/alex/SwarmUI/Models/Stable-Diffusion/v1-5-pruned-emaonly.safetensors`

### 2. Stable Diffusion XL (SDXL)
- **Status**: Fully working
- **Example**: `cargo run --example sdxl_generate`
- **Features**:
  - Text-to-image generation
  - 1024x1024 default resolution
  - Dual text encoders
  - Advanced conditioning
- **Model Path**: `/home/alex/SwarmUI/Models/Stable-Diffusion/sdXL_v10.safetensors`

### 3. Stable Diffusion 3.5
- **Status**: Fully working (via Candle integration)
- **Example**: `cargo run --example sd3_generate`
- **Features**:
  - Flow matching objective
  - MMDiT architecture
  - Multiple model sizes (Medium/Large/Large-Turbo)
  - T5 + CLIP text encoding
- **Model Path**: `/home/alex/SwarmUI/Models/diffusion_models/sd3.5_large.safetensors`

### 4. Flux (Dev/Schnell)
- **Status**: Working with basic inference
- **Example**: `cargo run --example flux_generate`
- **Features**:
  - Flow matching generation
  - Schnell variant (4 steps)
  - Dev variant (28+ steps)
  - T5 + CLIP conditioning
- **Model Path**: `/home/alex/SwarmUI/Models/diffusion_models/flux1-schnell.safetensors`

## 🚀 Quick Start

### Run All Models Demo
```bash
# List all available models
cargo run --example run_all_models -- list

# Run specific model
cargo run --example run_all_models -- sd15 --prompt "A beautiful sunset"
cargo run --example run_all_models -- sdxl --prompt "A futuristic city"
cargo run --example run_all_models -- sd35 --prompt "A magical forest"
cargo run --example run_all_models -- flux --prompt "A cyberpunk scene"

# Benchmark all models
cargo run --example run_all_models -- benchmark
```

### Unified Pipeline
```bash
# Use any model with unified interface
cargo run --example unified_pipeline -- --model sd15 --prompt "Your prompt"
cargo run --example unified_pipeline -- --model sdxl --prompt "Your prompt"
cargo run --example unified_pipeline -- --model sd35 --prompt "Your prompt"
cargo run --example unified_pipeline -- --model flux --prompt "Your prompt"
```

## 📊 Model Comparison

| Model | Resolution | Steps | Speed | Memory | Quality |
|-------|------------|-------|-------|--------|---------|
| SD 1.5 | 512x512 | 50 | Fast | ~1GB | Good |
| SDXL | 1024x1024 | 30 | Medium | ~7GB | Excellent |
| SD 3.5 | 1024x1024 | 28 | Medium | ~10GB | Best |
| Flux | 1024x1024 | 4-28 | Fast/Medium | ~12GB | Excellent |

## 🔧 Advanced Usage

### Custom Resolution
```bash
cargo run --example sdxl_generate -- \
  --prompt "A landscape" \
  --width 1920 \
  --height 1080
```

### Seed Control
```bash
cargo run --example sd35_generate -- \
  --prompt "A portrait" \
  --seed 42
```

### CPU Mode
```bash
cargo run --example sd15_generate -- \
  --prompt "A painting" \
  --cpu
```

## 🛠️ Implementation Details

All models use:
- Candle for tensor operations
- Safetensors for weight loading
- Pure Rust implementation
- CUDA support for GPU acceleration
- Efficient memory management

## 🚧 Models In Development

The following models have basic structure but need full implementation:
- PixArt-α/Σ (DiT-based)
- AuraFlow (Flow matching)
- Lumina (Next-gen DiT)
- OmniGen v2 (Multi-modal)

## 📝 Notes

- All models load from local safetensors files
- No automatic downloading - models must be pre-downloaded
- VRAM requirements vary by model (1GB to 12GB)
- All examples support both GPU and CPU execution