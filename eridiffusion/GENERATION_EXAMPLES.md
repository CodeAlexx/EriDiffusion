# AI-Toolkit Generation Examples

This directory contains several examples demonstrating how to generate images and videos with all 20 supported models in ai-toolkit-rs.

## 🎨 Image Generation Examples

### 1. Quick Generation Demo (`quick_generate.rs`)
Demonstrates basic generation with the most popular models:
- **SD1.5** - 512x512 classic stable diffusion
- **SDXL** - 1024x1024 high-resolution 
- **SD3.5** - Latest stable diffusion with flow matching
- **Flux** - State-of-the-art flow-based model

```bash
cargo run --example quick_generate
```

### 2. All Models Generation (`generate_all_models.rs`)
Generates images with all 20 supported architectures:

**Image Models:**
- SD1.5, SDXL, SD3, SD3.5
- Flux, Flux-Schnell (4-step), Flux-Dev
- PixArt-α, PixArt-Σ
- AuraFlow
- HiDream
- KonText (contextual control for Flux)
- OmniGen 2
- Flex 1, Flex 2
- Chroma
- Lumina

**Video Models (single frame):**
- Wan 2.1 (uses Flux VAE)
- LTX (Latent Text-to-Video)
- Hunyuan Video

```bash
cargo run --example generate_all_models -- \
    --prompt "Your prompt here" \
    --output-dir outputs \
    --models all  # or comma-separated list
```

### 3. Real Model Loading (`real_generation.rs`)
Shows how to load actual model weights from safetensors files:

```bash
cargo run --example real_generation -- \
    --model-dir /path/to/sd35/model \
    --model sd35 \
    --prompt "A beautiful landscape" \
    --adapter /path/to/lokr/adapter.safetensors \
    --steps 28 \
    --cfg 7.0
```

## 🎬 Video Generation Examples

### Video Generation Demo (`video_generation.rs`)
Demonstrates video generation with temporal models:

```bash
cargo run --example video_generation
```

Supports:
- **Wan 2.1** - 16 frames at 1024x576 (16:9)
- **LTX** - 24 frames at 768x512 (3:2)
- **Hunyuan Video** - 32 frames at 1280x720 (HD)

## 🚀 Quick Start

1. **Run all demos:**
   ```bash
   ./run_generation_demo.sh
   ```

2. **Generate with specific model:**
   ```bash
   cargo run --example generate_all_models -- --models sd35,flux
   ```

3. **Custom settings:**
   ```bash
   cargo run --example quick_generate -- \
       --prompt "cyberpunk city" \
       --steps 50 \
       --seed 12345
   ```

## 🔧 Model-Specific Settings

Each model has optimized default settings:

| Model | Resolution | Steps | CFG | Scheduler |
|-------|-----------|-------|-----|-----------|
| SD1.5 | 512x512 | 50 | 7.5 | DDIM |
| SDXL | 1024x1024 | 40 | 7.5 | DDIM |
| SD3/3.5 | 1024x1024 | 28 | 7.0 | Flow Matching |
| Flux | 1024x1024 | 20 | 3.5 | Flow Matching |
| Flux-Schnell | 1024x1024 | 4 | 0.0 | Flow Matching |

## 📝 Notes

- These examples use dummy models for demonstration
- For real generation, you need to download model weights
- Video models generate latent sequences that would be decoded to frames
- All models support LoKr/LoRA adapters for customization
- Memory requirements vary from ~1GB (SD1.5) to ~16GB (Hunyuan Video)