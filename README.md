# EriDiffusion

**PURE RUST DIFFUSION TRAINER**

## Features

- **100% Rust** - No Python dependencies whatsoever
- **All Diffusion Models** - SD 1.5, SD 2.1, SDXL, SD 3.5, Flux, and more
- **All Training Methods** - LoRA, LoKr, DoRA, LoCoN, DreamBooth, Textual Inversion
- **Production Ready** - 6.6+ it/s on RTX 3090 Ti
- **Memory Efficient** - Gradient checkpointing, mixed precision, and intelligent caching

## Quick Start

```bash
cd eridiffusion
cargo build --release --bin trainer
./target/release/trainer /path/to/config.yaml
```

Supports all diffusion models and training methods. Example configs in `/config/`

## Project Structure

- `eridiffusion/` - Main trainer implementation (Candle-based)
- `config/` - Training configuration examples
- `tokenizers/` - Required tokenizer files

## Requirements

- Rust 1.70+
- CUDA 11.8+ (for GPU training)
- 24GB+ VRAM recommended

## License

MIT OR Apache-2.0