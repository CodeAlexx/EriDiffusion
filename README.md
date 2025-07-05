# EriDiffusion

**PURE RUST DIFFUSION TRAINER**

## Features

- **100% Rust** - No Python dependencies whatsoever
- **SD 3.5 LoKr Training** - Currently implemented and working at 6.6+ it/s on RTX 3090 Ti
- **More Models Coming** - SDXL, Flux, and others in development
- **Memory Efficient** - Gradient checkpointing, mixed precision, and intelligent caching
- **Pure Candle Framework** - No PyTorch, no Python

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