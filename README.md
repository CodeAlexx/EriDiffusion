# EriDiffusion

Pure Rust implementation of modern diffusion model trainers. No Python dependencies.

## Features

- **SD 3.5 LoKr Training** - Production-ready trainer with 6.6+ it/s on RTX 3090 Ti
- **Pure Rust** - Built with Candle framework, no Python required
- **Memory Efficient** - Gradient checkpointing, mixed precision, and caching
- **Modern Models** - Supports SD 3.5, SDXL, and more coming soon

## Quick Start

### SD 3.5 LoKr Training

```bash
cd eridiffusion
cargo build --release --bin trainer
./target/release/trainer /path/to/config.yaml
```

Example config in `/config/eri1024.yaml`

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