# Changelog

All notable changes to AI-Toolkit-RS will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-01-29

### Added

#### Core Infrastructure (Weeks 1-2)
- Core traits for models, adapters, and plugins
- Zero-copy tensor operations with device abstraction
- Memory pool with CUDA unified memory support
- Plugin system with sandboxing
- Comprehensive error handling
- Logging and tracing infrastructure

#### Model Support (Week 3)
- Stable Diffusion 1.5 implementation
- Stable Diffusion 2.x with improved text encoder
- SDXL with dual text encoders
- SD3/SD3.5 with MMDiT architecture
- Flux (Base, Schnell, Dev) variants
- PixArt DiT-based architecture
- AuraFlow joint attention transformer
- Automatic model detection from weights

#### Training Infrastructure (Week 4)
- Complete training loop with validation
- Multiple loss functions (MSE, MAE, Huber, LPIPS, Flow Matching)
- Optimizer support (Adam, AdamW, SGD, Lion, AdaFactor, Prodigy)
- Checkpoint saving/loading with SafeTensors
- TensorBoard integration
- Data augmentation pipeline

#### Network Adapters (Weeks 5-8)
- LoRA with configurable rank/alpha
- DoRA (Weight-Decomposed LoRA)
- LoCoN for convolution layers
- LoKr with Kronecker product
- GLoRA with gating mechanisms
- ControlNet for spatial control
- IP-Adapter for image conditioning
- T2I-Adapter for lightweight control

#### Data Pipeline (Week 9)
- Async dataset loading
- Image folder dataset with captions
- HuggingFace dataset wrapper
- Bucketing system for aspect ratios
- Multi-level caching (memory + disk)
- Caption processing and augmentation
- Data validation and quality checks

#### Advanced Training (Week 10)
- Multi-GPU distributed training
- Gradient accumulation strategies
- Learning rate schedulers
- Mixed precision training (FP16/BF16)
- Callback system
- Early stopping
- Advanced metrics and logging

#### Inference Engine (Week 11)
- High-performance inference pipeline
- Batch inference with dynamic batching
- Model optimization (quantization, pruning)
- Caching strategies
- REST API server
- Client SDK
- Performance monitoring

#### CLI Application (Week 12)
- Train command with configuration
- Generate command for inference
- Convert command for model formats
- List command for available models
- Plugin management commands
- Serve command for API server

### Infrastructure
- Workspace with 8 specialized crates
- Comprehensive test suite
- Benchmark suite
- Example configurations
- Documentation

## [Unreleased]

### Planned
- Web UI with real-time monitoring
- Advanced model optimizations
- Cloud storage integrations
- Distributed inference
- Mobile deployment support
- Additional model architectures
- Enhanced plugin marketplace