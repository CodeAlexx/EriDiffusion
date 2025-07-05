# Changelog

## [Unreleased] - 2024-07-04

### Added
- **SD 3.5 LoKr Training** - Full implementation of Stable Diffusion 3.5 LoKr training in pure Rust
  - MMDiT model support with flow matching objective
  - LoKr (Low-rank Kronecker product) adaptation layers
  - Memory-efficient SimpleTuner-style sequential text encoder loading
  - Real CLIP-L, CLIP-G, and T5-XXL tokenizer support
  - Mixed precision training with BF16
  - Gradient checkpointing for 24GB GPUs
  - SNR weighting for improved convergence
  - SafeTensors checkpoint saving
  - Progress tracking with loss, speed, and ETA display

### Fixed
- Resolved infinity/NaN loss issues in training:
  - Added loss computation in F32 to prevent overflow
  - Implemented loss scaling for gradient stability
  - Added parameter norm monitoring
  - Skip batches with invalid loss values
- Fixed CPU bottleneck by using zero embeddings for T5
- Fixed pooled embeddings shape mismatch for SD 3.5
- Implemented CUDA RMS normalization to keep operations on GPU

### Performance
- Training speed: 6.6+ it/s on RTX 3090 Ti
- Memory usage: 20.6GB VRAM for SD 3.5 Large
- 100% GPU utilization with optimized pipeline

## Previous Versions

See git history for changes before SD 3.5 trainer implementation.