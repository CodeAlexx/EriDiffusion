# ERIDiffusion

ERIDiffusion is a CUDA-first training and inference stack for large diffusion models. It pairs a high-throughput trainer with the Flame tensor runtime to deliver pragmatic rebuilds of production-grade pipelines such as Flux LoRA, Stable Diffusion XL (SDXL), Stable Diffusion 3.5 (SD3.5), and Chroma variants.

## Key Goals
- **Deterministic rebuilds** – reproduce the original training behavior of our production pipelines without relying on third-party shims.
- **Flame alignment** – every trainer, loader, and streaming registry is written directly against `flame-core` devices, kernels, and tensor APIs.
- **GPU-first ergonomics** – the codebase assumes NVIDIA GPUs with CUDA 12+, bf16-capable hardware, and high-bandwidth storage for mmap’d weight streaming.

## Highlights
- **Flux LoRA trainer** rebuilt for sequential execution with optional INT8 path (`src/trainers/pipeline_flux_lora_sequential.rs`).
- **SDXL/SD3.5 streaming** registries ready for large-weight mmap loading (`crates/training/src/sdxl/*`, `crates/training/src/sd35/*`).
- **Chroma pipelines** integrated with adapter rotation, EMA support, and GradScaler-managed mixed precision (`crates/training/src/chroma/*`).
- **Core runtime** (`crates/core`) implements bf16-friendly kernels, cudarc wrappers, weight streaming, and adapter utilities aligned with Flame.
- **No FlashAttention dependency** – fused kernels are toggled off unless explicitly enabled; standard attention paths are the default.

## Repository Layout
- `crates/training/` – trainer infrastructure, schedulers, pipelines, checkpoint helpers.
- `crates/core/` – low-level tensor operations, CUDA kernels, memory pools, device abstractions.
- `crates/models/` – model-specific blocks (Flux, SDXL, Chroma) with dtype policies and IO utilities.
- `crates/inference/` – inference client, batching, optimization utilities.
- `src/trainers/` – top-level trainers, streaming cache managers, device diagnostics.
- `examples/` – smoke trainers and inference examples showcasing pipeline integration.
- `config/` – YAML configuration templates used during rebuild and validation.

## Dependencies
- **Rust** 1.75+ (uses edition 2021).
- **CUDA Toolkit** 12.x with nvcc, `libcudnn` (optional for cudnn-conv builds).
- **flame-core** – bundled in-tree, providing tensor, optimizer, and kernel primitives.
- **cudarc** 0.11+ – used for kernel launches and NVRTC compilation.
- Optional: `safetensors` for weight serialization, `tokio` (runtime for GradScaler tasks), `serde_yaml` for configuration loading.

## Build & Test
```bash
# Configure CUDA env (example)
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

# Run a type check for the training crate
default_target=eridiffusion
cargo check --manifest-path eridiffusion/Cargo.toml -p eridiffusion-training

# Full workspace check
default_target=eridiffusion
cargo check --manifest-path eridiffusion/Cargo.toml
```

To enable fused bf16 attention/ffn kernels, compile with `--features bf16_u16` and ensure NVRTC include paths are configured (`CUDA_HOME/include`).

## Pipelines
- **Flux LoRA Sequential**: focuses on low-latency LoRA training with streaming weights and INT8 loader for quantized baselines.
- **SDXL Trainer**: streaming block registry with per-device dual creation (`flame_core::Device`, `eridiffusion_core::Device`) at call sites to avoid type mismatches.
- **SD3.5 Trainer**: mirrors SDXL but tailored for the SD3.5 architecture and LyCORIS target modules.
- **Chroma Trainer**: adapter rotation, EMA, GradScaler-managed execution, and optional layer freezing.

## Strengths
- **Adapter-friendly**: adapters stored as `Arc<dyn Adapter + Send + Sync>` enabling cheap cloning and multi-owner orchestration.
- **Streaming-ready**: strict mmap loaders and weight providers designed for GPU-bound training with minimal host overhead.
- **Error consistency**: all safetensor and loader errors normalized to `flame_core::Error` variants.
- **Tripwire diagnostics**: `TensorDebugExt` guards and device mismatch checks to surface silent CUDA/hybrid bugs.

## Limitations / TODO
- Legacy demo binaries are intentionally removed from the default workspace; re-enable them via dedicated branches if needed.
- Warning cleanup (`cargo fix`) is still pending in several crates (especially training pipeline scaffolding).
- Runtime relies on NVIDIA GPUs; AMD/HIP paths are out of scope for the current rebuild.

## Getting Started
1. Clone the repository.
2. Point CUDA env vars to a valid Toolkit installation.
3. Run `cargo check --manifest-path eridiffusion/Cargo.toml -p eridiffusion-training`.
4. Inspect `examples/` for trainer entry points and pipeline configuration patterns.
