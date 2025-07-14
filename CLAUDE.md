# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal
Pure Rust trainer program for modern diffusion models:
- **Image models**: SDXL, Flux, SD 3.5, Flex, OmniGen 2, HiDream, Chroma, Sana, Kolors, Ommigen2
- **Video models**: Wan Vace 2.1, LTX, Hunyuan
- No ControlNet or IP-Adapter versions needed
- System specs: 24GB VRAM, 64GB RAM

## Cli usage
"trainer /path/config/xxxx.yaml"
or if in main directory -- "trainer /config/xxxx.yaml"
After building, the executable must be in the path or root of the project!

## Automatic model pipeline selector
The model it trains on is set in the .yaml file along with all its training parameters. 
EriDiffusion first reads the yaml, gets the model type and based on then routes to the correct pipeline for training.
example on .yaml
   name_or_path: "/home/alex/SwarmUI/Models/diffusion_models/chroma-unlocked-v37-detail-calibrated.safetensors"
        arch: "chroma"
 here it tell us it wants Chroma as the model to train for, along with the arch.
 With that, it will go to the Chroma model pipeline to do training. The parameters are the rest of the yaml, each in its own section,
 as EriDiffusion is .yaml file driven.

## Critical Development Rules
1. **NO MOCKS OR SIMULATIONS** - All code must be real and functional
    Bad code costs us time and for Claude code many errors in debugging. 
    To implement in the future code is FORBIDDEN. Come back to the code after the main parts are done and fix.
2. **NO PYTHON** - Pure Rust implementation only
3. **NO EXAGGERATION** - Be truthful about what is/isn't implemented
4. **SEARCH BEFORE IMPLEMENTING** - Check for existing crates on crates.io and GitHub
5. **LOCAL MODELS ONLY** - Never download from HuggingFace, use local paths

## REQUIRED READING for Training any model. Overcomes VarBuilder from candle

**CRITICAL**: Before working lora/full finetune training you MUST read:
1. `/home/alex/diffusers-rs/candle-fork/FINAL_SOLUTION.md` - The generic Linear<T> approach
2. `/home/alex/diffusers-rs/eridiffusion/GENERIC_CHECKPOINT_SOLUTION.md` - How it enables training -- started with SDXL 1024x1024 trainings

These documents explain the VarBuilder problem and the solution using generic types that enables true gradient checkpointing.

## TRUST RESTORATION GOAL
**The ONLY way to rebuild trust is to produce LoRAs that work in ComfyUI.**
- Must train real LoRAs on real data
- Must save in exact ComfyUI format
- Must be loadable and produce visible effects
- Include sampling during training to verify progress
- No placeholders, no shortcuts - working code only

Sub agents are used. They must be used to detect bad code, fake code, improper logic, missing features. Audits are regular. 
EVERYTHING DISCOVERED, EVERYTHING THAT WAS DONE THAT WAS MAJOR MUST, MUST BE DOCUMENTED!

## Model Weights Location
- Primary location: `/home/alex/SwarmUI/Models/`
- Configuration files: `/config/` directory
- Always use local paths, no hardcoded remote URLs

## Documentation Requirements
- `Changes.md` - Log all trainer updates
- `Models.md` - Model-specific changes
- Each model maintains its own changelog

## Important Notes on Model Usage

### SD 3.5 LoRA Trainer
A complete SD 3.5 LoRA trainer has been implemented in the `sd35-lora-trainer/` directory. Key features:
- **NO automatic downloads from HuggingFace** - all models must be provided locally
- Supports SD 3.5 Medium/Large/Large-Turbo variants
- LoRA training with configurable rank and alpha
- Flow matching objective with SNR weighting
- Security hardened with path traversal protection
- Local model paths must be specified via command line or config file

### FLUX Lora Trainer
-- add detailes


### Local Model Paths
**IMPORTANT: DO NOT DOWNLOAD MODELS - All required models are already available locally**
- Model files are located in `/home/alex/SwarmUI/Models/` 
- HuggingFace cache can also be checked for models
- When working with configurations, always use local paths
- Never attempt to download from HuggingFace or other remote sources

### EriDiffusion Implementation
- Pure Rust trainer (NO PYTHON ALLOWED)
- Configuration files in `/config/` directory
- Example SD 3.5 Large LoKr config: `/config/eri1024.yaml`
- Dataset format: image files with corresponding .txt caption files

## Candle Fork - Critical Information

### Why We Use a Forked Version of Candle
**Location**: `/home/alex/diffusers-rs/candle-fork/`

**VarBuilder Problem**: The official Candle's VarBuilder is designed for inference only and returns immutable `Tensor` objects instead of trainable `Var` objects. This makes training impossible because:
- `Tensor` = Immutable, no gradients, inference only
- `Var` = Mutable, tracks gradients, required for training

**Our Solution**: We created a fork that bypasses VarBuilder entirely, allowing us to:
1. Load weights directly with `candle::safetensors::load()` 
2. Create trainable parameters as `Var` objects manually
3. Implement custom forward passes with proper gradient tracking
4. Use Candle's autograd system for backpropagation

**Example of the difference**:
```rust
// WRONG - VarBuilder returns Tensor (no gradients!)
let vb = VarBuilder::from_tensors(weights, DType::F32, &device);
let weight = vb.get((out_dim, in_dim), "weight")?; // Returns Tensor, not Var!

// CORRECT - Direct approach returns Var (with gradients!)
let weight = Var::randn(0.0, 0.02, (out_dim, in_dim), &device)?; // Returns Var!
```

**Key Rule**: NEVER use VarBuilder for training code. Always create Var objects directly or load weights and wrap them in Var.

## Project Structure

### 1. Original diffusers-rs (PyTorch-based) 
- **Status**: Legacy, being replaced by Candle implementations

### 2. EriDiffusion (Main development focus)
- Located in `eridiffusion/` directory
- Pure Rust using Candle framework
- **Target models**: All modern diffusion models (SDXL, SD3.5, Flux, etc.)
- **Status**: Active development, Week 12 of 26-week roadmap complete
- **Architecture**: 8 crates (core, models, networks, training, data, inference, web, extensions)


## Common Development Commands

### Building
```bash
# Basic build
cargo build

# Release build
cargo build --release



### Testing and Quality
```bash
verbage need to be added

# Build the entire workspace
cd eridiffusion && cargo build --release

### Sampling/Inference Integration
The trainer now includes integrated sampling/inference for monitoring training progress:

**SD 3.5 Sampling**
- Flow matching with 16-channel latents
- Triple text encoding (CLIP-L, CLIP-G, T5-XXL)
- Configurable timestep schedules (including Turbo variants)

**Flux Sampling**
- Flow matching with patchified latents (2x2 patches)
- Shifted sigmoid timestep scheduling
- T5-XXL and CLIP text encoding
- Guidance embedding support (1.0 for Schnell, 3.5 for Dev)

**Usage in Training**
in the yaml add sample parameters
```rust
// Generate samples during training
let sample_paths = FluxTrainer::generate_samples(
    &model, &vae, &text_encoder, &config,
    &device, step, &output_dir
).await?;
```

### Working with Model Weights
**IMPORTANT**: All model weights must be provided locally. Do not use download scripts.
- Model location: `/home/alex/SwarmUI/Models/`
- Use `.safetensors` format for all models
- FP16 precision recommended for memory efficiency

## Architecture and Code Structure

### Core Components
- **src/models/**: Neural network implementations
  - `unet_2d.rs`: 2D U-Net architecture for diffusion
  - `unet_2d_blocks.rs`: Building blocks for U-Net (DownBlock2D, UpBlock2D, CrossAttnDownBlock2D, etc.)
  - `vae.rs`: Variational autoencoder for image encoding/decoding
  - `controlnet.rs`: ControlNet implementation
  - `attention.rs`: Attention mechanisms (including cross-attention)
  - `resnet.rs`: ResNet blocks
  - `embeddings.rs`: Time and text embeddings
  
- **src/pipelines/**: High-level pipelines that orchestrate the diffusion process
  - `stable_diffusion.rs`: Main text-to-image pipeline
  - `stable_diffusion_img2img.rs`: Image-to-image transformation
  - `stable_diffusion_inpaint.rs`: Inpainting pipeline
  
- **src/schedulers/**: Noise scheduling algorithms
  - Multiple scheduler implementations (DDIM, DDPM, Euler, etc.)
  - Each scheduler implements the `Scheduler` trait
  - Scheduler selection via `--sd-version` and `--scheduler` flags
  
- **src/transformers/**: Text encoding
  - `clip.rs`: CLIP model for text encoding

### Key Design Patterns
1. **Tensor Management**: All models work with `tch::Tensor` types from PyTorch
2. **Config Pattern**: Models use configuration structs (e.g., `UNetConfig`, `VaeConfig`)
3. **Pipeline Architecture**: Pipelines combine models, schedulers, and transformers
4. **Device Management**: Explicit GPU/CPU device handling via `tch::Device`
5. **Builder Pattern**: Schedulers use builder pattern for configuration

### Memory Considerations
- Default GPU execution requires 8GB+ VRAM
- Use `--cpu all` flag for CPU-only execution
- Mixed mode available: `--cpu vae --cpu clip` keeps UNet on GPU
- FP16 weights available for reduced memory usage
- For 8GB GPUs, recommended to use fp16 weights and possibly mixed mode

### Build Requirements
- Requires libtorch (automatically handled by `tch` crate)
- `build.rs` sets up proper linking and runtime paths for PyTorch libraries
- On Linux/macOS: Sets rpath for finding libtorch at runtime
- On Windows: Ensures proper DLL loading paths

### Error Handling
- Uses `anyhow::Result` for error propagation
- Custom error types via `thiserror` for model-specific errors
- Graceful fallbacks for missing optional components (e.g., safety checker)

## Key Technical Decisions

### Framework Choice
- **Legacy (diffusers-rs)**: PyTorch via `tch` crate
- **Current (ai-toolkit-rs)**: Candle framework for pure Rust
- **Rationale**: Candle provides better Rust integration and avoids Python dependencies

### Implementation Priorities
1. **Real functionality only** - No stubs, mocks, or placeholders
2. **Security first** - Path validation, input sanitization
3. **Memory efficiency** - Gradient checkpointing, mixed precision
4. **Performance** - Fused operations, static dispatch where possible
5. **Extensibility** - Plugin system for custom models/adapters

### Testing Strategy
- Unit tests for all core functionality
- Integration tests for training/inference pipelines
- Benchmarks for performance-critical paths
- No mock implementations in tests

### Development Workflow
1. Search for existing solutions before implementing
2. Use workspace structure for modular development
3. Document all API changes in Changes.md
4. Maintain backward compatibility where possible
5. Prioritize local model usage over downloads

## Configuration File Format

Training configurations use YAML format. Example structure from `/config/`:

```yaml
job: extension
config:
  name: "model_name"
  process:
    - type: 'sd_trainer'
      device: cuda:0
      low_vram: true  # Enable for 24GB cards
      trigger_word: "optional_trigger"
      network:
        type: "lora"  # or "lokr", "dora", "lokr_full_rank"
        linear: 16
        linear_alpha: 16
        lokr_factor: 4  # for lokr only
      save:
        dtype: float16
        save_every: 250
        max_step_saves_to_keep: 4
      datasets:
        - folder_path: "/absolute/path/to/images"
          caption_ext: "txt"
          caption_dropout_rate: 0.05
          shuffle_tokens: false
          cache_latents_to_disk: true
          resolution: [512, 768, 1024]  # multiple resolutions supported
      train:
        batch_size: 1
        steps: 2000
        gradient_accumulation: 1
        train_unet: true
        train_text_encoder: false
        gradient_checkpointing: true  # Required for 24GB VRAM
        noise_scheduler: "flowmatch"
        optimizer: "adamw8bit"
        lr: 1e-4
        dtype: bf16  # or fp16
        linear_timesteps: true  # For SD3.5
        bypass_guidance_embedding: true  # For Flux
      model:
        name_or_path: "/home/alex/SwarmUI/Models/..."
        is_flux: true  # Model-specific flags
        is_v3: true    # For SD3.5
        quantize: true  # 8-bit mixed precision
```

## Dataset Format Requirements

### Directory Structure
```
dataset_folder/
├── image1.jpg
├── image1.txt
├── image2.png
├── image2.txt
└── ...
```

### Requirements
- **Images**: `.jpg`, `.jpeg`, `.png` formats only
- **Captions**: `.txt` files with exact same name as image
- **Content**: Each caption file contains text description of the image
- **Trigger words**: Will be automatically added if not present in captions
- **Path format**: Always use absolute paths in configs

## Debugging and Development Commands

```bash
# Enable debug logging
RUST_LOG=debug cargo run --release -- train --config config/train.yaml

# Run with specific GPU
CUDA_VISIBLE_DEVICES=0 cargo run --release --bin train_sd3_lora

# Check CUDA availability
RUST_LOG=info cargo run --example check_cuda

# Run with backtrace
RUST_BACKTRACE=1 cargo run --release -- train --config config/train.yaml

# Memory debugging (for development)
CUDA_LAUNCH_BLOCKING=1 cargo run --release -- train --config config/train.yaml
```

## Common Troubleshooting

### Out of Memory Errors
- Enable `gradient_checkpointing: true` in config
- Reduce `batch_size` to 1
- Enable `low_vram: true` for 24GB cards
- Use `quantize: true` for 8-bit mixed precision
- Set `cache_latents_to_disk: true`

### CUDA Errors
- Ensure libtorch matches CUDA version
- Check `nvidia-smi` for driver compatibility
- Use `CUDA_VISIBLE_DEVICES` to select specific GPU

### Path Errors
- Always use absolute paths in configs
- Check file permissions
- Ensure caption files exist for all images

### Training Issues
- If loss is NaN: reduce learning rate
- If no progress: check if trigger word is in prompts
- For Flux: ensure `bypass_guidance_embedding: true`
- For SD3.5: use `linear_timesteps: true`

## Performance Guidelines (24GB VRAM)

### Recommended Settings by Model

**SD 3.5 Large**
```yaml
train:
  batch_size: 4
  gradient_checkpointing: true
  dtype: bf16
  linear_timesteps: true
model:
  quantize: true
  is_v3: true
```

**Flux**
```yaml
train:
  batch_size: 1
  gradient_checkpointing: true
  dtype: bf16
  bypass_guidance_embedding: true
model:
  quantize: true
  is_flux: true
```

**SDXL**
```yaml
train:
  batch_size: 2-4
  gradient_checkpointing: true
  dtype: fp16
```

### Memory Optimization Tips
- Use multiple resolutions for better bucketing efficiency
- Enable `cache_latents_to_disk: true`
- Use `adamw8bit` optimizer
- Keep `max_step_saves_to_keep` low to save disk space

## Active Development Status

### Currently Implemented
- **Legacy diffusers-rs**: SD 1.5, SD 2.1 (PyTorch-based)
- **SD 3.5 LoRA Trainer**: Fully functional, production-ready with integrated sampling
- **AI-Toolkit-RS Core**: Week 12 complete
  - Core infrastructure, device management, plugin system
  - Model architectures for SD, SDXL, SD3.5, Flux
  - Network adapters: LoRA, DoRA, LoKr, LoCoN
  - Training and inference pipelines
  - Data loading with bucketing
  - **Flux Sampling/Inference**: Complete implementation with flow matching
    - Patchification support for Flux latents
    - Shifted sigmoid timestep scheduling
    - Guidance embedding support
    - Integration with candle-transformers Flux model

### In Active Development
- **Sampling/Inference for all models**: Adding to trainers
  - ✅ SD 3.5 - Complete
  - ✅ Flux - Complete
  - 🚧 SDXL - In progress
  - 🚧 Other models - Pending
- **AI-Toolkit-RS Web UI**: Weeks 13-14
- **Performance optimizations**: CUDA kernels, Flash Attention 2
- **Advanced training features**: DreamBooth, Textual Inversion

### Planned Features
- Cloud deployment support (Weeks 19-20)
- Edge device optimization
- Research features (Weeks 23-24)
- Additional model support as they release

## Model-Specific Configuration Notes

### SD 3.5
- Set `is_v3: true` in model config
- Use `linear_timesteps: true` for better convergence
- Supports Medium/Large/Large-Turbo variants
- Requires `t5_max_length: 154` for text encoding
- Use `snr_gamma: 5` for signal-to-noise ratio weighting

### Flux
- Set `is_flux: true` in model config
- **Must** use `bypass_guidance_embedding: true`
- Requires `dtype: bf16` for stability
- Benefits from multiple resolutions in dataset
- Use `quantize_kwargs` to exclude time_text_embed from quantization

### SDXL
- Works with standard LoRA configurations
- Supports resolutions up to 1024x1024
- Can train text encoder (unlike Flux)
- Use `fp16` dtype for best compatibility

## Integration Testing

```bash
# Test SD 3.5 LoRA training
cd sd35-lora-trainer && cargo test --release

# Test AI-Toolkit-RS components
cd ai-toolkit-rs
cargo test --workspace --release

# Test specific integration
cargo test --test training_integration --release

# Run benchmarks
cargo bench --workspace
```

## Build Flags and Features

### Available Features
- `clap` - CLI argument parsing
- `imageproc` - Advanced image processing (for ControlNet)
- `accelerate` - GPU acceleration features
- `safetensors` - SafeTensors format support (default)

### Common Build Commands
```bash
# Minimal build
cargo build --release

# Full features
cargo build --release --features "clap,imageproc"

# Development build with debug symbols
cargo build --features clap

# Production build with all optimizations
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

## Error Handling Patterns

### Adding Context to Errors
```rust
use anyhow::{Context, Result};

// Add context when propagating errors
let model = load_model(path)
    .context("Failed to load diffusion model")?;

// Include relevant information
let config = load_config(&config_path)
    .with_context(|| format!("Failed to load config from: {}", config_path))?;
```

### Error Types Hierarchy
- `ModelError` - Model loading/inference failures
- `TrainingError` - Training process errors
- `NetworkError` - LoRA/adapter errors
- `DataError` - Dataset loading issues
- `ConfigError` - Configuration parsing errors

### Recovery Strategies
- For OOM: Automatically reduce batch size and retry
- For missing files: Provide helpful error with expected path
- For CUDA errors: Check CUDA installation and driver compatibility
- For config errors: Show example of correct format

## Training Without VarBuilder

### The Problem with VarBuilder
VarBuilder assumes inference-only usage and returns `Tensor` instead of `Var`, making training impossible without workarounds.

### The Solution: Direct Parameter Management
We bypass VarBuilder entirely for training by:

1. **Loading weights directly**: 
```rust
let weights = candle::safetensors::load("model.safetensors", &device)?;
```

2. **Creating trainable parameters as Var**:
```rust
// For LoRA
let lora_down = Var::randn(0.0, 0.02, (rank, in_features), DType::F32, device)?;
let lora_up = Var::zeros((out_features, rank), DType::F32, device)?;

// For full fine-tuning
let trainable_weight = Var::from_tensor(&frozen_weight)?;
```

3. **Custom forward passes**: Implement forward passes that inject LoRA or use trainable parameters directly

4. **Using Candle's autograd**:
```rust
let loss = compute_loss(&output, &target)?;
let grads = loss.backward()?;

// Update parameters
if let Some(grad) = grads.get(var.as_tensor()) {
    let new_value = var.as_tensor() - (grad * learning_rate)?;
    var.set(&new_value)?;
}
```

### LoRA Training Implementation Status

All LoRA trainers have been implemented following the VarBuilder-free pattern:

1. **SDXL LoRA** (`src/training/sdxl_lora.rs`)
   - Complete U-Net architecture with attention blocks
   - Support for down/mid/up blocks
   - Cross-attention and self-attention LoRA adapters
   - Gradient accumulation and checkpointing

2. **SD3.5 LoRA** (`src/training/sd35_lora.rs`)
   - MMDiT (Multimodal Diffusion Transformer) architecture
   - Joint image-text attention processing
   - RoPE (Rotary Position Embeddings) support
   - SNR-weighted loss for better convergence
   - Adaptive layer normalization with timestep modulation

3. **Flux LoRA** (`src/training/flux_lora.rs`)
   - Double stream blocks (separate image/text processing)
   - Single stream blocks (merged processing)
   - Flux-specific RoPE with multiple axis dimensions
   - Guidance embedding support (can be bypassed)
   - Linear attention for efficiency
   - 2x2 patch processing for 16-channel VAE

### Key Architecture Differences

**SDXL**: Traditional U-Net with ResNet blocks and attention layers
- Channel multipliers: [1, 2, 4]
- Context dimension: 2048
- Standard cross-attention

**SD3.5**: MMDiT architecture processing image and text jointly
- 16-channel VAE (vs 4 for SDXL)
- Hidden size: 1536 (Large variant)
- 38 transformer layers
- Joint attention on concatenated sequences

**Flux**: Hybrid architecture with double and single stream blocks
- 16-channel VAE with 2x2 patches (64 channels after patchify)
- Hidden size: 3072
- 19 double blocks + 38 single blocks
- Separate then merged processing

### Usage Example

```rust
use candle_fork::training::{train_sdxl_lora, train_sd35_lora, train_flux_lora};
use candle_fork::training::{SDXLConfig, SD35Config, FluxConfig};

// Train SDXL
let config = SDXLTrainingConfig::default();
train_sdxl_lora(unet_path, dataloader, None, config, &device)?;

// Train SD3.5 with SNR weighting
let mut config = SD35TrainingConfig::default();
config.snr_gamma = Some(5.0);
config.linear_timesteps = true;
train_sd35_lora(mmdit_path, dataloader, None, config, &device)?;

// Train Flux with guidance bypass
let mut config = FluxTrainingConfig::default();
config.bypass_guidance_embedding = true;
config.guidance_scale = 3.5;
train_flux_lora(flux_path, dataloader, None, config, &device)?;
```

This approach works for all models and training scenarios without needing VarBuilder modifications.

# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.
