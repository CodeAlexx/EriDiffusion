# EriDiffusion Code Metrics

## Overall Statistics
- **Total Rust files**: 84
- **Total lines of code**: 25,949
- **Directories**: 5 main modules (trainers, loaders, models, memory, bin)

## Training Binaries (src/bin/)

### Individual Trainer Binaries
```
train_sdxl_lora.rs    - 77 lines   - SDXL LoRA training entry point
train_sd35_lora.rs    - 311 lines  - SD 3.5 LoRA training entry point  
train_flux_lora.rs    - 223 lines  - Flux LoRA training entry point
trainer.rs            - 33 lines   - Main unified trainer (auto-detects model)
```

## Core Trainer Implementations (src/trainers/)

### Main Training Modules
```
sdxl_lora_trainer_fixed.rs  - 2,410 lines - Complete SDXL LoRA trainer
sd35_lora.rs                - 509 lines   - SD 3.5 LoRA trainer
flux_lora.rs                - 2,036 lines - Flux LoRA trainer
```

### Sampling Infrastructure
```
sdxl_sampling_complete.rs   - 696 lines  - SDXL sampling implementation
flux_sampling.rs            - 274 lines  - Flux sampling utilities
sdxl_forward_sampling.rs    - 197 lines  - SDXL forward pass for sampling
sampling_utils.rs           - 82 lines   - Common sampling utilities
```

### Support Modules
```
adam8bit.rs                 - 244 lines  - 8-bit Adam optimizer
ddpm_scheduler.rs           - 165 lines  - DDPM noise scheduler
enhanced_data_loader.rs     - 626 lines  - Enhanced data loading
flux_data_loader.rs         - 418 lines  - Flux-specific data loading
text_encoders.rs            - 329 lines  - CLIP/T5 text encoding
memory_utils.rs             - 91 lines   - Memory management utilities
```

## Model Implementations (src/models/)

### Key Model Files
```
flux_custom/
├── mod.rs                  - 1,073 lines - Flux model architecture
├── lora.rs                 - 503 lines   - Flux LoRA implementation
└── attention.rs            - 409 lines   - Flux attention mechanisms

sdxl/
├── unet_2d_condition.rs    - 892 lines   - SDXL U-Net implementation
└── attention_processor.rs  - 567 lines   - SDXL attention processing
```

## Memory Management (src/memory/)
```
config.rs                   - 142 lines  - Memory configuration
quantization.rs             - 387 lines  - Model quantization
optimizer.rs                - 298 lines  - Memory optimization
```

## Loaders (src/loaders/)
```
sdxl_checkpoint_loader.rs   - 157 lines  - SDXL model loading
sdxl_weight_remapper.rs     - 224 lines  - Weight remapping for SDXL
sdxl_full_remapper.rs       - 286 lines  - Full model remapping
```

## Directory Structure
```
EriDiffusion/
├── src/
│   ├── bin/         (4 files, 644 lines)    - Training executables
│   ├── trainers/    (27 files, 10,976 lines) - Training implementations
│   ├── models/      (45 files, 12,156 lines) - Model architectures
│   ├── memory/      (5 files, 1,423 lines)   - Memory management
│   └── loaders/     (3 files, 667 lines)     - Model loaders
├── config/          (5 YAML files)            - Training configurations
└── Cargo.toml       (70 lines)                - Project configuration
```

## Largest Files (Top 10)
1. `sdxl_lora_trainer_fixed.rs` - 2,410 lines
2. `flux_lora.rs` - 2,036 lines
3. `flux_custom/mod.rs` - 1,073 lines
4. `unet_2d_condition.rs` - 892 lines
5. `sdxl_vae_native.rs` - 689 lines
6. `sdxl_sampling_complete.rs` - 696 lines
7. `enhanced_data_loader.rs` - 626 lines
8. `attention_processor.rs` - 567 lines
9. `sd35_lora.rs` - 509 lines
10. `flux_custom/lora.rs` - 503 lines

## Configuration Files
```
sdxl_lora_24gb_optimized.yaml - 107 lines
sd35_lora_training.yaml       - 91 lines
flux_lora_24gb.yaml          - 98 lines
example_sdxl_lora.yaml       - 49 lines
t5_config.json               - 25 lines
```

## Missing Implementations Status
- ✅ SDXL: Fully implemented with sampling
- ⚠️ SD 3.5: Training works, sampling placeholder
- ⚠️ Flux: Training works, sampling placeholder (memory constraints)

## Build Commands
```bash
# Main trainer (auto-detects model type)
cargo build --release --bin trainer

# Individual model trainers
cargo build --release --bin train_sdxl_lora
cargo build --release --bin train_sd35_lora  
cargo build --release --bin train_flux_lora
```