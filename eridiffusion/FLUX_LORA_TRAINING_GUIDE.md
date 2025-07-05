# Flux LoRA Training Guide - AI-Toolkit Compatible

This guide explains how to train Flux LoRA models using the same YAML configuration format as the Python AI-Toolkit.

## Overview

This implementation:
- ✅ Parses the exact same YAML format as Python AI-Toolkit
- ✅ Fits training in 24GB VRAM using preprocessing
- ✅ Supports LoRA and DoRA adapters
- ✅ Implements all key features (sampling, EMA, gradient checkpointing)
- ✅ Produces compatible .safetensors files

## Quick Start

### 1. Use Your Existing AI-Toolkit Config

```bash
# Use the same YAML config as Python AI-Toolkit!
cargo run --release --bin flux_train -- config/train_lora_flux_24gb.yaml
```

### 2. First Run - Preprocessing

On first run, you'll need to provide encoder models:

```bash
cargo run --release --bin flux_train -- \
    config/train_lora_flux_24gb.yaml \
    --vae-path /path/to/ae.safetensors \
    --t5-path /path/to/t5-v1_1-xxl.safetensors \
    --clip-path /path/to/clip_l.safetensors \
    --t5-tokenizer /path/to/t5_tokenizer.json \
    --clip-tokenizer /path/to/clip_tokenizer.json
```

This preprocessing step:
1. Loads VAE, T5, and CLIP (temporarily)
2. Encodes all images and captions
3. Saves to cache directory
4. Frees all encoders from memory

### 3. Subsequent Runs

After preprocessing, just run:

```bash
cargo run --release --bin flux_train -- config/train_lora_flux_24gb.yaml
```

## Configuration File Format

The configuration uses the **exact same format** as Python AI-Toolkit:

```yaml
---
job: extension
config:
  name: "my_flux_lora"
  process:
    - type: 'sd_trainer'
      training_folder: "output"
      device: cuda:0
      trigger_word: "my_trigger"
      
      network:
        type: "lora"
        linear: 16          # LoRA rank
        linear_alpha: 16    # LoRA alpha
        
      save:
        dtype: float16
        save_every: 250
        max_step_saves_to_keep: 4
        
      datasets:
        - folder_path: "/path/to/images"
          caption_ext: "txt"
          caption_dropout_rate: 0.05
          cache_latents_to_disk: true  # REQUIRED for 24GB
          resolution: [512, 768, 1024]
          
      train:
        batch_size: 1
        steps: 2000
        gradient_accumulation_steps: 1
        train_unet: true
        gradient_checkpointing: true  # REQUIRED for 24GB
        optimizer: "adamw8bit"
        lr: 1e-4
        dtype: bf16
        
      model:
        name_or_path: "black-forest-labs/FLUX.1-dev"
        is_flux: true
        quantize: true
        
      sample:
        sampler: "flowmatch"
        sample_every: 250
        width: 1024
        height: 1024
        prompts:
          - "a photo of [trigger]"
          - "a painting of [trigger] in the style of van gogh"
```

## Key Settings for 24GB VRAM

### Required Settings
```yaml
cache_latents_to_disk: true      # MUST be true
gradient_checkpointing: true     # MUST be true
batch_size: 1                    # Max 2 for 24GB
train_text_encoder: false        # Not supported with Flux
```

### Recommended Settings
```yaml
dtype: bf16                      # Better than fp16 for Flux
optimizer: "adamw8bit"           # Saves memory
gradient_accumulation_steps: 4   # Effective batch size of 4
linear: 16                       # Good balance for LoRA rank
```

## Memory Breakdown

With these settings on 24GB VRAM:

```
Flux Model (BF16):     ~12 GB
LoRA Parameters:       ~0.1 GB
Optimizer (8-bit):     ~3 GB
Gradients:             ~3 GB (with checkpointing)
Activations:           ~2 GB
Free:                  ~3-4 GB
```

## Model Paths

The trainer will look for models in these locations:
1. Exact path if provided
2. `~/SwarmUI/Models/unet/` (for Flux models)
3. `~/ComfyUI/models/unet/`
4. `~/models/`

For HuggingFace model names, it maps:
- `black-forest-labs/FLUX.1-dev` → `flux1-dev.safetensors`
- `black-forest-labs/FLUX.1-schnell` → `flux1-schnell.safetensors`

## Output Files

Training produces:
```
output/my_flux_lora/
├── cache/                    # Preprocessed data (reusable)
├── samples/                  # Generated samples
├── checkpoint-250/
│   ├── flux_lora.safetensors       # Raw LoRA weights
│   └── my_flux_lora.safetensors    # AI-Toolkit compatible
├── checkpoint-500/
└── metrics.csv              # Training metrics
```

## Advanced Usage

### Force Reprocessing
```bash
cargo run --release --bin flux_train -- \
    config.yaml \
    --force-preprocess
```

### Skip Preprocessing Check
```bash
cargo run --release --bin flux_train -- \
    config.yaml \
    --skip-preprocess
```

### Custom Cache Directory
```bash
cargo run --release --bin flux_train -- \
    config.yaml \
    --cache-dir /path/to/cache
```

## Differences from Python AI-Toolkit

### What's the Same:
- ✅ YAML configuration format
- ✅ Output file format
- ✅ Training methodology
- ✅ Memory optimization approach
- ✅ LoRA implementation

### What's Different:
- 🦀 Written in Rust (faster, safer)
- 📦 Single binary deployment
- 🚀 Better memory management
- ⚡ Faster preprocessing
- 🔧 No Python dependencies

## Troubleshooting

### Out of Memory
1. Ensure `gradient_checkpointing: true`
2. Reduce `batch_size` to 1
3. Use `dtype: bf16` or `fp16`
4. Use `optimizer: adamw8bit`

### Preprocessing Fails
1. Check encoder model paths
2. Ensure enough disk space for cache
3. Try smaller batch size for preprocessing

### Training Doesn't Start
1. Check if preprocessing completed
2. Verify cache directory has files
3. Look for error messages about missing models

## Performance Tips

1. **Use NVMe SSD** for cache directory
2. **Preprocessing is one-time** - reuse for multiple training runs
3. **BF16 is preferred** over FP16 for Flux
4. **Sample less frequently** to speed up training

## Deployment

Once built, the binary can be deployed without Rust:

```bash
# Build release binary
cargo build --release --bin flux_train

# Copy to deployment location
cp target/release/flux_train /usr/local/bin/

# Run anywhere with just the config
flux_train /path/to/config.yaml
```

No need for Python, Conda, or virtual environments!