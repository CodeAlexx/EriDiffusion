# Flux Sampling Implementation

## Overview
Successfully implemented Flux sampling/inference pipeline for generating validation images during training.

## Key Components Implemented

### 1. FluxSampler Class (`src/trainers/flux_sampling.rs`)
- Complete flow matching denoising loop
- Proper patchification (16-channel latents → 64-channel patches)
- Position embeddings for both image and text
- Shifted sigmoid timestep scheduling (Flux-specific)
- VAE decoding to generate images

### 2. Patchification Process
- Converts 16-channel latents to 64-channel patches (16 × 2 × 2)
- Reshapes [B, C, H, W] → [B, H*W/4, C*4] for model input
- Unpatchifies model output back to standard format

### 3. Position Embeddings
- Image IDs: 3D embeddings [batch, seq_len, 3] with (batch_idx, y_pos, x_pos)
- Text IDs: Zero-initialized for text sequence

### 4. Integration with FluxLoRATrainer
- Added `generate_samples()` method to trainer
- Integrated with training loop at `sample_every` steps
- Handles model references (quantized or standard)
- Creates timestamped output directories

## Usage

### Configuration
```yaml
sample:
  sampler: "euler"
  sample_every: 50
  sample_steps: 28
  guidance_scale: 3.5
  prompts:
    - "a majestic mountain landscape"
    - "a futuristic city at night"
  width: 1024
  height: 1024
```

### During Training
The sampler automatically generates validation images at specified intervals:
- Creates `samples_step_XXXXXX` directories
- Saves images as `step_XXXXXX_sample_XX.png`
- Prints generation progress and file paths

## Technical Details

### Model Forward Pass
```rust
model.forward(
    &img_patches,    // Patchified latents
    &img_ids,        // Position embeddings
    &text_embeds,    // T5-XXL text embeddings
    &txt_ids,        // Text position embeddings
    &timestep,       // Current denoising timestep
    &pooled_embeds,  // CLIP pooled embeddings
    &guidance,       // Guidance scale tensor
)
```

### Flow Matching Update
- Uses Euler integration: x_{t+1} = x_t + dt * v_t
- Velocity prediction from model output
- Shifted sigmoid schedule for timesteps

## Memory Efficiency
- Designed for 24GB VRAM constraints
- Uses existing loaded models (no duplicate loading)
- Efficient tensor operations with proper device management

## Next Steps
- Add support for negative prompts
- Implement proper random seeding for reproducibility
- Add more sampling schedulers (DDIM, DPM++)
- Support for different image resolutions during sampling