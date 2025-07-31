# Inference Integration Plan for EriDiffusion

## Overview
This document outlines the plan to integrate working inference code from candle-fork examples into the EriDiffusion training pipeline.

## Current Working Examples Location
- **SDXL**: `/home/alex/diffusers-rs/candle-fork/candle-examples/examples/stable-diffusion/main.rs`
- **SD 3.5**: `/home/alex/diffusers-rs/candle-fork/candle-examples/examples/stable-diffusion-3/main.rs`
- **Flux**: `/home/alex/diffusers-rs/candle-fork/candle-examples/examples/flux/main.rs`

## Integration Strategy

### 1. Core Components to Copy

#### From SDXL Example:
- `build_stable_diffusion_unet` function (lines 366-438)
- `run` function's denoising loop (lines 828-925)
- CLIP text encoding setup (lines 658-704)
- VAE decoding logic (lines 941-951)

#### From SD 3.5 Example:
- `run` function with triple text encoding (lines 217-334)
- MMDiT model loading and configuration
- Flow matching denoising loop
- TAESD3 VAE scaling

#### From Flux Example:
- `patchify` and `unpatchify` functions
- Flux model architecture setup
- Shifted sigmoid timestep scheduling
- Guidance embedding handling

### 2. Architecture Design

```rust
// New trait for unified inference
pub trait DiffusionInference {
    fn load_model(&mut self, config: &ModelConfig) -> Result<()>;
    fn encode_prompt(&self, prompt: &str) -> Result<Tensor>;
    fn denoise(&self, latents: &Tensor, text_embeds: &Tensor, steps: usize) -> Result<Tensor>;
    fn decode_vae(&self, latents: &Tensor) -> Result<Tensor>;
    fn apply_lora(&mut self, lora_weights: &HashMap<String, Tensor>, scale: f32) -> Result<()>;
}

// Implementations
pub struct SDXLInference { /* fields from example */ }
pub struct SD35Inference { /* fields from example */ }
pub struct FluxInference { /* fields from example */ }
```

### 3. YAML Config Integration

```yaml
sample:
  prompts:
    - "a white swan on mars"
  sample_every: 100
  sample_steps: 30
  cfg_scale: 7.5
  seed: 42
  # LoRA settings
  use_lora: true
  lora_scale: 1.0
```

### 4. File Structure

```
src/inference/
├── mod.rs              # Trait definitions
├── sdxl.rs            # SDXL implementation
├── sd35.rs            # SD 3.5 implementation
├── flux.rs            # Flux implementation
├── utils.rs           # Shared utilities
└── lora_adapter.rs    # LoRA weight injection
```

## Implementation Steps

### Phase 1: SDXL Integration
1. Copy working SDXL inference code from candle example
2. Create `SDXLInference` struct implementing `DiffusionInference`
3. Integrate with existing `sdxl_lora_trainer_fixed.rs`
4. Test with real image generation
5. Verify LoRA weight application

### Phase 2: SD 3.5 Integration
1. Copy SD 3.5 inference code from candle example
2. Create `SD35Inference` struct
3. Handle triple text encoding properly
4. Test with 16-channel VAE
5. Verify flow matching works

### Phase 3: Flux Integration
1. Copy Flux inference code from candle example
2. Create `FluxInference` struct
3. Implement patchification logic
4. Test with shifted sigmoid scheduling
5. Verify guidance embeddings

## LoRA Integration Strategy

For each model, LoRA weights will be injected at:
- **SDXL**: CrossAttention layers in U-Net blocks
- **SD 3.5**: Joint attention blocks in MMDiT
- **Flux**: Double and single stream attention blocks

```rust
// Example LoRA injection
fn apply_lora_to_linear(
    linear: &mut Linear,
    lora_down: &Tensor,
    lora_up: &Tensor,
    scale: f32
) -> Result<()> {
    // w' = w + scale * (lora_up @ lora_down)
    let delta = lora_up.matmul(&lora_down)? * scale;
    linear.weight = (linear.weight + delta)?;
    Ok(())
}
```

## Verification Tests

Each integration will include:
1. Model loading test
2. Text encoding test
3. Single denoising step test
4. Full generation test with prompt
5. LoRA weight application test
6. Memory usage verification

## Success Criteria
- Generated images are recognizable and match prompts
- No placeholder functions
- All forward() calls use real implementations
- LoRA weights properly affect output
- Memory usage stays within 24GB VRAM limit