# Sampling Implementation Status

## Overview
This document tracks the status of sampling/inference implementation during training for each supported model.

## Implementation Status

### ✅ SDXL - WORKING
- **Status**: Fully implemented and functional
- **Implementation**: Uses `TrainingSampler` from `sdxl_sampling_complete.rs`
- **Features**:
  - DDIM scheduler with 30 steps
  - Classifier-free guidance (scale 7.5)
  - Saves samples to `output/[model]/samples/step_XXXXXX/`
  - Generates samples based on validation prompts from config
  - Supports VAE decoding to images
  - Creates both .png images and .txt prompt files

### ⚠️ SD 3.5 - PLACEHOLDER
- **Status**: Not yet implemented
- **Current behavior**: Logs placeholder message during training
- **Reason**: SD 3.5 requires:
  - Triple text encoding (CLIP-L, CLIP-G, T5-XXL)
  - Flow matching instead of DDPM
  - 16-channel VAE support
  - Different conditioning format
- **Next steps**: Implement SD35Sampler similar to SDXLSampler

### ⚠️ Flux - PLACEHOLDER  
- **Status**: Not yet implemented
- **Current behavior**: Logs placeholder message during training
- **Reason**: Memory-efficient mode requires model swapping:
  - Need to unload Flux model
  - Load T5 + CLIP for text encoding
  - Load VAE for decoding
  - Generate samples
  - Unload and reload Flux
- **Next steps**: Implement model swapping or use separate sampling process

## Configuration

Sampling is configured in the YAML files under the `sample` section:

```yaml
sample:
  sampler: "ddim"
  sample_every: 250
  sample_steps: 30
  guidance_scale: 7.5
  prompts:
    - "a photo of {trigger_word}"
    - "a painting of {trigger_word} in the style of Van Gogh"
  neg: "low quality, blurry"
  width: 1024
  height: 1024
```

## Technical Details

### SDXL Sampling Flow
1. Check if VAE and text encoders are loaded
2. Use `TrainingSampler` to generate samples
3. Text encoding with CLIP-L and CLIP-G
4. Initialize random latents
5. Denoise using DDIM scheduler
6. Apply LoRA weights during forward pass
7. Decode latents to images using VAE
8. Save images and prompts to disk

### Forward Pass Compatibility
- Created `sdxl_forward_sampling.rs` for sampling-specific forward pass
- Supports additional SDXL conditioning (pooled embeddings, time IDs)
- Simplified implementation focused on inference rather than training

## Usage

For SDXL, sampling works automatically when configured in the YAML:
- Samples are generated every `sample_every` steps
- Images saved as PNG files
- Prompts saved as text files
- Directory structure: `samples/step_XXXXXX/`

For SD 3.5 and Flux:
- Training continues normally without sampling
- Placeholder messages indicate why sampling is not available
- LoRA weights are still saved at checkpoints

## Future Improvements

1. **SD 3.5 Sampling**:
   - Implement flow matching scheduler
   - Add T5 text encoding support
   - Handle 16-channel VAE latents
   - Create SD35Sampler class

2. **Flux Sampling**:
   - Implement model swapping mechanism
   - Or create separate sampling script that runs in parallel
   - Handle patchified latents (2x2 patches)
   - Support guidance embedding

3. **General Improvements**:
   - Add more scheduler options (DPM++, Euler A)
   - Support dynamic resolution sampling
   - Add batch sampling for efficiency
   - Implement prompt templating with wildcards