# Comprehensive Pipeline Analysis

This document provides a detailed analysis of the SD 3.5 and Flux pipelines in EriDiffusion, along with insights from SimpleTuner's implementation.

## Table of Contents
1. [SD 3.5 Pipeline Analysis](#sd-35-pipeline-analysis)
2. [Flux Pipeline Analysis](#flux-pipeline-analysis)
3. [SimpleTuner Implementation Analysis](#simpletuner-implementation-analysis)
4. [Comparative Analysis](#comparative-analysis)
5. [Key Findings and Recommendations](#key-findings-and-recommendations)

---

## SD 3.5 Pipeline Analysis

### Working Components ✅

#### 1. LoKr Training Infrastructure
- **Location**: `eridiffusion/src/trainers/sd35_lokr.rs`
- **Performance**: ~0.6 iterations/second on 24GB VRAM
- **Key Features**:
  - Full LoKr (Low-rank Kronecker product) adapter implementation
  - Flow matching objective with SNR gamma weighting
  - Memory-optimized training with VAE CPU offloading
  - Latent caching to disk for efficiency
  - Triple text encoder integration (CLIP-L, CLIP-G, T5-XXL)
  - Checkpoint saving in safetensors format

#### 2. Memory Management System
- **VAE Offloading**: SimpleTuner-style CPU offloading to save ~2-3GB VRAM
- **Latent Caching**: Pre-computed latents stored to disk
- **Gradient Checkpointing**: Reduces activation memory at 20-30% speed cost
- **Optimized for 24GB**: Careful memory allocation with batch size 1

#### 3. Text Encoding Pipeline
- **Multi-Encoder Support**:
  ```rust
  // From sd35_lokr.rs
  let clip_g_embeds = self.encode_prompts_clip_g(&prompts)?;
  let clip_l_embeds = self.encode_prompts_clip_l(&prompts)?;
  let t5_embeds = self.encode_prompts_t5(&prompts)?;
  ```
- **Pooled Embeddings**: Proper generation for all three encoders
- **Token Caching**: Encoded embeddings cached during training

### Non-Functional Components ❌

#### 1. Inference/Sampling Pipeline
- **Critical Issue**: No native SD 3.5 inference implementation
- **Current Workaround**: External candle binary invocation
  ```rust
  // From sd35_lokr.rs:1310
  let status = std::process::Command::new(&sampler_path)
      .arg(&self.model_path)
      .arg(&temp_lora_path)  // LoKr weights passed but not used
      .status()?;
  ```
- **Problems**:
  - Cannot apply trained LoKr weights during inference
  - Relies on external process spawning
  - Memory inefficient (drops/reloads model)
  - Output limited to PPM format

#### 2. LoKr Weight Application
- **MMDiT Forward Pass Issue**:
  ```rust
  // From sampling.rs:152
  fn mmdit_forward(...) -> Result<Tensor> {
      // This is a placeholder - actual implementation depends on MMDiT interface
      // For now, return a dummy velocity
      Ok(Tensor::randn_like(latent, 0.0, 1.0)?)
  }
  ```
- **No Layer Interception**: Cannot inject LoKr into model layers
- **Weight Merging Non-Functional**: Merge utility exists but doesn't work

#### 3. Model Architecture Limitations
- **RMS Norm Workaround**: Custom implementation due to CUDA issues
  ```rust
  // From sd35_lokr.rs
  fn forward_with_rms_norm_workaround(...) -> Result<Tensor> {
      // Workaround for RMS norm CUDA compatibility
  }
  ```
- **Incomplete MMDiT Wrapper**: `mmdit_lokr_wrapper.rs` is scaffolding only

### Missing Components 🚧
1. **Native Diffusion Sampling Loop**
2. **Proper MMDiT Layer Hooks**
3. **Other Network Adapters** (LoRA, DoRA, LoCoN)
4. **Image Format Support** (PNG/JPEG)
5. **Validation Sampling During Training**

---

## Flux Pipeline Analysis

### Working Components ✅

#### 1. Complete Inference Pipeline
- **Location**: `eridiffusion/crates/training/src/pipelines/sampling.rs`
- **Key Features**:
  ```rust
  // From sampling.rs:308
  fn generate_flux_latent(...) -> Result<Tensor> {
      // Flux uses 64 channels in latent space (16 * 2 * 2 from patchification)
      let latent_channels = 16;
      let latent_height = self.config.height / 16; // VAE + patchify
      let latent_width = self.config.width / 16;
      
      // Patchify latents for Flux (2x2 patches)
      let patchified = self.patchify_for_flux(&latent)?;
      
      // Flux uses a shifted sigmoid schedule
      let timesteps = self.get_flux_timesteps(num_steps, latent_height * latent_width);
  }
  ```

#### 2. Model Infrastructure
- **Model Loading**: Complete safetensors support via `flux_model_loader.rs`
- **Text Encoders**: Both T5-XXL and CLIP-L fully integrated
- **VAE Wrapper**: 16-channel support with proper encoding/decoding
- **Variant Support**: Base, Dev, and Schnell models

#### 3. Training Infrastructure
- **FluxTrainer** (`flux_trainer.rs`):
  - Rectified flow training implementation
  - Guidance dropout (10% rate)
  - Flow matching loss computation
  - Mixed precision (BF16) support
  - Checkpoint save/load infrastructure

### Partially Implemented ⚠️

#### 1. LoRA Training Structure
- **Location**: `flux_lora_trainer_24gb.rs`
- **Issues**:
  ```rust
  // Line 150 - LoRA initialization works but not connected
  let down_tensor = Tensor::randn(0.0, scale, (rank, in_features), device)?;
  let up_tensor = Tensor::zeros((out_features, rank), dtype, device)?;
  
  // But forward pass returns dummy loss:
  pub fn forward_training_step(...) -> Result<Tensor> {
      Tensor::randn(0.0, 0.1, &[], device)  // Dummy implementation
  }
  ```

#### 2. Model Integration
- **Parameter Access Problem**:
  ```rust
  // FluxDiffusionModel wrapper
  fn parameters(&self) -> Vec<Tensor> {
      vec![]  // Returns empty - no access to internal parameters
  }
  ```

### Non-Functional Components ❌

#### 1. LoRA Injection Mechanism
- **No Layer Access**: Cannot modify Flux transformer blocks
- **No Parameter Tracking**: Empty parameter vectors
- **No Gradient Flow**: LoRA weights exist but disconnected

#### 2. Preprocessing Pipeline
- **Dummy Data Generation**:
  ```rust
  // From flux_train.rs
  latents: Tensor::randn(...),     // Should be VAE encoded
  text_embeds: Tensor::randn(...),  // Should be T5 encoded
  ```

### Key Difference from SD 3.5
- **Flux has working integrated sampling** ✅
- **SD 3.5 relies on external binaries** ❌
- Both suffer from LoRA/LoKr application issues

---

## SimpleTuner Implementation Analysis

### 1. Architecture Overview
- **Entry Point**: `train.py` → `Trainer` class
- **Model Loading**: `helpers/models/flux/model.py`
- **Transformer**: Custom `FluxTransformer2DModel` implementation
- **Framework**: PyTorch with PEFT for LoRA injection

### 2. Memory Management Excellence

#### A. Gradient Checkpointing
```python
# From config example
"gradient_checkpointing": true,
"gradient_checkpointing_interval": 2,  # Checkpoint every 2 blocks
```
- Reduces memory by ~40-50%
- Performance impact: ~20-30% slower
- Interval-based for fine-tuning memory/speed tradeoff

#### B. VAE Optimization
```python
# VAE handling strategy
if not args.keep_vae_loaded:
    init_unload_vae()  # Frees ~2-3GB VRAM after encoding
    
# On-demand caching
if args.vae_cache_ondemand:
    encode_and_cache_on_access()
```

#### C. Quantization Strategy
```python
# Quantization configs for 24GB VRAM
"base_model_precision": "int8-quanto",  # ~18GB VRAM
# or
"base_model_precision": "nf4-bnb",      # ~13GB VRAM
# or  
"base_model_precision": "int2-quanto",  # ~9GB VRAM

# Exclusions for critical layers
"--quantize_via": "quanto",
"--quantization_excluded_layers": ["norm", "img_in", "time_in", "proj_out"]
```

### 3. LoRA Injection Methods

#### A. Target Module Configuration
```python
# SimpleTuner's flexible targeting
flux_lora_target_map = {
    "mmdit": ["to_k", "to_q", "to_v", "to_out.0"],
    "all": "all-linear",
    "all+ffs": "all-linear+ffs", 
    "context": "text_model",
    "tiny": ["transformer_blocks.0", "transformer_blocks.1"],
    "nano": ["transformer_blocks.0"],
    "ai-toolkit": ["to_k", "to_q", "to_v", "to_out.0", "add_k_proj", "add_q_proj", "add_v_proj"]
}
```

#### B. PEFT Integration
```python
# How SimpleTuner injects LoRA
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=rank,
    lora_alpha=alpha,
    target_modules=target_modules,
    lora_dropout=dropout
)
model = get_peft_model(model, lora_config)
```

### 4. Flux-Specific Optimizations

#### A. Flow Matching
```python
# Timestep shifting for flow matching
if args.flow_schedule_shift:
    timesteps = shift_flux_schedule(timesteps, shift_amount)
    
# Loss computation
velocity_target = (noise - latents) / (1 - timesteps)
loss = F.mse_loss(velocity_pred, velocity_target)
```

#### B. Attention Optimizations
```python
# Fused QKV projections
if args.fuse_qkv_projections:
    processor = FluxFusedFlashAttnProcessor()
    model.set_attn_processor(processor)
```

### 5. Training Configuration for 24GB VRAM

```yaml
# Optimal settings from SimpleTuner
gradient_checkpointing: true
gradient_checkpointing_interval: 3
base_model_precision: "int8-quanto"
train_batch_size: 1
gradient_accumulation_steps: 4
vae_batch_size: 4
optimizer: "adamw_bf16"
mixed_precision: "bf16"
offload_during_startup: true
flux_lora_target: "mmdit"  # Or "tiny" for minimal VRAM
```

---

## Comparative Analysis

### Memory Management Comparison

| Feature | EriDiffusion SD3.5 | EriDiffusion Flux | SimpleTuner |
|---------|-------------------|-------------------|-------------|
| VAE Offloading | ✅ Manual drop/reload | ❌ Not implemented | ✅ Automatic |
| Gradient Checkpointing | ✅ Basic support | ✅ Config flag | ✅ Interval-based |
| Quantization | ❌ Not implemented | ❌ Not implemented | ✅ Multiple backends |
| Latent Caching | ✅ Disk caching | ❌ Not implemented | ✅ Smart caching |
| Memory Usage (24GB) | ~22GB (fits) | Unknown | 9-18GB (quantized) |

### LoRA Implementation Comparison

| Feature | EriDiffusion | SimpleTuner |
|---------|--------------|-------------|
| Framework | Candle (Rust) | PyTorch + PEFT |
| LoRA Injection | ❌ Cannot modify layers | ✅ PEFT handles it |
| Target Flexibility | ❌ Hardcoded attempts | ✅ Highly configurable |
| Gradient Flow | ❌ Broken | ✅ Automatic via PyTorch |
| Weight Merging | ❌ Non-functional | ✅ PEFT built-in |

### Architecture Differences

1. **Framework Limitations**:
   - **Candle**: Static model architecture, no dynamic modification
   - **PyTorch**: Dynamic graphs, easy module replacement
   
2. **LoRA Application**:
   - **EriDiffusion**: Post-hoc application (workaround)
   - **SimpleTuner**: In-place layer modification (proper)

3. **Memory Strategy**:
   - **EriDiffusion**: Manual memory management
   - **SimpleTuner**: Automated with multiple strategies

---

## Key Findings and Recommendations

### 1. Core Issue: Candle Framework Limitations

The fundamental problem is Candle's design philosophy:
- **Static Architecture**: Models are compiled, not dynamic
- **No Module Hooks**: Cannot intercept layer forward passes
- **Limited Parameter Access**: Cannot modify internal weights

### 2. Why SimpleTuner Works

SimpleTuner succeeds because PyTorch provides:
- **Dynamic Module System**: Can replace layers at runtime
- **PEFT Library**: Handles all LoRA injection complexity
- **Automatic Differentiation**: Gradients flow naturally

### 3. Potential Solutions for EriDiffusion

#### Option A: Fork Candle Framework
- Modify Candle to support dynamic module replacement
- Add hooks for layer interception
- Significant engineering effort

#### Option B: Re-implement Models
- Create custom Flux/SD3.5 implementations
- Build in LoRA support from ground up
- Full control but massive work

#### Option C: Hybrid Approach
- Keep inference in Candle
- Use PyTorch bindings for training only
- Breaks "pure Rust" goal

#### Option D: Alternative Framework
- Consider Burn or other Rust ML frameworks
- May have better extensibility
- Migration effort required

### 4. Immediate Actionable Steps

1. **For SD 3.5**:
   - Implement native sampling to replace external binary
   - Fix MMDiT forward pass
   - Complete LoKr weight application

2. **For Flux**:
   - Connect preprocessing to actual VAE/text encoders
   - Implement proper forward pass
   - Add layer access mechanism

3. **Memory Optimizations**:
   - Port SimpleTuner's quantization approach
   - Implement interval-based checkpointing
   - Add automatic VAE offloading

### 5. Conclusion

EriDiffusion has solid training infrastructure but is fundamentally limited by Candle's architecture. The gap between what's implemented (training loops, optimizers, data loading) and what works (actual gradient updates) stems from the framework's inability to dynamically modify models. SimpleTuner's success demonstrates that the algorithms are sound - the limitation is purely technical.

To achieve functional LoRA training in pure Rust, either Candle needs significant modifications or the models need complete reimplementation with adapter support built-in from the start.