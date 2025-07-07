# SD 3.5 LoRA Training Implementation Plan

## Executive Summary
This document outlines a comprehensive plan to enable functional LoRA training for SD 3.5 models in EriDiffusion. While SD 3.5 already has a working LoKr trainer, we need to implement standard LoRA and fix the inference pipeline.

## Current State Analysis

### Working Components
- ✅ LoKr training pipeline (0.6 it/s on 24GB VRAM)
- ✅ Text encoding (CLIP-L, CLIP-G, T5-XXL)
- ✅ VAE encoding with CPU offloading
- ✅ Memory management and latent caching
- ✅ Flow matching training objective

### Non-Functional Components
- ❌ Standard LoRA implementation (only LoKr works)
- ❌ Native inference pipeline (uses external binary)
- ❌ LoRA/LoKr weight application during inference
- ❌ MMDiT forward pass in sampling
- ❌ Proper layer interception for adapters

### Root Cause
The current implementation uses a workaround approach where LoKr is applied post-hoc after the base model forward pass. This works for training but prevents proper inference integration.

---

## Implementation Strategy

Build upon the existing SD 3.5 infrastructure by:
1. Implementing standard LoRA using the same patterns as LoKr
2. Creating a native inference pipeline to replace external binary dependency
3. Implementing proper MMDiT layer wrapping for adapter injection
4. Ensuring compatibility with trained weights

---

## Phase 1: Standard LoRA Implementation (Week 1)

### Objective
Implement regular LoRA for SD 3.5 following the successful LoKr pattern.

### 1.1 Create SD 3.5 LoRA Adapter

**File**: `eridiffusion/crates/networks/src/sd35_lora.rs`

```rust
use candle_core::{Tensor, Module, Var, Result, Device, DType};
use std::collections::HashMap;

pub struct SD35LoRALayer {
    // LoRA parameters
    pub lora_a: Var,  // [rank, in_features]
    pub lora_b: Var,  // [out_features, rank]
    
    // Configuration
    pub in_features: usize,
    pub out_features: usize,
    pub rank: usize,
    pub alpha: f32,
    pub dropout: f32,
}

impl SD35LoRALayer {
    pub fn new(
        in_features: usize,
        out_features: usize,
        rank: usize,
        alpha: f32,
        device: &Device,
    ) -> Result<Self> {
        // Initialize A with normal distribution
        let std_dev = (1.0 / (in_features as f32)).sqrt();
        let lora_a = Var::new(
            Tensor::randn(0.0, std_dev as f64, &[rank, in_features], device)?,
            device,
        )?;
        
        // Initialize B with zeros (important for stability)
        let lora_b = Var::new(
            Tensor::zeros(&[out_features, rank], DType::F32, device)?,
            device,
        )?;
        
        Ok(Self {
            lora_a,
            lora_b,
            in_features,
            out_features,
            rank,
            alpha,
            dropout: 0.0,
        })
    }
    
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [batch, seq_len, in_features]
        // Compute x @ A^T @ B^T
        let lora_out = x
            .matmul(&self.lora_a.as_tensor().t()?)?
            .matmul(&self.lora_b.as_tensor().t()?)?;
        
        // Scale by alpha/rank
        let scale = self.alpha / self.rank as f32;
        lora_out.affine(scale as f64, 0.0)
    }
}

pub struct SD35LoRAAdapter {
    pub layers: HashMap<String, SD35LoRALayer>,
    pub rank: usize,
    pub alpha: f32,
    pub target_modules: Vec<String>,
}

impl SD35LoRAAdapter {
    pub fn new(
        model_config: &SD35Config,
        rank: usize,
        alpha: f32,
        target_modules: Vec<String>,
        device: &Device,
    ) -> Result<Self> {
        let mut layers = HashMap::new();
        
        // Create LoRA layers for each target module
        for module_name in &target_modules {
            let (in_features, out_features) = Self::get_module_dimensions(model_config, module_name)?;
            
            let layer = SD35LoRALayer::new(
                in_features,
                out_features,
                rank,
                alpha,
                device,
            )?;
            
            layers.insert(module_name.clone(), layer);
        }
        
        Ok(Self {
            layers,
            rank,
            alpha,
            target_modules,
        })
    }
    
    fn get_module_dimensions(config: &SD35Config, module_name: &str) -> Result<(usize, usize)> {
        // Map module names to dimensions based on SD 3.5 architecture
        let hidden_size = config.hidden_size; // 2048 for SD 3.5 Large
        
        match module_name {
            "attn.to_q" | "attn.to_k" | "attn.to_v" => Ok((hidden_size, hidden_size)),
            "attn.to_out.0" => Ok((hidden_size, hidden_size)),
            "ff.net.0" => Ok((hidden_size, hidden_size * 4)),
            "ff.net.2" => Ok((hidden_size * 4, hidden_size)),
            _ => Err(Error::Msg(format!("Unknown module: {}", module_name))),
        }
    }
    
    pub fn trainable_parameters(&self) -> Vec<&Var> {
        let mut params = Vec::new();
        for layer in self.layers.values() {
            params.push(&layer.lora_a);
            params.push(&layer.lora_b);
        }
        params
    }
}
```

### 1.2 MMDiT Block with LoRA Support

**File**: `eridiffusion/crates/models/src/sd35_lora/mmdit_block.rs`

```rust
pub struct MMDiTBlockWithLoRA {
    // Base components (frozen)
    norm1: RMSNorm,
    attn: Attention,
    norm2: RMSNorm,
    ff: FeedForward,
    
    // LoRA adapters
    lora_adapters: HashMap<String, SD35LoRALayer>,
    
    // Configuration
    block_idx: usize,
}

impl MMDiTBlockWithLoRA {
    pub fn forward(
        &self,
        x: &Tensor,
        c: &Tensor,
        timestep_emb: &Tensor,
    ) -> Result<Tensor> {
        // 1. Pre-norm and attention
        let norm_x = self.norm1.forward(x)?;
        
        // Apply modulation from timestep
        let (shift, scale, gate) = self.get_modulation(timestep_emb)?;
        let modulated_x = apply_modulation(&norm_x, &shift, &scale)?;
        
        // Attention with LoRA
        let attn_out = self.forward_attention_with_lora(&modulated_x, c)?;
        
        // Gated residual
        let x = x + gate * attn_out;
        
        // 2. Pre-norm and feedforward
        let norm_x = self.norm2.forward(&x)?;
        let (shift2, scale2, gate2) = self.get_modulation_ff(timestep_emb)?;
        let modulated_x = apply_modulation(&norm_x, &shift2, &scale2)?;
        
        // Feedforward with LoRA
        let ff_out = self.forward_ff_with_lora(&modulated_x)?;
        
        // Gated residual
        Ok(x + gate2 * ff_out)
    }
    
    fn forward_attention_with_lora(
        &self,
        x: &Tensor,
        context: &Tensor,
    ) -> Result<Tensor> {
        // Get Q, K, V projections
        let q = self.attn.to_q.forward(x)?;
        let k = self.attn.to_k.forward(context)?;
        let v = self.attn.to_v.forward(context)?;
        
        // Add LoRA adaptations if present
        if let Some(lora_q) = self.lora_adapters.get(&format!("block_{}.attn.to_q", self.block_idx)) {
            q = q.add(&lora_q.forward(x)?)?;
        }
        if let Some(lora_k) = self.lora_adapters.get(&format!("block_{}.attn.to_k", self.block_idx)) {
            k = k.add(&lora_k.forward(context)?)?;
        }
        if let Some(lora_v) = self.lora_adapters.get(&format!("block_{}.attn.to_v", self.block_idx)) {
            v = v.add(&lora_v.forward(context)?)?;
        }
        
        // Compute attention
        let attn_out = scaled_dot_product_attention(&q, &k, &v)?;
        
        // Output projection with LoRA
        let out = self.attn.to_out.forward(&attn_out)?;
        if let Some(lora_out) = self.lora_adapters.get(&format!("block_{}.attn.to_out.0", self.block_idx)) {
            out = out.add(&lora_out.forward(&attn_out)?)?;
        }
        
        Ok(out)
    }
}
```

### 1.3 SD 3.5 Model with LoRA

**File**: `eridiffusion/crates/models/src/sd35_lora/model.rs`

```rust
pub struct SD35ModelWithLoRA {
    // Input layers
    x_embedder: PatchEmbed,
    t_embedder: TimestepEmbedder,
    y_embedder: VectorEmbedder,
    
    // Transformer blocks with LoRA
    blocks: Vec<MMDiTBlockWithLoRA>,
    
    // Output layers
    final_layer: FinalLayer,
    
    // LoRA configuration
    lora_adapter: SD35LoRAAdapter,
    
    // Base configuration
    config: SD35Config,
}

impl SD35ModelWithLoRA {
    pub fn from_pretrained(
        model_path: &Path,
        lora_config: LoRAConfig,
        device: &Device,
    ) -> Result<Self> {
        // Load base model weights
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_path], DType::F32, device)? };
        
        // Create LoRA adapter
        let lora_adapter = SD35LoRAAdapter::new(
            &config,
            lora_config.rank,
            lora_config.alpha,
            lora_config.target_modules,
            device,
        )?;
        
        // Create transformer blocks with LoRA support
        let mut blocks = Vec::new();
        for i in 0..config.num_layers {
            let block = MMDiTBlockWithLoRA::new(
                &config,
                i,
                &lora_adapter,
                vb.pp(&format!("transformer_blocks.{}", i)),
            )?;
            blocks.push(block);
        }
        
        Ok(Self {
            x_embedder,
            t_embedder,
            y_embedder,
            blocks,
            final_layer,
            lora_adapter,
            config,
        })
    }
    
    pub fn trainable_parameters(&self) -> Vec<&Var> {
        self.lora_adapter.trainable_parameters()
    }
}
```

---

## Phase 2: Native Inference Pipeline (Week 2)

### Objective
Implement native SD 3.5 sampling to replace external binary dependency.

### 2.1 Complete SD 3.5 Sampler

**File**: `eridiffusion/src/trainers/sd35_sampling_native.rs`

```rust
use candle_core::{Tensor, Module, Device, DType};
use crate::models::{SD35ModelWithLoRA, VAE};

pub struct SD35NativeSampler {
    model: SD35ModelWithLoRA,
    vae: VAE,
    device: Device,
}

impl SD35NativeSampler {
    pub fn generate(
        &self,
        prompt_embeds: &Tensor,      // [batch, seq_len, 4096]
        pooled_embeds: &Tensor,      // [batch, 2048]
        height: usize,
        width: usize,
        num_steps: usize,
        guidance_scale: f32,
        seed: Option<u64>,
    ) -> Result<Tensor> {
        let batch_size = prompt_embeds.dims()[0];
        
        // Initialize latents
        let latent_height = height / 8;
        let latent_width = width / 8;
        let latent_channels = 16; // SD 3.5 uses 16 channels
        
        let mut latents = if let Some(s) = seed {
            Tensor::randn_with_seed(
                s,
                0.0,
                1.0,
                &[batch_size, latent_channels, latent_height, latent_width],
                &self.device,
            )?
        } else {
            Tensor::randn(
                0.0,
                1.0,
                &[batch_size, latent_channels, latent_height, latent_width],
                &self.device,
            )?
        };
        
        // Create timestep schedule (linear for SD 3.5)
        let timesteps = self.get_linear_timesteps(num_steps);
        
        // Sampling loop
        for (i, &t) in timesteps.iter().enumerate() {
            let timestep = Tensor::new(&[t], &self.device)?
                .to_dtype(DType::F32)?
                .unsqueeze(0)?
                .repeat(&[batch_size])?;
            
            // Prepare conditional and unconditional inputs
            let latent_input = if guidance_scale > 1.0 {
                Tensor::cat(&[&latents, &latents], 0)?
            } else {
                latents.clone()
            };
            
            let embeds_input = if guidance_scale > 1.0 {
                let uncond_embeds = Tensor::zeros_like(prompt_embeds)?;
                Tensor::cat(&[prompt_embeds, &uncond_embeds], 0)?
            } else {
                prompt_embeds.clone()
            };
            
            let pooled_input = if guidance_scale > 1.0 {
                let uncond_pooled = Tensor::zeros_like(pooled_embeds)?;
                Tensor::cat(&[pooled_embeds, &uncond_pooled], 0)?
            } else {
                pooled_embeds.clone()
            };
            
            let timestep_input = if guidance_scale > 1.0 {
                Tensor::cat(&[&timestep, &timestep], 0)?
            } else {
                timestep.clone()
            };
            
            // Model forward pass (with LoRA!)
            let velocity_pred = self.model.forward(
                &latent_input,
                &timestep_input,
                &embeds_input,
                &pooled_input,
            )?;
            
            // Apply classifier-free guidance
            let velocity = if guidance_scale > 1.0 {
                let cond_v = velocity_pred.narrow(0, 0, batch_size)?;
                let uncond_v = velocity_pred.narrow(0, batch_size, batch_size)?;
                
                // v = uncond + scale * (cond - uncond)
                let diff = (&cond_v - &uncond_v)?;
                uncond_v.add(&diff.affine(guidance_scale as f64, 0.0)?)?
            } else {
                velocity_pred
            };
            
            // Update latents using flow matching
            let dt = 1.0 / num_steps as f32;
            latents = (&latents - velocity.affine(dt as f64, 0.0)?)?;
        }
        
        // Decode latents to image
        let images = self.vae.decode(&latents)?;
        
        // Convert from [-1, 1] to [0, 1]
        let images = images.affine(0.5, 0.5)?;
        
        Ok(images)
    }
    
    fn get_linear_timesteps(&self, num_steps: usize) -> Vec<f32> {
        // Linear timesteps from 1.0 to 0.0
        (0..num_steps)
            .map(|i| 1.0 - (i as f32 / (num_steps - 1) as f32))
            .collect()
    }
}
```

### 2.2 Fix MMDiT Forward Implementation

**File**: `eridiffusion/crates/models/src/sd35_lora/mmdit_forward.rs`

```rust
impl SD35ModelWithLoRA {
    pub fn forward(
        &self,
        x: &Tensor,           // latents: [batch, 16, h, w]
        t: &Tensor,           // timesteps: [batch]
        y: &Tensor,           // text_embeds: [batch, seq_len, 4096]
        pooled_y: &Tensor,    // pooled_embeds: [batch, 2048]
    ) -> Result<Tensor> {
        // 1. Patch embedding
        let x = self.x_embedder.forward(x)?; // [batch, num_patches, hidden_size]
        
        // 2. Time embedding
        let t_emb = self.t_embedder.forward(t)?; // [batch, hidden_size]
        
        // 3. Text embedding projection
        let y_emb = self.y_embedder.forward(pooled_y)?; // [batch, hidden_size]
        
        // 4. Combine time and text embeddings
        let c = (t_emb + y_emb)?; // [batch, hidden_size]
        
        // 5. Pass through transformer blocks
        let mut x = x;
        for block in &self.blocks {
            x = block.forward(&x, y, &c)?;
        }
        
        // 6. Final layer
        let x = self.final_layer.forward(&x, &c)?;
        
        // 7. Unpatchify
        let x = self.unpatchify(x)?; // [batch, 16, h, w]
        
        Ok(x)
    }
    
    fn unpatchify(&self, x: &Tensor) -> Result<Tensor> {
        // Reverse of patchification
        let batch_size = x.dims()[0];
        let num_patches = x.dims()[1];
        let patch_size = 2; // SD 3.5 uses 2x2 patches
        
        // Calculate spatial dimensions
        let h = (num_patches as f32).sqrt() as usize;
        let w = h;
        
        // Reshape to image format
        x.reshape(&[batch_size, h, w, patch_size, patch_size, 16])?
            .permute(&[0, 5, 1, 3, 2, 4])?
            .reshape(&[batch_size, 16, h * patch_size, w * patch_size])
    }
}
```

---

## Phase 3: Training Integration (Week 3)

### Objective
Update the training pipeline to use standard LoRA instead of LoKr.

### 3.1 SD 3.5 LoRA Trainer

**File**: `eridiffusion/src/trainers/sd35_lora_trainer.rs`

```rust
pub struct SD35LoRATrainer {
    // Models
    model: SD35ModelWithLoRA,
    vae: VAE,
    text_encoders: TextEncoders,
    
    // Training components
    optimizer: AdamW,
    ema: Option<EMAModel>,
    
    // Configuration
    config: SD35LoRAConfig,
    device: Device,
    
    // State
    global_step: usize,
}

impl SD35LoRATrainer {
    pub fn new(
        model_path: &Path,
        vae_path: &Path,
        text_encoder_paths: TextEncoderPaths,
        config: SD35LoRAConfig,
    ) -> Result<Self> {
        let device = Device::new_cuda(config.device_id)?;
        
        // Load models
        let model = SD35ModelWithLoRA::from_pretrained(
            model_path,
            config.lora_config.clone(),
            &device,
        )?;
        
        let vae = load_vae(vae_path, &device)?;
        let text_encoders = load_text_encoders(text_encoder_paths, &device)?;
        
        // Create optimizer for LoRA parameters only
        let lora_params = model.trainable_parameters();
        let var_map = create_var_map_from_params(&lora_params);
        let optimizer = AdamW::new(
            var_map.all_vars(),
            ParamsAdamW {
                lr: config.learning_rate,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
                weight_decay: 0.01,
            },
        )?;
        
        Ok(Self {
            model,
            vae,
            text_encoders,
            optimizer,
            ema: None,
            config,
            device,
            global_step: 0,
        })
    }
    
    pub fn train_step(&mut self, batch: TrainingBatch) -> Result<f32> {
        // 1. Encode images to latents (with caching)
        let latents = if let Some(cached) = batch.cached_latents {
            cached
        } else {
            self.vae.encode(&batch.pixel_values)?
        };
        
        // 2. Encode text (already cached)
        let text_embeds = batch.text_embeds;
        let pooled_embeds = batch.pooled_embeds;
        
        // 3. Sample timesteps
        let batch_size = latents.dims()[0];
        let timesteps = sample_timesteps(batch_size, &self.device)?;
        
        // 4. Add noise (flow matching)
        let noise = Tensor::randn_like(&latents)?;
        let noisy_latents = add_flow_matching_noise(&latents, &noise, &timesteps)?;
        
        // 5. Forward pass
        let velocity_pred = self.model.forward(
            &noisy_latents,
            &timesteps,
            &text_embeds,
            &pooled_embeds,
        )?;
        
        // 6. Compute loss
        let velocity_target = compute_velocity_target(&latents, &noise, &timesteps)?;
        let loss = mse_loss(&velocity_pred, &velocity_target)?;
        
        // 7. Backward pass
        self.optimizer.zero_grad();
        loss.backward()?;
        
        // 8. Gradient clipping
        if let Some(max_norm) = self.config.gradient_clip_val {
            clip_grad_norm(&self.model.trainable_parameters(), max_norm)?;
        }
        
        // 9. Optimizer step
        self.optimizer.step()?;
        
        // 10. Update EMA
        if let Some(ema) = &mut self.ema {
            ema.update(&self.model)?;
        }
        
        self.global_step += 1;
        
        Ok(loss.to_scalar::<f32>()?)
    }
}
```

### 3.2 Integrated Sampling During Training

**File**: `eridiffusion/src/trainers/sd35_training_with_sampling.rs`

```rust
impl SD35LoRATrainer {
    pub fn generate_samples(
        &self,
        prompts: &[String],
        step: usize,
        output_dir: &Path,
    ) -> Result<Vec<PathBuf>> {
        // Use native sampler instead of external binary
        let sampler = SD35NativeSampler::new(
            self.model.clone(),
            self.vae.clone(),
            self.device.clone(),
        );
        
        let mut saved_paths = Vec::new();
        
        for (i, prompt) in prompts.iter().enumerate() {
            // Encode prompt
            let (text_embeds, pooled_embeds) = self.encode_prompt(prompt)?;
            
            // Generate image
            let image = sampler.generate(
                &text_embeds,
                &pooled_embeds,
                1024,  // height
                1024,  // width
                50,    // steps
                7.5,   // guidance
                Some(42 + i as u64),
            )?;
            
            // Save image
            let path = output_dir.join(format!("step_{:06}_sample_{:02}.png", step, i));
            save_tensor_as_image(&image, &path)?;
            saved_paths.push(path);
        }
        
        Ok(saved_paths)
    }
}
```

---

## Phase 4: Testing and Validation (Week 4)

### Objective
Ensure the implementation works correctly.

### 4.1 Gradient Flow Test

```rust
#[test]
fn test_sd35_lora_gradients() {
    let model = create_test_sd35_model_with_lora();
    let params = model.trainable_parameters();
    
    // Forward pass
    let loss = compute_test_loss(&model)?;
    
    // Backward
    loss.backward()?;
    
    // Check all LoRA parameters have gradients
    for (i, param) in params.iter().enumerate() {
        let grad = param.grad().expect("Missing gradient");
        let grad_norm = grad.abs()?.mean_all()?.to_scalar::<f32>()?;
        assert!(grad_norm > 1e-8, "Zero gradient for param {}", i);
    }
}
```

### 4.2 Training Convergence Test

```rust
#[test]
fn test_sd35_lora_training() {
    let mut trainer = create_test_trainer()?;
    let dataset = create_small_dataset()?;
    
    let mut losses = Vec::new();
    
    // Train for 100 steps
    for batch in dataset.take(100) {
        let loss = trainer.train_step(batch)?;
        losses.push(loss);
    }
    
    // Verify loss decreases
    let avg_first_10: f32 = losses[..10].iter().sum::<f32>() / 10.0;
    let avg_last_10: f32 = losses[90..].iter().sum::<f32>() / 10.0;
    
    assert!(avg_last_10 < avg_first_10 * 0.7, "Loss did not decrease");
}
```

---

## Phase 5: Optimization and Polish (Week 5)

### Objective
Optimize for production use and add quality-of-life features.

### 5.1 Memory Optimization

```rust
pub struct SD35MemoryOptimizer {
    pub gradient_checkpointing: bool,
    pub gradient_checkpointing_blocks: Vec<usize>,
    pub cpu_offload_optimizer: bool,
    pub activation_checkpointing: bool,
}

impl SD35MemoryOptimizer {
    pub fn apply_to_model(&self, model: &mut SD35ModelWithLoRA) {
        if self.gradient_checkpointing {
            for (i, block) in model.blocks.iter_mut().enumerate() {
                if self.gradient_checkpointing_blocks.contains(&i) {
                    block.enable_gradient_checkpointing();
                }
            }
        }
    }
}
```

### 5.2 Multi-Resolution Training

```rust
pub struct MultiResolutionConfig {
    pub resolutions: Vec<(usize, usize)>,
    pub batch_sizes: HashMap<(usize, usize), usize>,
}

impl SD35LoRATrainer {
    pub fn train_step_multiresolution(
        &mut self,
        batch: MultiResolutionBatch,
    ) -> Result<f32> {
        // Group by resolution
        let mut total_loss = 0.0;
        let mut total_samples = 0;
        
        for (resolution, samples) in batch.grouped_by_resolution() {
            let loss = self.train_step_single_resolution(samples, resolution)?;
            total_loss += loss * samples.len() as f32;
            total_samples += samples.len();
        }
        
        Ok(total_loss / total_samples as f32)
    }
}
```

---

## Configuration Templates

### Minimal Configuration (Testing)
```yaml
model:
  name_or_path: "/home/alex/SwarmUI/Models/stable-diffusion/sd3.5_large.safetensors"
  is_v3: true

lora:
  rank: 4
  alpha: 4.0
  target_modules: ["attn.to_q", "attn.to_v"]
  
training:
  batch_size: 1
  learning_rate: 1e-4
  num_steps: 100
  gradient_checkpointing: true
```

### Standard Configuration (Recommended)
```yaml
model:
  name_or_path: "/home/alex/SwarmUI/Models/stable-diffusion/sd3.5_large.safetensors"
  is_v3: true

lora:
  rank: 32
  alpha: 32.0
  target_modules: [
    "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0"
  ]
  
training:
  batch_size: 1
  gradient_accumulation_steps: 4
  learning_rate: 5e-5
  num_steps: 2000
  gradient_checkpointing: true
  gradient_checkpointing_blocks: [0, 4, 8, 12, 16, 20]
  
sampling:
  every_n_steps: 250
  prompts: [
    "a photo of sks person",
    "sks person wearing a hat",
    "sks person smiling"
  ]
```

---

## Success Metrics

1. **LoRA weights update properly** ✓
2. **Native inference works** ✓
3. **Training loss converges** ✓
4. **Can load and merge LoRA weights** ✓
5. **Memory usage < 24GB** ✓
6. **Generation quality matches LoKr** ✓

---

## Migration Path from LoKr

For users with existing LoKr models:
1. Continue using LoKr trainer for now
2. New models can use standard LoRA
3. Provide conversion utility if needed
4. Both will use the same inference pipeline

---

## Timeline

- **Week 1**: Implement standard LoRA adapter
- **Week 2**: Native inference pipeline
- **Week 3**: Training integration
- **Week 4**: Testing and validation
- **Week 5**: Optimization and polish

Total: 5 weeks to complete SD 3.5 LoRA implementation

This plan builds on the existing successful LoKr implementation while adding standard LoRA support and fixing the inference pipeline issues.