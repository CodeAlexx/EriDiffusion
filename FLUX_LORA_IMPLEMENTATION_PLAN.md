# Flux LoRA Training Implementation Plan

## Executive Summary
This document outlines a comprehensive plan to enable functional LoRA training for Flux models in EriDiffusion. The plan addresses the fundamental limitation of Candle's static architecture by creating a custom Flux implementation with built-in LoRA support.

## Current State Analysis

### Working Components
- ✅ Flux inference with pre-trained models
- ✅ Complete sampling pipeline with patchification
- ✅ Text encoding (T5-XXL and CLIP-L)
- ✅ VAE encoding/decoding
- ✅ Training loop structure

### Non-Functional Components
- ❌ LoRA weight injection into model
- ❌ Parameter access from Flux model
- ❌ Gradient flow through LoRA adapters
- ❌ Real data preprocessing (uses dummy tensors)
- ❌ Actual loss computation (returns random values)

### Root Cause
Candle's Flux model is a sealed implementation that doesn't expose internal layers or parameters. This prevents any form of adapter injection or parameter modification.

---

## Implementation Strategy

Instead of trying to modify the existing Candle Flux model, we will:
1. Create a new Flux implementation with LoRA built-in from the ground up
2. Maintain compatibility with existing Flux weights
3. Ensure all components use real implementations (no dummy code)

---

## Phase 1: Foundation - LoRA-Capable Modules (Week 1)

### Objective
Create base building blocks that support LoRA adaptation natively.

### 1.1 Create LoRA-Aware Linear Module

**File**: `eridiffusion/crates/networks/src/lora_layers.rs`

```rust
use candle_core::{Tensor, Module, Var, Result, Device, DType};
use candle_nn::VarBuilder;

pub struct LinearWithLoRA {
    // Base layer parameters
    weight: Tensor,
    bias: Option<Tensor>,
    
    // LoRA parameters (trainable)
    lora_a: Option<Var>,  // [rank, in_features]
    lora_b: Option<Var>,  // [out_features, rank]
    
    // Configuration
    in_features: usize,
    out_features: usize,
    rank: Option<usize>,
    alpha: f32,
    dropout: f32,
    enabled: bool,
}

impl LinearWithLoRA {
    pub fn new(
        weight: Tensor,
        bias: Option<Tensor>,
        rank: Option<usize>,
        alpha: f32,
        device: &Device,
    ) -> Result<Self> {
        let shape = weight.shape();
        let out_features = shape.dims()[0];
        let in_features = shape.dims()[1];
        
        let (lora_a, lora_b) = if let Some(r) = rank {
            // Initialize LoRA weights
            let a = Var::new(
                Tensor::randn(0f32, 1f32, &[r, in_features], device)?,
                device,
            )?;
            let b = Var::new(
                Tensor::zeros(&[out_features, r], DType::F32, device)?,
                device,
            )?;
            (Some(a), Some(b))
        } else {
            (None, None)
        };
        
        Ok(Self {
            weight,
            bias,
            lora_a,
            lora_b,
            in_features,
            out_features,
            rank,
            alpha,
            dropout: 0.0,
            enabled: true,
        })
    }
    
    pub fn trainable_parameters(&self) -> Vec<&Var> {
        let mut params = Vec::new();
        if let Some(a) = &self.lora_a {
            params.push(a);
        }
        if let Some(b) = &self.lora_b {
            params.push(b);
        }
        params
    }
    
    pub fn disable_lora(&mut self) {
        self.enabled = false;
    }
    
    pub fn enable_lora(&mut self) {
        self.enabled = true;
    }
}

impl Module for LinearWithLoRA {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Base forward pass
        let mut output = x.matmul(&self.weight.t()?)?;
        
        // Add LoRA if enabled and present
        if self.enabled {
            if let (Some(a), Some(b)) = (&self.lora_a, &self.lora_b) {
                // LoRA forward: x @ A^T @ B^T
                let lora_out = x
                    .matmul(&a.as_tensor().t()?)?
                    .matmul(&b.as_tensor().t()?)?;
                
                // Scale by alpha/rank
                let scale = self.alpha / self.rank.unwrap_or(1) as f32;
                output = output.add(&lora_out.affine(scale as f64, 0.0)?)?;
            }
        }
        
        // Add bias if present
        if let Some(bias) = &self.bias {
            output = output.broadcast_add(bias)?;
        }
        
        Ok(output)
    }
}
```

### 1.2 Create Multi-Head Attention with LoRA

**File**: `eridiffusion/crates/networks/src/lora_attention.rs`

```rust
pub struct AttentionWithLoRA {
    to_q: LinearWithLoRA,
    to_k: LinearWithLoRA,
    to_v: LinearWithLoRA,
    to_out: LinearWithLoRA,
    
    num_heads: usize,
    head_dim: usize,
    dropout: f32,
}

impl AttentionWithLoRA {
    pub fn new(
        dim: usize,
        num_heads: usize,
        rank: Option<usize>,
        alpha: f32,
        vb: VarBuilder,
    ) -> Result<Self> {
        let head_dim = dim / num_heads;
        
        // Load base weights
        let q_weight = vb.get((dim, dim), "to_q.weight")?;
        let k_weight = vb.get((dim, dim), "to_k.weight")?;
        let v_weight = vb.get((dim, dim), "to_v.weight")?;
        let out_weight = vb.get((dim, dim), "to_out.0.weight")?;
        
        let device = vb.device();
        
        Ok(Self {
            to_q: LinearWithLoRA::new(q_weight, None, rank, alpha, device)?,
            to_k: LinearWithLoRA::new(k_weight, None, rank, alpha, device)?,
            to_v: LinearWithLoRA::new(v_weight, None, rank, alpha, device)?,
            to_out: LinearWithLoRA::new(out_weight, None, rank, alpha, device)?,
            num_heads,
            head_dim,
            dropout: 0.0,
        })
    }
    
    pub fn trainable_parameters(&self) -> Vec<&Var> {
        let mut params = Vec::new();
        params.extend(self.to_q.trainable_parameters());
        params.extend(self.to_k.trainable_parameters());
        params.extend(self.to_v.trainable_parameters());
        params.extend(self.to_out.trainable_parameters());
        params
    }
}
```

### 1.3 Testing Foundation

**File**: `eridiffusion/crates/networks/tests/lora_layers_test.rs`

```rust
#[test]
fn test_lora_forward_pass() {
    let device = Device::Cpu;
    let batch_size = 2;
    let seq_len = 10;
    let in_features = 512;
    let out_features = 512;
    let rank = 16;
    
    // Create base weight
    let weight = Tensor::randn(0f32, 1f32, &[out_features, in_features], &device).unwrap();
    
    // Create layer with LoRA
    let layer = LinearWithLoRA::new(weight.clone(), None, Some(rank), 16.0, &device).unwrap();
    
    // Test input
    let x = Tensor::randn(0f32, 1f32, &[batch_size, seq_len, in_features], &device).unwrap();
    
    // Forward pass
    let output = layer.forward(&x).unwrap();
    assert_eq!(output.shape().dims(), &[batch_size, seq_len, out_features]);
    
    // Verify LoRA parameters are created
    assert_eq!(layer.trainable_parameters().len(), 2);
}

#[test]
fn test_lora_gradient_flow() {
    // Test that gradients flow through LoRA weights
    // Implementation here
}
```

---

## Phase 2: Flux Model Architecture with LoRA (Week 2)

### Objective
Implement Flux transformer blocks with built-in LoRA support.

### 2.1 Flux Double Block with LoRA

**File**: `eridiffusion/crates/models/src/flux_lora/double_block.rs`

```rust
use crate::lora_layers::{LinearWithLoRA, AttentionWithLoRA};

pub struct FluxDoubleBlockWithLoRA {
    // Self-attention for image
    img_attn: AttentionWithLoRA,
    img_mlp: FluxMLPWithLoRA,
    
    // Self-attention for text
    txt_attn: AttentionWithLoRA,
    txt_mlp: FluxMLPWithLoRA,
    
    // Normalization layers (no LoRA)
    img_norm1: LayerNorm,
    img_norm2: LayerNorm,
    txt_norm1: LayerNorm,
    txt_norm2: LayerNorm,
}

impl FluxDoubleBlockWithLoRA {
    pub fn new(
        dim: usize,
        num_heads: usize,
        mlp_ratio: f32,
        lora_config: &LoRAConfig,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mlp_dim = (dim as f32 * mlp_ratio) as usize;
        
        Ok(Self {
            img_attn: AttentionWithLoRA::new(
                dim, 
                num_heads, 
                lora_config.rank,
                lora_config.alpha,
                vb.pp("img_attn"),
            )?,
            img_mlp: FluxMLPWithLoRA::new(
                dim,
                mlp_dim,
                lora_config.rank,
                lora_config.alpha,
                vb.pp("img_mlp"),
            )?,
            txt_attn: AttentionWithLoRA::new(
                dim,
                num_heads,
                lora_config.rank,
                lora_config.alpha,
                vb.pp("txt_attn"),
            )?,
            txt_mlp: FluxMLPWithLoRA::new(
                dim,
                mlp_dim,
                lora_config.rank,
                lora_config.alpha,
                vb.pp("txt_mlp"),
            )?,
            img_norm1: LayerNorm::new(dim, 1e-6, vb.pp("img_norm1"))?,
            img_norm2: LayerNorm::new(dim, 1e-6, vb.pp("img_norm2"))?,
            txt_norm1: LayerNorm::new(dim, 1e-6, vb.pp("txt_norm1"))?,
            txt_norm2: LayerNorm::new(dim, 1e-6, vb.pp("txt_norm2"))?,
        })
    }
    
    pub fn forward(
        &self,
        img: &Tensor,
        txt: &Tensor,
        vec: &Tensor,
        pe: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        // Image path with modulation
        let img_mod1 = self.img_norm1.forward(&vec)?;
        let img_norm = self.img_norm1.forward(&img)?;
        let img_modulated = apply_modulation(&img_norm, &img_mod1)?;
        let img_attn_out = self.img_attn.forward(&img_modulated, Some(pe))?;
        let img = img.add(&img_attn_out)?;
        
        // Image MLP
        let img_mod2 = self.img_norm2.forward(&vec)?;
        let img_norm2 = self.img_norm2.forward(&img)?;
        let img_modulated2 = apply_modulation(&img_norm2, &img_mod2)?;
        let img_mlp_out = self.img_mlp.forward(&img_modulated2)?;
        let img = img.add(&img_mlp_out)?;
        
        // Similar for text path...
        // (Implementation continues)
        
        Ok((img, txt))
    }
    
    pub fn trainable_parameters(&self) -> Vec<&Var> {
        let mut params = Vec::new();
        params.extend(self.img_attn.trainable_parameters());
        params.extend(self.img_mlp.trainable_parameters());
        params.extend(self.txt_attn.trainable_parameters());
        params.extend(self.txt_mlp.trainable_parameters());
        params
    }
}
```

### 2.2 Complete Flux Model with LoRA

**File**: `eridiffusion/crates/models/src/flux_lora/model.rs`

```rust
pub struct FluxModelWithLoRA {
    // Input layers
    img_in: Conv2d,
    txt_in: Linear,
    time_in: FluxTimeEmbedding,
    guidance_in: Option<Linear>,
    
    // Transformer blocks
    double_blocks: Vec<FluxDoubleBlockWithLoRA>,
    single_blocks: Vec<FluxSingleBlockWithLoRA>,
    
    // Output layers
    final_norm: LayerNorm,
    proj_out: Linear,
    
    // Configuration
    config: FluxConfig,
    lora_config: LoRAConfig,
}

impl FluxModelWithLoRA {
    pub fn new(
        config: FluxConfig,
        lora_config: LoRAConfig,
        vb: VarBuilder,
    ) -> Result<Self> {
        // Load base model weights
        let img_in = Conv2d::new(
            config.in_channels,
            config.hidden_size,
            3,
            Default::default(),
            vb.pp("img_in"),
        )?;
        
        // Create transformer blocks with LoRA
        let mut double_blocks = Vec::new();
        for i in 0..config.num_double_blocks {
            double_blocks.push(FluxDoubleBlockWithLoRA::new(
                config.hidden_size,
                config.num_heads,
                config.mlp_ratio,
                &lora_config,
                vb.pp(&format!("double_blocks.{}", i)),
            )?);
        }
        
        // (Continue implementation...)
        
        Ok(Self {
            img_in,
            txt_in,
            time_in,
            guidance_in,
            double_blocks,
            single_blocks,
            final_norm,
            proj_out,
            config,
            lora_config,
        })
    }
    
    pub fn trainable_parameters(&self) -> Vec<&Var> {
        let mut params = Vec::new();
        
        // Collect LoRA parameters from all blocks
        for block in &self.double_blocks {
            params.extend(block.trainable_parameters());
        }
        for block in &self.single_blocks {
            params.extend(block.trainable_parameters());
        }
        
        params
    }
    
    pub fn forward(&self, inputs: &ModelInputs) -> Result<ModelOutput> {
        // Actual forward implementation
        // No dummy tensors!
        
        // 1. Patchify and embed images
        let img_emb = self.embed_image(&inputs.latents)?;
        
        // 2. Embed text
        let txt_emb = self.embed_text(&inputs.encoder_hidden_states)?;
        
        // 3. Time and guidance embeddings
        let time_emb = self.time_in.forward(&inputs.timestep)?;
        let vec = self.compute_modulation_vector(&time_emb, &inputs.pooled_projections)?;
        
        // 4. Transformer blocks
        let (img, txt) = self.forward_blocks(img_emb, txt_emb, vec)?;
        
        // 5. Output projection
        let output = self.proj_out.forward(&img)?;
        
        Ok(ModelOutput {
            sample: output,
            ..Default::default()
        })
    }
}
```

---

## Phase 3: Training Pipeline Integration (Week 3)

### Objective
Connect the LoRA-capable model to the training pipeline with real data flow.

### 3.1 Fix Preprocessing - Real VAE and Text Encoding

**File**: `eridiffusion/crates/training/src/flux_preprocessor_real.rs`

```rust
pub struct RealFluxPreprocessor {
    vae: Arc<dyn VAE>,
    text_encoder_t5: Arc<dyn TextEncoder>,
    text_encoder_clip: Arc<dyn TextEncoder>,
    device: Device,
}

impl RealFluxPreprocessor {
    pub async fn preprocess_batch(
        &self,
        images: Vec<DynamicImage>,
        captions: Vec<String>,
    ) -> Result<PreprocessedFluxBatch> {
        // 1. Encode images with VAE (no dummy tensors!)
        let image_tensors = images_to_tensor(&images, &self.device)?;
        let latents = self.vae.encode(&image_tensors)?;
        
        // 2. Encode text with both encoders
        let (t5_embeds, t5_pooled) = self.text_encoder_t5.encode(&captions)?;
        let (clip_embeds, clip_pooled) = self.text_encoder_clip.encode(&captions)?;
        
        // 3. Combine embeddings as Flux expects
        let text_embeds = self.combine_text_embeddings(&t5_embeds, &clip_embeds)?;
        let pooled_embeds = self.combine_pooled_embeddings(&t5_pooled, &clip_pooled)?;
        
        Ok(PreprocessedFluxBatch {
            latents,
            text_embeds,
            pooled_embeds,
            original_sizes: images.iter().map(|img| (img.width(), img.height())).collect(),
            crop_coords: vec![(0, 0); images.len()],
        })
    }
}
```

### 3.2 Update Flux LoRA Trainer

**File**: `eridiffusion/crates/training/src/flux_lora_trainer_real.rs`

```rust
pub struct FluxLoRATrainerReal {
    model: FluxModelWithLoRA,
    preprocessor: RealFluxPreprocessor,
    optimizer: OptimizerWrapper,
    config: FluxLoRATraining24GBConfig,
    device: Device,
}

impl FluxLoRATrainerReal {
    pub fn new(
        model_path: &Path,
        vae_path: &Path,
        text_encoder_paths: (&Path, &Path),
        config: FluxLoRATraining24GBConfig,
    ) -> Result<Self> {
        // Load models
        let device = Device::new_cuda(config.device_id)?;
        
        // Load base Flux model and wrap with LoRA
        let base_model = load_flux_model(model_path, &device)?;
        let lora_config = LoRAConfig {
            rank: config.lora_rank,
            alpha: config.lora_alpha,
            target_modules: vec!["to_q", "to_k", "to_v", "to_out.0"],
            ..Default::default()
        };
        let model = FluxModelWithLoRA::from_base(base_model, lora_config)?;
        
        // Create real preprocessor
        let vae = load_vae(vae_path, &device)?;
        let (t5_encoder, clip_encoder) = load_text_encoders(text_encoder_paths, &device)?;
        let preprocessor = RealFluxPreprocessor::new(vae, t5_encoder, clip_encoder);
        
        // Create optimizer for LoRA parameters only
        let lora_params = model.trainable_parameters();
        let optimizer = create_optimizer(&config, lora_params)?;
        
        Ok(Self {
            model,
            preprocessor,
            optimizer,
            config,
            device,
        })
    }
    
    pub fn forward_training_step(
        &mut self,
        batch: &PreprocessedFluxBatch,
    ) -> Result<Tensor> {
        // 1. Sample timesteps
        let batch_size = batch.latents.dims()[0];
        let timesteps = self.sample_timesteps(batch_size)?;
        
        // 2. Add noise (flow matching)
        let noise = Tensor::randn_like(&batch.latents)?;
        let noisy_latents = self.add_flow_noise(&batch.latents, &noise, &timesteps)?;
        
        // 3. Create model inputs
        let inputs = ModelInputs {
            latents: noisy_latents,
            timestep: timesteps.clone(),
            encoder_hidden_states: Some(batch.text_embeds.clone()),
            pooled_projections: Some(batch.pooled_embeds.clone()),
            ..Default::default()
        };
        
        // 4. Forward pass through model (with LoRA!)
        let model_output = self.model.forward(&inputs)?;
        
        // 5. Compute flow matching loss
        let velocity_target = self.compute_velocity_target(&batch.latents, &noise, &timesteps)?;
        let loss = (model_output.sample - velocity_target).sqr()?.mean_all()?;
        
        // 6. Backward pass
        self.optimizer.zero_grad()?;
        loss.backward()?;
        
        // 7. Optimizer step (updates LoRA weights only)
        self.optimizer.step()?;
        
        Ok(loss)
    }
}
```

### 3.3 Memory Optimization

**File**: `eridiffusion/crates/training/src/memory_optimizer.rs`

```rust
pub struct FluxMemoryOptimizer {
    gradient_checkpointing_interval: Option<usize>,
    vae_offload: bool,
    activation_offload: bool,
}

impl FluxMemoryOptimizer {
    pub fn configure_model(&self, model: &mut FluxModelWithLoRA) {
        if let Some(interval) = self.gradient_checkpointing_interval {
            // Enable gradient checkpointing every N blocks
            for (i, block) in model.double_blocks.iter_mut().enumerate() {
                if i % interval == 0 {
                    block.enable_gradient_checkpointing();
                }
            }
        }
    }
    
    pub fn offload_vae(&self, vae: &mut dyn VAE) {
        if self.vae_offload {
            vae.to_device(&Device::Cpu).ok();
        }
    }
}
```

---

## Phase 4: Validation and Testing (Week 4)

### Objective
Ensure the implementation works correctly and matches expected behavior.

### 4.1 Gradient Flow Test

**File**: `eridiffusion/tests/flux_lora_gradient_test.rs`

```rust
#[test]
fn test_flux_lora_gradient_flow() {
    // Create model with LoRA
    let model = create_test_flux_model_with_lora();
    
    // Get LoRA parameters
    let lora_params = model.trainable_parameters();
    let initial_weights: Vec<_> = lora_params.iter()
        .map(|p| p.as_tensor().mean_all().unwrap().to_scalar::<f32>().unwrap())
        .collect();
    
    // Forward pass
    let inputs = create_test_inputs();
    let output = model.forward(&inputs).unwrap();
    
    // Compute loss
    let target = Tensor::randn_like(&output.sample).unwrap();
    let loss = (output.sample - target).sqr().unwrap().mean_all().unwrap();
    
    // Backward pass
    loss.backward().unwrap();
    
    // Check gradients exist and are non-zero
    for (i, param) in lora_params.iter().enumerate() {
        let grad = param.grad().unwrap();
        let grad_norm = grad.sqr().unwrap().sum_all().unwrap().sqrt().unwrap();
        let grad_norm_scalar = grad_norm.to_scalar::<f32>().unwrap();
        
        assert!(grad_norm_scalar > 0.0, "LoRA parameter {} has zero gradient", i);
    }
    
    // Simulate optimizer step
    for param in lora_params {
        let grad = param.grad().unwrap();
        let new_value = param.as_tensor() - grad.affine(0.01, 0.0).unwrap();
        param.set(&new_value).unwrap();
    }
    
    // Verify weights changed
    let updated_weights: Vec<_> = lora_params.iter()
        .map(|p| p.as_tensor().mean_all().unwrap().to_scalar::<f32>().unwrap())
        .collect();
    
    for (i, (initial, updated)) in initial_weights.iter().zip(updated_weights.iter()).enumerate() {
        assert!((initial - updated).abs() > 1e-6, "LoRA parameter {} did not update", i);
    }
}
```

### 4.2 End-to-End Training Test

**File**: `eridiffusion/tests/flux_lora_training_test.rs`

```rust
#[test]
fn test_flux_lora_training_convergence() {
    // Create trainer
    let config = create_test_config();
    let trainer = FluxLoRATrainerReal::new(
        &test_model_path(),
        &test_vae_path(),
        (&test_t5_path(), &test_clip_path()),
        config,
    ).unwrap();
    
    // Create small dataset
    let dataset = create_test_dataset(10); // 10 samples
    
    // Track loss
    let mut losses = Vec::new();
    
    // Train for a few steps
    for epoch in 0..5 {
        for batch in dataset.iter_batches() {
            let loss = trainer.forward_training_step(&batch).unwrap();
            losses.push(loss.to_scalar::<f32>().unwrap());
        }
    }
    
    // Verify loss decreases
    let initial_loss = losses[0];
    let final_loss = losses[losses.len() - 1];
    assert!(final_loss < initial_loss * 0.8, "Loss did not decrease sufficiently");
}
```

### 4.3 Inference with Trained LoRA

**File**: `eridiffusion/tests/flux_lora_inference_test.rs`

```rust
#[test]
fn test_flux_lora_inference() {
    // Load trained model
    let model = load_trained_flux_lora_model(&checkpoint_path()).unwrap();
    
    // Generate image
    let sampler = FluxSampler::new(model, vae, text_encoder);
    let prompt = "a beautiful sunset over mountains";
    let image = sampler.generate(prompt, 28, 3.5).unwrap();
    
    // Verify output
    assert_eq!(image.dims(), &[3, 1024, 1024]);
    
    // Compare with base model
    model.disable_lora();
    let base_image = sampler.generate(prompt, 28, 3.5).unwrap();
    
    // LoRA should produce different output
    let diff = (image - base_image).abs().unwrap().mean_all().unwrap();
    assert!(diff.to_scalar::<f32>().unwrap() > 0.01, "LoRA had no effect");
}
```

---

## Phase 5: Integration and Optimization (Week 5)

### Objective
Integrate with existing infrastructure and optimize for 24GB VRAM.

### 5.1 Integration with Existing Training Loop

**File**: `eridiffusion/crates/training/src/trainers/flux_trainer_integrated.rs`

```rust
impl Trainer for FluxLoRATrainerReal {
    fn train_step(&mut self, batch: DataBatch) -> Result<TrainingMetrics> {
        // Preprocess batch
        let preprocessed = self.preprocessor.preprocess_batch(
            batch.images,
            batch.captions,
        ).await?;
        
        // Forward training step
        let loss = self.forward_training_step(&preprocessed)?;
        
        // Return metrics
        Ok(TrainingMetrics {
            loss: loss.to_scalar()?,
            learning_rate: self.optimizer.current_lr(),
            grad_norm: self.compute_grad_norm()?,
            ..Default::default()
        })
    }
    
    fn save_checkpoint(&self, path: &Path) -> Result<()> {
        // Save only LoRA weights
        let lora_state_dict = self.model.get_lora_state_dict()?;
        safetensors::save(lora_state_dict, path)?;
        Ok(())
    }
}
```

### 5.2 Memory Profiling and Optimization

**File**: `eridiffusion/crates/training/src/memory_profiler.rs`

```rust
pub fn profile_flux_lora_memory() -> MemoryReport {
    // Profile memory usage at each stage
    let mut report = MemoryReport::new();
    
    // Model loading
    let model_memory = measure_memory(|| {
        load_flux_model_with_lora()
    });
    report.add("model_loading", model_memory);
    
    // Forward pass
    let forward_memory = measure_memory(|| {
        model.forward(&test_inputs)
    });
    report.add("forward_pass", forward_memory);
    
    // Backward pass
    let backward_memory = measure_memory(|| {
        loss.backward()
    });
    report.add("backward_pass", backward_memory);
    
    report
}
```

---

## Configuration Examples

### Minimal LoRA (for testing)
```yaml
lora_rank: 4
lora_alpha: 4
target_modules: ["to_q", "to_v"]  # Minimal targets
gradient_checkpointing: true
batch_size: 1
```

### Standard LoRA (recommended)
```yaml
lora_rank: 32
lora_alpha: 32
target_modules: ["to_q", "to_k", "to_v", "to_out.0"]
gradient_checkpointing: true
gradient_checkpointing_interval: 2
batch_size: 1
gradient_accumulation_steps: 4
```

### Advanced LoRA (all targets)
```yaml
lora_rank: 64
lora_alpha: 64
target_modules: ["to_q", "to_k", "to_v", "to_out.0", "ff.0", "ff.2"]
gradient_checkpointing: true
gradient_checkpointing_interval: 1
batch_size: 1
gradient_accumulation_steps: 8
mixed_precision: "bf16"
```

---

## Success Criteria

1. **LoRA weights update during training** ✓
   - Verify via gradient flow tests
   - Check weight changes after optimizer steps

2. **Loss decreases properly** ✓
   - Monitor training loss curve
   - Compare with SimpleTuner baseline

3. **Can generate images with trained LoRA** ✓
   - Load checkpoint and generate
   - Verify style/concept learned

4. **Memory usage under 24GB** ✓
   - Profile peak memory usage
   - Implement optimizations as needed

5. **No dummy/fake implementations** ✓
   - All forward passes use real computation
   - All data is properly preprocessed

---

## Risk Mitigation

1. **Gradient Vanishing/Exploding**
   - Initialize LoRA B matrix to zeros
   - Use proper scaling (alpha/rank)
   - Clip gradients if needed

2. **Memory Overflow**
   - Start with rank=4 for testing
   - Enable aggressive checkpointing
   - Profile each component

3. **Compatibility Issues**
   - Test weight loading from original Flux
   - Verify output matches base model (when LoRA disabled)
   - Cross-validate with SimpleTuner outputs

---

## Timeline

- **Week 1**: Foundation modules (LinearWithLoRA, AttentionWithLoRA)
- **Week 2**: Flux model architecture with LoRA
- **Week 3**: Training pipeline integration
- **Week 4**: Validation and testing
- **Week 5**: Integration and optimization

Total: 5 weeks to fully functional Flux LoRA training

---

## Next Steps

1. Start with Phase 1 - implement LinearWithLoRA
2. Write comprehensive tests for each component
3. Verify gradient flow before moving to next phase
4. Profile memory usage at each stage
5. Document any Candle limitations encountered

This plan ensures we build a working Flux LoRA training system with no fake code, proper gradient flow, and optimized memory usage for 24GB GPUs.