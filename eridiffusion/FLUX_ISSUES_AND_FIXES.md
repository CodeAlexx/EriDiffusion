# Flux Training Issues and Fixes

## Identified Issues

### 1. Dataloader Issues

#### Problem: Empty/NaN Latents
The preprocessor doesn't validate encoded latents, which can lead to NaN values propagating through training.

**Fix needed in `flux_preprocessor.rs`:**
```rust
// After encoding latents, add validation:
let latent = vae.encode(&image)?;
// Check for NaN or Inf
if latent.to_vec1::<f32>()?.iter().any(|x| x.is_nan() || x.is_infinite()) {
    return Err(Error::InvalidData("NaN or Inf detected in encoded latents"));
}
```

#### Problem: Incorrect Latent Scaling
Flux uses a different scaling factor (0.3611) compared to other models, but this isn't consistently applied.

**Fix needed:**
```rust
// In VAEPreprocessor for Flux
let scaled = latent.affine(0.3611 as f64, 0.0)?; // Flux-specific scaling
```

### 2. Training Loop Issues

#### Problem: Incorrect Noise Scheduling
The current implementation uses simple linear interpolation for flow matching, but Flux requires shifted timestep sampling.

**Fix needed in `training_step`:**
```rust
// Replace simple uniform sampling with shifted sampling
let timesteps: Vec<f32> = (0..batch_size)
    .map(|_| {
        let u = rng.gen_range(0.0..1.0);
        // Apply shift function for Flux
        let shift = 1.0;
        let t = (shift * u) / (1.0 + (shift - 1.0) * u);
        t
    })
    .collect();
```

#### Problem: Incorrect Patchification
The current reshape operation for patchifying is incorrect:
```rust
// Current (wrong):
let img = noisy_latents.reshape((batch_size, c, h / 2, 2, w / 2, 2))?

// Should be:
let img = noisy_latents
    .reshape((batch_size, c, h, w))?
    .unfold(2, 2, 2)?  // unfold height dimension
    .unfold(3, 2, 2)?  // unfold width dimension
    .permute((0, 2, 3, 1, 4, 5))? // [b, h/2, w/2, c, 2, 2]
    .reshape((batch_size, seq_len, 4 * c))?; // [b, seq, c*4]
```

#### Problem: Missing Gradient Clipping
NaN losses often occur due to gradient explosion.

**Fix needed:**
```rust
// After loss computation, before optimizer step:
if self.config.max_grad_norm > 0.0 {
    let grad_norm = clip_grad_norm_(&self.lora_adapter.parameters(), self.config.max_grad_norm)?;
    if grad_norm.is_nan() || grad_norm.is_infinite() {
        warn!("Gradient norm is NaN/Inf, skipping step");
        return Ok(0.0);
    }
}
```

### 3. Device Issues

#### Problem: Device Shows as GPU 1 Instead of 0
This is likely a display issue in Candle's device enumeration.

**Fix:**
```rust
// Force explicit device creation:
let device = candle_core::Device::cuda_if_available(0)
    .unwrap_or(candle_core::Device::Cpu);
```

### 4. Checkpoint Saving Issues

#### Problem: Empty 16-byte Files
The VarMap isn't properly saving LoRA weights.

**Fix needed in `save_checkpoint`:**
```rust
// Ensure tensors are materialized before saving
for (name, var) in self.lora_adapter.var_map.data().lock().unwrap().iter() {
    // Force computation if lazy
    let _ = var.as_tensor().to_vec1::<f32>()?;
}
// Then save
self.lora_adapter.var_map.save(&lora_path)?;
```

### 5. Memory Optimization

#### Problem: OOM with 24GB
The current implementation loads all models simultaneously.

**Fix suggestions:**
1. Use gradient checkpointing more aggressively
2. Clear intermediate tensors:
```rust
// After using tensors that won't be needed for backward:
drop(noisy_latents);
drop(noise);
```
3. Use mixed precision more effectively:
```rust
// Ensure all intermediate computations use BF16
let noise = Tensor::randn_like(&latents, 0.0, 1.0)?
    .to_dtype(DType::BF16)?;
```

## Implementation Priority

1. **Critical**: Fix NaN detection and gradient clipping
2. **High**: Fix patchification and noise scheduling  
3. **Medium**: Fix checkpoint saving
4. **Low**: Device display issue

## Testing Strategy

1. Start with small test (10 steps) to verify no NaN
2. Test checkpoint saving after 10 steps
3. Run 100 steps to check stability
4. Full 2000 step run only after above pass