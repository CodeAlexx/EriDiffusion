# Candle Memory Management Implementation Guide

## Implementation Details for Memory-Efficient Training

### 1. Core Memory Manager Implementation

We created a `MemoryManager` struct that provides PyTorch-like memory management functions for Candle:

```rust
// src/memory/manager.rs
pub struct MemoryManager;

impl MemoryManager {
    /// Forces CUDA to release cached memory (torch.cuda.empty_cache() equivalent)
    pub fn empty_cache() -> Result<()> {
        if let Ok(device) = Device::cuda_if_available(0) {
            device.synchronize()?;
        }
        crate::memory::cuda::empty_cache()?;
        Ok(())
    }
    
    /// Comprehensive cleanup
    pub fn cleanup() -> Result<()> {
        Self::empty_cache()?;
        Self::ipc_collect()?;
        Ok(())
    }
    
    /// Memory usage logging
    pub fn log_memory_usage(prefix: &str) -> Result<()> {
        if let Ok((free, total)) = Self::memory_stats(0) {
            let used = total - free;
            let used_gb = used as f32 / 1024.0 / 1024.0 / 1024.0;
            let total_gb = total as f32 / 1024.0 / 1024.0 / 1024.0;
            log::info!("{}: {:.2} GB / {:.2} GB", prefix, used_gb, total_gb);
        }
        Ok(())
    }
}
```

### 2. Phase-Based Model Structure

The key innovation is using `Option<Model>` for all components:

```rust
pub struct FluxLoRATrainer {
    // Core models - loaded only when needed
    model: Option<FluxModelWithLoRA>,
    vae: Option<AutoencoderKL>,
    vae_cpu: Option<AutoencoderKL>,  // For CPU offloading
    text_encoders: Option<TextEncoders>,
    optimizer: Option<AdamW>,
    
    // Persistent caches
    latent_cache: HashMap<usize, Tensor>,
    text_embed_cache: HashMap<usize, (Tensor, Tensor)>,
    cache_dir: PathBuf,
    
    // Memory management
    memory_pool: Arc<RwLock<MemoryPool>>,
    block_swap_manager: Option<BlockSwapManager>,
}
```

### 3. Three-Phase Training Implementation

#### Phase 1: Preprocessing
```rust
fn preprocess_and_cache(&mut self) -> Result<()> {
    // Step 1: Image encoding
    println!("--- Step 1: Encoding Images with VAE ---");
    self.load_vae_to_gpu()?;
    
    for batch in data_loader {
        let latents = self.vae.as_ref().unwrap().encode(&images)?;
        self.latent_cache.insert(idx, latents);
    }
    
    self.unload_vae_from_gpu()?;
    MemoryManager::cleanup()?;
    MemoryManager::log_memory_usage("After VAE cleanup")?;
    
    // Step 2: Text encoding
    println!("--- Step 2: Encoding Text ---");
    self.load_text_encoders()?;
    
    for (idx, caption) in captions.iter().enumerate() {
        let (context, pooled) = self.text_encoders.encode(caption)?;
        self.text_embed_cache.insert(idx, (context, pooled));
    }
    
    self.unload_text_encoders()?;
    MemoryManager::cleanup()?;
    MemoryManager::log_memory_usage("After text encoder cleanup")?;
    
    // Step 3: Save cache
    self.save_cache(&cache_dir)?;
    
    println!("=== Preprocessing Complete! ===");
    Ok(())
}
```

#### Phase 2: Model Loading
```rust
fn load_training_models(&mut self) -> Result<()> {
    println!("=== PHASE 2: Loading Flux Model ===");
    MemoryManager::log_memory_usage("Before loading Flux")?;
    
    // Only load what we need for training
    self.load_flux_model()?;
    self.create_lora_layers()?;
    self.setup_optimizer()?;
    
    MemoryManager::log_memory_usage("After loading training models")?;
    Ok(())
}
```

#### Phase 3: Training Loop
```rust
fn train(&mut self) -> Result<()> {
    println!("=== PHASE 3: Training ===");
    
    for epoch in 0..num_epochs {
        for batch_idx in 0..num_batches {
            // Load cached data
            let latents = self.load_cached_latents(batch_idx)?;
            let (text_embeds, pooled) = self.load_cached_text(batch_idx)?;
            
            // Forward pass (no VAE/text encoders needed!)
            let noise_pred = self.model.forward(&latents, &text_embeds)?;
            
            // Training step
            let loss = self.compute_loss(noise_pred, target)?;
            loss.backward()?;
            self.optimizer.step()?;
        }
    }
    Ok(())
}
```

### 4. Memory-Efficient Helper Functions

#### Loading with CPU staging:
```rust
fn load_vae_to_gpu(&mut self) -> Result<()> {
    // First load to CPU
    println!("Loading VAE to CPU for memory efficiency...");
    self.vae_cpu = Some(load_vae_to_device(&Device::Cpu)?);
    
    // Then move to GPU
    println!("Moving VAE to GPU...");
    let vae_gpu = self.vae_cpu.as_ref().unwrap().to_device(&Device::Cuda(0))?;
    self.vae = Some(vae_gpu);
    
    Ok(())
}
```

#### Unloading with cleanup:
```rust
fn unload_vae_from_gpu(&mut self) -> Result<()> {
    println!("Unloading VAE from GPU...");
    
    // Drop GPU reference
    self.vae = None;
    
    // Force synchronization
    if let Ok(device) = Device::cuda_if_available(0) {
        device.synchronize()?;
    }
    
    // Clean up memory
    cuda::empty_cache()?;
    
    Ok(())
}
```

### 5. Caching Implementation

#### Saving cache:
```rust
fn save_cache(&self, cache_dir: &Path) -> Result<()> {
    // Save latents
    for (idx, latent) in &self.latent_cache {
        let path = cache_dir.join(format!("latent_{}.safetensors", idx));
        safetensors::save(&HashMap::from([("latent", latent)]), &path)?;
    }
    
    // Save text embeddings
    for (idx, (context, pooled)) in &self.text_embed_cache {
        let path = cache_dir.join(format!("text_embed_{}.safetensors", idx));
        safetensors::save(&HashMap::from([
            ("context", context),
            ("pooled", pooled),
        ]), &path)?;
    }
    
    println!("Saved {} latents and {} text embeddings", 
             self.latent_cache.len(), self.text_embed_cache.len());
    Ok(())
}
```

#### Loading cache:
```rust
fn load_cache(&mut self, cache_dir: &Path) -> Result<()> {
    // Check for latent files
    for entry in fs::read_dir(cache_dir)? {
        let path = entry?.path();
        if path.file_name().unwrap().to_str().unwrap().starts_with("latent_") {
            let tensors = safetensors::load(&path, &self.device)?;
            let idx = extract_index_from_filename(&path)?;
            self.latent_cache.insert(idx, tensors["latent"].clone());
        }
    }
    
    println!("Loaded {} cached latents", self.latent_cache.len());
    Ok(())
}
```

### 6. Memory Tracking Macros

```rust
/// Track memory usage around operations
track_memory!("VAE encoding", {
    let latents = vae.encode(&images)?;
});

/// Run with automatic cleanup
with_memory_cleanup!({
    // Do memory-intensive operation
    let result = process_batch()?;
    result
});
```

### 7. Configuration for Memory Management

```yaml
# Memory-optimized settings
train:
  gradient_checkpointing: true  # Essential for memory
  gradient_accumulation: 4      # Reduce memory per step
  batch_size: 1                 # Minimum batch size
  
memory:
  cache_latents_to_disk: true   # Don't keep in RAM
  offload_during_startup: true  # SimpleTuner-style
  empty_cache_after_phase: true # Force cleanup
  
# For 24GB cards
model:
  quantize: true                # Would need int8
  dtype: bf16                   # Better than fp32
```

### 8. Key Implementation Patterns

#### Pattern 1: Guard-based loading
```rust
struct ModelGuard<'a> {
    model: &'a mut Option<Model>,
}

impl<'a> Drop for ModelGuard<'a> {
    fn drop(&mut self) {
        *self.model = None;
        let _ = MemoryManager::cleanup();
    }
}
```

#### Pattern 2: Staged operations
```rust
fn process_in_stages<T, F>(&mut self, loader: F) -> Result<T>
where
    F: FnOnce(&mut Self) -> Result<T>,
{
    let result = loader(self)?;
    self.cleanup_stage()?;
    Ok(result)
}
```

### 9. Memory Requirements Tracking

```rust
impl FluxLoRATrainer {
    fn estimate_memory_requirements(&self) -> MemoryRequirements {
        MemoryRequirements {
            vae: 1_500_000_000,        // 1.5GB
            text_encoders: 6_000_000_000, // 6GB (CLIP + T5)
            flux_model: 12_000_000_000,   // 12GB
            optimizer: 2_000_000_000,     // 2GB
            gradients: 2_000_000_000,     // 2GB
            workspace: 1_000_000_000,     // 1GB
            total: 24_500_000_000,        // 24.5GB
        }
    }
}
```

### 10. Integration with Existing Code

The memory management can be retrofitted to existing trainers:

```rust
// Before (memory-hungry)
let vae = load_vae()?;
let text_encoder = load_text_encoder()?;
let model = load_model()?;
train(model, vae, text_encoder)?;

// After (memory-efficient)
let mut trainer = PhaseBasedTrainer::new();
trainer.preprocess_and_cache()?;  // Load, process, unload
trainer.load_training_models()?;   // Only load what's needed
trainer.train()?;                  // Use cached data
```

## Conclusion

This implementation demonstrates that Candle can achieve the same memory efficiency as PyTorch/SimpleTuner through:
1. Explicit phase-based loading with `Option<Model>`
2. Aggressive cleanup between phases
3. Comprehensive caching strategies
4. Memory tracking and logging

The patterns established here can be applied to any large model training scenario where memory is constrained.