# Candle VAE CPU Offloading Strategy

## SimpleTuner's Approach

SimpleTuner uses PyTorch's "meta" device to effectively unload models from GPU memory:

```python
def unload_vae(self):
    if self.vae is not None:
        if hasattr(self.vae, "to"):
            self.vae.to("meta")  # Move to meta device (no actual memory allocation)
        self.vae = None
```

The "meta" device holds only tensor metadata (shape, dtype) without allocating actual memory for data.

## Candle Limitations

Candle doesn't have a "meta" device concept. Models are either:
- On a device (GPU/CPU) with memory allocated
- Dropped entirely (memory freed)

## Proposed Candle Implementation

### Option 1: CPU Offloading (SimpleTuner-style)

```rust
pub struct SD35LoKrTrainer {
    // ... other fields ...
    vae: Option<AutoEncoderKL>,
    vae_on_cpu: Option<AutoEncoderKL>,  // CPU-cached version
}

impl SD35LoKrTrainer {
    fn offload_vae_to_cpu(&mut self) -> Result<()> {
        if let Some(vae) = self.vae.take() {
            // Move VAE to CPU
            println!("Offloading VAE to CPU...");
            let vae_cpu = vae.to_device(&Device::Cpu)?;
            self.vae_on_cpu = Some(vae_cpu);
            self.vae = None;
            
            // Force CUDA to free memory
            thread::sleep(Duration::from_millis(500));
        }
        Ok(())
    }
    
    fn load_vae_from_cpu(&mut self) -> Result<()> {
        if let Some(vae_cpu) = self.vae_on_cpu.take() {
            println!("Loading VAE from CPU to GPU...");
            let vae_gpu = vae_cpu.to_device(&self.device)?;
            self.vae = Some(vae_gpu);
            self.vae_on_cpu = None;
        } else {
            // Load from disk if not in CPU cache
            self.load_vae()?;
        }
        Ok(())
    }
}
```

### Option 2: Full Model Swapping (Current Implementation)

Drop models entirely and reload from disk when needed:

```rust
// Before sampling
drop(self.mmdit.take());
self.mmdit = None;

// Load VAE for sampling
let vae = load_vae()?;

// After sampling
drop(vae);

// Reload MMDiT
self.load_mmdit()?;
```

### Option 3: Lazy VAE Loading (Recommended)

Keep VAE on CPU by default, only move to GPU when needed:

```rust
impl SD35LoKrTrainer {
    fn new(config: &Config, process: &ProcessConfig) -> Result<Self> {
        // ... initialization ...
        
        // Load VAE to CPU initially
        let vae_cpu = {
            let vae_config = AutoEncoderKLConfig { /* ... */ };
            let vae_vb = unsafe {
                VarBuilder::from_mmaped_safetensors(&[&model_path], DType::F16, &Device::Cpu)?
            };
            let vb_vae = vae_vb.rename_f(sd3_vae_rename).pp("first_stage_model");
            AutoEncoderKL::new(vb_vae, 3, 3, vae_config)?
        };
        
        Ok(Self {
            // ... other fields ...
            vae: None,  // Start with no GPU VAE
            vae_cpu: Some(vae_cpu),  // Keep CPU version
        })
    }
    
    fn get_vae(&mut self) -> Result<&AutoEncoderKL> {
        if self.vae.is_none() {
            // Move from CPU to GPU on demand
            if let Some(vae_cpu) = &self.vae_cpu {
                println!("Moving VAE from CPU to GPU...");
                let vae_gpu = vae_cpu.to_device(&self.device)?;
                self.vae = Some(vae_gpu);
            } else {
                return Err(anyhow::anyhow!("No VAE available"));
            }
        }
        Ok(self.vae.as_ref().unwrap())
    }
    
    fn unload_vae_from_gpu(&mut self) -> Result<()> {
        if self.vae.is_some() {
            println!("Unloading VAE from GPU (keeping CPU copy)...");
            self.vae = None;
            // CPU copy remains in self.vae_cpu
        }
        Ok(())
    }
}
```

## Memory Impact

### SimpleTuner Approach:
- VAE on GPU: ~800MB-1GB VRAM
- VAE on CPU: ~800MB-1GB RAM
- VAE on "meta": ~0 memory (metadata only)

### Candle Approach:
- VAE on GPU: ~800MB-1GB VRAM
- VAE on CPU: ~800MB-1GB RAM
- VAE dropped: 0 memory (but requires reload from disk)

## Recommended Implementation for EriDiffusion

For the SD 3.5 LoKr trainer, I recommend **Option 3** (Lazy VAE Loading):

1. **Initialize VAE on CPU** during trainer construction
2. **Move to GPU only when needed** (during sampling)
3. **Drop GPU copy after sampling** (keep CPU copy)
4. **Reuse CPU copy** for next sampling iteration

This approach:
- Minimizes GPU memory usage during training
- Avoids repeated disk I/O
- Provides fast VAE access when needed
- Similar to SimpleTuner's behavior but adapted for Candle

## Implementation Steps

1. Modify trainer initialization to load VAE to CPU
2. Add `vae_cpu` field to store CPU version
3. Implement `get_vae()` for lazy GPU loading
4. Update `generate_samples()` to use `get_vae()` and `unload_vae_from_gpu()`
5. Keep MMDiT swapping for additional memory savings

This hybrid approach combines the best of both worlds: SimpleTuner's CPU offloading pattern with Candle's constraints.