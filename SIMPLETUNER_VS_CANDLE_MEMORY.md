# SimpleTuner vs Candle Memory Management

## How SimpleTuner Avoids OOM During Sampling

### SimpleTuner's Strategy (PyTorch)

1. **Meta Device "Parking"**
   ```python
   # Move to meta device (unload from GPU)
   vae.to('meta')
   text_encoder.to('meta')
   
   # Only load when needed
   vae.to('cuda')
   # Do sampling
   vae.to('meta')  # Unload again
   ```

2. **Pipeline Creation/Deletion**
   ```python
   # Create pipeline only for sampling
   pipeline = StableDiffusionPipeline(...)
   # Generate samples
   del pipeline
   pipeline = None
   torch.cuda.empty_cache()
   ```

3. **Memory Reclamation**
   ```python
   gc.collect()
   torch.cuda.empty_cache()
   torch.cuda.ipc_collect()  # Multi-GPU
   ```

4. **Configuration Options**
   - `keep_vae_loaded=False` - Don't keep VAE in memory
   - `vae_cache_ondemand=True` - Load only when needed

## Why Candle Can't Do The Same

### 1. **No Meta Device**
- PyTorch has 'meta' device that holds tensor metadata without data
- Candle doesn't have this concept - tensors are either on device or not

### 2. **No Manual Cache Management**
- PyTorch: `torch.cuda.empty_cache()` forces GPU memory release
- Candle: Memory is managed by CUDA driver automatically

### 3. **Different Memory Models**
- PyTorch: Reference counting + Python GC + manual cache control
- Candle: Rust ownership model + automatic CUDA management

### 4. **No Dynamic Model Loading**
- PyTorch: Can move models between devices dynamically
- Candle: Models are loaded once, device is fixed

## What We Could Do in Candle

### Option 1: Drop and Reload Models
```rust
// Before sampling
drop(self.mmdit);  // Free MMDiT from memory
self.mmdit = None;

// Load VAE for sampling
let vae = load_vae()?;
// Do sampling
drop(vae);

// Reload MMDiT for training
self.mmdit = Some(load_mmdit()?);
```

### Option 2: Two-Stage Process
1. Training stage: Only MMDiT + LoKr loaded
2. Sampling stage: Save checkpoint, restart with only VAE + sampling

### Option 3: CPU Offloading
```rust
// Move tensors to CPU during sampling
let mmdit_cpu = self.mmdit.to_device(&Device::Cpu)?;
drop(self.mmdit);

// Do sampling with freed GPU memory

// Move back to GPU
self.mmdit = mmdit_cpu.to_device(&Device::Cuda(0))?;
```

## The Fundamental Difference

**SimpleTuner**: "Load what you need, when you need it, then unload it"
**Candle**: "Load everything at start, keep it loaded"

This is why SimpleTuner can sample without OOM - it's essentially running two separate programs:
1. Training program (MMDiT + optimizer in memory)
2. Sampling program (VAE + text encoders in memory)

They never overlap in memory because of the meta device trick.