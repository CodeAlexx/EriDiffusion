# Candle Framework Bugs and Workarounds

This document details specific bugs and limitations we encountered in Candle while implementing Flux LoRA training, along with our workarounds.

## 1. 3D @ 2D Matrix Multiplication Not Supported

### Bug Description
Candle's matmul operation doesn't support broadcasting for 3D @ 2D matrix multiplication, which is common in transformer architectures.

### Error
```
Error: ShapeMismatchBinaryOp { lhs: [2, 4096, 3072], rhs: [3072, 9216], op: "matmul" }
```

### Expected Behavior
In PyTorch/NumPy, this would broadcast correctly:
- Input: [batch_size, seq_len, dim] @ [dim, out_dim]
- Output: [batch_size, seq_len, out_dim]

### Workaround
```rust
// In eridiffusion/src/models/flux_custom/lora.rs

pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
    let x_shape = x.dims();
    let x_ndims = x_shape.len();
    
    // Check if we need to reshape 3D to 2D
    let (reshaped_x, original_shape) = if x_ndims == 3 {
        let batch_size = x_shape[0];
        let seq_len = x_shape[1];
        let in_features = x_shape[2];
        
        // Reshape to 2D: [batch * seq_len, in_features]
        let reshaped = x.reshape((batch_size * seq_len, in_features))?;
        (reshaped, Some((batch_size, seq_len)))
    } else {
        (x.clone(), None)
    };
    
    // Now do 2D matmul
    let out = self.base.forward(&reshaped_x)?;
    
    // Add LoRA
    let lora_out = reshaped_x.matmul(&self.lora_a)?
        .matmul(&self.lora_b)?
        .broadcast_mul(&self.scale)?;
    
    let out = (out + lora_out)?;
    
    // Reshape back to 3D if needed
    if let Some((batch_size, seq_len)) = original_shape {
        let out_features = out.dim(D::Minus1)?;
        Ok(out.reshape((batch_size, seq_len, out_features))?)
    } else {
        Ok(out)
    }
}
```

### Impact
- Every linear layer in Flux needs this workaround
- Slight performance overhead from reshape operations
- Makes code more complex

## 2. Empty VarMap Causes CUDA Context Errors

### Bug Description
Creating a model with an empty VarMap can cause mysterious CUDA_ERROR_NOT_FOUND errors, even though the error occurs much later during tensor operations.

### Error
```
Error: Cuda(Cuda(DriverError(CUDA_ERROR_NOT_FOUND, "named symbol not found")))
```

### Root Cause
When VarBuilder is created from an empty VarMap, some internal CUDA context initialization might be skipped or improperly set up.

### Workaround
```rust
// In eridiffusion/src/trainers/flux_init_weights.rs

pub fn initialize_flux_weights_minimal(vb: &VarBuilder, config: &FluxConfig) -> Result<()> {
    let init = Init::Const(0.0);
    
    // Initialize at least one tensor to ensure proper CUDA context
    vb.get_with_hints(
        &[config.hidden_size, 256], 
        "time_in.0.weight", 
        init
    )?;
    
    // Initialize more minimal weights as needed
    vb.get_with_hints(&[256], "time_in.0.bias", init)?;
    vb.get_with_hints(
        &[config.hidden_size, config.vec_in_dim], 
        "vector_in.0.weight", 
        init
    )?;
    
    Ok(())
}
```

### Impact
- Must always initialize some weights, even for LoRA-only training
- Adds unnecessary memory usage (though minimal)
- Complicates the "pure LoRA" approach

## 3. Device Consistency Issues

### Bug Description
Candle doesn't always maintain device consistency when creating new tensors, especially with operations like `randn` or `zeros`.

### Symptoms
- Tensors unexpectedly created on wrong device
- Cross-device operation errors
- Inconsistent device assignment

### Workaround
```rust
// In eridiffusion/src/trainers/cached_device.rs

use std::sync::Mutex;
use candle_core::Device;

static CACHED_DEVICE: Mutex<Option<Device>> = Mutex::new(None);

pub fn get_single_device() -> Result<Device> {
    let mut cache = CACHED_DEVICE.lock().unwrap();
    
    if let Some(device) = cache.as_ref() {
        return Ok(device.clone());
    }
    
    // Force device 0
    cuda::set_device(0)?;
    let device = Device::cuda_if_available(0)?;
    *cache = Some(device.clone());
    
    Ok(device)
}

// Usage throughout code:
let device = get_single_device()?;
let tensor = Tensor::randn(..., &device)?;
```

### Impact
- Must use cached device pattern everywhere
- Can't rely on default device selection
- Requires explicit device management

## 4. Tensor Creation on Wrong Device

### Bug Description
Some tensor creation functions ignore the device parameter or create on CPU by default.

### Example
```rust
// This might create on CPU even with CUDA device
let noise = Tensor::randn(0.0, 1.0, shape, &cuda_device)?;
```

### Workaround
```rust
// Create on CPU first, then move
let noise_cpu = Tensor::randn(0.0, 1.0, shape, &Device::Cpu)?;
let noise = noise_cpu.to_device(&cuda_device)?;
```

### Impact
- Extra memory copies
- Performance overhead
- More verbose code

## 5. Limited Optimizer Support

### Bug Description
Candle's optimizers don't support mixed precision or CPU offloading out of the box.

### Workaround
```rust
// In eridiffusion/src/trainers/optimizer_cpu_offload.rs

pub struct CPUOffloadedOptimizer {
    params: Vec<Var>,
    moment1: Vec<Tensor>,  // On CPU
    moment2: Vec<Tensor>,  // On CPU
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
    step: usize,
}

impl CPUOffloadedOptimizer {
    pub fn step(&mut self, grads: &Gradients) -> Result<()> {
        for (idx, param) in self.params.iter().enumerate() {
            if let Some(grad) = grads.get(param) {
                // Move gradient to CPU
                let grad_cpu = grad.to_device(&Device::Cpu)?;
                
                // Update moments on CPU
                let m1 = &mut self.moment1[idx];
                *m1 = ((m1 * self.beta1)? + 
                      (grad_cpu * (1.0 - self.beta1))?)?;
                
                // ... Adam update logic ...
                
                // Move update back to GPU
                let update_gpu = update.to_device(param.device())?;
                param.set(&(param.as_tensor() - update_gpu)?)?;
            }
        }
        Ok(())
    }
}
```

## 6. ModuleT Trait Limitations

### Bug Description
Candle's ModuleT trait expects modules to own their parameters, making it difficult to implement features like LoRA that modify existing modules.

### Workaround
Instead of implementing ModuleT, create custom forward methods:
```rust
pub struct FluxBlockWithLoRA {
    // Don't use ModuleT, just store what we need
    base_block: FluxBlock,
    lora_adapters: HashMap<String, LoRALayer>,
}

impl FluxBlockWithLoRA {
    pub fn forward(&self, x: &Tensor, ...) -> Result<Tensor> {
        // Custom forward implementation
    }
}
```

## 7. No Native Flash Attention

### Bug Description
Candle doesn't have built-in Flash Attention support, limiting memory efficiency for long sequences.

### Current Status
- Using standard attention (O(n²) memory)
- Limits sequence length on 24GB GPUs
- No easy workaround without custom CUDA kernels

## Recommendations for Candle

1. **Add 3D @ 2D matmul support** - Critical for transformer models
2. **Fix device consistency** - Ensure all operations respect device parameter
3. **Improve VarMap initialization** - Don't require dummy tensors
4. **Add Flash Attention** - Essential for modern transformers
5. **Better error messages** - CUDA_ERROR_NOT_FOUND is not helpful
6. **Mixed precision support** - Native FP16/BF16 operations
7. **CPU offloading utilities** - For large model training

## Summary

Despite these limitations, we successfully implemented Flux LoRA training by:
- Working around every limitation with custom code
- Implementing our own device management
- Creating reshape wrappers for linear layers
- Building CPU-offloaded optimizers
- Carefully managing memory with LoRA-only loading

The resulting implementation works but is more complex than it needs to be. Many of these workarounds wouldn't be necessary with a more mature framework.