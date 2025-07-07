# Candle Device Bug Analysis

## Key Finding
The Candle device mismatch bug (`CUDA_ERROR_NOT_FOUND`) is NOT a fundamental limitation of Candle. Our test program shows that cross-device tensor operations work correctly in isolation.

## Test Results
```
Device 1: Cuda(CudaDevice(DeviceId(2)))
Device 2: Cuda(CudaDevice(DeviceId(3)))
✅ Cross-device matmul works
```

This proves that tensors created with different `Device::new_cuda(0)` calls CAN interact successfully.

## Root Cause
The issue appears during the Flux LoRA training forward pass, specifically when:
1. Cached latents/embeddings are loaded (created during preprocessing)
2. These tensors interact with model weights/parameters (created during model loading)
3. The error occurs during the forward pass

## Potential Causes
1. **Tensor Lifetime Issues**: Tensors created in one phase may have device references that become invalid
2. **VarMap/VarBuilder Issues**: The way Candle's VarBuilder creates parameters might be incompatible
3. **Memory Mapping**: When loading from SafeTensors files, device contexts might not transfer correctly
4. **Gradient Tracking**: The Var type used for trainable parameters might have stricter device requirements

## Evidence from Training
From the error location:
```
=== Using LoRA-Only Training for Flux ===
✅ LoRA-only model created successfully!
Starting forward pass preparation
ERROR in train_step_cached: DriverError(CUDA_ERROR_NOT_FOUND, "named symbol not found")
```

The error occurs AFTER successful model creation but DURING the forward pass when cached tensors interact with model parameters.

## Solution Approaches
1. **Immediate Fix**: Ensure ALL tensors (cached data, model weights, temporary tensors) use the exact same Device instance
2. **Debug Steps**:
   - Print device info for all tensors before forward pass
   - Check if VarBuilder creates tensors with different device contexts
   - Verify SafeTensors loading preserves device consistency

3. **Workaround**: Create all tensors on CPU first, then move to a single GPU device instance

## Code to Debug
Add this before the forward pass:
```rust
println!("Latent device: {:?}", latent.device());
println!("Model weight device example: {:?}", model.some_weight.device());
println!("Are devices equal? {}", 
    format!("{:?}", latent.device()) == format!("{:?}", model.some_weight.device()));
```

## Conclusion
This is not a fundamental Candle limitation but rather a specific issue with how tensors are created across different phases of the training pipeline. The fix involves ensuring true device consistency throughout the entire pipeline.