# Device Debugging Summary for Flux LoRA Training

## Problem
- Multiple CUDA devices appearing (DeviceId 2, 6, 7) when only one RTX 3090 exists
- CUDA kernel error: "named symbol not found" due to device mismatch
- Tensors being created on different devices despite forcing device 0

## Debugging Added

### 1. Main Entry Point (`src/main.rs`)
- Sets `CUDA_VISIBLE_DEVICES=0` programmatically
- Prints environment variable status
- **NEW**: Enumerates all devices that Candle thinks exist
```rust
// Check what Candle thinks about devices
for i in 0..10 {
    match candle_core::Device::new_cuda(i) {
        Ok(device) => println!("Candle thinks device {} exists: {:?}", i, device),
        Err(e) => {
            println!("Device {} error: {}", i, e);
            break;
        }
    }
}
```

### 2. Training Entry (`train_flux_lora`)
- Initializes device fix module
- Prints CUDA environment details
- Shows device count estimate
- Forces cuda::set_device(0)
- Forces all device creation to use cuda:0

### 3. Cached Data Loading (`load_cached_data`)
- Prints cuda::current_device before/after operations
- Uses `device_fix::load_safetensors_forced` to ensure correct device
- Shows device details for each loaded tensor
- Forces tensors to cuda:0 if loaded on wrong device

### 4. Training Step (`train_step_cached`)
- Enhanced latent retrieval debugging with device details
- Forces latents to cuda:0 before stacking
- Detailed noise generation debugging
- Explicit device forcing for noise and timesteps
- Shows CUDA device details for all tensors

### 5. Device Fix Module (`src/trainers/device_fix.rs`)
- Global device singleton to ensure consistency
- `init_global_device()` - Initialize once at startup
- `get_device()` - Always returns cuda:0
- `ensure_device()` - Moves tensors to correct device
- `load_safetensors_forced()` - Loads with forced device

### 6. Memory Module Updates
- Added `device_count()` function to check visible devices
- Enhanced debugging in memory pool operations

## What to Look For in Output

1. **Device Detection Phase**:
   - How many devices does Candle think exist?
   - Are there errors after device 0?

2. **Cache Loading Phase**:
   - Are tensors being loaded on device 0?
   - Any warnings about moving tensors?

3. **Training Step Phase**:
   - Device IDs for latents, noise, timesteps
   - Any device movement warnings

## Hypothesis
The issue is likely that:
1. Candle is detecting phantom devices despite CUDA_VISIBLE_DEVICES=0
2. The safetensors loading might be ignoring the device parameter
3. Random tensor generation (randn/rand) might be creating on wrong devices

## Next Steps After Running
1. If multiple devices are detected in enumeration, we need to investigate Candle's device detection
2. If tensors are still on wrong devices after forced loading, we need a different approach
3. Consider patching Candle itself or using a wrapper that intercepts all device creation

## Run Command
```bash
cd /home/alex/diffusers-rs/eridiffusion
./trainer config/train.yaml 2>&1 | tee training_debug.log
```

Then search the log for:
- "Candle thinks device"
- "CUDA device details"
- "WARNING:"
- "Moving"
- "DeviceId"