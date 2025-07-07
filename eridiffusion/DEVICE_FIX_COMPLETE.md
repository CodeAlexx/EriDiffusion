# Candle Device Fix Complete

## Problem Solved
- Candle creates a new DeviceId for every `Device::new_cuda()` call
- This caused "device mismatch" errors between tensors
- Fixed by implementing a single cached device pattern

## Solution Implemented
1. Created `cached_device.rs` module with `get_single_device()` function
2. All code now uses the same cached Device instance
3. This ensures all tensors have the same DeviceId

## Current Issue
- Cached latents from previous runs have old DeviceIds
- These need to be cleared and regenerated

## How to Run Training

### Option 1: Use the clear cache script
```bash
cd /home/alex/diffusers-rs/eridiffusion
./clear_cache_and_train.sh config/train.yaml
```

### Option 2: Manual steps
```bash
# Clear the cache
rm -rf /home/alex/diffusers-rs/datasets/40_woman/.latent_cache

# Run training
cd /home/alex/diffusers-rs/eridiffusion
CUDA_VISIBLE_DEVICES=0 ./target/release/trainer config/train.yaml
```

## What to Expect
1. First run will be slower as it regenerates the latent cache
2. All tensors will now have consistent DeviceId(1)
3. No more device mismatch errors
4. Training should proceed to forward pass and start optimizing

## Next Steps After Training Works
1. Integrate your masked attention code for performance
2. Optimize loading speed (currently slower than Python)
3. Benchmark training speed (it/s)
4. Report Candle bug upstream