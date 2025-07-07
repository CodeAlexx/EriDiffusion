# Candle Device ID Bug

## Issue
When `CUDA_VISIBLE_DEVICES=0` is set, Candle's `Device::new_cuda(0)` returns `DeviceId(3)` instead of `DeviceId(0)`.

This appears to be a bug where Candle is using physical device IDs instead of logical device IDs when CUDA_VISIBLE_DEVICES is set.

## Evidence
```
=== CANDLE DEVICE DETECTION ===
Candle thinks device 0 exists: Cuda(CudaDevice(DeviceId(3)))
Device 1 error: Cuda error cudaSetDevice(1)
```

## Why This Happens
Your RTX 3090 is physically device 3 in your system, but when you set CUDA_VISIBLE_DEVICES=0, it should become logical device 0. Candle seems to be returning the physical ID instead.

## Workaround
Since we know device 0 maps to DeviceId(3), we can:
1. Use device 3 directly
2. Or detect the mapping at runtime and use the correct device

## Long-term Fix
This needs to be reported to the Candle project as a bug.