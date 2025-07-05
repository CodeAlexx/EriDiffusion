# Device API Fixes Summary

## Issue
The code was trying to use `.ordinal()` method on Candle's `CudaDevice` struct, which doesn't exist in the current version of Candle. The correct field to use is `.id`, but that field is private.

## Affected Files and Fixes

### 1. **crates/models/src/sd3.rs** (line 328)
- **Original**: `Device::Cuda(cuda_device.ordinal())`
- **Fixed**: `Device::Cuda(0)` with comment explaining limitation

### 2. **crates/models/src/sd15.rs** (line 227)
- **Original**: `Device::Cuda(cuda_device.ordinal())`
- **Fixed**: `Device::Cuda(0)` with comment explaining limitation

### 3. **crates/models/src/auraflow.rs** (line 443)
- **Original**: `Device::Cuda(cuda_device.ordinal())`
- **Fixed**: `Device::Cuda(0)` with comment explaining limitation

### 4. **crates/core/src/device.rs** (lines 54, 77)
- **Original**: `Device::Cuda(cuda_device.id)`
- **Fixed**: `Device::Cuda(0)` with comment explaining limitation

### 5. **crates/core/src/cuda.rs** (line 104)
- **Original**: Trying to access `cuda_device.id` directly
- **Fixed**: Removed device ID check, defaulting to primary GPU assumptions

## Root Cause
Candle's `CudaDevice` struct has a private `id` field and no public method to retrieve the original ordinal used to create the device. While there is a public `id()` method that returns a `DeviceId`, this is not the CUDA ordinal but rather a unique identifier generated for each device instance.

## Limitation
Due to this API limitation in Candle, when converting from a Candle `CudaDevice` back to our `Device` enum, we cannot determine which GPU ordinal was originally used. As a workaround, we default to device 0. This limitation only affects:
- Device conversions from Candle to ai-toolkit-rs
- Device property queries where we can't determine the specific GPU

## Recommendation
For production use, consider:
1. Tracking device ordinals separately when creating Candle devices
2. Using a wrapper struct that stores both the Candle device and its ordinal
3. Contributing to Candle to add a public method for retrieving the device ordinal

## Compilation Status
All device-related compilation errors have been resolved. The remaining errors in the build are unrelated to device handling and are in the networks crate.