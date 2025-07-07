# Candle Bug Report: Device ID Increments on Each Creation

## Bug Description
Every call to `Device::new_cuda(ordinal)` creates a new `DeviceId` with an incrementing counter, even when requesting the same GPU ordinal. This causes "device mismatch" errors when tensors created with different `Device::new_cuda(0)` calls are used together.

## Root Cause
In `candle-core/src/cuda_backend/device.rs`:
```rust
impl DeviceId {
    fn new() -> Self {
        static COUNTER: atomic::AtomicUsize = atomic::AtomicUsize::new(1);
        Self(COUNTER.fetch_add(1, atomic::Ordering::Relaxed))
    }
}
```

Each `CudaDevice::new()` creates a fresh DeviceId, regardless of the GPU ordinal.

## Impact
- Tensors created with different `Device::new_cuda(0)` calls are considered on "different devices"
- CUDA kernels fail with "named symbol not found" when operating on these tensors
- Makes it impossible to use Candle in a modular way where different components create their own devices

## Reproduction
```rust
let device1 = Device::new_cuda(0)?; // DeviceId(1)
let device2 = Device::new_cuda(0)?; // DeviceId(2) - different!

let tensor1 = Tensor::zeros(&[10], DType::F32, &device1)?;
let tensor2 = Tensor::zeros(&[10], DType::F32, &device2)?;

// This will fail with device mismatch!
let result = (&tensor1 + &tensor2)?;
```

## Workaround
Cache the device instance and reuse it throughout the application:
```rust
static CACHED_DEVICE: OnceLock<Device> = OnceLock::new();

pub fn get_device() -> Device {
    CACHED_DEVICE.get_or_init(|| Device::new_cuda(0).unwrap()).clone()
}
```

## Suggested Fix
Candle should cache devices internally per GPU ordinal, similar to PyTorch's behavior.