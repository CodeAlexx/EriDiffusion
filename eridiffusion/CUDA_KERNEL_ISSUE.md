# CUDA Kernel Issue Analysis

## Problem
"named symbol not found" error occurs during Flux forward pass despite all tensors being on DeviceId(1).

## Root Cause
Custom CUDA kernels in `src/kernels/*.cu` are compiled statically and linked with the binary. These kernels:
- `group_norm_forward_f32`
- `rope_forward_f32` 
- `rms_norm_f32`

Were compiled with one CUDA context but are being called with tensors from a different context (even though both show DeviceId(1)).

## Solutions

### 1. Quick Fix - Disable CUDA feature
```bash
cargo build --release --bin trainer --no-default-features
```

### 2. Rebuild CUDA kernels
```bash
rm -rf target/
cargo clean
cargo build --release --bin trainer
```

### 3. Use environment variable to force single context
```bash
export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID
```

## Investigation Steps
1. The error happens right at forward pass
2. Custom CUDA kernels are defined in src/kernels/mod.rs
3. These are FFI calls to compiled CUDA code
4. The DeviceId mismatch causes CUDA to not find the compiled symbols