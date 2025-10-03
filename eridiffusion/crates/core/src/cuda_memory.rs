//! CUDA memory utilities (feature-gated)
//! These helpers are only compiled when `cuda-raw` feature is enabled.

#[allow(unused_imports)]
use crate::{Result, Error};

#[cfg(all(feature = "cuda-raw", feature = "alloc-debug"))]
pub unsafe fn cuda_memset_async(dev: usize, _ptr: *mut u8, _byte: u8, _size: usize) -> Result<()> {
    // NOTE: Cudarc driver API does not currently expose a direct memset
    // for arbitrary raw pointers not owned by a CudaSlice. Implementing a
    // robust memset would require either NVRTC to launch a small kernel or
    // wrapping the raw pointer in a DevicePtr and using low-level driver calls.
    // For now, compile-time stub that succeeds. Replace with a real memset
    // if/when raw-pointer ops are stabilized in your backend.
    let _ = dev; // silence unused in stub
    Ok(())
}

#[cfg(all(feature = "cuda-raw", feature = "alloc-debug"))]
pub unsafe fn cuda_fill_nan_bf16(dev: usize, _ptr: *mut u8, _elems: usize) -> Result<()> {
    // See note above; stub implementation.
    let _ = dev;
    Ok(())
}

#[cfg(all(feature = "cuda-raw"))]
pub unsafe fn cuda_free(_dev: usize, _ptr: *mut u8) -> Result<()> {
    // If you move to owner-aware allocations, track allocations and free here.
    // Stubbed OK; actual freeing would require tracking the CudaSlice or
    // wrapping the raw pointer for cudarc::driver to accept.
    Ok(())
}

// ===== NVRTC compile + launch (optional) =====
#[cfg(feature = "nvrtc-fill")]
mod nvrtc_fill {
    use super::*;
    use cudarc::driver::{CudaDevice, LaunchAsync};
    use cudarc::nvrtc::compile_ptx;
    use std::sync::OnceLock;

    const KERNEL_SRC: &str = r#"
    extern "C" __global__
    void fill_bytes(unsigned char* dst, unsigned char val, size_t n) {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) dst[i] = val;
    }
    
    extern "C" __global__
    void fill_bf16_nan(unsigned short* dst, size_t n) {
        // 0x7FC1 is a canonical bf16 NaN bit-pattern
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) dst[i] = 0x7FC1u;
    }
    "#;

    static INIT: OnceLock<()> = OnceLock::new();
    static mut DEV: Option<CudaDevice> = None;

    pub fn ensure_module(dev_idx: usize) -> Result<()> {
        INIT.get_or_init(|| {
            let dev = CudaDevice::new(dev_idx).expect("cuda device");
            let ptx = compile_ptx(KERNEL_SRC).expect("nvrtc compile");
            dev.load_ptx(ptx, "nvrtc_fill", &["fill_bytes", "fill_bf16_nan"]).expect("load ptx");
            unsafe { DEV = Some(dev); }
        });
        Ok(())
    }

    pub fn memset_async(dev_idx: usize, ptr: *mut u8, byte: u8, size: usize) -> Result<()> {
        ensure_module(dev_idx)?;
        unsafe {
            let dev = DEV.as_ref().unwrap();
            let func = dev.get_func("nvrtc_fill", "fill_bytes").map_err(|e| Error::Device(e.to_string()))?;
            let block = 256u32;
            let grid = ((size as u64 + block as u64 - 1) / block as u64) as u32;
            // Launch signature: (dst, val, n)
            unsafe { func.launch((grid,1,1), (block,1,1), 0, (&ptr, &byte, &size)).map_err(|e| Error::Device(e.to_string()))?; }
        }
        Ok(())
    }

    pub fn fill_bf16_nan_async(dev_idx: usize, ptr: *mut u8, elems: usize) -> Result<()> {
        ensure_module(dev_idx)?;
        unsafe {
            let dev = DEV.as_ref().unwrap();
            let func = dev.get_func("nvrtc_fill", "fill_bf16_nan").map_err(|e| Error::Device(e.to_string()))?;
            let block = 256u32;
            let grid = ((elems as u64 + block as u64 - 1) / block as u64) as u32;
            let ptr_u16 = ptr as *mut u16;
            unsafe { func.launch((grid,1,1), (block,1,1), 0, (&ptr_u16, &elems)).map_err(|e| Error::Device(e.to_string()))?; }
        }
        Ok(())
    }
}

#[cfg(feature = "nvrtc-fill")]
pub use nvrtc_fill::{memset_async as nvrtc_memset_async, fill_bf16_nan_async as nvrtc_fill_bf16_nan};
