#[cfg(all(feature = "cuda-raw", feature = "nvrtc-fill"))]
#[test]
fn nvrtc_fill_bytes_and_nan() {
    use eridiffusion_core::cuda;
    use eridiffusion_core::cuda_memory;

    unsafe {
        let dev = 0usize;
        let szb = 2048usize;
        // Allocate a zeroed device buffer (stubbed when cuda-raw is off)
        let p = cuda::cuda_malloc_zeroed(dev, szb).expect("cuda malloc zeroed");

        // Byte fill via NVRTC kernel
        cuda_memory::nvrtc_memset_async(dev, p, 0xAB, szb).expect("nvrtc fill bytes");

        // BF16 NaN fill (interprets ptr as u16 stream)
        cuda_memory::nvrtc_fill_bf16_nan(dev, p, szb / 2).expect("nvrtc fill bf16 nan");

        // Free (stubbed when using synthetic handles)
        #[cfg(feature = "cuda-raw")]
        {
            let _ = cuda_memory::cuda_free(dev, p as *mut u8);
        }
    }
}

