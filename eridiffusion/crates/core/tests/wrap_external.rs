#[cfg(all(feature = "cuda-raw", feature = "nvrtc-fill", feature = "external-wrap"))]
#[test]
fn wrap_external_memory_into_tensor() {
    use eridiffusion_core as core;
    use core::cuda;
    use core::cuda_memory;
    use flame_core::{Shape, DType, CudaDevice, Tensor};

    unsafe {
        let dev_idx = 0usize;
        let elems = 256usize;
        let bytes = elems * std::mem::size_of::<f32>();

        let ptr = cuda::cuda_malloc_zeroed(dev_idx, bytes).expect("cuda malloc zeroed");
        // Fill with a pattern (noop if already zeroed)
        let _ = cuda_memory::nvrtc_memset_async(dev_idx, ptr, 0x00, bytes);

        let shape = Shape::from_dims(&[elems]);
        let device = CudaDevice::new(dev_idx).expect("cuda dev");

        let t = Tensor::from_device_ptr_unsafe(ptr, shape, DType::F32, device).expect("wrap ptr");
        // We can't deref the data here; just ensure Tensor constructed.
        let _ = t.shape();

        // Free external memory explicitly (Tensor shouldn't free external ptr)
        #[cfg(feature = "cuda-raw")]
        {
            let _ = core::cuda_memory::cuda_free(dev_idx, ptr as *mut u8);
        }
    }
}

