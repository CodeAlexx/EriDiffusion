use std::mem::size_of;

use eridiffusion_core::{device::{require_cuda_device, shared_cuda_device}, DType};
use flame_core::{
    device::CudaStreamRawPtrExt,
    memcpy_async_device_to_host,
    memcpy_async_host_to_device,
    PinnedAllocFlags,
    PinnedHostBuffer,
    Shape,
    Tensor,
};

#[test]
fn pinned_memcpy_roundtrip() {
    require_cuda_device(0);
    let cuda = shared_cuda_device().expect("cuda available");
    let shape = Shape::from_dims(&[4usize, 8usize]);
    let elem_count = shape.elem_count();
    let bytes = elem_count * size_of::<f32>();

    let mut tensor = Tensor::zeros_dtype(shape, DType::F32, cuda.clone()).expect("device tensor");
    let device_ptr = tensor.cuda_ptr_mut() as *mut core::ffi::c_void;

    let mut pinned =
        PinnedHostBuffer::<f32>::with_capacity_elems(elem_count, PinnedAllocFlags::WRITE_COMBINED)
            .expect("pinned alloc");
    unsafe { pinned.set_len(elem_count); }
    let host_slice = pinned.as_mut_slice();
    for (i, v) in host_slice.iter_mut().enumerate() {
        *v = (i as f32) * 0.25 - 3.0;
    }

    let stream = tensor.device().cuda_stream_raw_ptr();
    memcpy_async_host_to_device(device_ptr, pinned.as_ptr() as *const _, bytes, stream)
        .expect("h2d copy");
    cuda.synchronize().expect("sync after h2d");

    let device_vec = tensor.to_vec().expect("device to host");
    for (i, v) in device_vec.iter().enumerate() {
        assert!(
            (v - host_slice[i]).abs() < 1e-6,
            "device mismatch at {}: {} vs {}",
            i,
            v,
            host_slice[i]
        );
    }

    // Zero host buffer and copy back from device
    for v in host_slice.iter_mut() {
        *v = 0.0;
    }
    memcpy_async_device_to_host(pinned.as_mut_ptr() as *mut _, device_ptr, bytes, stream)
        .expect("d2h copy");
    cuda.synchronize().expect("sync after d2h");

    for (i, v) in host_slice.iter().enumerate() {
        assert!(
            (v - device_vec[i]).abs() < 1e-6,
            "host mismatch at {}: {} vs {}",
            i,
            v,
            device_vec[i]
        );
    }
}
