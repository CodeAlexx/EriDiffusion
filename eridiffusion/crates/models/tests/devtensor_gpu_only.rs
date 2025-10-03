#![cfg(feature = "cuda")]

use eridiffusion_core::Device;
use eridiffusion_models::devtensor::{
    copy_to_device,
    shape3,
    tensor_from_vec_on,
    to_device_dtype,
    zeros_on,
    BF16,
    F32_,
};
use flame_core::DType;

#[test]
fn devtensor_gpu_zero_and_cast() {
    let device = Device::Cuda(0);

    let zeros = zeros_on(shape3(2, 3, 4), &device, BF16).expect("zeros_on cuda");
    assert!(zeros.device().is_cuda(), "zeros tensor must live on CUDA");
    assert_eq!(zeros.dtype(), DType::BF16, "zeros_on must preserve BF16 storage");

    // Host upload must land on CUDA and keep dtype stable
    let host = vec![0.5f32; 6];
    let tensor = tensor_from_vec_on(host, shape3(1, 2, 3), &device, BF16)
        .expect("tensor_from_vec_on cuda");
    assert!(tensor.device().is_cuda(), "tensor_from_vec_on should yield CUDA tensors");
    assert_eq!(tensor.dtype(), DType::BF16, "upload should materialize BF16 storage");

    // Explicit dtype change should succeed and remain on the same device
    let tensor32 = to_device_dtype(&tensor, &device, F32_)
        .expect("to_device_dtype");
    assert_eq!(tensor32.dtype(), DType::F32, "to_device_dtype must honor requested dtype");
    assert!(tensor32.device().is_cuda(), "to_device_dtype must preserve device");

    // copy_to_device should be a no-op clone when already on target device
    let cloned = copy_to_device(&tensor32, &device).expect("copy_to_device same ordinal");
    assert_eq!(cloned.device().ordinal(), tensor32.device().ordinal());
    assert_eq!(cloned.dtype(), tensor32.dtype());
}
