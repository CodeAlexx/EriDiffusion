#![cfg(feature = "sdxl")]

use eridiffusion_models::sdxl::adaln::adaln_modulate_true;
use flame_core::{device::Device, DType, Shape, Tensor};

#[test]
fn adaln_returns_bf16_and_no_panic() {
    let Ok(device) = Device::cuda(0) else {
        // Skip when CUDA is unavailable (e.g., doc builds).
        return;
    };
    let cuda = device.cuda_device().clone();

    let (b, t, c) = (1, 8, 64);
    let x = Tensor::randn(Shape::from_dims(&[b, t, c]), 0.0, 1.0, cuda.clone())
        .and_then(|t| t.to_dtype(DType::BF16))
        .unwrap();
    let aff = Tensor::randn(Shape::from_dims(&[b, 3 * c]), 0.0, 1.0, cuda)
        .and_then(|t| t.to_dtype(DType::BF16))
        .unwrap();
    let (y, gate) = adaln_modulate_true(&x, &aff, 1e-6).unwrap();
    assert_eq!(y.dtype(), DType::BF16);
    assert_eq!(gate.dtype(), DType::BF16);
    assert_eq!(y.shape(), x.shape());
    assert_eq!(gate.shape(), x.shape());
}
