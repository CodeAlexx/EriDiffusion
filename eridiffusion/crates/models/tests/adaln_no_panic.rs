use flame_core::{DType, Tensor};
use eridiffusion_models::sdxl::adaln::adaln_modulate_true;

#[test]
fn adaln_returns_bf16_and_no_panic() {
    let (b, t, c) = (1, 8, 64);
    let x = Tensor::randn(&[b, t, c], DType::BF16).unwrap();
    let aff = Tensor::randn(&[b, 3 * c], DType::BF16).unwrap();
    let (y, gate) = adaln_modulate_true(&x, &aff, 1e-6).unwrap();
    assert_eq!(y.dtype(), DType::BF16);
    assert_eq!(gate.dtype(), DType::BF16);
    assert_eq!(y.shape(), x.shape());
    assert_eq!(gate.shape(), x.shape());
}
