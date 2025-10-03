#[cfg(feature = "bf16_u16")]
use flame_core::DType;
#[cfg(feature = "bf16_u16")]
use flame_core::{bf16_clamp, bf16_elementwise as be, bf16_ops};
use flame_core::{Result, Tensor};

/// BF16/legacy-aware add (broadcast-safe when BF16)
#[inline]
pub fn add(x: &Tensor, y: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "bf16_u16")]
    if x.dtype() == DType::BF16 && y.dtype() == DType::BF16 {
        return be::add_bf16(x, y);
    }
    x.add(y)
}

/// BF16/legacy-aware mul (broadcast-safe when BF16)
#[inline]
pub fn mul(x: &Tensor, y: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "bf16_u16")]
    if x.dtype() == DType::BF16 && y.dtype() == DType::BF16 {
        return be::mul_bf16(x, y);
    }
    x.mul(y)
}

/// BF16/legacy-aware ge compare (Bool output)
#[inline]
pub fn ge(x: &Tensor, y: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "bf16_u16")]
    if x.dtype() == DType::BF16 && y.dtype() == DType::BF16 {
        return be::ge_bf16(x, y);
    }
    x.ge(y)
}

/// BF16/legacy-aware GELU
#[inline]
pub fn gelu(x: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "bf16_u16")]
    if x.dtype() == DType::BF16 {
        return bf16_ops::gelu_bf16(x);
    }
    x.gelu()
}

/// BF16/legacy-aware symmetric clamp
#[inline]
pub fn clamp(x: &Tensor, limit: f32) -> Result<Tensor> {
    #[cfg(feature = "bf16_u16")]
    if x.dtype() == DType::BF16 {
        return bf16_clamp::clamp_bf16(x, -limit, limit);
    }
    // Fallback: compute-only clamp via engine path
    let dev = x.device().clone();
    let lo = Tensor::from_scalar(-limit, dev.clone())?;
    let hi = Tensor::from_scalar(limit, dev.clone())?;
    x.maximum(&lo)?.minimum(&hi)
}

/// BF16/legacy-aware linear (matmul)
#[inline]
pub fn linear(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "bf16_u16")]
    if x.dtype() == DType::BF16 && w.dtype() == DType::BF16 {
        return x.matmul_bf16(w);
    }
    x.matmul(w)
}
