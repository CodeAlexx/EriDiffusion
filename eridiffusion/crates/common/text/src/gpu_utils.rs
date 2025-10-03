use anyhow::{bail, Result};
use flame_core::{DType, Tensor};

/// Ensure the tensor resides on CUDA. Our text encoders currently operate only on CUDA tensors.
#[inline]
pub fn ensure_cuda(_: &str, _t: &Tensor) -> Result<()> {
    // All tensors produced in this pipeline are CUDA-backed. If CPU support is
    // ever added, wire a real check here.
    Ok(())
}

/// Convert a tensor to an owning BF16 buffer (no views, storage == logical dtype).
#[inline]
pub fn to_bf16_owned(tag: &str, x: &Tensor) -> Result<Tensor> {
    let y = if x.dtype() == DType::BF16 {
        x.clone_result()?
    } else {
        x.to_dtype(DType::BF16)?
    };
    if y.dtype() != DType::BF16 {
        bail!("{tag}: expected BF16 result, got {:?}", y.dtype());
    }
    Ok(y)
}

pub use flame_core::tensor_ext::to_owning_fp32_strong;
