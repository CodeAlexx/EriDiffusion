use anyhow::Result;
use flame_core::{DType, Tensor};
use crate::devtensor::to_dtype;

#[inline]
pub fn align_for_matmul(lhs: &Tensor, rhs: &Tensor, compute: DType) -> Result<(Tensor, Tensor)> {
    debug_assert!(compute == DType::BF16 || compute == DType::F32);
    let lhs_cast = if lhs.dtype() != compute { to_dtype(lhs, compute)? } else { lhs.clone() };
    let rhs_cast = if rhs.dtype() != compute { to_dtype(rhs, compute)? } else { rhs.clone() };
    Ok((lhs_cast, rhs_cast))
}
