use anyhow::Result;
use flame_core::{DType, Tensor};

/// Convert NHWC tensor to NCHW.
pub fn nhwc_to_nchw(t: &Tensor) -> Result<Tensor> {
    Ok(t.permute(&[0, 3, 1, 2])?)
}

/// Convert NCHW tensor to NHWC.
pub fn nchw_to_nhwc(t: &Tensor) -> Result<Tensor> {
    Ok(t.permute(&[0, 2, 3, 1])?)
}

/// Convert tensor to desired dtype if needed.
pub fn to_dtype(t: &Tensor, dtype: DType) -> Result<Tensor> {
    if t.dtype() == dtype {
        Ok(t.clone())
    } else {
        Ok(t.to_dtype(dtype)?)
    }
}
