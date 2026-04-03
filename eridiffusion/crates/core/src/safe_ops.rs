use crate::{Result, Error};
use flame_core::{Tensor, DType, CudaDevice};
use std::sync::Arc;

#[inline]
fn as_f32(x: &Tensor) -> Result<Tensor> { x.to_dtype(DType::F32).map_err(Error::from) }

#[inline]
fn scalar(dev: Arc<CudaDevice>, v: f32) -> Result<Tensor> {
    Tensor::from_scalar(v, dev).map_err(Error::from)
}

/// Safe elementwise division: x / clamp_min(y, eps)
pub fn safe_div(x: &Tensor, y: &Tensor, eps: f32) -> Result<Tensor> {
    let eps_t = scalar(y.device().clone(), eps)?;
    let y_safe = y.maximum(&eps_t).map_err(Error::from)?;
    x.div(&y_safe).map_err(Error::from)
}

/// Safe log: log(clamp_min(x, eps))
pub fn safe_log(x: &Tensor, eps: f32) -> Result<Tensor> {
    let eps_t = scalar(x.device().clone(), eps)?;
    x.maximum(&eps_t).map_err(Error::from)?.log().map_err(Error::from)
}

/// Safe sqrt: sqrt(clamp_min(x, eps))
pub fn safe_sqrt(x: &Tensor, eps: f32) -> Result<Tensor> {
    let eps_t = scalar(x.device().clone(), eps)?;
    x.maximum(&eps_t).map_err(Error::from)?.sqrt().map_err(Error::from)
}

/// `sum(x, dim, keepdim=true)` in FP32
pub fn sum_keepdim_fp32(x: &Tensor, dim: usize) -> Result<Tensor> {
    let x32 = as_f32(x)?;
    x32.sum_dim_keepdim(dim).map_err(Error::from)
}

/// `mean(x, dim, keepdim=true)` in FP32
pub fn mean_keepdim_fp32(x: &Tensor, dim: usize) -> Result<Tensor> {
    let x32 = as_f32(x)?;
    x32.mean_dim(&[dim], true).map_err(Error::from)
}

/// Stable softmax in FP32 with keepdim sums.
pub fn softmax_stable(x: &Tensor, dim: isize) -> Result<Tensor> {
    let x32 = as_f32(x)?;
    let rank = x32.shape().dims().len();
    let axis: usize = if dim < 0 { (rank as isize + dim) as usize } else { dim as usize };
    let x_max = x32.max_dim(axis, true).map_err(Error::from)?;
    let z = x32.sub(&x_max).map_err(Error::from)?;
    let e = z.exp().map_err(Error::from)?;
    let s = e.sum_dim_keepdim(axis).map_err(Error::from)?;
    // clamp sum to avoid divide-by-zero
    let s_safe = {
        let eps = scalar(e.device().clone(), 1e-12f32)?;
        s.maximum(&eps).map_err(Error::from)?
    };
    let y32 = e.div(&s_safe).map_err(Error::from)?;
    y32.to_dtype(x.dtype()).map_err(Error::from)
}

/// Stable log_softmax via softmax_stable then log.
pub fn log_softmax_safe(x: &Tensor, dim: isize) -> Result<Tensor> {
    let sm = softmax_stable(x, dim)?;
    safe_log(&sm, 1e-12)
}

/// LayerNorm with FP32 stats, keepdim, and safe eps on the last dimension.
pub fn layer_norm_safe(x: &Tensor, eps: f32) -> Result<Tensor> {
    let x32 = as_f32(x)?;
    let last = x32.shape().dims().len().saturating_sub(1);
    let mean = mean_keepdim_fp32(&x32, last)?;
    let diff = x32.sub(&mean).map_err(Error::from)?;
    let var = diff.square().map_err(Error::from)?.mean_dim(&[last], true).map_err(Error::from)?;
    let denom = safe_sqrt(&var.add_scalar(eps).map_err(Error::from)?, 1e-20)?;
    let y32 = diff.div(&denom).map_err(Error::from)?;
    y32.to_dtype(x.dtype()).map_err(Error::from)
}

/// Normalize vector along last dim: x / ||x|| (FP32 + safe eps)
pub fn safe_normalize(x: &Tensor, eps: f32) -> Result<Tensor> {
    let x32 = as_f32(x)?;
    let last = x32.shape().dims().len().saturating_sub(1);
    let nrm = x32.square().map_err(Error::from)?.sum_dim_keepdim(last).map_err(Error::from)?;
    let nrm = safe_sqrt(&nrm, eps)?;
    x32.div(&nrm).map_err(Error::from)?.to_dtype(x.dtype()).map_err(Error::from)
}
