use flame_core::{Tensor, DType, Shape, Result as CoreResult, Error as CoreError};

type FlameResult<T> = CoreResult<T>;

fn fused_enabled() -> bool {
    std::env::var("ERID_FUSE").ok().as_deref() == Some("1")
}

/// LayerNorm over the last dimension. FP32 stats and affine.
pub fn layer_norm(x: &Tensor, gamma: &Tensor, beta: &Tensor, eps: f32) -> CoreResult<Tensor> {
    if fused_enabled() { layer_norm_fused(x, gamma, beta, eps) } else { layer_norm_ref(x, gamma, beta, eps) }
}

fn layer_norm_ref(x: &Tensor, gamma: &Tensor, beta: &Tensor, eps: f32) -> CoreResult<Tensor> {
    let dims = x.shape().dims().to_vec();
    let d = *dims.last().ok_or_else(|| CoreError::InvalidInput("layer_norm: empty shape".into()))?;
    if gamma.shape().elem_count() != d || beta.shape().elem_count() != d {
        return Err(CoreError::InvalidInput("layer_norm: gamma/beta mismatch".into()));
    }
    let x32 = x.to_dtype(DType::F32)?;
    let last = dims[dims.len() - 1] as f32;
    let mean = x32.sum_dim_keepdim(dims.len() - 1)?.div_scalar(last)?; // [*,1]
    let xc = x32.sub(&mean)?;
    let var = xc.square()?.sum_dim_keepdim(dims.len() - 1)?.div_scalar(last)?;
    let inv = var.add_scalar(eps)?.rsqrt()?;
    let y32 = xc.mul(&inv)?;
    let y = y32.to_dtype(x.dtype())?;
    // affine
    let gamma_b = broadcast_to_last(gamma, &dims)?;
    let beta_b = broadcast_to_last(beta, &dims)?;
    y.mul(&gamma_b)?.add(&beta_b)
}

fn layer_norm_fused(x: &Tensor, gamma: &Tensor, beta: &Tensor, eps: f32) -> CoreResult<Tensor> {
    // For now, same as ref path but kept distinct for gating/kernels later
    layer_norm_ref(x, gamma, beta, eps)
}

/// RMSNorm over the last dimension. FP32 stats and affine (gamma, optional beta).
pub fn rms_norm(x: &Tensor, gamma: &Tensor, beta: Option<&Tensor>, eps: f32) -> FlameResult<Tensor> {
    if fused_enabled() { rms_norm_fused(x, gamma, beta, eps) } else { rms_norm_ref(x, gamma, beta, eps) }
}

fn rms_norm_ref(x: &Tensor, gamma: &Tensor, beta: Option<&Tensor>, eps: f32) -> FlameResult<Tensor> {
    let dims = x.shape().dims().to_vec();
    let d = *dims.last().ok_or_else(|| CoreError::InvalidInput("rms_norm: empty shape".into()))?;
    if gamma.shape().elem_count() != d { return Err(CoreError::InvalidInput("rms_norm: gamma mismatch".into())); }
    if let Some(b) = beta { if b.shape().elem_count() != d { return Err(CoreError::InvalidInput("rms_norm: beta mismatch".into())); } }
    let x32 = x.to_dtype(DType::F32)?;
    let last = dims[dims.len() - 1] as f32;
    let mean_sq = x32.square()?.sum_dim_keepdim(dims.len() - 1)?.div_scalar(last)?;
    let inv_rms = mean_sq.add_scalar(eps)?.rsqrt()?;
    let y32 = x32.mul(&inv_rms)?;
    let y = y32.to_dtype(x.dtype())?;
    let gamma_b = broadcast_to_last(gamma, &dims)?;
    let mut out = y.mul(&gamma_b)?;
    if let Some(b) = beta { let beta_b = broadcast_to_last(b, &dims)?; out = out.add(&beta_b)?; }
    Ok(out)
}

fn rms_norm_fused(x: &Tensor, gamma: &Tensor, beta: Option<&Tensor>, eps: f32) -> FlameResult<Tensor> {
    rms_norm_ref(x, gamma, beta, eps)
}

fn broadcast_to_last(param: &Tensor, dims: &[usize]) -> FlameResult<Tensor> {
    // param shape [D] -> [*, D]
    let mut target = vec![1usize; dims.len()];
    target[dims.len() - 1] = dims[dims.len() - 1];
    let p = param.reshape(&[1, dims[dims.len() - 1]])?;
    p.broadcast_to(&Shape::from_dims(dims))
}
