use std::sync::Arc;

use eridiffusion_core::{Error, Result};
use flame_core::{
    bf16_elementwise::{add_bf16, mul_bf16},
    ops::{elt, reduce::sum_dim_keepdim_as, tile::tile_bc_to_bhwc_f32},
    tensor_ext::to_owning_fp32_strong,
    CudaDevice, DType, Shape, Tensor,
};

#[derive(Clone, Copy, Debug)]
enum BinaryOp {
    Add,
    Mul,
}

/// Elementwise add with NumPy-style broadcasting semantics.
pub fn broadcast_add(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    broadcast_binary(a, b, BinaryOp::Add)
}

/// Elementwise multiply with NumPy-style broadcasting semantics.
pub fn broadcast_mul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    broadcast_binary(a, b, BinaryOp::Mul)
}

/// Thin wrapper around the CUDA in-place add helper to keep error types aligned.
pub fn add_inplace_same_dtype(dst: &mut Tensor, src: &Tensor) -> Result<()> {
    elt::add_inplace_same_dtype(dst, src).map_err(Error::from)
}

/// Broadcast tensor to a target shape while ensuring storage dtype matches `dtype`.
pub fn broadcast_to_as(a: &Tensor, target: &[usize], dtype: DType) -> Result<Tensor> {
    let cast = if a.dtype() == dtype { a.clone_result()? } else { a.to_dtype(dtype)? };
    if std::env::var("SDXL_DEBUG_SHAPES").ok().as_deref() == Some("1") {
        eprintln!(
            "[broadcast_to_as] len={} in dtype {:?} -> {:?} target {:?}",
            target.len(),
            a.dtype(),
            cast.dtype(),
            target
        );
    }
    let shape = shape_from_usize(target);
    let out = cast.broadcast_to(&shape).map_err(Error::from)?;
    let out = if dtype == DType::BF16 && out.storage_dtype() == DType::BF16 {
        out.clone_result().map_err(Error::from)?
    } else {
        out
    };
    if std::env::var("SDXL_DEBUG_SHAPES").ok().as_deref() == Some("1") {
        eprintln!(
            "[broadcast_to_as] result dtype {:?} storage {:?} shape {:?}",
            out.dtype(),
            out.storage_dtype(),
            out.shape().dims()
        );
    }
    Ok(out)
}

pub fn materialize_full_f32(
    vec_bc: &Tensor,
    b: usize,
    h: usize,
    w: usize,
    c: usize,
) -> Result<Tensor> {
    let cast = to_owning_fp32_strong(vec_bc)?;
    if std::env::var("SDXL_DEBUG_SHAPES").ok().as_deref() == Some("1") {
        eprintln!(
            "[materialize_full_f32] input dtype {:?} storage {:?} shape {:?}",
            cast.dtype(),
            cast.storage_dtype(),
            cast.shape().dims()
        );
    }
    if std::env::var("SDXL_DEBUG_SHAPES").ok().as_deref() == Some("1") {
        eprintln!("[materialize_full_f32] params b={} h={} w={} c={}", b, h, w, c);
    }
    let tiled =
        tile_bc_to_bhwc_f32(&cast, b, h, w, c).map_err(|e| Error::InvalidInput(e.to_string()))?;
    if std::env::var("SDXL_DEBUG_SHAPES").ok().as_deref() == Some("1") {
        eprintln!(
            "[materialize_full_f32] tiled dtype {:?} storage {:?} shape {:?}",
            tiled.dtype(),
            tiled.storage_dtype(),
            tiled.shape().dims()
        );
    }
    let tiled = to_owning_fp32_strong(&tiled)?;
    if std::env::var("SDXL_DEBUG_SHAPES").ok().as_deref() == Some("1") {
        eprintln!(
            "[materialize_full_f32] output dtype {:?} storage {:?} shape {:?}",
            tiled.dtype(),
            tiled.storage_dtype(),
            tiled.shape().dims()
        );
    }
    Ok(tiled)
}

fn broadcast_binary(a: &Tensor, b: &Tensor, op: BinaryOp) -> Result<Tensor> {
    ensure_same_device(a, b, "broadcast_binary")?;
    ensure_same_dtype(a, b, "broadcast_binary")?;

    match a.dtype() {
        DType::BF16 => broadcast_binary_bf16(a, b, op),
        DType::F32 => broadcast_binary_f32(a, b, op),
        other => {
            Err(Error::InvalidInput(format!("broadcast_binary: unsupported dtype {:?}", other)))
        }
    }
}

fn broadcast_binary_bf16(a: &Tensor, b: &Tensor, op: BinaryOp) -> Result<Tensor> {
    if std::env::var("SDXL_DEBUG_SHAPES").ok().as_deref() == Some("1") {
        eprintln!(
            "[broadcast_binary_bf16] op={:?} a={:?} b={:?}",
            op,
            a.shape().dims(),
            b.shape().dims()
        );
    }

    let out = match op {
        BinaryOp::Add => add_bf16(a, b),
        BinaryOp::Mul => mul_bf16(a, b),
    }
    .map_err(|e| Error::InvalidInput(format!("broadcast bf16 failed: {e}")))?;

    if std::env::var("SDXL_DEBUG_SHAPES").ok().as_deref() == Some("1") {
        eprintln!("[broadcast_binary_bf16] result shape {:?}", out.shape().dims());
    }

    Ok(out)
}

fn broadcast_binary_f32(a: &Tensor, b: &Tensor, op: BinaryOp) -> Result<Tensor> {
    ensure_supported_dtype(a.dtype())?;

    if std::env::var("SDXL_DEBUG_SHAPES").ok().as_deref() == Some("1") {
        eprintln!(
            "[broadcast_binary_f32] op={:?} a={:?} b={:?}",
            op,
            a.shape().dims(),
            b.shape().dims()
        );
    }

    let target = compute_broadcast_shape(a.shape().dims(), b.shape().dims()).ok_or_else(|| {
        Error::InvalidShape(format!(
            "broadcast: shapes {:?} vs {:?} are incompatible",
            a.shape().dims(),
            b.shape().dims()
        ))
    })?;

    let a_view = broadcast_to_shape(a, &target)?;
    let b_view = broadcast_to_shape(b, &target)?;

    let out = match op {
        BinaryOp::Add => a_view.add(&b_view),
        BinaryOp::Mul => a_view.mul(&b_view),
    }
    .map_err(|e| Error::InvalidInput(format!("broadcast op failed: {e}")))?;

    if std::env::var("SDXL_DEBUG_SHAPES").ok().as_deref() == Some("1") {
        eprintln!(
            "[broadcast_binary_f32] result dtype {:?} shape {:?}",
            out.dtype(),
            out.shape().dims()
        );
    }

    Ok(out)
}

fn broadcast_to_shape(t: &Tensor, target: &[usize]) -> Result<Tensor> {
    let mut current = t.clone();
    if current.shape().dims() == target {
        return Ok(current);
    }

    if current.shape().dims().len() > target.len() {
        return Err(Error::InvalidShape(format!(
            "broadcast: rank {} cannot fit target {:?}",
            current.shape().dims().len(),
            target
        )));
    }

    if current.shape().dims().len() < target.len() {
        let mut padded = vec![1usize; target.len()];
        let offset = target.len() - current.shape().dims().len();
        for (idx, &dim) in current.shape().dims().iter().enumerate() {
            padded[offset + idx] = dim;
        }
        let shape = shape_from_usize(&padded);
        current = current.reshape(shape.dims()).map_err(Error::from)?;
    }

    if current.shape().dims() == target {
        return Ok(current);
    }

    let shape = shape_from_usize(target);
    current.broadcast_to(&shape).map_err(Error::from)
}

fn compute_broadcast_shape(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    let rank = a.len().max(b.len());
    let mut out = vec![1usize; rank];
    for i in 0..rank {
        let da = if i < a.len() { a[a.len() - 1 - i] } else { 1 };
        let db = if i < b.len() { b[b.len() - 1 - i] } else { 1 };
        if da == db || da == 1 || db == 1 {
            out[rank - 1 - i] = da.max(db);
        } else {
            return None;
        }
    }
    Some(out)
}

fn shape_from_usize(dims: &[usize]) -> Shape {
    let dims_usize = dims.to_vec();
    Shape::from_dims(&dims_usize)
}

fn ensure_same_device(a: &Tensor, b: &Tensor, ctx: &str) -> Result<()> {
    let da = a.device();
    let db = b.device();
    if da.ordinal() != db.ordinal() {
        return Err(Error::InvalidInput(format!(
            "{}: tensors on different devices cuda:{} vs cuda:{}",
            ctx,
            da.ordinal(),
            db.ordinal()
        )));
    }
    Ok(())
}

fn ensure_same_dtype(a: &Tensor, b: &Tensor, ctx: &str) -> Result<()> {
    if a.dtype() != b.dtype() {
        return Err(Error::InvalidInput(format!(
            "{}: dtype mismatch {:?} vs {:?}",
            ctx,
            a.dtype(),
            b.dtype()
        )));
    }
    Ok(())
}

fn ensure_supported_dtype(dtype: DType) -> Result<()> {
    match dtype {
        DType::BF16 | DType::F16 | DType::F32 => Ok(()),
        other => Err(Error::Unsupported(format!("broadcast: unsupported dtype {:?}", other))),
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn broadcast_mul_expands_row_vector() {
        eprintln!("skipping broadcast_mul_expands_row_vector (GPU-only test)");
    }

    #[test]
    fn broadcast_mul_qk_norm_shapes() {
        eprintln!("skipping broadcast_mul_qk_norm_shapes (GPU-only test)");
    }
}

#[inline]
pub fn sum_keepdim_fp32(x: &Tensor, dim: usize) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    if dims.is_empty() {
        return Err(Error::InvalidInput("sum_keepdim_fp32: tensor rank must be >= 1".into()));
    }
    if dim >= dims.len() {
        return Err(Error::InvalidInput(format!("sum_keepdim_fp32: dim {dim} out of range")));
    }
    if dims[dim] == 0 {
        return Err(Error::InvalidInput(
            "sum_keepdim_fp32: reduction axis must have non-zero length".into(),
        ));
    }
    if x.dtype() != DType::BF16 {
        return Err(Error::InvalidInput(format!(
            "sum_keepdim_fp32: expected BF16 tensor, got {:?}",
            x.dtype()
        )));
    }

    let (prepared, restore_shape, inv_perm, _, moved) = bf16_reduce_prepare(x, dim)?;
    let sum = sum_dim_keepdim_as(&prepared, 2, DType::BF16).map_err(Error::from)?;
    let sum = sum.reshape(&restore_shape).map_err(Error::from)?;
    let result = if moved { sum.permute(&inv_perm).map_err(Error::from)? } else { sum };
    Ok(result)
}

#[inline]
pub fn mean_keepdim_fp32(x: &Tensor, dim: usize) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    if dims.is_empty() {
        return Err(Error::InvalidInput("mean_keepdim_fp32: tensor rank must be >= 1".into()));
    }
    if dim >= dims.len() {
        return Err(Error::InvalidInput(format!("mean_keepdim_fp32: dim {dim} out of range")));
    }
    if dims[dim] == 0 {
        return Err(Error::InvalidInput(
            "mean_keepdim_fp32: reduction axis must have non-zero length".into(),
        ));
    }
    if x.dtype() != DType::BF16 {
        return Err(Error::InvalidInput(format!(
            "mean_keepdim_fp32: expected BF16 tensor, got {:?}",
            x.dtype()
        )));
    }

    let (prepared, restore_shape, inv_perm, reduce_len, moved) = bf16_reduce_prepare(x, dim)?;
    let sum = sum_dim_keepdim_as(&prepared, 2, DType::BF16).map_err(Error::from)?;
    let mean = sum
        .to_dtype(DType::F32)
        .map_err(Error::from)?
        .div_scalar(reduce_len as f32)
        .map_err(Error::from)?
        .to_dtype(DType::BF16)
        .map_err(Error::from)?;
    let mean = mean.reshape(&restore_shape).map_err(Error::from)?;
    let result = if moved { mean.permute(&inv_perm).map_err(Error::from)? } else { mean };
    Ok(result)
}

#[inline]
pub fn mean_keepdim_bf16(x: &Tensor, dim: usize) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    if dims.is_empty() {
        return Err(Error::InvalidInput("mean_keepdim_bf16: tensor rank must be >= 1".into()));
    }
    if dim >= dims.len() {
        return Err(Error::InvalidInput(format!("mean_keepdim_bf16: dim {dim} out of range")));
    }
    if dims[dim] == 0 {
        return Err(Error::InvalidInput(
            "mean_keepdim_bf16: reduction axis must have non-zero length".into(),
        ));
    }
    if x.dtype() != DType::BF16 {
        return Err(Error::InvalidInput(format!(
            "mean_keepdim_bf16: expected BF16 tensor, got {:?}",
            x.dtype()
        )));
    }

    let (prepared, restore_shape, inv_perm, reduce_len, moved) = bf16_reduce_prepare(x, dim)?;
    let sum = sum_dim_keepdim_as(&prepared, 2, DType::BF16).map_err(Error::from)?;
    let mean = sum.div_scalar(reduce_len as f32).map_err(Error::from)?;
    let mean = mean.reshape(&restore_shape).map_err(Error::from)?;
    let result = if moved { mean.permute(&inv_perm).map_err(Error::from)? } else { mean };
    Ok(result)
}

#[inline]
pub fn sum_all_bf16(x: &Tensor) -> Result<Tensor> {
    let numel = x.shape().elem_count();
    if numel == 0 {
        return Err(Error::InvalidInput(
            "sum_all_bf16: zero-sized tensor not supported".into(),
        ));
    }
    if x.dtype() != DType::BF16 {
        return Err(Error::InvalidInput(format!(
            "sum_all_bf16: expected BF16 tensor, got {:?}",
            x.dtype()
        )));
    }
    let flat = x.reshape(&[1, 1, numel]).map_err(Error::from)?;
    sum_dim_keepdim_as(&flat, 2, DType::BF16).map_err(Error::from)
}

#[inline]
pub fn softmax_stable(x: &Tensor, dim: i64) -> Result<Tensor> {
    // Cast to F32 for numerical stability
    let x32 = x.to_dtype(DType::F32).map_err(Error::from)?;
    // translate potentially-negative dim to absolute axis
    let rank = x32.shape().dims().len();
    let axis: usize = if dim < 0 { (rank as i64 + dim) as usize } else { dim as usize };
    let x_max = x32.max_dim(axis, true).map_err(Error::from)?;
    let z = x32.sub(&x_max).map_err(Error::from)?;
    let e = z.exp().map_err(Error::from)?;
    let s = e.sum_dim_keepdim(axis).map_err(Error::from)?;
    // clamp min to avoid divide-by-zero
    let dev = e.device().clone();
    let eps = Tensor::from_scalar(1e-12f32, dev).map_err(Error::from)?;
    let s_safe = s.maximum(&eps).map_err(Error::from)?;
    let y32 = e.div(&s_safe).map_err(Error::from)?;
    y32.to_dtype(x.dtype()).map_err(Error::from)
}

fn bf16_reduce_prepare(
    x: &Tensor,
    dim: usize,
) -> Result<(Tensor, Vec<usize>, Vec<usize>, usize, bool)> {
    let dims = x.shape().dims().to_vec();
    let rank = dims.len();
    let mut perm: Vec<usize> = (0..rank).collect();
    let moved = dim != rank.saturating_sub(1);
    if moved {
        let axis = perm.remove(dim);
        perm.push(axis);
    }

    let permuted = if moved {
        x.permute(&perm).map_err(Error::from)?
    } else {
        x.clone_result().map_err(Error::from)?
    };
    let perm_dims = permuted.shape().dims().to_vec();
    let total: usize = perm_dims.iter().product();
    if total == 0 {
        return Err(Error::InvalidInput(
            "bf16_reduce_prepare: zero-sized tensor not supported in BF16 path".into(),
        ));
    }

    let reduce_len = *perm_dims
        .last()
        .ok_or_else(|| Error::InvalidInput("bf16_reduce_prepare: empty shape".into()))?;
    let outer = total
        .checked_div(reduce_len)
        .ok_or_else(|| Error::InvalidInput("bf16_reduce_prepare: invalid reshape".into()))?;

    let reshaped = permuted.reshape(&[1, outer, reduce_len]).map_err(Error::from)?;
    let mut restore_shape = perm_dims;
    let last_idx = restore_shape.len() - 1;
    restore_shape[last_idx] = 1;

    let inv_perm = if moved { invert_permutation(&perm) } else { (0..rank).collect() };

    Ok((reshaped, restore_shape, inv_perm, reduce_len, moved))
}

fn invert_permutation(perm: &[usize]) -> Vec<usize> {
    let mut inv = vec![0usize; perm.len()];
    for (i, &axis) in perm.iter().enumerate() {
        inv[axis] = i;
    }
    inv
}

#[inline]
pub fn scalar_f32(v: f32, device: Arc<CudaDevice>) -> Result<Tensor> {
    Tensor::from_scalar(v, device).map_err(Error::from)
}

#[inline]
pub fn scalar_i32(v: i32, device: Arc<CudaDevice>) -> Result<Tensor> {
    let t = Tensor::from_scalar(v as f32, device).map_err(Error::from)?;
    t.to_dtype(DType::I32).map_err(Error::from)
}

// --- Safe numeric helpers for training paths ---

#[inline]
pub fn safe_log(x: &Tensor) -> Result<Tensor> {
    // log(clamp_min(x, 1e-12))
    let dev = x.device().clone();
    let eps = Tensor::from_scalar(1e-12f32, dev).map_err(Error::from)?;
    x.maximum(&eps).map_err(Error::from)?.log().map_err(Error::from)
}

#[inline]
pub fn safe_sqrt_pos(x: &Tensor) -> Result<Tensor> {
    // sqrt(max(x, 0)) with small epsilon to avoid gradients exploding at 0
    let dev = x.device().clone();
    let eps = Tensor::from_scalar(1e-12f32, dev.clone()).map_err(Error::from)?;
    x.maximum(&eps).map_err(Error::from)?.sqrt().map_err(Error::from)
}

#[inline]
pub fn safe_div_eps(num: &Tensor, den: &Tensor, eps_val: f32) -> Result<Tensor> {
    // num / max(den, eps)
    let dev = den.device().clone();
    let eps = Tensor::from_scalar(eps_val, dev).map_err(Error::from)?;
    let den_safe = den.maximum(&eps).map_err(Error::from)?;
    num.div(&den_safe).map_err(Error::from)
}

#[inline]
pub fn reciprocal_eps(x: &Tensor, eps_val: f32) -> Result<Tensor> {
    let dev = x.device().clone();
    let eps = Tensor::from_scalar(eps_val, dev).map_err(Error::from)?;
    let x_safe = x.maximum(&eps).map_err(Error::from)?;
    Tensor::from_scalar(1.0f32, x_safe.device().clone())
        .map_err(Error::from)?
        .div(&x_safe)
        .map_err(Error::from)
}

#[inline]
pub fn clamp01_eps(x: &Tensor, eps_val: f32) -> Result<Tensor> {
    let dev = x.device().clone();
    let eps = Tensor::from_scalar(eps_val, dev.clone()).map_err(Error::from)?;
    let one_minus = Tensor::from_scalar(1.0f32 - eps_val, dev.clone()).map_err(Error::from)?;
    let y = x.maximum(&eps).map_err(Error::from)?;
    y.minimum(&one_minus).map_err(Error::from)
}
