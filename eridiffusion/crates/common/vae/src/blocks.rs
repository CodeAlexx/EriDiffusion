use anyhow::{bail, Result};
use flame_core::{tensor_ext, DType, Tensor};
use std::sync::OnceLock;

fn trace_verbose() -> bool {
    static ONCE: OnceLock<bool> = OnceLock::new();
    *ONCE.get_or_init(|| std::env::var("VAE_TRACE_VERBOSE").ok().as_deref() == Some("1"))
}

/// Ensure NHWC BF16 input, cast to F32 for math
pub fn ensure_nhwc_f32(x: &Tensor) -> Result<Tensor> {
    let d = x.shape().dims().to_vec();
    if d.len() != 4 {
        bail!("expected NHWC rank-4 tensor, got {:?}", d);
    }
    if x.dtype() == DType::F32 {
        Ok(x.clone())
    } else {
        Ok(x.to_dtype(DType::F32)?)
    }
}

/// Convert to BF16 output (owning)
pub fn to_bf16(x: &Tensor) -> Result<Tensor> {
    if x.dtype() == DType::BF16 && x.storage_dtype() == DType::BF16 {
        Ok(x.clone_result()?)
    } else {
        Ok(x.to_dtype(DType::BF16)?.clone_result()?)
    }
}

/// Downsample NHWC by integer factor using bilinear
pub fn downsample_nhwc(x: &Tensor, factor: usize) -> Result<Tensor> {
    let d = x.shape().dims().to_vec();
    let (_b, h, w, _c) = (d[0], d[1], d[2], d[3]);
    let (oh, ow) = (h / factor, w / factor);
    Ok(flame_core::image_ops_nhwc::resize_bilinear_nhwc(x, oh, ow, false)?)
}

/// Upsample NHWC to the requested (H,W) using bilinear
pub fn upsample_nhwc(x: &Tensor, out_h: usize, out_w: usize) -> Result<Tensor> {
    Ok(flame_core::image_ops_nhwc::resize_bilinear_nhwc(x, out_h, out_w, false)?)
}

/// Adjust last-dim channels to out_c by slice or zero-pad (GPU-only path)
pub fn project_channels_nhwc(x: &Tensor, out_c: usize) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    if dims.len() != 4 {
        bail!("project_channels_nhwc expects NHWC tensor; got {:?}", dims);
    }
    let current = dims[3];
    let projected = if current == out_c {
        x.clone_result()?
    } else if current > out_c {
        tensor_ext::slice_channels(x, out_c)?
    } else {
        tensor_ext::pad_channels(x, out_c)?
    };

    if projected.dtype() == DType::BF16 && projected.storage_dtype() == DType::BF16 {
        Ok(projected)
    } else {
        Ok(projected.to_dtype(DType::BF16)?.clone_result()?)
    }
}

/// Log basic stats for quick sanity (GPU-safe by default)
pub fn log_stats(tag: &str, x: &Tensor) -> Result<()> {
    if !trace_verbose() {
        return Ok(());
    }
    eprintln!(
        "[vae.stats] {tag} dtype={:?} storage={:?} shape={:?}",
        x.dtype(),
        x.storage_dtype(),
        x.shape().dims()
    );

    #[cfg(feature = "vae_debug_stats")]
    {
        log_stats_debug(tag, x)?;
    }

    Ok(())
}

#[cfg(feature = "vae_debug_stats")]
fn log_stats_debug(tag: &str, x: &Tensor) -> Result<()> {
    let v = x.to_vec()?;
    if v.is_empty() {
        return Ok(());
    }
    let n = v.len().min(1_000_000);
    let mut mn = v[0];
    let mut mx = v[0];
    let mut sum = 0.0f32;
    let mut sq = 0.0f32;
    for &val in &v[..n] {
        mn = mn.min(val);
        mx = mx.max(val);
        sum += val;
        sq += val * val;
    }
    let nf = n as f32;
    let mean = sum / nf;
    let std = ((sq / nf) - mean * mean).max(0.0).sqrt();
    eprintln!("[vae.stats] {tag} mean={mean:.4} std={std:.4} min={mn:.4} max={mx:.4}");
    Ok(())
}
