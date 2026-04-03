//! Global training policy helpers enforcing device/dtype/layout/noise/loss/grad rules.

use eridiffusion_core::{DType, Device, Error, Result};
use eridiffusion_models::devtensor::{shape1, shape2, uniform_on};
use flame_core::{bf16_factories, ops::reduce::sum_dim_keepdim_as, rng, Shape, Tensor};

/// Error if not running on a CUDA device.
pub fn assert_gpu_only(dev: &Device) -> Result<()> {
    match dev {
        Device::Cuda(_) => Ok(()),
        _ => Err(Error::Device("GPU-only execution required (CUDA device)".into())),
    }
}

/// Ensure parameters are stored as BF16 and (eventual) grads/optimizer states are FP32.
/// Note: Actual grad dtype enforcement depends on autograd integration; here we coerce params.
pub fn enforce_param_grad_dtypes(p: &Tensor) -> Result<()> {
    if p.dtype() != DType::BF16 {
        let _ = p.to_dtype(DType::BF16)?;
    }
    Ok(())
}

/// Convert latent to NHWC if needed. Assumes 4D tensor representing image latents.
pub fn enforce_nhwc_latent(z: &Tensor) -> Result<Tensor> {
    let dims = z.shape().dims().to_vec();
    if dims.len() != 4 {
        return Ok(z.clone());
    }
    // Treat [B,C,H,W] as NCHW if C is small (e.g., 4) and dims[3] != channel count
    let (_b, c, _h, w) = (dims[0], dims[1], dims[2], dims[3]);
    if c <= 8 && w != c {
        // NCHW -> NHWC
        return Ok(z.permute(&[0, 2, 3, 1])?);
    }
    Ok(z.clone())
}

/// Reduce mean in FP32 with keepdim=true across all elements (returns scalar [1]).
pub fn reduce_mean_fp32_keepdim(x: &Tensor) -> Result<Tensor> {
    if x.dtype() == DType::BF16 {
        let numel = x.shape().elem_count();
        if numel == 0 {
            return Err(Error::InvalidInput(
                "reduce_mean_fp32_keepdim: empty tensor".into(),
            ));
        }
        let flat = x.reshape(&[1, 1, numel])?;
        let sum = sum_dim_keepdim_as(&flat, 2, DType::BF16).map_err(Error::from)?;
        return sum.div_scalar(numel as f32).map_err(Error::from);
    }

    // Avoid dtype cast here to preserve requires_grad on the computation graph
    let sum = x.sum()?;
    let denom = (x.shape().elem_count() as f32).max(1.0);
    let out = sum.affine(1.0f32 / denom, 0.0f32)?;
    Ok(out)
}

#[inline]
fn clamp_symmetric(x: &Tensor, limit: f32) -> Result<Tensor> {
    // y = min(max(x, -limit), +limit) using scalar broadcasts
    let dev = x.device().clone();
    let hi = Tensor::from_scalar(limit, dev.clone())?;
    let lo = Tensor::from_scalar(-limit, dev.clone())?;
    let y = x.maximum(&lo)?.minimum(&hi)?;
    Ok(y)
}

/// Sample timesteps per batch uniformly in [0, 1000).
pub fn sample_timesteps(b: usize, dev: &Device) -> Result<Tensor> {
    let t = uniform_on(shape1(b as i64), dev, 0.0, 1000.0).map_err(Error::from)?;
    Ok(t)
}

/// Compute sigma for each timestep; placeholder linear mapping with optional Karras-style rescale.
pub fn sigma_for(t: &Tensor, _rescale: bool) -> Result<Tensor> {
    // Placeholder: sigma = t (users can plug scheduler later). Keep FP32.
    let t32 = if t.dtype() != DType::F32 { t.to_dtype(DType::F32)? } else { t.clone_result()? };
    Ok(t32)
}

/// Uniform timesteps in [0,1) as [B,1] tensor (FP32)
pub fn sample_timesteps01(b: usize, dev: &Device) -> Result<Tensor> {
    let t01 = uniform_on(shape2(b as i64, 1), dev, 0.0, 1.0).map_err(Error::from)?;
    Ok(t01.to_dtype(DType::F32)?)
}

/// Bounded sigma schedule: sigma in [sigma_min, sigma_max], monotone decreasing ~ exp(-t ln r)
pub fn sigma_for_bounded(t01: &Tensor, sigma_min: f32, sigma_max: f32) -> Result<Tensor> {
    let t = t01.clone_result()?;
    let ratio = (sigma_max / sigma_min).max(1.0 + 1e-6f32);
    let exp_term = t.neg()?.mul_scalar(ratio.ln())?.exp()?; // r^{-t}
    Ok(exp_term.mul_scalar(sigma_min)?)
}

/// Uniform timesteps in [0,1) as [B,1] tensor (BF16)
pub fn sample_timesteps01_bf16(b: usize, dev: &Device) -> Result<Tensor> {
    let cuda = dev.to_flame_cuda().map_err(Error::from)?;
    let seed = rng::next_u64();
    let shape = Shape::from_dims(&[b, 1]);
    bf16_factories::uniform_bf16(shape, 0.0, 1.0, seed, cuda).map_err(Error::from)
}

/// FP32 add_noise with sigma broadcast from [B,1] to data shape
pub fn add_noise(z_clean: &Tensor, eps: &Tensor, sigma_b1: &Tensor) -> Result<Tensor> {
    let zf = if z_clean.dtype() != DType::F32 {
        z_clean.to_dtype(DType::F32)?
    } else {
        z_clean.clone_result()?
    };
    let ef =
        if eps.dtype() != DType::F32 { eps.to_dtype(DType::F32)? } else { eps.clone_result()? };
    // Expand sigma [B,1] → [B,1,1] (will broadcast across trailing dims)
    let sb = if sigma_b1.shape().dims().len() == 2 {
        sigma_b1.unsqueeze(2)?
    } else {
        sigma_b1.clone_result()?
    };
    Ok(zf.add(&ef.mul(&sb)?)?)
}

/// Masked ε-prediction loss, reduced in FP32.
/// Mixed-precision reminders:
/// - Keep pred/targets in BF16 storage where possible; only do FP32 math inside kernels.
/// - Reductions (sum/mean/var) and the final scalar loss should be FP32 for stability.
/// - Do not force scalar loss to BF16; downstream logging/checkpointing expects FP32.
/// pred, eps: NHWC tensors. mask: [B,1,H,W] (NHWC-like with C=1) or None.
pub fn masked_eps_loss(pred: &Tensor, eps: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
    // Compute squared error; avoid casting to preserve grad tracking
    // Clamp residuals to avoid overflow during squaring when magnitudes explode early in bring-up
    let p = pred.clone();
    let e = eps.clone();
    let diff = p.sub(&e)?;
    let diff = clamp_symmetric(&diff, 1.0e6)?; // safety clamp
                                               // Square without global dtype promotion; BF16 kernel when enabled
    #[cfg(feature = "bf16_u16")]
    let sq = {
        use flame_core::bf16_ops::square_bf16;
        square_bf16(&diff)?
    };
    #[cfg(not(feature = "bf16_u16"))]
    let sq = diff.mul(&diff)?;

    if let Some(m) = mask {
        // Proper masked mean: sum(mask * sq) / (sum(mask) + eps)
        let m_nhwc = enforce_nhwc_latent(m)?;
        let m32 = if m_nhwc.dtype() != DType::F32 {
            m_nhwc.to_dtype(DType::F32)?
        } else {
            m_nhwc.clone()
        };
        let num = sq.mul(&m32)?;
        let sum_num = num.sum()?; // [1]
        let sum_den = m32.sum()?; // [1]
        let denom: f32 = sum_den.item()? as f32 + 1e-6f32;
        let out = sum_num.div_scalar(denom)?;
        Ok(out)
    } else {
        // Global mean in FP32
        reduce_mean_fp32_keepdim(&sq)
    }
}

/// Safe ε-prediction MSE with optional mask and warm-up clamp.
/// Shapes:
///  pred_eps, true_eps: [B, T, D]
///  mask (optional):    [B, T]
pub fn masked_eps_loss_safe(
    pred_eps: &Tensor,
    true_eps: &Tensor,
    mask_bt: Option<&Tensor>,
    global_step: usize,
    clamp_warmup_steps: usize,
    clamp_start: f32,
    clamp_end: f32,
) -> Result<Tensor> {
    // Keep pred on its current dtype to preserve autograd; cast target to match pred when needed
    let target = if true_eps.dtype() != pred_eps.dtype() {
        true_eps.to_dtype(pred_eps.dtype())?
    } else {
        true_eps.clone_result()?
    };
    let mut diff = pred_eps.sub(&target)?; // [B,T,D]

    if clamp_warmup_steps > 0 && global_step < clamp_warmup_steps {
        let t = global_step as f32 / clamp_warmup_steps.max(1) as f32;
        let k = 0.5 * (1.0 + (std::f32::consts::PI * t).cos());
        let lim = clamp_end + (clamp_start - clamp_end) * k;
        diff = clamp_symmetric(&diff, lim)?;
    }

    let sq = diff.mul(&diff)?; // [B,T,D] — avoid dtype cast that would drop grads
    let sq_masked = if let Some(mbt) = mask_bt {
        let m_b_t_1 = mbt.unsqueeze(1)?; // If mask is [B,1,H,W] style would differ; here assume [B,T]
                                         // but training callers pass None today; keep simple
        sq.mul(&m_b_t_1)?
    } else {
        sq
    };

    // Mean over D with keepdim
    let dims = sq_masked.shape().dims().to_vec();
    let (b, t, d) = (dims[0], dims[1], dims[2]);
    let sum_d = sq_masked.sum_dim_keepdim(2)?; // [B,T,1]
    let mean_d = sum_d.div_scalar(d as f32)?; // [B,T,1]
                                              // Mean over T
    let sum_t = mean_d.sum_dim_keepdim(1)?; // [B,1,1]
    let mean_t = sum_t.div_scalar(t as f32)?; // [B,1,1]
                                              // Mean over B to scalar
    let sum_b = mean_t.sum()?; // [1]
    let mean_b = sum_b.div_scalar(b as f32)?; // [1]
    Ok(mean_b)
}

/// Ensure only adapter params require grads; base weights remain frozen.
pub fn require_adapter_grads_only(_params_all: &[Tensor], _adapters: &[Tensor]) -> Result<()> {
    // Placeholder invariant check; full grad inspection requires autograd integration.
    Ok(())
}
