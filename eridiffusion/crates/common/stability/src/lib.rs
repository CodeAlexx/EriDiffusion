use anyhow::Result;
use once_cell::sync::Lazy;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use tracing::info;
use flame_core::{Tensor, DType};
use eridiffusion_common_weights::ParamRegistry;

static TF32_ALLOWED: AtomicBool = AtomicBool::new(false);
static SNR_GAMMA_X1000: AtomicU32 = AtomicU32::new(0);

pub fn allow_tf32(on: bool) -> Result<()> {
    TF32_ALLOWED.store(on, Ordering::Relaxed);
    info!("tf32_allowed={}", on);
    Ok(())
}

pub fn sanitize_attn_mask(mask: &Tensor) -> Result<Tensor> {
    // Replace large negative (e.g., -inf) with -1e4 to stabilize exp
    let clamped = mask.clamp(-1.0e4, 0.0)?;
    Ok(clamped)
}

pub fn layernorm_stats_fp32(on: bool) {
    info!("layernorm_stats_fp32={}", on);
}

pub fn compute_loss_fp32(eps: &Tensor) -> Result<Tensor> {
    // Cast to f32 for loss math, then mean
    let t = flame_core::fp16::cast_tensor(eps, DType::F32)?;
    let loss = t.mul(&t)?.mean()?;
    Ok(loss)
}

pub fn clip_grad_global_norm(_reg: &ParamRegistry, _max_norm: f32) -> Result<()> {
    // TODO: integrate with GradientMap when wired; placeholder
    Ok(())
}

pub fn set_snr_gamma(gamma: f32) {
    let v = (gamma * 1000.0).max(0.0) as u32;
    SNR_GAMMA_X1000.store(v, Ordering::Relaxed);
    info!("snr_gamma={}", gamma);
}

