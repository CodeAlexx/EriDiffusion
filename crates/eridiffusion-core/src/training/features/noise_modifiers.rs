//! Noise modifiers — extra perturbations applied to the sampled training noise
//! before the forward pass. Two independent operators:
//!
//!   1. Offset noise: with probability `prob`, add a per-channel scalar
//!      `weight * randn([B, C, 1, 1])` broadcast over spatial dims, encouraging
//!      the model to learn variable-brightness samples (Hang et al. 2023).
//!   2. Input perturbation: noise + γ * randn(noise.shape()) — small
//!      additive noise to noise itself (Ning et al. 2023, "Input Perturbation
//!      Reduces Exposure Bias"). Applied AFTER offset noise.
//!
//! Phase: 1
//! Config flags:
//!   - `offset_noise_weight: f64` (existing) — magnitude
//!   - `noise_offset_probability: f32` (Phase 0) — Bernoulli gate
//!   - `gamma_input_perturbation: f32` (Phase 0) — perturbation γ
//!
//! Default-off invariance:
//!   - `offset_noise_weight <= 0.0` OR `prob <= 0.0` short-circuits with
//!     `Ok(noise.clone())` and does NOT consume rng or alloc tensors.
//!   - `gamma <= 0.0` short-circuits the same way.
//!
//! Reference:
//!   - SimpleTuner `helpers/training/diffusion_model.py` offset noise +
//!     input perturbation paths.

use flame_core::upsampling::{Upsample2d, Upsample2dConfig, UpsampleMode};
use flame_core::{Result, Shape, Tensor};
use rand::rngs::StdRng;
use rand::Rng;

/// Apply offset noise: `noise + weight * per_channel_offset` with probability
/// `prob`.
///
/// `per_channel_offset` is a `[B, C, 1, 1]` (or `[B, C, 1]` for 3D tensors)
/// randn broadcast over the spatial dims.
///
/// When `weight <= 0.0` or `prob <= 0.0`, returns `Ok(noise.clone())` with no
/// rng consumption — preserving default-off byte invariance for trainers that
/// never opted in.
pub fn maybe_apply_offset_noise(
    noise: &Tensor,
    weight: f32,
    prob: f32,
    rng: &mut StdRng,
) -> Result<Tensor> {
    if weight <= 0.0 || prob <= 0.0 {
        return Ok(noise.clone());
    }
    if rng.r#gen::<f32>() >= prob {
        return Ok(noise.clone());
    }
    let dims = noise.shape().dims();
    let per_channel_shape = match dims.len() {
        4 => Shape::from_dims(&[dims[0], dims[1], 1, 1]),
        3 => Shape::from_dims(&[dims[0], dims[1], 1]),
        2 => Shape::from_dims(&[dims[0], dims[1]]),
        _ => return Ok(noise.clone()),
    };
    let offset = Tensor::randn(per_channel_shape, 0.0, 1.0, noise.device().clone())?
        .mul_scalar(weight)?;
    let offset = offset.to_dtype(noise.dtype())?;
    let broadcast = offset.broadcast_to(noise.shape())?;
    noise.add(&broadcast)
}

/// Apply input perturbation: `noise + gamma * randn(noise.shape())`.
///
/// When `gamma <= 0.0`, returns `Ok(noise.clone())` with no rng consumption.
pub fn maybe_apply_input_perturbation(
    noise: &Tensor,
    gamma: f32,
    _rng: &mut StdRng,
) -> Result<Tensor> {
    if gamma <= 0.0 {
        return Ok(noise.clone());
    }
    let perturbation = Tensor::randn(noise.shape().clone(), 0.0, 1.0, noise.device().clone())?
        .mul_scalar(gamma)?
        .to_dtype(noise.dtype())?;
    noise.add(&perturbation)
}

/// Apply pyramid (multi-resolution) noise modifier.
///
/// Adds `discount^k * upsample(randn(scaled))` for `k ∈ 1..=levels` on top of
/// the input noise, where each `scaled` shape halves spatial dims relative to
/// the previous level. The upsample is bilinear back to the input's H/W.
///
/// `noise_out = noise + sum_{k=1..levels} discount^k * bilinear_up(randn(B, C, H/2^k, W/2^k))`
///
/// Same semantics as OneTrainer `apply_multi_resolution_noise` (additive,
/// no post-`std()` normalization). Kohya's implementation rescales the
/// total to unit-variance via `noise / noise.std()`; this implementation
/// does not — at the typical default discount=0.3 the variance growth is
/// `1 + 0.09 + 0.0081 + ... < 1.1`, well within sigma's training range.
///
/// Default-off invariance: `levels = 0` OR `discount <= 0.0` returns
/// `Ok(noise.clone())` and consumes no rng.
///
/// Inner loop terminates early when either spatial dim collapses to 1
/// (further halvings would just resample the same constant).
///
/// Reference: Kohya `library/multires_noise_pl.py`,
/// SimpleTuner `helpers/training/multires_noise.py`,
/// OneTrainer `modules/util/loss/MultiResolutionNoiseUtil.py`.
pub fn maybe_apply_multires_noise(
    noise: &Tensor,
    levels: usize,
    discount: f32,
    _rng: &mut StdRng,
) -> Result<Tensor> {
    if levels == 0 || discount <= 0.0 {
        return Ok(noise.clone());
    }
    let dims = noise.shape().dims();
    if dims.len() != 4 {
        return Ok(noise.clone());
    }
    let (b, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);
    let upsampler = Upsample2d::new(
        Upsample2dConfig::new(UpsampleMode::Bilinear).with_size((h, w)),
    );
    let mut out = noise.clone();
    for k in 1..=levels {
        let scale = 1usize << k;
        let h_k = (h / scale).max(1);
        let w_k = (w / scale).max(1);
        let scaled = Tensor::randn(
            Shape::from_dims(&[b, c, h_k, w_k]),
            0.0,
            1.0,
            noise.device().clone(),
        )?;
        let scaled = scaled.to_dtype(noise.dtype())?;
        let upsampled = upsampler.forward(&scaled)?;
        let weighted = upsampled.mul_scalar(discount.powi(k as i32))?;
        out = out.add(&weighted)?;
        if h_k == 1 || w_k == 1 {
            break;
        }
    }
    Ok(out)
}

// ── Legacy skeleton names (Phase 0 callers) ─────────────────────────────────

/// Legacy skeleton name — forwards to [`maybe_apply_offset_noise`].
pub fn offset_noise(
    noise: &Tensor,
    weight: f32,
    prob: f32,
    rng: &mut StdRng,
) -> Result<Tensor> {
    maybe_apply_offset_noise(noise, weight, prob, rng)
}

/// Legacy skeleton name — forwards to [`maybe_apply_input_perturbation`].
pub fn input_perturbation(noise: &Tensor, gamma: f32, rng: &mut StdRng) -> Result<Tensor> {
    maybe_apply_input_perturbation(noise, gamma, rng)
}
