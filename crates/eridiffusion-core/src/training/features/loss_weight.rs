//! Loss weighting and combined MSE+MAE+Huber loss.
//!
//! Phase: 1
//!
//! Implementations:
//!   - `min_snr_weight(sigma, gamma, is_v_prediction)` — MIN-SNR γ
//!   - `debiased_weight(sigma)`                       — debiased estimation
//!   - `apply_loss_weight(loss, sigma, …)`            — dispatch from
//!     `config.loss_weight_fn` enum + `min_snr_gamma` Option override
//!   - `combined_loss(pred, target, mse_s, mae_s, huber_s)` — MSE + MAE + Huber
//!
//! Default-off invariance: when `loss_weight_fn=Constant` and
//! `min_snr_gamma=None`, [`apply_loss_weight`] returns `Ok(loss.clone())`
//! and is byte-identical to "do nothing". When `mse=1.0, mae=0.0, huber=0.0`,
//! [`combined_loss`] is byte-identical to the previous bare
//! `(pred-target).square().mean()`.
//!
//! Reference:
//!   - SimpleTuner `helpers/training/diffusion_model.py` (search `min_snr_gamma`)
//!   - OneTrainer `modules/modelSetup/mixin/ModelSetupDiffusionLossMixin.py:_diffusion_losses`
//!
//! SNR derivation for FLOW MATCHING (Klein/Flux/Z-Image/SD3.5):
//!     snr(sigma) = ((1 - sigma) / sigma)^2
//! For ε-prediction (DDPM-style; SDXL):
//!     snr(t)     = alpha_bar(t) / (1 - alpha_bar(t))
//! Caller passes `sigma` for flow matching; for ε-pred trainers wishing to
//! use these weights, convert via the standard mapping before calling.

use flame_core::{Result, Tensor};

use crate::config::LossWeight;

/// MIN-SNR γ weight per Hang et al. 2023.
///
/// `is_v_prediction = false` (ε-prediction): w = min(snr, γ) / snr
/// `is_v_prediction = true`  (v-prediction / flow matching): w = min(snr, γ) / (snr + 1)
pub fn min_snr_weight(sigma: f32, gamma: f32, is_v_prediction: bool) -> f32 {
    let s = sigma.max(1e-8);
    let snr = ((1.0 - s) / s).powi(2);
    min_snr_weight_from_snr(snr, gamma, is_v_prediction)
}

/// MIN-SNR γ weight from a raw SNR value. Use this for ε-prediction
/// (DDPM-style) trainers where SNR = ᾱ / (1 - ᾱ) and the flow-matching
/// `sigma` formulation does not apply.
pub fn min_snr_weight_from_snr(snr: f32, gamma: f32, is_v_prediction: bool) -> f32 {
    let cap = snr.min(gamma);
    if is_v_prediction {
        cap / (snr + 1.0)
    } else {
        cap / snr.max(1e-8)
    }
}

/// Debiased-estimation weight per OneTrainer
/// `modules/modelSetup/mixin/ModelSetupDiffusionLossMixin.py:226-241`:
///     w = 1 / sqrt(min(snr, 1e3) + (1 if v_pred else 0))
///
/// The 1e3 clip prevents extreme upweighting at very-high-SNR (small sigma)
/// steps; the +1 v-prediction adjustment matches the v-loss derivation
/// (see Salimans & Ho 2022).
pub fn debiased_weight(sigma: f32, is_v_prediction: bool) -> f32 {
    let s = sigma.max(1e-8);
    let snr = ((1.0 - s) / s).powi(2);
    debiased_weight_from_snr(snr, is_v_prediction)
}

/// Debiased-estimation weight from a raw SNR value (with the OT clip + v-pred
/// adjustment applied).
pub fn debiased_weight_from_snr(snr: f32, is_v_prediction: bool) -> f32 {
    let snr_clamped = snr.min(1e3);
    let snr_adjusted = if is_v_prediction { snr_clamped + 1.0 } else { snr_clamped };
    1.0 / snr_adjusted.sqrt().max(1e-8)
}

/// Apply per-step loss weighting selected by `weight_fn` with optional
/// `gamma_override` (CLI / config `min_snr_gamma`).
///
/// When `weight_fn = Constant` AND `gamma_override = None`, the function
/// short-circuits with `Ok(loss.clone())` — default-off byte invariance.
///
/// `is_v_prediction`: pass `true` for flow-matching trainers (Klein, Flux,
/// Z-Image, SD3.5, ERNIE, Qwen, ACE-Step, LTX2, Anima); pass `false` for
/// SDXL/SD2-style ε-prediction.
pub fn apply_loss_weight(
    loss: &Tensor,
    sigma: f32,
    weight_fn: LossWeight,
    gamma_override: Option<f32>,
    is_v_prediction: bool,
) -> Result<Tensor> {
    // Override path: --min-snr-gamma X forces MIN-SNR γ regardless of the
    // configured `loss_weight_fn`. This matches the user expectation that
    // the CLI flag is a one-shot toggle.
    if let Some(g) = gamma_override {
        let w = min_snr_weight(sigma, g, is_v_prediction);
        return loss.mul_scalar(w);
    }
    match weight_fn {
        LossWeight::Constant => Ok(loss.clone()),
        LossWeight::MinSnrGamma => {
            // Enum carries no gamma; use a sensible default (5.0 — Hang et
            // al. recommended). User wanting non-5.0 should pass `--min-snr-gamma`.
            let w = min_snr_weight(sigma, 5.0, is_v_prediction);
            loss.mul_scalar(w)
        }
        LossWeight::P2 => {
            // P2 needs (gamma, k); not in current enum. Defer to a later phase.
            // Default-off: behave as Constant.
            Ok(loss.clone())
        }
        LossWeight::Sigma => {
            // Sigma weighting needs an exponent; not in current enum. Defer.
            Ok(loss.clone())
        }
    }
}

/// Same as [`apply_loss_weight`] but takes a raw SNR rather than a flow-style
/// sigma. Use for ε-prediction (DDPM-style) trainers where
/// SNR = ᾱ / (1 - ᾱ).
pub fn apply_loss_weight_from_snr(
    loss: &Tensor,
    snr: f32,
    weight_fn: LossWeight,
    gamma_override: Option<f32>,
    is_v_prediction: bool,
) -> Result<Tensor> {
    if let Some(g) = gamma_override {
        let w = min_snr_weight_from_snr(snr, g, is_v_prediction);
        return loss.mul_scalar(w);
    }
    match weight_fn {
        LossWeight::Constant => Ok(loss.clone()),
        LossWeight::MinSnrGamma => {
            let w = min_snr_weight_from_snr(snr, 5.0, is_v_prediction);
            loss.mul_scalar(w)
        }
        LossWeight::P2 | LossWeight::Sigma => Ok(loss.clone()),
    }
}

/// Combined MSE + MAE + Huber loss with per-term strengths.
///
/// Default-off invariance: when called with `mse_strength=1.0, mae_strength=0.0,
/// huber_strength=0.0`, this returns exactly `(pred - target).square().mean()`,
/// byte-identical to the existing trainer loss line.
///
/// `pred` and `target` should already be at F32 (trainers up-cast before this
/// call, matching the existing pattern).
///
/// Huber loss with δ=1 is implemented in closed form using only operators
/// flame-core supports natively while preserving autograd:
///     Huber(x) = 0.5 * min(|x|, 1)^2  +  max(|x| - 1, 0)
/// For |x| ≤ 1 the second term is zero so this reduces to 0.5*x²; for |x| > 1
/// the first term saturates at 0.5 and the second contributes |x| - 1, giving
/// |x| - 0.5. Both branches are non-negative.
pub fn combined_loss(
    pred: &Tensor,
    target: &Tensor,
    mse_strength: f32,
    mae_strength: f32,
    huber_strength: f32,
) -> Result<Tensor> {
    let diff = pred.sub(target)?;

    // MSE term — short-circuit to the canonical formula when strengths
    // collapse to defaults so byte-equivalence is guaranteed.
    if mae_strength == 0.0 && huber_strength == 0.0 {
        let mse = diff.square()?.mean()?;
        if mse_strength == 1.0 {
            return Ok(mse);
        }
        return mse.mul_scalar(mse_strength);
    }

    let mut total: Option<Tensor> = None;
    if mse_strength != 0.0 {
        let mse = diff.square()?.mean()?.mul_scalar(mse_strength)?;
        total = Some(mse);
    }
    if mae_strength != 0.0 {
        let mae = diff.abs()?.mean()?.mul_scalar(mae_strength)?;
        total = Some(match total {
            Some(t) => t.add(&mae)?,
            None => mae,
        });
    }
    if huber_strength != 0.0 {
        // Closed-form Huber (δ=1): 0.5 * min(|x|, 1)^2 + max(|x| - 1, 0).
        // Pure arithmetic + autograd-recording ops (clamp, relu, square, mean).
        let abs = diff.abs()?;
        let abs_clamped = abs.clamp(0.0, 1.0)?;
        let sq_part = abs_clamped.square()?.mul_scalar(0.5)?;
        let lin_excess = abs.sub_scalar(1.0)?.relu()?;
        let huber = sq_part.add(&lin_excess)?.mean()?.mul_scalar(huber_strength)?;
        total = Some(match total {
            Some(t) => t.add(&huber)?,
            None => huber,
        });
    }
    // If every strength was 0, return a 0-loss scalar so backward is a no-op.
    match total {
        Some(t) => Ok(t),
        None => diff.square()?.mean()?.mul_scalar(0.0),
    }
}

// ── Legacy skeleton API kept compiling for any Phase-0 caller ───────────────
#[derive(Debug, Clone, Copy)]
pub enum LossWeightFn {
    Constant,
    MinSnrGamma(f32),
    DebiasedEstimation,
    P2(f32, f32),
    Sigma(f32),
}

/// Legacy skeleton — preserved so Phase-0 imports keep compiling.
/// Phase 1 callers should use [`apply_loss_weight`] instead.
pub fn apply(loss: &Tensor, weight_fn: LossWeightFn, sigma: f32) -> Result<Tensor> {
    match weight_fn {
        LossWeightFn::Constant => Ok(loss.clone()),
        LossWeightFn::MinSnrGamma(g) => loss.mul_scalar(min_snr_weight(sigma, g, true)),
        LossWeightFn::DebiasedEstimation => loss.mul_scalar(debiased_weight(sigma, true)),
        LossWeightFn::P2(_, _) | LossWeightFn::Sigma(_) => Ok(loss.clone()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn min_snr_eps_pred_caps_low_snr() {
        // For very low SNR (sigma close to 1, mostly noise), MIN-SNR-γ should
        // cap the weight at γ/snr — but since cap = min(snr, γ) = snr when
        // snr < γ, and weight = cap/snr = 1.0, low-SNR steps get weight 1.
        let w = min_snr_weight(0.99, 5.0, false);
        assert!((w - 1.0).abs() < 1e-6, "low-snr eps-pred weight = {}", w);
    }

    #[test]
    fn min_snr_eps_pred_caps_high_snr() {
        // High SNR (sigma small): snr huge → cap = γ; weight = γ/snr ≪ 1.
        let w = min_snr_weight(0.01, 5.0, false);
        let s: f32 = 0.01;
        let snr = ((1.0 - s) / s).powi(2);
        let expected = 5.0 / snr;
        assert!((w - expected).abs() < 1e-5, "high-snr eps-pred: {} != {}", w, expected);
    }

    #[test]
    fn min_snr_v_pred_uses_snr_plus_one() {
        // v-pred: weight = min(snr, γ) / (snr + 1)
        let w = min_snr_weight(0.5, 5.0, true);
        let snr = ((1.0 - 0.5_f32) / 0.5).powi(2); // = 1.0
        let expected = snr.min(5.0) / (snr + 1.0); // = 0.5
        assert!((w - expected).abs() < 1e-6);
    }

    #[test]
    fn debiased_is_inv_sqrt_snr_eps_pred() {
        // ε-prediction: w = 1 / sqrt(min(snr, 1e3))
        let w = debiased_weight(0.25, false);
        let snr = ((1.0 - 0.25_f32) / 0.25).powi(2); // = 9
        let expected = 1.0 / snr.sqrt(); // = 1/3
        assert!((w - expected).abs() < 1e-6);
    }

    #[test]
    fn debiased_v_pred_adds_one() {
        // v-prediction: w = 1 / sqrt(snr + 1)
        let w = debiased_weight(0.25, true);
        let snr = ((1.0 - 0.25_f32) / 0.25).powi(2); // = 9
        let expected = 1.0 / (snr + 1.0).sqrt(); // = 1/sqrt(10)
        assert!((w - expected).abs() < 1e-6);
    }

    #[test]
    fn debiased_clips_at_1e3() {
        // Very small sigma → snr ≫ 1e3, must clip.
        let w = debiased_weight(1e-6, false);
        let expected = 1.0 / (1e3_f32).sqrt();
        assert!((w - expected).abs() < 1e-4, "got {} expected {}", w, expected);
    }

    #[test]
    fn huber_no_negative_loss() {
        use flame_core::{global_cuda_device, Shape, Tensor};
        let device = global_cuda_device();
        // Test for |x| in {0.1, 0.3, 0.5, 1.0, 2.0}. Pred = x, target = 0
        // → diff = x. Huber with δ=1: 0.5 x² for |x| ≤ 1, |x| - 0.5 otherwise.
        for x in [0.1f32, 0.3, 0.5, 1.0, 2.0] {
            let pred = Tensor::from_vec(vec![x], Shape::from_dims(&[1]), device.clone()).unwrap();
            let target = Tensor::from_vec(vec![0.0], Shape::from_dims(&[1]), device.clone()).unwrap();
            // mse=0, mae=0, huber=1 → forces the Huber branch.
            let loss = combined_loss(&pred, &target, 0.0, 0.0, 1.0).unwrap();
            let v = loss.to_vec().unwrap()[0];
            let expected = if x.abs() <= 1.0 { 0.5 * x * x } else { x.abs() - 0.5 };
            assert!(v >= 0.0, "Huber returned negative for x={x}: {v}");
            assert!(
                (v - expected).abs() < 1e-5,
                "Huber x={x}: got {v}, expected {expected}"
            );
        }
    }

    #[test]
    fn ddpm_snr_form_matches_alphabar_form() {
        // DDPM SNR = ᾱ / (1 - ᾱ). For ᾱ = 0.5, SNR = 1.
        let snr = 1.0_f32;
        let w = min_snr_weight_from_snr(snr, 5.0, false);
        // cap = min(1, 5) = 1; weight = 1 / 1 = 1.
        assert!((w - 1.0).abs() < 1e-6);
    }
}

