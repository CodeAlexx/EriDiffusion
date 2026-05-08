//! Masked loss — multiply per-pixel diff by a foreground/region mask plus a
//! global blend weight, so background gradient contribution is reduced.
//!
//! Phase 3.
//!
//! Default-off invariance: when `weight = 0.0`, [`apply_loss_mask`] returns
//! `Ok(diff.clone())` and the trainer's loss path is byte-identical to the
//! prior commit.
//!
//! Reference: SimpleTuner `helpers/training/diffusion_model.py` masked-loss
//! handling and OneTrainer's per-pixel weight maps (`alpha_mask` field).

use std::collections::HashMap;

use flame_core::{DType, Result, Shape, Tensor};

/// Apply per-pixel mask weighting to a per-element diff `(pred - target)`.
///
/// `mask` shape must broadcast against `diff` (typically same shape as the
/// latent: `[B, C, H, W]`). `weight` blends the masked diff with the unmasked
/// diff:
///
/// ```text
///   result = weight * (mask * diff) + (1 - weight) * diff
/// ```
///
/// Edge cases:
///   - `weight <= 0.0`  → returns `diff.clone()` (no-op, byte-invariant).
///   - `weight >= 1.0`  → returns `mask * diff` (full masking).
///   - `0 < weight < 1` → blends.
///
/// The caller is responsible for `square()` + `mean()` after this call.
pub fn apply_loss_mask(diff: &Tensor, mask: &Tensor, weight: f32) -> Result<Tensor> {
    if weight <= 0.0 {
        return Ok(diff.clone());
    }
    let masked = diff.mul(mask)?;
    if weight >= 1.0 {
        return Ok(masked);
    }
    let blend_mask = masked.mul_scalar(weight)?;
    let blend_orig = diff.mul_scalar(1.0 - weight)?;
    blend_mask.add(&blend_orig)
}

/// Pull a mask tensor from a loaded sample's safetensors map.
///
/// Looks up `latent_mask` first, then `mask` for backwards compatibility.
/// When neither key is present, returns a tensor of ones with the supplied
/// `latent_shape`. The trainer is expected to have already up-cast the
/// latent to its working dtype, but the mask is returned as F32 so the
/// caller can multiply by an F32 diff in the masked-loss path.
pub fn load_mask(
    sample: &HashMap<String, Tensor>,
    latent_shape: &Shape,
    device: std::sync::Arc<cudarc::driver::CudaDevice>,
) -> Result<Tensor> {
    if let Some(m) = sample.get("latent_mask").or_else(|| sample.get("mask")) {
        // Up-cast (or retain) at F32 — caller's diff will be F32 in Klein/etc.
        return m.to_dtype(DType::F32);
    }
    Tensor::ones_dtype(latent_shape.clone(), DType::F32, device)
}

// ── Legacy skeleton API kept compiling for any pre-Phase-3 caller ───────────
/// Legacy skeleton — preserved so Phase-0 imports keep compiling.
/// Returns `loss.clone()` regardless of mask/weight. Phase-3 callers should
/// use [`apply_loss_mask`] on the per-element diff instead.
pub fn apply_mask(loss: &Tensor, _mask: &Tensor, _weight: f32) -> Result<Tensor> {
    Ok(loss.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use flame_core::{global_cuda_device, Shape, Tensor};

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn weight_zero_is_noop() {
        let device = global_cuda_device();
        let diff = Tensor::from_vec(vec![1.0, 2.0, -3.0], Shape::from_dims(&[3]), device.clone())
            .unwrap();
        let mask = Tensor::from_vec(vec![0.0, 1.0, 0.5], Shape::from_dims(&[3]), device.clone())
            .unwrap();
        let out = apply_loss_mask(&diff, &mask, 0.0).unwrap();
        let v = out.to_vec().unwrap();
        assert!(approx_eq(v[0], 1.0, 1e-6));
        assert!(approx_eq(v[1], 2.0, 1e-6));
        assert!(approx_eq(v[2], -3.0, 1e-6));
    }

    #[test]
    fn weight_one_full_mask() {
        let device = global_cuda_device();
        let diff = Tensor::from_vec(vec![1.0, 2.0, -3.0], Shape::from_dims(&[3]), device.clone())
            .unwrap();
        let mask = Tensor::from_vec(vec![0.0, 1.0, 0.5], Shape::from_dims(&[3]), device.clone())
            .unwrap();
        let out = apply_loss_mask(&diff, &mask, 1.0).unwrap();
        let v = out.to_vec().unwrap();
        assert!(approx_eq(v[0], 0.0, 1e-6));
        assert!(approx_eq(v[1], 2.0, 1e-6));
        assert!(approx_eq(v[2], -1.5, 1e-6));
    }

    #[test]
    fn weight_half_blends() {
        let device = global_cuda_device();
        let diff =
            Tensor::from_vec(vec![1.0, 2.0], Shape::from_dims(&[2]), device.clone()).unwrap();
        let mask =
            Tensor::from_vec(vec![0.0, 1.0], Shape::from_dims(&[2]), device.clone()).unwrap();
        let out = apply_loss_mask(&diff, &mask, 0.5).unwrap();
        let v = out.to_vec().unwrap();
        // result = 0.5 * (mask * diff) + 0.5 * diff
        // index 0: 0.5*0 + 0.5*1 = 0.5
        // index 1: 0.5*2 + 0.5*2 = 2
        assert!(approx_eq(v[0], 0.5, 1e-6));
        assert!(approx_eq(v[1], 2.0, 1e-6));
    }

    #[test]
    fn ones_mask_at_full_weight_is_identity() {
        let device = global_cuda_device();
        let diff =
            Tensor::from_vec(vec![1.0, -2.0, 3.0], Shape::from_dims(&[3]), device.clone()).unwrap();
        let mask = Tensor::ones(Shape::from_dims(&[3]), device.clone()).unwrap();
        let out = apply_loss_mask(&diff, &mask, 1.0).unwrap();
        let v = out.to_vec().unwrap();
        assert!(approx_eq(v[0], 1.0, 1e-6));
        assert!(approx_eq(v[1], -2.0, 1e-6));
        assert!(approx_eq(v[2], 3.0, 1e-6));
    }

    #[test]
    fn load_mask_falls_back_to_ones() {
        let device = global_cuda_device();
        let map: HashMap<String, Tensor> = HashMap::new();
        let shape = Shape::from_dims(&[2, 3]);
        let mask = load_mask(&map, &shape, device.clone()).unwrap();
        assert_eq!(mask.shape().dims(), &[2, 3]);
        let v = mask.to_vec().unwrap();
        for x in v {
            assert!(approx_eq(x, 1.0, 1e-6));
        }
    }
}
