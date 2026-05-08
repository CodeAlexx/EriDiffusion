//! Slider-LoRA loss helper (Phase 5+ feature).
//!
//! Status: WIRED (Klein only via the `train_slider_klein` binary).
//!
//! A Slider-LoRA is a single-axis concept controller. Given paired prompts
//! (`positive` / `negative`), the training procedure makes a LoRA whose
//! magnitude controls how strongly the model leans toward the positive
//! direction. At inference, scaling the LoRA by α ∈ [-1, +1] interpolates
//! between the two concepts.
//!
//! Reference paper: "Concept Sliders" — Gandikota et al., 2023
//!   <https://arxiv.org/abs/2311.12092>
//!
//! Reference implementation:
//!   `SimpleTuner/helpers/distillation/slider/`
//!
//! Loss formulation (per step, all tensors in F32 to keep gradient math
//! numerically clean):
//!
//! ```text
//!   direction   = ε_pos - ε_neg                    # detached
//!   target_pos  = ε_pos      + scale * direction    # detached
//!   target_neg  = ε_neg      - scale * direction    # detached
//!   loss = mean((ε_pos_lora - target_pos)²)
//!        + mean((ε_neg_lora - target_neg)²)
//! ```
//!
//! `eps_pos` and `eps_neg` are produced under `AutogradContext::no_grad` by
//! the caller, so gradients flow only through `eps_pos_lora` /
//! `eps_neg_lora`.

use flame_core::Tensor;

use crate::Result;

/// Compute slider-LoRA loss given the four predictions.
///
/// All inputs may be in any dtype; the math is performed in F32 to match the
/// loss-weight pipeline. The base predictions (`eps_pos`, `eps_neg`) are
/// expected to come from a `no_grad` forward pass, so they detach naturally
/// when subtracted from the with-LoRA branches.
pub fn slider_loss(
    eps_pos_lora: &Tensor,
    eps_neg_lora: &Tensor,
    eps_pos: &Tensor,
    eps_neg: &Tensor,
    scale: f32,
) -> Result<Tensor> {
    use flame_core::DType;

    // Cast to F32 only when needed to avoid creating no-op autograd nodes
    // that can confuse the leaf-id lookup at backward time.
    let cast = |t: &Tensor| -> Result<Tensor> {
        if t.dtype() == DType::F32 {
            Ok(t.clone())
        } else {
            Ok(t.to_dtype(DType::F32)?)
        }
    };
    let eps_pos_lora_f = cast(eps_pos_lora)?;
    let eps_neg_lora_f = cast(eps_neg_lora)?;
    let eps_pos_f = cast(eps_pos)?;
    let eps_neg_f = cast(eps_neg)?;

    let direction = eps_pos_f.sub(&eps_neg_f)?;
    let scaled = direction.mul_scalar(scale)?;
    let target_pos = eps_pos_f.add(&scaled)?;
    let target_neg = eps_neg_f.sub(&scaled)?;

    let loss_pos = eps_pos_lora_f.sub(&target_pos)?.square()?.mean()?;
    let loss_neg = eps_neg_lora_f.sub(&target_neg)?.square()?.mean()?;
    Ok(loss_pos.add(&loss_neg)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use flame_core::{global_cuda_device, Shape, Tensor};

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn scale_zero_with_matching_outputs_is_zero_loss() {
        // When scale=0, target_pos = eps_pos and target_neg = eps_neg.
        // If eps_pos_lora == eps_pos and eps_neg_lora == eps_neg, loss = 0.
        let device = global_cuda_device();
        let eps_pos =
            Tensor::from_vec(vec![1.0, -2.0, 3.0], Shape::from_dims(&[3]), device.clone())
                .unwrap();
        let eps_neg =
            Tensor::from_vec(vec![0.5, 0.5, -1.0], Shape::from_dims(&[3]), device.clone())
                .unwrap();
        let eps_pos_lora = eps_pos.clone();
        let eps_neg_lora = eps_neg.clone();
        let loss = slider_loss(&eps_pos_lora, &eps_neg_lora, &eps_pos, &eps_neg, 0.0).unwrap();
        let v = loss.to_vec().unwrap();
        assert!(approx_eq(v[0], 0.0, 1e-6), "loss={}", v[0]);
    }

    #[test]
    fn nonzero_scale_with_lora_eq_base_is_predictable() {
        // When eps_pos_lora == eps_pos and eps_neg_lora == eps_neg but scale > 0,
        // the loss equals 2 * scale^2 * mean(direction^2).
        // Here eps_pos = [1, -2, 3], eps_neg = [0, 0, 0] → direction = [1, -2, 3]
        // mean(direction^2) = (1 + 4 + 9) / 3 = 14/3.
        // scale = 0.5 → contribution per side = 0.25 * 14/3 = 7/6.
        // Total = 14/6 ≈ 2.3333.
        let device = global_cuda_device();
        let eps_pos =
            Tensor::from_vec(vec![1.0, -2.0, 3.0], Shape::from_dims(&[3]), device.clone())
                .unwrap();
        let eps_neg = Tensor::zeros(Shape::from_dims(&[3]), device.clone()).unwrap();
        let loss = slider_loss(&eps_pos, &eps_neg, &eps_pos, &eps_neg, 0.5).unwrap();
        let v = loss.to_vec().unwrap();
        let expected = 2.0 * 0.25 * (1.0 + 4.0 + 9.0) / 3.0;
        assert!(
            approx_eq(v[0], expected, 1e-5),
            "got {} expected {}",
            v[0],
            expected
        );
    }

    #[test]
    fn random_inputs_produce_finite_loss() {
        let device = global_cuda_device();
        let shape = Shape::from_dims(&[2, 4, 8]);
        let eps_pos_lora = Tensor::randn(shape.clone(), 0.0, 1.0, device.clone()).unwrap();
        let eps_neg_lora = Tensor::randn(shape.clone(), 0.0, 1.0, device.clone()).unwrap();
        let eps_pos = Tensor::randn(shape.clone(), 0.0, 1.0, device.clone()).unwrap();
        let eps_neg = Tensor::randn(shape.clone(), 0.0, 1.0, device.clone()).unwrap();
        let loss =
            slider_loss(&eps_pos_lora, &eps_neg_lora, &eps_pos, &eps_neg, 2.0).unwrap();
        let v = loss.to_vec().unwrap();
        assert!(v[0].is_finite(), "loss not finite: {}", v[0]);
        assert!(v[0] >= 0.0, "loss negative: {}", v[0]);
    }

    #[test]
    fn loss_propagates_requires_grad() {
        // When the LoRA inputs require_grad, the resulting loss must too.
        // When all four inputs are detached, the loss must NOT.
        let device = global_cuda_device();
        let shape = Shape::from_dims(&[4]);

        let eps_pos_lora =
            Tensor::from_vec(vec![0.1, 0.2, -0.3, 0.4], shape.clone(), device.clone())
                .unwrap()
                .requires_grad_(true);
        let eps_neg_lora =
            Tensor::from_vec(vec![-0.2, 0.3, 0.1, -0.5], shape.clone(), device.clone())
                .unwrap()
                .requires_grad_(true);
        let eps_pos =
            Tensor::from_vec(vec![1.0, -1.0, 0.5, 0.5], shape.clone(), device.clone()).unwrap();
        let eps_neg =
            Tensor::from_vec(vec![0.0, 0.0, 0.0, 0.0], shape.clone(), device.clone()).unwrap();

        let loss_with_grad =
            slider_loss(&eps_pos_lora, &eps_neg_lora, &eps_pos, &eps_neg, 1.0).unwrap();
        assert!(
            loss_with_grad.requires_grad(),
            "loss must require_grad when LoRA inputs do"
        );

        // All-detached case: no autograd attachment → loss is a leaf scalar.
        let loss_detached = slider_loss(&eps_pos, &eps_neg, &eps_pos, &eps_neg, 1.0).unwrap();
        assert!(
            !loss_detached.requires_grad(),
            "loss must NOT require_grad when all inputs are detached"
        );
    }
}
