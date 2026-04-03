use crate::Result;
use flame_core::{DType, Device, Tensor, TensorId};
use std::collections::HashMap;

/// Trait for sanity checking trainable models
pub trait TrainerSanityCheck {
    /// Assert that all base weights are frozen and only LoRA weights are trainable
    fn assert_trainable_is_only_lora(&self) -> Result<()>;

    /// Assert that all weights are finite (no NaN/Inf)
    fn assert_all_weights_finite(&self) -> Result<()>;

    /// Assert that no LoRA weights are zero-initialized
    fn assert_no_zero_lora(&self) -> Result<()>;

    /// Assert that gradients only exist for LoRA parameters
    fn assert_only_lora_has_gradients(&self, gradients: &HashMap<TensorId, Tensor>) -> Result<()>;

    /// Assert that loss is valid (finite and scalar)
    fn assert_loss_is_valid(&self, loss: &Tensor) -> Result<()>;
}

/// Generic implementation for models with named tensors
pub struct ModelSanityChecker<'a> {
    pub named_tensors: Vec<(&'a str, &'a Tensor)>,
}

impl<'a> TrainerSanityCheck for ModelSanityChecker<'a> {
    fn assert_trainable_is_only_lora(&self) -> Result<()> {
        for (name, tensor) in &self.named_tensors {
            let is_lora = name.contains("lora");
            let requires_grad = tensor.requires_grad();

            if !is_lora && requires_grad {
                panic!("❌ Base weight `{}` is not frozen! requires_grad={}", name, requires_grad);
            }

            if is_lora && !requires_grad {
                panic!("❌ LoRA weight `{}` is frozen! requires_grad={}", name, requires_grad);
            }
        }

        println!("✅ All base weights are frozen, only LoRA weights are trainable");
        Ok(())
    }

    fn assert_all_weights_finite(&self) -> Result<()> {
        for (name, tensor) in &self.named_tensors {
            // Check for NaN/Inf by comparing with itself (NaN != NaN)
            let data = tensor.to_vec1::<f32>()?;
            let has_nan_inf = data.iter().any(|&x| !x.is_finite());

            if has_nan_inf {
                panic!("❌ Weight `{}` contains NaN or Inf values!", name);
            }

            // Additional check for extreme values
            let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let min_val = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));

            if max_val.abs() > 1e6 || min_val.abs() > 1e6 {
                eprintln!(
                    "⚠️  Weight `{}` has extreme values: [{:.2e}, {:.2e}]",
                    name, min_val, max_val
                );
            }
        }

        println!("✅ All weights are finite");
        Ok(())
    }

    fn assert_no_zero_lora(&self) -> Result<()> {
        for (name, tensor) in &self.named_tensors {
            if name.contains("lora") {
                let data = tensor.to_vec1::<f32>()?;
                let abs_sum: f32 = data.iter().map(|x| x.abs()).sum();

                if abs_sum == 0.0 {
                    panic!("❌ LoRA tensor `{}` is all zeros!", name);
                }

                // Also check if values are too small (essentially zero)
                if abs_sum < 1e-10 * data.len() as f32 {
                    eprintln!(
                        "⚠️  LoRA tensor `{}` has extremely small values (abs_sum={})",
                        name, abs_sum
                    );
                }
            }
        }

        println!("✅ No LoRA weights are zero-initialized");
        Ok(())
    }

    fn assert_only_lora_has_gradients(&self, gradients: &HashMap<TensorId, Tensor>) -> Result<()> {
        for (name, tensor) in &self.named_tensors {
            let tensor_id = tensor.id();
            let has_gradient = gradients.contains_key(&tensor_id);
            let is_lora = name.contains("lora");

            if is_lora && !has_gradient {
                panic!("❌ Missing gradient for LoRA tensor `{}`", name);
            }

            if !is_lora && has_gradient {
                panic!("❌ Unexpected gradient for base tensor `{}`", name);
            }
        }

        println!("✅ Only LoRA parameters have gradients");
        Ok(())
    }

    fn assert_loss_is_valid(&self, loss: &Tensor) -> Result<()> {
        // Check shape is scalar
        let shape = loss.shape();
        if shape.elem_count() != 1 {
            panic!("❌ Loss is not scalar! Shape: {:?}", shape);
        }

        // Check value is finite
        let loss_val = loss.to_scalar::<f32>()?;
        if !loss_val.is_finite() {
            panic!("❌ Loss is NaN or Inf: {}", loss_val);
        }

        // Warn on extreme values
        if loss_val.abs() > 1e6 {
            eprintln!("⚠️  Loss has extreme value: {:.2e}", loss_val);
        }

        Ok(())
    }
}

/// Helper function to run all sanity checks
pub fn run_training_sanity_checks<T: TrainerSanityCheck>(
    model: &T,
    loss: &Tensor,
    gradients: &HashMap<TensorId, Tensor>,
    step: usize,
) -> Result<()> {
    println!("\n🔍 Running sanity checks at step {}...", step);

    // Only run expensive checks every N steps or at the beginning
    if step == 0 || step % 100 == 0 {
        model.assert_trainable_is_only_lora()?;
        model.assert_all_weights_finite()?;
        model.assert_no_zero_lora()?;
    }

    // Always check loss and gradients
    model.assert_loss_is_valid(loss)?;
    model.assert_only_lora_has_gradients(gradients)?;

    println!("✅ All sanity checks passed!\n");
    Ok(())
}

/// Optimizer sanity check
pub fn assert_optimizer_only_touches_lora(optimizer_params: &[(&str, &Tensor)]) -> Result<()> {
    for (name, _tensor) in optimizer_params {
        if !name.contains("lora") {
            panic!("❌ Optimizer includes non-LoRA weight `{}`", name);
        }
    }

    println!("✅ Optimizer only contains LoRA parameters");
    Ok(())
}
