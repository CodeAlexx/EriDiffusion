use anyhow::Result;
use flame_core::{Tensor, Shape, Device};

/// Stable identifiers for LoRA-adapted parameters in Chroma.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ParamId {
    // Attention projections
    AttnQ,
    AttnK,
    AttnV,
    AttnO,
    // MLP projections
    MlpFc1,
    MlpFc2,
}

/// Simple LoRA adapter container
pub struct LoRAAdapter {
    pub a: Tensor, // [in, r]
    pub b: Tensor, // [r, out]
}

impl LoRAAdapter {
    /// Zero-init LoRA so initial delta is zero.
    pub fn zero(a_shape: &[usize], b_shape: &[usize], device: &Device) -> Result<Self> {
        let a = Tensor::zeros(Shape::from_dims(a_shape), device.cuda_device().clone())?;
        let b = Tensor::zeros(Shape::from_dims(b_shape), device.cuda_device().clone())?;
        Ok(Self { a, b })
    }
}

/// Trait for models that expose LoRA-only trainable params
pub trait HasLoRA {
    /// Return stable ParamIds and their adapters
    fn lora_params(&self) -> Vec<(ParamId, &LoRAAdapter)>;

    /// Set grads requirement: only LoRA adapters require grads
    fn restrict_grads_to_lora(&self);
}
