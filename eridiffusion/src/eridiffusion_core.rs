use flame_core::device::Device;
use flame_core::optimizers::{Adam, SGD};
use flame_core::{DType, Result, Shape, Tensor};

// Compatibility module for eridiffusion_core imports
// Maps disabled crate imports to local types

// Re-export common types

/// Model inputs for diffusion models
#[derive(Debug)]
pub struct ModelInputs {
    pub latents: Tensor,
    pub timesteps: Tensor,
    pub context: Tensor,
    pub additional_inputs: std::collections::HashMap<String, Tensor>,
}

/// Model outputs
#[derive(Debug)]
pub struct ModelOutput {
    pub noise_pred: Tensor,
}

/// Base trait for diffusion models
pub trait DiffusionModel {
    fn forward(&self, inputs: &ModelInputs) -> flame_core::Result<Tensor>;
}
