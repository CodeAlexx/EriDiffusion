//! Trait for Flux models with or without LoRA

use candle_core::{Tensor, Result, Var};

/// Common trait for Flux models
pub trait FluxModel: Send + Sync {
    /// Forward pass
    fn forward(
        &self,
        img: &Tensor,
        img_ids: &Tensor,
        txt: &Tensor,
        txt_ids: &Tensor,
        timesteps: &Tensor,
        y: &Tensor,
        guidance: Option<&Tensor>,
    ) -> Result<Tensor>;
    
    /// Get trainable parameters (empty for base model, LoRA params for adapted model)
    fn trainable_parameters(&self) -> Vec<Var>;
}