use flame_core::device::Device;
use flame_core::Result;
use flame_core::{DType, Error, Parameter};
use serde::{Deserialize, Serialize};

use crate::trainers::adam8bit::Adam8bit;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerConfig {
    pub optimizer_type: String,
    pub learning_rate: f32,
    pub weight_decay: Option<f32>,
    pub betas: Option<(f32, f32)>,
    pub eps: Option<f32>,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            optimizer_type: "adamw8bit".to_string(),
            learning_rate: 1e-4,
            weight_decay: Some(0.01),
            betas: Some((0.9, 0.999)),
            eps: Some(1e-8),
        }
    }
}

/// Create optimizer based on configuration
pub fn create_optimizer(
    config: &OptimizerConfig,
    parameters: Vec<&Parameter>,
) -> flame_core::Result<Box<dyn OptimizerTrait>> {
    match config.optimizer_type.as_str() {
        "adam8bit" | "adamw8bit" => {
            let optimizer = Adam8bit::new(
                parameters,
                config.learning_rate,
                config.betas.unwrap_or((0.9, 0.999)),
                config.eps.unwrap_or(1e-8),
                config.weight_decay.unwrap_or(0.0),
            )?;
            Ok(Box::new(optimizer))
        }
        "sgd" => {
            // TODO: Implement SGD optimizer
            return Err(flame_core::Error::InvalidOperation(
                "SGD optimizer not yet implemented".to_string(),
            ));
        }
        _ => {
            return Err(flame_core::Error::InvalidOperation(format!(
                "Unknown optimizer: {}",
                config.optimizer_type
            )))
        }
    }
}

/// Trait for all optimizers
pub trait OptimizerTrait {
    fn step(&mut self, param: &Parameter, grad: &flame_core::Tensor) -> flame_core::Result<()>;
    fn zero_grad(&mut self) -> flame_core::Result<()>;
    fn get_lr(&self) -> f32;
    fn set_lr(&mut self, lr: f32);
    fn state_dict(
        &self,
    ) -> flame_core::Result<std::collections::HashMap<String, flame_core::Tensor>>;
    fn load_state_dict(
        &mut self,
        state: std::collections::HashMap<String, flame_core::Tensor>,
    ) -> flame_core::Result<()>;
}

impl OptimizerTrait for Adam8bit {
    fn step(&mut self, param: &Parameter, grad: &flame_core::Tensor) -> flame_core::Result<()> {
        // Adam8bit implementation would handle this
        Ok(())
    }

    fn zero_grad(&mut self) -> flame_core::Result<()> {
        Ok(())
    }

    fn get_lr(&self) -> f32 {
        1e-4 // Placeholder
    }

    fn set_lr(&mut self, lr: f32) {
        // Set learning rate
    }

    fn state_dict(
        &self,
    ) -> flame_core::Result<std::collections::HashMap<String, flame_core::Tensor>> {
        Ok(std::collections::HashMap::new())
    }

    fn load_state_dict(
        &mut self,
        state: std::collections::HashMap<String, flame_core::Tensor>,
    ) -> flame_core::Result<()> {
        Ok(())
    }
}
