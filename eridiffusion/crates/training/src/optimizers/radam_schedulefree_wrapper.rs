//! Wrapper to adapt schedule-free RAdam to our Optimizer trait (minimal stub).
use eridiffusion_core::Result;
use flame_core::Tensor;

use crate::optimizer::{Optimizer, OptimizerConfig, OptimizerState};

/// Wrapper for RAdamScheduleFree that implements our Optimizer trait
pub struct RAdamScheduleFreeWrapper {
    #[allow(dead_code)]
    config: OptimizerConfig,
    state: OptimizerState,
}

impl RAdamScheduleFreeWrapper {
    pub fn new(config: OptimizerConfig, _params: &[&Tensor]) -> Result<Self> {
        Ok(Self { config, state: OptimizerState::new() })
    }
}

impl Optimizer for RAdamScheduleFreeWrapper {
    fn step(&mut self, params: &[&Tensor], grads: &[Tensor], lr: f64) -> Result<()> {
        let _ = (params, grads, lr); // no-op stub
        self.state.step += 1;
        Ok(())
    }

    fn name(&self) -> &str {
        "RAdamScheduleFree"
    }

    fn state(&self) -> &OptimizerState {
        &self.state
    }

    fn set_state(&mut self, state: OptimizerState) -> Result<()> {
        self.state = state;
        Ok(())
    }
}
