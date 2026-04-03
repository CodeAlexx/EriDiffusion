//! ProdigyScheduleFree optimizer wrapper (compile-safe stub).
//! Integrates with the unified OptimizerWrapper so all models share one API.

use anyhow::Result;
use flame_core::Parameter;

use super::wrapper::OptimizerWrapper;

#[derive(Debug, Clone)]
pub struct ProdigyScheduleFreeConfig {
    pub lr: f32,
    pub weight_decay: f32,
}

impl Default for ProdigyScheduleFreeConfig {
    fn default() -> Self {
        Self { lr: 1e-4, weight_decay: 0.0 }
    }
}

pub struct ProdigyScheduleFreeWrapper {
    params: Vec<Parameter>,
    pub cfg: ProdigyScheduleFreeConfig,
    step: usize,
}

impl ProdigyScheduleFreeWrapper {
    pub fn new(cfg: ProdigyScheduleFreeConfig, params: &[Parameter]) -> Result<Self> {
        Ok(Self { cfg, params: params.to_vec(), step: 0 })
    }
}

impl OptimizerWrapper for ProdigyScheduleFreeWrapper {
    fn step(&mut self) -> Result<()> {
        self.step += 1;
        Ok(())
    }
    fn zero_grad(&mut self) { /* grads managed externally */
    }
    fn vars(&mut self) -> &mut [Parameter] {
        self.params.as_mut_slice()
    }
}
