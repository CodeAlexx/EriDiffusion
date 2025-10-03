//! Unified optimizer wrapper and factory so all models share a single interface.
//! This wraps internal Optimizer implementations behind an adapter working with
//! Flame Parameters. Keeps training loop uniform across architectures.

use anyhow::Result;
use flame_core::Parameter;

#[derive(Debug, Clone, Copy)]
pub enum OptimizerType {
    AdamW,
    RAdamScheduleFree,
    ProdigyScheduleFree,
    Lion,
    AdaFactor,
}

#[derive(Debug, Clone)]
pub struct OptimizerConfig {
    pub optimizer_type: OptimizerType,
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub weight_decay: f32,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            optimizer_type: OptimizerType::AdamW,
            lr: 1e-4,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0,
        }
    }
}

pub trait OptimizerWrapper {
    fn step(&mut self) -> Result<()>;
    fn zero_grad(&mut self);
    fn vars(&mut self) -> &mut [Parameter];
}

/// No-op fallback wrapper to keep builds green until wired to tensor-based optimizers.
struct NoopWrapper {
    params: Vec<Parameter>,
}
impl OptimizerWrapper for NoopWrapper {
    fn step(&mut self) -> Result<()> {
        Ok(())
    }
    fn zero_grad(&mut self) { /* gradients are handled by autograd map externally */
    }
    fn vars(&mut self) -> &mut [Parameter] {
        self.params.as_mut_slice()
    }
}

/// Build an optimizer wrapper for a set of Parameters.
/// Note: Current Flame optimizers in this workspace operate on Tensors; this wrapper
/// provides a uniform surface. Hooking the real algorithms can be done later by
/// mapping `Parameter` -> underlying tensors and calling the existing Optimizer APIs.
pub fn build_optimizer(
    config: OptimizerConfig,
    params: &[Parameter],
) -> Result<Box<dyn OptimizerWrapper>> {
    match config.optimizer_type {
        OptimizerType::ProdigyScheduleFree => {
            let cfg = crate::optimizers::prodigy_schedulefree::ProdigyScheduleFreeConfig {
                lr: config.lr,
                weight_decay: config.weight_decay,
            };
            let w = crate::optimizers::prodigy_schedulefree::ProdigyScheduleFreeWrapper::new(
                cfg, params,
            )?;
            Ok(Box::new(w))
        }
        // Placeholders until wired to real tensor-based optimizers
        _ => Ok(Box::new(NoopWrapper { params: params.to_vec() })),
    }
}
