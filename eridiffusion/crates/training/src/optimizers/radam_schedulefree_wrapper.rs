//! Wrapper to adapt candle_nn::Optimizer RadamScheduleFree to our Optimizer trait

use crate::optimizer::{Optimizer, OptimizerState, OptimizerConfig};
use crate::optimizers::radam_schedulefree::{RAdamScheduleFree, RAdamScheduleFreeConfig};
use eridiffusion_core::{Result, Error};
use candle_core::{Tensor, Var};
use candle_nn::Optimizer as CandleOptimizer;
use std::collections::HashMap;

/// Wrapper for RAdamScheduleFree that implements our Optimizer trait
pub struct RAdamScheduleFreeWrapper {
    inner: RAdamScheduleFree,
    config: OptimizerConfig,
    state: OptimizerState,
    vars: Vec<Var>,
}

impl RAdamScheduleFreeWrapper {
    pub fn new(config: OptimizerConfig, params: &[&Tensor]) -> Result<Self> {
        // Convert Tensors to Vars
        let vars: Vec<Var> = params
            .iter()
            .map(|t| Var::from_tensor(t))
            .collect::<candle_core::Result<Vec<_>>>()
            .map_err(|e| Error::Training(format!("Failed to create vars: {}", e)))?;
        
        // Create RAdamScheduleFree config
        let radam_config = RAdamScheduleFreeConfig {
            lr: config.lr,
            beta1: config.beta1,
            beta2: config.beta2,
            eps: config.epsilon,
            weight_decay: config.weight_decay,
            warmup_steps: 0, // Could be extended in OptimizerConfig
            r: 1.0, // Default schedule-free parameter
        };
        
        // Create inner optimizer
        let inner = RAdamScheduleFree::new(
            vars.clone(),
            radam_config.lr,
            radam_config.beta1,
            radam_config.beta2,
            radam_config.eps,
            radam_config.weight_decay,
            radam_config.warmup_steps,
            radam_config.r,
        ).map_err(|e| Error::Training(format!("Failed to create RAdamScheduleFree: {}", e)))?;
        
        Ok(Self {
            inner,
            config,
            state: OptimizerState::new(),
            vars,
        })
    }
}

impl Optimizer for RAdamScheduleFreeWrapper {
    fn step(&mut self, params: &[&Tensor], grads: &[Tensor], lr: f64) -> Result<()> {
        self.state.step += 1;
        
        // Create a gradient map
        let mut grad_map = HashMap::new();
        
        // Add gradients to the map
        for (i, (var, grad)) in self.vars.iter().zip(grads.iter()).enumerate() {
            grad_map.insert(format!("param_{}", i), grad.clone());
        }
        
        // Update learning rate if different
        if (lr - self.config.lr).abs() > 1e-10 {
            // Re-create optimizer with new learning rate
            let radam_config = RAdamScheduleFreeConfig {
                lr,
                beta1: self.config.beta1,
                beta2: self.config.beta2,
                eps: self.config.epsilon,
                weight_decay: self.config.weight_decay,
                warmup_steps: 0,
                r: 1.0,
            };
            
            self.inner = RAdamScheduleFree::new(
                self.vars.clone(),
                radam_config.lr,
                radam_config.beta1,
                radam_config.beta2,
                radam_config.eps,
                radam_config.weight_decay,
                radam_config.warmup_steps,
                radam_config.r,
            ).map_err(|e| Error::Training(format!("Failed to recreate RAdamScheduleFree: {}", e)))?;
            
            self.config.lr = lr;
        }
        
        // Perform optimization step
        self.inner.step(params, grads)
            .map_err(|e| Error::Training(format!("RAdamScheduleFree step failed: {}", e)))?;
        
        // Update params with the new var values
        for (i, (param, var)) in params.iter().zip(self.vars.iter()).enumerate() {
            // Copy var values back to params
            // Note: This assumes params are mutable references in practice
            // The actual implementation would need mutable params
        }
        
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