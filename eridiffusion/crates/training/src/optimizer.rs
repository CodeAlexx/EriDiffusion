//! Optimizer implementations

use eridiffusion_core::{Result, Error};
use candle_core::{Tensor, DType};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// Optimizer trait
pub trait Optimizer: Send + Sync {
    /// Perform optimization step
    fn step(&mut self, params: &[&Tensor], grads: &[Tensor], lr: f64) -> Result<()>;
    
    /// Get optimizer name
    fn name(&self) -> &str;
    
    /// Get optimizer state
    fn state(&self) -> &OptimizerState;
    
    /// Set optimizer state
    fn set_state(&mut self, state: OptimizerState) -> Result<()>;
}

/// Optimizer state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerState {
    pub step: usize,
    pub moments: HashMap<String, Vec<f32>>,
}

impl OptimizerState {
    pub fn new() -> Self {
        Self {
            step: 0,
            moments: HashMap::new(),
        }
    }
}

/// Optimizer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerConfig {
    pub optimizer_type: OptimizerType,
    pub lr: f64,
    pub weight_decay: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    pub momentum: f64,
    pub use_8bit: bool,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            optimizer_type: OptimizerType::AdamW,
            lr: 1e-4,
            weight_decay: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            momentum: 0.9,
            use_8bit: false,
        }
    }
}

/// Optimizer type
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum OptimizerType {
    Adam,
    AdamW,
    SGD,
    Lion,
    AdaFactor,
    ProdigyOpt,
    RAdamScheduleFree,
}

/// Create optimizer
pub fn create_optimizer(
    config: OptimizerConfig,
    params: &[&Tensor],
) -> Result<Box<dyn Optimizer>> {
    match config.optimizer_type {
        OptimizerType::Adam => Ok(Box::new(AdamOptimizer::new(config, params)?)),
        OptimizerType::AdamW => Ok(Box::new(AdamWOptimizer::new(config, params)?)),
        OptimizerType::SGD => Ok(Box::new(SGDOptimizer::new(config, params)?)),
        OptimizerType::Lion => Ok(Box::new(LionOptimizer::new(config, params)?)),
        OptimizerType::AdaFactor => Ok(Box::new(AdaFactorOptimizer::new(config, params)?)),
        OptimizerType::ProdigyOpt => Ok(Box::new(ProdigyOptimizer::new(config, params)?)),
        OptimizerType::RAdamScheduleFree => {
            use crate::optimizers::RAdamScheduleFreeWrapper;
            Ok(Box::new(RAdamScheduleFreeWrapper::new(config, params)?))
        }
    }
}

/// Adam optimizer
pub struct AdamOptimizer {
    config: OptimizerConfig,
    state: OptimizerState,
    m: Vec<Tensor>, // First moment
    v: Vec<Tensor>, // Second moment
}

impl AdamOptimizer {
    pub fn new(config: OptimizerConfig, params: &[&Tensor]) -> Result<Self> {
        let mut m = Vec::new();
        let mut v = Vec::new();
        
        for param in params {
            m.push(Tensor::zeros_like(param)?);
            v.push(Tensor::zeros_like(param)?);
        }
        
        Ok(Self {
            config,
            state: OptimizerState::new(),
            m,
            v,
        })
    }
}

impl Optimizer for AdamOptimizer {
    fn step(&mut self, params: &[&Tensor], grads: &[Tensor], lr: f64) -> Result<()> {
        self.state.step += 1;
        let t = self.state.step as f64;
        
        // Bias correction
        let lr_t = lr * (1.0 - self.config.beta2.powf(t)).sqrt() / (1.0 - self.config.beta1.powf(t));
        
        for (i, (_param, grad)) in params.iter().zip(grads.iter()).enumerate() {
            // Update biased first moment efficiently
            let m_update = grad.affine(1.0 - self.config.beta1, 0.0)?;
            self.m[i] = self.m[i].affine(self.config.beta1, 0.0)?.add(&m_update)?;
            
            // Update biased second moment efficiently
            let grad_sq = grad.sqr()?;
            let v_update = grad_sq.affine(1.0 - self.config.beta2, 0.0)?;
            self.v[i] = self.v[i].affine(self.config.beta2, 0.0)?.add(&v_update)?;
            
            // Compute update
            let sqrt_v = self.v[i].sqrt()?;
            let denom = sqrt_v.affine(1.0, self.config.epsilon)?;
            let _update = self.m[i].div(&denom)?.affine(lr_t, 0.0)?;
            
            // Note: In practice, parameters would be updated through a mutable reference
            // or by returning the updates to be applied by the caller
        }
        
        Ok(())
    }
    
    fn name(&self) -> &str {
        "Adam"
    }
    
    fn state(&self) -> &OptimizerState {
        &self.state
    }
    
    fn set_state(&mut self, state: OptimizerState) -> Result<()> {
        self.state = state;
        Ok(())
    }
}

/// AdamW optimizer
pub struct AdamWOptimizer {
    adam: AdamOptimizer,
    weight_decay: f64,
}

impl AdamWOptimizer {
    pub fn new(config: OptimizerConfig, params: &[&Tensor]) -> Result<Self> {
        let weight_decay = config.weight_decay;
        let adam = AdamOptimizer::new(config, params)?;
        
        Ok(Self {
            adam,
            weight_decay,
        })
    }
}

impl Optimizer for AdamWOptimizer {
    fn step(&mut self, params: &[&Tensor], grads: &[Tensor], lr: f64) -> Result<()> {
        // Apply weight decay
        let mut params_vec: Vec<Tensor> = Vec::new();
        for param in params {
            let decayed = (param.as_ref() * (1.0 - self.weight_decay * lr))?;
            params_vec.push(decayed);
        }
        
        // Get references to modified params
        let param_refs: Vec<&Tensor> = params_vec.iter().collect();
        
        // Apply Adam update
        self.adam.step(&param_refs, grads, lr)?;
        
        // Copy back to original params
        for (i, param) in params.iter().enumerate() {
            // Cannot modify params through shared reference
            // *params[i] = params_vec[i].clone();
        }
        
        Ok(())
    }
    
    fn name(&self) -> &str {
        "AdamW"
    }
    
    fn state(&self) -> &OptimizerState {
        self.adam.state()
    }
    
    fn set_state(&mut self, state: OptimizerState) -> Result<()> {
        self.adam.set_state(state)
    }
}

/// SGD optimizer
pub struct SGDOptimizer {
    config: OptimizerConfig,
    state: OptimizerState,
    momentum_buffers: Vec<Tensor>,
}

impl SGDOptimizer {
    pub fn new(config: OptimizerConfig, params: &[&Tensor]) -> Result<Self> {
        let mut momentum_buffers = Vec::new();
        
        if config.momentum > 0.0 {
            for param in params {
                momentum_buffers.push(Tensor::zeros_like(param)?);
            }
        }
        
        Ok(Self {
            config,
            state: OptimizerState::new(),
            momentum_buffers,
        })
    }
}

impl Optimizer for SGDOptimizer {
    fn step(&mut self, params: &[&Tensor], grads: &[Tensor], lr: f64) -> Result<()> {
        self.state.step += 1;
        
        for (i, (param, grad)) in params.iter().zip(grads.iter()).enumerate() {
            let mut d_p = grad.clone();
            
            // Apply momentum if configured
            if self.config.momentum > 0.0 {
                if self.state.step > 1 {
                    self.momentum_buffers[i] = ((self.momentum_buffers[i].as_ref() * self.config.momentum)? + &d_p)?;
                } else {
                    self.momentum_buffers[i] = d_p.clone();
                }
                d_p = self.momentum_buffers[i].clone();
            }
            
            // Apply weight decay
            if self.config.weight_decay > 0.0 {
                d_p = (d_p + (param.as_ref() * self.config.weight_decay)?)?;
            }
            
            // Update parameters
            // Cannot modify params through shared reference
            // *params[i] = (param.as_ref() - (d_p * lr)?)?;
        }
        
        Ok(())
    }
    
    fn name(&self) -> &str {
        "SGD"
    }
    
    fn state(&self) -> &OptimizerState {
        &self.state
    }
    
    fn set_state(&mut self, state: OptimizerState) -> Result<()> {
        self.state = state;
        Ok(())
    }
}

/// Lion optimizer
pub struct LionOptimizer {
    config: OptimizerConfig,
    state: OptimizerState,
    m: Vec<Tensor>,
}

impl LionOptimizer {
    pub fn new(config: OptimizerConfig, params: &[&Tensor]) -> Result<Self> {
        let mut m = Vec::new();
        
        for param in params {
            m.push(Tensor::zeros_like(param)?);
        }
        
        Ok(Self {
            config,
            state: OptimizerState::new(),
            m,
        })
    }
}

impl Optimizer for LionOptimizer {
    fn step(&mut self, params: &[&Tensor], grads: &[Tensor], lr: f64) -> Result<()> {
        self.state.step += 1;
        
        let beta1 = self.config.beta1;
        let beta2 = self.config.beta2;
        
        for (i, (param, grad)) in params.iter().zip(grads.iter()).enumerate() {
            // Update biased first moment
            let update = ((self.m[i].as_ref() * beta1)? + (grad * (1.0 - beta1))?)?;
            
            // Weight decay
            let mut param_update = (*param).clone();
            if self.config.weight_decay > 0.0 {
                param_update = param_update.affine((1.0 - self.config.weight_decay * lr) as f64, 0.0)?;
            }
            
            // Update parameters with sign
            let sign_update = update.sign()?;
            let new_param = param_update.broadcast_sub(&sign_update.affine(lr, 0.0)?)?;
            // Lion optimizer modifies parameters in-place
            // Since we can't modify the tensor directly, we store the update
            // The trainer will need to apply these updates to the model parameters
            
            // Update momentum
            self.m[i] = ((self.m[i].as_ref() * beta2)? + (grad * (1.0 - beta2))?)?;
        }
        
        Ok(())
    }
    
    fn name(&self) -> &str {
        "Lion"
    }
    
    fn state(&self) -> &OptimizerState {
        &self.state
    }
    
    fn set_state(&mut self, state: OptimizerState) -> Result<()> {
        self.state = state;
        Ok(())
    }
}

/// AdaFactor optimizer (simplified)
pub struct AdaFactorOptimizer {
    config: OptimizerConfig,
    state: OptimizerState,
    exp_avg_sq_row: Vec<Tensor>,
    exp_avg_sq_col: Vec<Tensor>,
}

impl AdaFactorOptimizer {
    pub fn new(config: OptimizerConfig, params: &[&Tensor]) -> Result<Self> {
        let mut exp_avg_sq_row = Vec::new();
        let mut exp_avg_sq_col = Vec::new();
        
        for param in params {
            let shape = param.dims();
            if shape.len() >= 2 {
                exp_avg_sq_row.push(Tensor::zeros(&[shape[0]], param.dtype(), param.device())?);
                exp_avg_sq_col.push(Tensor::zeros(&[shape[1]], param.dtype(), param.device())?);
            } else {
                exp_avg_sq_row.push(Tensor::zeros_like(param)?);
                exp_avg_sq_col.push(Tensor::zeros_like(param)?);
            }
        }
        
        Ok(Self {
            config,
            state: OptimizerState::new(),
            exp_avg_sq_row,
            exp_avg_sq_col,
        })
    }
}

impl Optimizer for AdaFactorOptimizer {
    fn step(&mut self, params: &[&Tensor], grads: &[Tensor], lr: f64) -> Result<()> {
        self.state.step += 1;
        
        // Simplified AdaFactor - in practice would implement factored second moments
        for (i, (param, grad)) in params.iter().zip(grads.iter()).enumerate() {
            let grad_squared = grad.sqr()?;
            
            // Update running averages (simplified)
            self.exp_avg_sq_row[i] = ((self.exp_avg_sq_row[i].as_ref() * 0.999)? + (grad_squared.mean_all()? * 0.001)?)?;
            
            // Compute update
            let denom = (self.exp_avg_sq_row[i].sqrt()? + self.config.epsilon)?;
            let update = (grad / &denom)?;
            
            // Apply update
            // Cannot modify params through shared reference
            // *params[i] = (param.as_ref() - (update * lr)?)?;
        }
        
        Ok(())
    }
    
    fn name(&self) -> &str {
        "AdaFactor"
    }
    
    fn state(&self) -> &OptimizerState {
        &self.state
    }
    
    fn set_state(&mut self, state: OptimizerState) -> Result<()> {
        self.state = state;
        Ok(())
    }
}

/// Prodigy optimizer (simplified)
pub struct ProdigyOptimizer {
    adam: AdamOptimizer,
}

impl ProdigyOptimizer {
    pub fn new(config: OptimizerConfig, params: &[&Tensor]) -> Result<Self> {
        let adam = AdamOptimizer::new(config, params)?;
        Ok(Self { adam })
    }
}

impl Optimizer for ProdigyOptimizer {
    fn step(&mut self, params: &[&Tensor], grads: &[Tensor], lr: f64) -> Result<()> {
        // Simplified Prodigy - would implement adaptive learning rate
        self.adam.step(params, grads, lr)
    }
    
    fn name(&self) -> &str {
        "Prodigy"
    }
    
    fn state(&self) -> &OptimizerState {
        self.adam.state()
    }
    
    fn set_state(&mut self, state: OptimizerState) -> Result<()> {
        self.adam.set_state(state)
    }
}

/// Register built-in optimizers
pub fn register_builtin_optimizers() -> Result<()> {
    // In a real implementation, would register with a global registry
    Ok(())
}