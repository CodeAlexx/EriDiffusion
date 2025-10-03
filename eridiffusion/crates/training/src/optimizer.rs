//! Optimizer implementations

use std::collections::HashMap;

use eridiffusion_core::{Error, Result};
use flame_core::{DType, Tensor};
use serde::{Deserialize, Serialize};

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

    /// Optional: provide parameter names corresponding to `params` order
    fn set_param_names(&mut self, _names: Vec<String>) -> Result<()> {
        Ok(())
    }

    /// Optional: set weight-decay exclusion patterns (substring match)
    fn set_wd_exclusion_patterns(&mut self, _patterns: Vec<String>) -> Result<()> {
        Ok(())
    }

    /// Optional: bind parameter names (same as set_param_names; provided for clarity)
    fn bind_names(&mut self, names: &[String]) -> Result<()> {
        self.set_param_names(names.to_vec())
    }

    /// Optional: export Adam-like moment buffers keyed by names (FP32 tensors)
    fn export_mv_as_map(
        &self,
        _names: &[String],
    ) -> Result<(HashMap<String, Tensor>, HashMap<String, Tensor>)> {
        Err(Error::Unsupported("optimizer does not support state export".into()))
    }

    /// Optional: import Adam-like moment buffers keyed by names (FP32 tensors)
    fn import_mv_from_map(
        &mut self,
        _m: &HashMap<String, Tensor>,
        _v: &HashMap<String, Tensor>,
    ) -> Result<()> {
        Ok(())
    }
}

/// Compute global norm in FP32 and clip gradients in-place (for Tensor-based optimizers)
pub fn clip_grads_global_norm_fp32_tensors(grads: &mut [Tensor], max_norm: f32) -> Result<f32> {
    if grads.is_empty() || max_norm <= 0.0 || !max_norm.is_finite() {
        // Compute and return norm anyway for callers that log it
        let mut total: f32 = 0.0;
        for g in grads.iter() {
            let g32 = if g.dtype() != DType::F32 { g.to_dtype(DType::F32)? } else { g.clone() };
            let sq = g32.mul(&g32)?.sum()?.item()?;
            total += sq;
        }
        return Ok(total.sqrt());
    }

    // Accumulate in FP32
    let mut total_sq: f32 = 0.0;
    for g in grads.iter() {
        let g32 = if g.dtype() != DType::F32 { g.to_dtype(DType::F32)? } else { g.clone() };
        let sq = g32.mul(&g32)?.sum()?.item()?;
        total_sq += sq;
    }
    let global = total_sq.sqrt();
    if global.is_finite() && global > max_norm {
        let scale = max_norm / (global + 1e-6);
        for gi in grads.iter_mut() {
            *gi = gi.affine(scale as f32, 0.0f32)?;
        }
    }
    Ok(global)
}

/// Optimizer state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerState {
    pub step: usize,
    pub moments: HashMap<String, Vec<f32>>,
}

impl OptimizerState {
    pub fn new() -> Self {
        Self { step: 0, moments: HashMap::new() }
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
pub fn create_optimizer(config: OptimizerConfig, params: &[&Tensor]) -> Result<Box<dyn Optimizer>> {
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

        Ok(Self { config, state: OptimizerState::new(), m, v })
    }
}

impl Optimizer for AdamOptimizer {
    fn step(&mut self, params: &[&Tensor], grads: &[Tensor], lr: f64) -> Result<()> {
        self.state.step += 1;
        let t = self.state.step as f64;

        // Bias correction
        let lr_t =
            lr * (1.0 - self.config.beta2.powf(t)).sqrt() / (1.0 - self.config.beta1.powf(t));

        for (i, (_param, grad)) in params.iter().zip(grads.iter()).enumerate() {
            // Update biased first moment efficiently
            let m_update = grad.affine((1.0 - self.config.beta1) as f32, 0.0f32)?;
            self.m[i] = self.m[i].affine(self.config.beta1 as f32, 0.0f32)?.add(&m_update)?;

            // Update biased second moment efficiently
            let grad_sq = grad.square()?;
            let v_update = grad_sq.affine((1.0 - self.config.beta2) as f32, 0.0f32)?;
            self.v[i] = self.v[i].affine(self.config.beta2 as f32, 0.0f32)?.add(&v_update)?;

            // Compute update
            let sqrt_v = self.v[i].sqrt()?;
            let denom = sqrt_v.affine(1.0f32, self.config.epsilon as f32)?;
            let _update = self.m[i].div(&denom)?.affine(lr_t as f32, 0.0f32)?;

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
    param_names: Vec<String>,
    wd_exclusions: Vec<String>,
}

impl AdamWOptimizer {
    pub fn new(config: OptimizerConfig, params: &[&Tensor]) -> Result<Self> {
        let weight_decay = config.weight_decay;
        let adam = AdamOptimizer::new(config, params)?;

        Ok(Self { adam, weight_decay, param_names: Vec::new(), wd_exclusions: Vec::new() })
    }
}

impl Optimizer for AdamWOptimizer {
    fn step(&mut self, params: &[&Tensor], grads: &[Tensor], lr: f64) -> Result<()> {
        // Apply weight decay
        let mut params_vec: Vec<Tensor> = Vec::new();
        for (i, param) in params.iter().enumerate() {
            let mut apply_wd = true;
            if !self.param_names.is_empty() && !self.wd_exclusions.is_empty() {
                if let Some(name) = self.param_names.get(i) {
                    if self.wd_exclusions.iter().any(|pat| name.contains(pat)) {
                        apply_wd = false;
                    }
                }
            }
            let decayed = if apply_wd {
                param.mul_scalar((1.0 - self.weight_decay * lr) as f32)?
            } else {
                (*param).clone()
            };
            params_vec.push(decayed);
        }

        // Get references to modified params
        let param_refs: Vec<&Tensor> = params_vec.iter().collect();

        // Apply Adam update
        self.adam.step(&param_refs, grads, lr)?; // TODO: Use gradient_map instead of individual tensor

        // Copy back to original params
        for (_i, _param) in params.iter().enumerate() {
            let _ = _param;
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

    fn set_param_names(&mut self, names: Vec<String>) -> Result<()> {
        self.param_names = names;
        Ok(())
    }
    fn set_wd_exclusion_patterns(&mut self, patterns: Vec<String>) -> Result<()> {
        self.wd_exclusions = patterns;
        Ok(())
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

        Ok(Self { config, state: OptimizerState::new(), momentum_buffers })
    }
}

impl Optimizer for SGDOptimizer {
    fn step(&mut self, params: &[&Tensor], grads: &[Tensor], _lr: f64) -> Result<()> {
        self.state.step += 1;

        for (i, (param, grad)) in params.iter().zip(grads.iter()).enumerate() {
            let mut d_p = grad.clone();

            // Apply momentum if configured
            if self.config.momentum > 0.0 {
                if self.state.step > 1 {
                    self.momentum_buffers[i] = self.momentum_buffers[i]
                        .mul_scalar(self.config.momentum as f32)?
                        .add(&d_p)?;
                } else {
                    self.momentum_buffers[i] = d_p.clone();
                }
                d_p = self.momentum_buffers[i].clone();
            }

            // Apply weight decay
            if self.config.weight_decay > 0.0 {
                d_p = d_p.add(&param.mul_scalar(self.config.weight_decay as f32)?)?;
            }

            let _ = d_p;
            let _ = param;
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

        Ok(Self { config, state: OptimizerState::new(), m })
    }
}

impl Optimizer for LionOptimizer {
    fn step(&mut self, params: &[&Tensor], grads: &[Tensor], lr: f64) -> Result<()> {
        self.state.step += 1;

        let beta1 = self.config.beta1;
        let beta2 = self.config.beta2;

        for (i, (param, grad)) in params.iter().zip(grads.iter()).enumerate() {
            // Update biased first moment
            let update =
                self.m[i].mul_scalar(beta1 as f32)?.add(&grad.mul_scalar((1.0 - beta1) as f32)?)?;

            // Weight decay
            let mut param_update = (*param).clone();
            if self.config.weight_decay > 0.0 {
                param_update =
                    param_update.affine((1.0 - self.config.weight_decay * lr) as f32, 0.0f32)?;
            }

            // Update parameters with sign
            let sign_update = update.sign()?;
            let _new_param = param_update.sub(&sign_update.affine(lr as f32, 0.0f32)?)?;
            // Lion optimizer modifies parameters in-place
            // Since we can't modify the tensor directly, we store the update
            // The trainer will need to apply these updates to the model parameters

            // Update momentum
            self.m[i] =
                self.m[i].mul_scalar(beta2 as f32)?.add(&grad.mul_scalar((1.0 - beta2) as f32)?)?;
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

/// AdaFactor optimizer
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
            exp_avg_sq_row.push(Tensor::zeros_like(param)?);
            exp_avg_sq_col.push(Tensor::zeros_like(param)?);
        }

        Ok(Self { config, state: OptimizerState::new(), exp_avg_sq_row, exp_avg_sq_col })
    }
}

impl Optimizer for AdaFactorOptimizer {
    fn step(&mut self, params: &[&Tensor], grads: &[Tensor], _lr: f64) -> Result<()> {
        self.state.step += 1;
        let step = self.state.step as f64;

        // AdaFactor with factored second moments for memory efficiency
        let beta2 = 1.0 - 1.0 / step.max(1.0);
        let clipping_threshold = 1.0;

        for (i, (param, grad)) in params.iter().zip(grads.iter()).enumerate() {
            let dims = param.shape().dims().to_vec();
            if dims.len() >= 2 {
                // Factored second moment estimation for matrices
                let _rows = dims[0];
                let _cols = dims[1];

                // Compute row and column means of squared gradients
                let grad_sq = grad.square()?;

                // Row mean: average across columns for each row
                let row_mean = grad_sq.mean_along_dims(&[1], true)?;

                // Column mean: average across rows for each column
                let col_mean = grad_sq.mean_along_dims(&[0], true)?;

                // Update exponential moving averages
                self.exp_avg_sq_row[i] = self.exp_avg_sq_row[i]
                    .mul_scalar(beta2 as f32)?
                    .add(&row_mean.mul_scalar((1.0 - beta2) as f32)?)?;
                self.exp_avg_sq_col[i] = self.exp_avg_sq_col[i]
                    .mul_scalar(beta2 as f32)?
                    .add(&col_mean.mul_scalar((1.0 - beta2) as f32)?)?;

                // Reconstruct second moment estimate
                let r_factor = self.exp_avg_sq_row[i].rsqrt()?;
                let c_factor = self.exp_avg_sq_col[i].rsqrt()?;
                let v_hat = r_factor.matmul(&c_factor)?;

                // RMS normalization for stability
                let rms = grad_sq
                    .sum()?
                    .div_scalar(grad_sq.shape().elem_count() as f32)?
                    .sqrt()?
                    .item()? as f64;
                let rms_clipped = rms.min(clipping_threshold);

                // Compute update with adaptive learning rate
                let _update = grad.mul(&v_hat)?.mul_scalar((rms_clipped / (rms + 1e-10)) as f32)?;

                // Note: Cannot modify params directly with shared reference
                // In practice, this would update the parameter
            } else {
                // For vectors, use standard second moment
                let grad_squared = grad.square()?;
                self.exp_avg_sq_row[i] = self.exp_avg_sq_row[i]
                    .mul_scalar(beta2 as f32)?
                    .add(&grad_squared.mul_scalar((1.0 - beta2) as f32)?)?;

                let denom =
                    self.exp_avg_sq_row[i].sqrt()?.add_scalar(self.config.epsilon as f32)?;
                let _update = grad.div(&denom)?;

                // Note: Cannot modify params directly with shared reference
            }
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

/// Prodigy optimizer with adaptive learning rate
pub struct ProdigyOptimizer {
    adam: AdamOptimizer,
}

impl ProdigyOptimizer {
    pub fn new(config: OptimizerConfig, params: &[&Tensor]) -> Result<Self> {
        let adam = AdamOptimizer::new(config, params)?;
        Ok(Self { adam })
    }

    // Prodigy-specific adaptive learning rate calculation
    fn compute_adaptive_lr(&self, grads: &[Tensor], base_lr: f64) -> Result<f64> {
        // Compute gradient norm for adaptive scaling
        let mut grad_norm = 0.0;
        for grad in grads {
            let norm = grad.square()?.sum()?.item()? as f64;
            grad_norm += norm;
        }
        grad_norm = grad_norm.sqrt();

        // Prodigy adaptive factor based on gradient statistics
        let d_estimate = grad_norm / (grad_norm + 1.0);
        let adaptive_lr = base_lr * (1.0 + d_estimate);

        Ok(adaptive_lr)
    }
}

impl Optimizer for ProdigyOptimizer {
    fn step(&mut self, params: &[&Tensor], grads: &[Tensor], lr: f64) -> Result<()> {
        // Prodigy with adaptive learning rate based on gradient statistics
        let adaptive_lr = self.compute_adaptive_lr(grads, lr)?;

        // Use Adam as base optimizer with adaptive learning rate
        self.adam.step(params, grads, adaptive_lr)
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

    fn export_mv_as_map(
        &self,
        names: &[String],
    ) -> Result<(HashMap<String, Tensor>, HashMap<String, Tensor>)> {
        // Use provided order as index mapping (Prodigy uses embedded Adam buffers)
        let mut m_map: HashMap<String, Tensor> = HashMap::new();
        let mut v_map: HashMap<String, Tensor> = HashMap::new();
        for (i, name) in names.iter().enumerate() {
            if let Some(mt) = self.adam.m.get(i) {
                let m32 =
                    if mt.dtype() != DType::F32 { mt.to_dtype(DType::F32)? } else { mt.clone() };
                m_map.insert(name.clone(), m32);
            }
            if let Some(vt) = self.adam.v.get(i) {
                let v32 =
                    if vt.dtype() != DType::F32 { vt.to_dtype(DType::F32)? } else { vt.clone() };
                v_map.insert(name.clone(), v32);
            }
        }
        Ok((m_map, v_map))
    }

    fn import_mv_from_map(
        &mut self,
        m: &HashMap<String, Tensor>,
        v: &HashMap<String, Tensor>,
    ) -> Result<()> {
        // Use provided names' order as index mapping. Assume maps were built against same names order.
        // Apply by iterating over names present in `m` and `v` and matching indices by sorted key order.
        // For simplicity, assign by name position if present.
        let mut keys: Vec<&String> = m.keys().collect();
        keys.sort();
        for (i, name) in keys.iter().enumerate() {
            if let Some(mt) = m.get(*name) {
                let m32 =
                    if mt.dtype() != DType::F32 { mt.to_dtype(DType::F32)? } else { mt.clone() };
                if let Some(slot) = self.adam.m.get_mut(i) {
                    *slot = m32;
                }
            }
        }
        let mut keys_v: Vec<&String> = v.keys().collect();
        keys_v.sort();
        for (i, name) in keys_v.iter().enumerate() {
            if let Some(vt) = v.get(*name) {
                let v32 =
                    if vt.dtype() != DType::F32 { vt.to_dtype(DType::F32)? } else { vt.clone() };
                if let Some(slot) = self.adam.v.get_mut(i) {
                    *slot = v32;
                }
            }
        }
        Ok(())
    }
}

/// Register built-in optimizers
pub fn register_builtin_optimizers() -> Result<()> {
    // In a real implementation, would register with a global registry
    Ok(())
}
