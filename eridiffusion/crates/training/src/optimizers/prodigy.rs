//! Prodigy optimizer implementation for Candle
//! Based on "Prodigy: An Expeditiously Adaptive Parameter-Free Learner"
//! https://arxiv.org/abs/2306.06101

use candle_core::{DType, Device, Result, Tensor, D, Var};
use candle_nn::{Optimizer as CanOptimizer, VarMap};
type CanResult<T> = std::result::Result<T, candle_core::Error>;
use std::collections::HashMap;

/// Prodigy optimizer - Parameter-free adaptive learning rate optimizer
/// Automatically estimates and adjusts learning rate during training
pub struct Prodigy {
    var_map: VarMap,
    d0: f64,              // Initial D estimate
    growth_rate: f64,     // Growth rate for D
    beta1: f64,           // Exponential moving average coefficient for gradient
    beta2: f64,           // Exponential moving average coefficient for squared gradient
    beta3: f64,           // Exponential moving average coefficient for D estimation
    epsilon: f64,         // Small constant for numerical stability
    weight_decay: f64,    // Weight decay coefficient
    step: usize,          // Current step count
    
    // Optimizer states
    exp_avg: HashMap<String, Tensor>,           // First moment estimates (m)
    exp_avg_sq: HashMap<String, Tensor>,        // Second moment estimates (v)
    d_numerator: HashMap<String, Tensor>,       // Numerator for D estimation
    d_denominator: HashMap<String, Tensor>,     // Denominator for D estimation
    
    // Global learning rate state
    d_estimate: f64,      // Current D estimate (adaptive learning rate)
    s_estimate: f64,      // Current S estimate (scale factor)
}

impl Prodigy {
    pub fn new(
        var_map: VarMap,
        d0: f64,
        growth_rate: f64,
        beta1: f64,
        beta2: f64,
        beta3: f64,
        epsilon: f64,
        weight_decay: f64,
    ) -> Result<Self> {
        Ok(Self {
            var_map,
            d0,
            growth_rate,
            beta1,
            beta2,
            beta3,
            epsilon,
            weight_decay,
            step: 0,
            exp_avg: HashMap::new(),
            exp_avg_sq: HashMap::new(),
            d_numerator: HashMap::new(),
            d_denominator: HashMap::new(),
            d_estimate: d0,
            s_estimate: 0.0,
        })
    }
    
    /// Initialize optimizer states for a variable
    fn init_state(&mut self, var_id: String, shape: &[usize], device: &Device) -> Result<()> {
        if !self.exp_avg.contains_key(&var_id) {
            let zeros = Tensor::zeros(shape, DType::F32, device)?;
            self.exp_avg.insert(var_id.clone(), zeros.clone());
            self.exp_avg_sq.insert(var_id.clone(), zeros.clone());
            self.d_numerator.insert(var_id.clone(), zeros.clone());
            self.d_denominator.insert(var_id.clone(), zeros);
        }
        Ok(())
    }
    
    /// Update the global D estimate using all parameter groups
    fn update_d_estimate(&mut self) -> Result<()> {
        let mut global_d_numerator = 0.0;
        let mut global_d_denominator = 0.0;
        
        // Aggregate D estimates from all parameter groups
        for var_id in self.d_numerator.keys() {
            let d_num = &self.d_numerator[var_id];
            let d_den = &self.d_denominator[var_id];
            
            // Sum over all elements
            let num_sum: f32 = d_num.flatten_all()?.to_vec1::<f32>()?.iter().sum();
            let den_sum: f32 = d_den.flatten_all()?.to_vec1::<f32>()?.iter().sum();
            
            global_d_numerator += num_sum as f64;
            global_d_denominator += den_sum as f64;
        }
        
        // Update D estimate with growth rate and numerical stability
        if global_d_denominator > self.epsilon {
            let d_hat = global_d_numerator / global_d_denominator;
            self.d_estimate = (self.d_estimate * self.growth_rate).max(d_hat);
        }
        
        Ok(())
    }
    
    /// Get the current effective learning rate
    pub fn current_learning_rate(&self) -> f64 {
        self.d_estimate
    }
    
    /// Get current S estimate (useful for monitoring)
    pub fn current_s_estimate(&self) -> f64 {
        self.s_estimate
    }
}

impl CanOptimizer for Prodigy {
    type Config = ProdigyConfig;

    fn new(vars: Vec<Var>, config: Self::Config) -> Result<Self> {
        let mut var_map = VarMap::new();
        for var in vars {
            var_map.insert(var);
        }
        
        Self::new(
            var_map,
            config.d0,
            config.growth_rate,
            config.beta1,
            config.beta2,
            config.beta3,
            config.epsilon,
            config.weight_decay,
        )
    }

    fn learning_rate(&self) -> f64 {
        self.d_estimate
    }

    fn set_learning_rate(&mut self, _lr: f64) {
        // Prodigy automatically manages learning rate, but we can adjust d_estimate
        // In practice, you might want to scale d0 instead
        eprintln!("Warning: Prodigy manages learning rate automatically. Consider adjusting d0 instead.");
    }

    fn step(&mut self, grads: &candle_core::backprop::GradStore) -> Result<()> {
        self.step += 1;
        
        let beta1 = self.beta1;
        let beta2 = self.beta2;
        let beta3 = self.beta3;
        let eps = self.epsilon;
        let weight_decay = self.weight_decay;
        let step = self.step as f64;
        
        // Bias correction factors
        let bias_correction1 = 1.0 - beta1.powf(step);
        let bias_correction2 = 1.0 - beta2.powf(step);
        let bias_correction3 = 1.0 - beta3.powf(step);
        
        // Process each parameter
        for (idx, var) in self.var_map.all_vars().iter().enumerate() {
            let var_id = format!("var_{}", idx);
            let param = var.as_tensor();
            
            // Get gradient
            let grad = match grads.get(var) {
                Some(g) => g,
                None => continue,
            };
            
            // Apply weight decay to gradient (L2 regularization)
            let grad = if weight_decay > 0.0 {
                (grad + (param * weight_decay)?)?
            } else {
                grad.clone()
            };
            
            // Initialize states if needed
            self.init_state(var_id.clone(), param.dims(), param.device())?;
            
            // Get states
            let exp_avg = &mut self.exp_avg.get_mut(&var_id).unwrap();
            let exp_avg_sq = &mut self.exp_avg_sq.get_mut(&var_id).unwrap();
            let d_numerator = &mut self.d_numerator.get_mut(&var_id).unwrap();
            let d_denominator = &mut self.d_denominator.get_mut(&var_id).unwrap();
            
            // Update biased first moment estimate (exponential moving average of gradient)
            *exp_avg = ((exp_avg.as_ref() * beta1)? + (grad.as_ref() * (1.0 - beta1))?)?;
            
            // Update biased second raw moment estimate (exponential moving average of squared gradient)
            let grad_sq = grad.sqr()?;
            *exp_avg_sq = ((exp_avg_sq.as_ref() * beta2)? + (grad_sq * (1.0 - beta2))?)?;
            
            // Bias-corrected first and second moment estimates
            let exp_avg_corrected = (exp_avg.as_ref() / bias_correction1)?;
            let exp_avg_sq_corrected = (exp_avg_sq.as_ref() / bias_correction2)?;
            
            // Compute the preconditioned gradient (similar to Adam)
            let denom = (exp_avg_sq_corrected.sqrt()? + eps)?;
            let preconditioned_grad = exp_avg_corrected.div(&denom)?;
            
            // Update D estimation components
            // d_numerator tracks the dot product of consecutive preconditioned gradients
            // d_denominator tracks the squared norm of preconditioned gradients
            if self.step > 1 {
                // Get previous preconditioned gradient (stored in d_numerator temporarily)
                let prev_preconditioned = d_numerator.as_ref();
                
                // Compute dot product for numerator
                let dot_product = (prev_preconditioned * &preconditioned_grad)?;
                *d_numerator = ((d_numerator.as_ref() * beta3)? + (dot_product * (1.0 - beta3))?)?;
                
                // Compute squared norm for denominator
                let sq_norm = preconditioned_grad.sqr()?;
                *d_denominator = ((d_denominator.as_ref() * beta3)? + (sq_norm * (1.0 - beta3))?)?;
            } else {
                // First step: initialize with current preconditioned gradient
                *d_numerator = preconditioned_grad.clone();
                *d_denominator = preconditioned_grad.sqr()?;
            }
        }
        
        // Update global D estimate
        self.update_d_estimate()?;
        
        // Apply parameter updates using current D estimate
        for (idx, var) in self.var_map.all_vars().iter().enumerate() {
            let var_id = format!("var_{}", idx);
            let param = var.as_tensor();
            
            if let Some(grad) = grads.get(var) {
                // Apply weight decay to gradient
                let grad = if weight_decay > 0.0 {
                    (grad + (param * weight_decay)?)?
                } else {
                    grad.clone()
                };
                
                // Get states
                let exp_avg = &self.exp_avg[&var_id];
                let exp_avg_sq = &self.exp_avg_sq[&var_id];
                
                // Bias-corrected estimates
                let exp_avg_corrected = (exp_avg / bias_correction1)?;
                let exp_avg_sq_corrected = (exp_avg_sq / bias_correction2)?;
                
                // Compute update
                let denom = (exp_avg_sq_corrected.sqrt()? + eps)?;
                let update = exp_avg_corrected.div(&denom)?;
                
                // Apply update with current D estimate as learning rate
                let new_param = (param - (update * self.d_estimate)?)?;
                var.set(&new_param)?;
            }
        }
        
        Ok(())
    }
    
    fn backward_step(&mut self, loss: &Tensor) -> Result<()> {
        let grads = loss.backward()?;
        self.step(&grads)
    }
}

/// Configuration for Prodigy optimizer
#[derive(Debug, Clone)]
pub struct ProdigyConfig {
    pub d0: f64,            // Initial D estimate (default: 1e-6)
    pub growth_rate: f64,   // Growth rate for D (default: inf, meaning no upper bound)
    pub beta1: f64,         // Exponential moving average coefficient for gradient (default: 0.9)
    pub beta2: f64,         // Exponential moving average coefficient for squared gradient (default: 0.999)
    pub beta3: f64,         // Exponential moving average coefficient for D estimation (default: 0.999)
    pub epsilon: f64,       // Small constant for numerical stability (default: 1e-8)
    pub weight_decay: f64,  // Weight decay coefficient (default: 0.0)
}

impl Default for ProdigyConfig {
    fn default() -> Self {
        Self {
            d0: 1e-6,
            growth_rate: f64::INFINITY,
            beta1: 0.9,
            beta2: 0.999,
            beta3: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0,
        }
    }
}

/// Prodigy with different presets for common use cases
impl ProdigyConfig {
    /// Configuration for language model training
    pub fn for_language_models() -> Self {
        Self {
            d0: 1e-6,
            growth_rate: f64::INFINITY,
            beta1: 0.9,
            beta2: 0.999,
            beta3: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.01,
        }
    }
    
    /// Configuration for computer vision tasks
    pub fn for_vision() -> Self {
        Self {
            d0: 1e-6,
            growth_rate: f64::INFINITY,
            beta1: 0.9,
            beta2: 0.999,
            beta3: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.05,
        }
    }
    
    /// Configuration for fine-tuning (more conservative)
    pub fn for_finetuning() -> Self {
        Self {
            d0: 1e-7,
            growth_rate: 10.0,  // Limited growth for stability
            beta1: 0.9,
            beta2: 0.999,
            beta3: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.01,
        }
    }
}

/// Utility function to create Prodigy with preset configurations
pub fn create_prodigy_preset(
    vars: Vec<Var>,
    preset: &str,
) -> Result<Prodigy> {
    let config = match preset {
        "language" | "llm" | "nlp" => ProdigyConfig::for_language_models(),
        "vision" | "cv" | "image" => ProdigyConfig::for_vision(),
        "finetune" | "ft" => ProdigyConfig::for_finetuning(),
        "default" => ProdigyConfig::default(),
        _ => return Err(candle_core::Error::Msg(format!("Unknown Prodigy preset: {}", preset))),
    };
    
    Prodigy::new(vars, config)
}

/// Monitor Prodigy optimization progress
pub struct ProdigyMonitor {
    d_history: Vec<f64>,
    s_history: Vec<f64>,
    step_history: Vec<usize>,
}

impl ProdigyMonitor {
    pub fn new() -> Self {
        Self {
            d_history: Vec::new(),
            s_history: Vec::new(),
            step_history: Vec::new(),
        }
    }
    
    /// Record current optimizer state
    pub fn record(&mut self, optimizer: &Prodigy) {
        self.d_history.push(optimizer.current_learning_rate());
        self.s_history.push(optimizer.current_s_estimate());
        self.step_history.push(optimizer.step);
    }
    
    /// Get summary statistics
    pub fn summary(&self) -> ProdigySummary {
        if self.d_history.is_empty() {
            return ProdigySummary::default();
        }
        
        let d_mean = self.d_history.iter().sum::<f64>() / self.d_history.len() as f64;
        let d_min = self.d_history.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let d_max = self.d_history.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        
        ProdigySummary {
            total_steps: self.step_history.len(),
            d_mean,
            d_min,
            d_max,
            final_d: *self.d_history.last().unwrap_or(&0.0),
        }
    }
}

#[derive(Debug, Default)]
pub struct ProdigySummary {
    pub total_steps: usize,
    pub d_mean: f64,
    pub d_min: f64,
    pub d_max: f64,
    pub final_d: f64,
}

impl std::fmt::Display for ProdigySummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Prodigy Optimization Summary:")?;
        writeln!(f, "  Total steps: {}", self.total_steps)?;
        writeln!(f, "  Learning rate (D) statistics:")?;
        writeln!(f, "    Mean: {:.2e}", self.d_mean)?;
        writeln!(f, "    Min:  {:.2e}", self.d_min)?;
        writeln!(f, "    Max:  {:.2e}", self.d_max)?;
        writeln!(f, "    Final: {:.2e}", self.final_d)?;
        Ok(())
    }
}