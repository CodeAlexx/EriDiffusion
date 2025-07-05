//! Lion (EvoLved Sign Momentum) optimizer implementation for Candle
//! Based on "Symbolic Discovery of Optimization Algorithms"
//! https://arxiv.org/abs/2302.06675

use candle_core::{DType, Device, Result, Tensor, D, Var};
use candle_nn::{Optimizer as CanOptimizer, VarMap};
use std::collections::HashMap;

/// Lion (EvoLved Sign Momentum) optimizer
/// 
/// A simple and memory-efficient optimizer that uses only the sign of gradients
/// and momentum. Lion requires only momentum states, saving 33% memory compared to Adam.
/// 
/// Key characteristics:
/// - Uses sign of interpolated gradient and momentum
/// - Requires smaller learning rates than Adam (typically 3-10x smaller)
/// - More robust to hyperparameter choices
/// - Excellent performance on large models
pub struct Lion {
    var_map: VarMap,
    learning_rate: f64,
    beta1: f64,  // Momentum for update direction
    beta2: f64,  // Momentum for momentum update
    weight_decay: f64,
    step: usize,
    
    // Single momentum state (33% less memory than Adam)
    momentum: HashMap<String, Tensor>,
}

impl Lion {
    pub fn new(
        var_map: VarMap,
        learning_rate: f64,
        beta1: f64,
        beta2: f64,
        weight_decay: f64,
    ) -> Result<Self> {
        Ok(Self {
            var_map,
            learning_rate,
            beta1,
            beta2,
            weight_decay,
            step: 0,
            momentum: HashMap::new(),
        })
    }
    
    /// Initialize momentum state for a variable
    fn init_momentum(&mut self, var_id: String, shape: &[usize], device: &Device) -> Result<()> {
        if !self.momentum.contains_key(&var_id) {
            let zeros = Tensor::zeros(shape, DType::F32, device)?;
            self.momentum.insert(var_id, zeros);
        }
        Ok(())
    }
    
    /// Get total memory usage (parameters + momentum only)
    pub fn memory_usage(&self) -> usize {
        let param_size: usize = self.var_map.all_vars().iter()
            .map(|var| var.as_tensor().elem_count())
            .sum();
        param_size * 8  // 4 bytes for params + 4 bytes for momentum
    }
}

impl CanOptimizer for Lion {
    type Config = LionConfig;

    fn new(vars: Vec<Var>, config: Self::Config) -> Result<Self> {
        let mut var_map = VarMap::new();
        for var in vars {
            var_map.insert(var);
        }
        
        Ok(Self::new(
            var_map,
            config.lr,
            config.beta1,
            config.beta2,
            config.weight_decay,
        )?)
    }

    fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.learning_rate = lr;
    }

    fn step(&mut self, grads: &candle_core::backprop::GradStore) -> Result<()> {
        self.step += 1;
        
        let lr = self.learning_rate;
        let beta1 = self.beta1;
        let beta2 = self.beta2;
        let wd = self.weight_decay;
        
        for (idx, var) in self.var_map.all_vars().iter().enumerate() {
            let var_id = format!("var_{}", idx);
            let tensor = var.as_tensor();
            
            // Get gradient
            let grad = match grads.get(var) {
                Some(g) => g,
                None => continue,
            };
            
            // Initialize momentum if needed
            self.init_momentum(var_id.clone(), tensor.dims(), tensor.device())?;
            
            // Get momentum
            let momentum = &self.momentum[&var_id];
            
            // Lion update algorithm:
            // 1. Interpolate between momentum and gradient
            let interpolated = ((momentum * beta1)? + (grad * (1.0 - beta1))?)?;
            
            // 2. Take sign of interpolated value
            let update = interpolated.sign()?;
            
            // 3. Apply weight decay (decoupled)
            let update = if wd > 0.0 {
                (update + (tensor.sign()? * wd)?)?
            } else {
                update
            };
            
            // 4. Update parameters
            let new_tensor = (tensor - (update * lr)?)?;
            var.set(&new_tensor)?;
            
            // 5. Update momentum (different from update direction!)
            let new_momentum = ((momentum * beta2)? + (grad * (1.0 - beta2))?)?;
            self.momentum.insert(var_id, new_momentum);
        }
        
        Ok(())
    }
    
    fn backward_step(&mut self, loss: &Tensor) -> Result<()> {
        let grads = loss.backward()?;
        self.step(&grads)
    }
}

/// Configuration for Lion optimizer
#[derive(Debug, Clone)]
pub struct LionConfig {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub weight_decay: f64,
}

impl Default for LionConfig {
    fn default() -> Self {
        Self {
            lr: 1e-4,      // Much smaller than Adam's typical 1e-3
            beta1: 0.9,    // Momentum for update direction
            beta2: 0.99,   // Momentum for momentum update
            weight_decay: 0.0,
        }
    }
}

/// Preset configurations for different domains
pub enum LionPreset {
    Language,   // For large language models
    Vision,     // For vision models
    FineTune,   // For fine-tuning pre-trained models
    Default,    // General purpose
}

impl LionPreset {
    pub fn config(&self) -> LionConfig {
        match self {
            Self::Language => LionConfig {
                lr: 3e-4,
                beta1: 0.95,
                beta2: 0.98,
                weight_decay: 0.01,
            },
            Self::Vision => LionConfig {
                lr: 1e-4,
                beta1: 0.9,
                beta2: 0.99,
                weight_decay: 0.05,
            },
            Self::FineTune => LionConfig {
                lr: 3e-5,
                beta1: 0.9,
                beta2: 0.999,
                weight_decay: 0.001,
            },
            Self::Default => LionConfig::default(),
        }
    }
}

/// Learning rate scheduler for Lion
pub enum LionSchedule {
    Constant,
    Linear { warmup_steps: usize, total_steps: usize },
    Cosine { total_steps: usize, min_lr: f64 },
    ExponentialDecay { decay_rate: f64, decay_steps: usize },
}

impl LionSchedule {
    pub fn get_lr(&self, base_lr: f64, step: usize) -> f64 {
        match self {
            Self::Constant => base_lr,
            Self::Linear { warmup_steps, total_steps } => {
                if step < *warmup_steps {
                    base_lr * (step as f64 / *warmup_steps as f64)
                } else {
                    let progress = (step - warmup_steps) as f64 / (*total_steps - warmup_steps) as f64;
                    base_lr * (1.0 - progress).max(0.0)
                }
            },
            Self::Cosine { total_steps, min_lr } => {
                let progress = (step as f64 / *total_steps as f64).min(1.0);
                let cosine_decay = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
                min_lr + (base_lr - min_lr) * cosine_decay
            },
            Self::ExponentialDecay { decay_rate, decay_steps } => {
                let exponent = step as f64 / *decay_steps as f64;
                base_lr * decay_rate.powf(exponent)
            },
        }
    }
}

/// Monitor for tracking Lion optimizer progress
pub struct LionMonitor {
    lr_history: Vec<f64>,
    step_times: Vec<std::time::Duration>,
    last_time: Option<std::time::Instant>,
}

impl LionMonitor {
    pub fn new() -> Self {
        Self {
            lr_history: Vec::new(),
            step_times: Vec::new(),
            last_time: None,
        }
    }
    
    pub fn record(&mut self, optimizer: &Lion) {
        self.lr_history.push(optimizer.learning_rate());
        
        if let Some(last) = self.last_time {
            self.step_times.push(last.elapsed());
        }
        self.last_time = Some(std::time::Instant::now());
    }
    
    pub fn summary(&self) -> String {
        let avg_step_time = if !self.step_times.is_empty() {
            self.step_times.iter().sum::<std::time::Duration>() / self.step_times.len() as u32
        } else {
            std::time::Duration::from_secs(0)
        };
        
        format!(
            "Lion Optimizer Summary:\n\
             Total steps: {}\n\
             Final LR: {:.2e}\n\
             Avg step time: {:.2}ms",
            self.lr_history.len(),
            self.lr_history.last().unwrap_or(&0.0),
            avg_step_time.as_secs_f64() * 1000.0
        )
    }
}

/// Training guide and tips for Lion optimizer
pub struct LionTrainingGuide;

impl LionTrainingGuide {
    /// Convert Adam learning rate to Lion learning rate
    pub fn convert_adam_lr(adam_lr: f64) -> f64 {
        adam_lr / 10.0  // Lion typically uses 10x smaller LR
    }
    
    /// Print training tips
    pub fn print_tips() {
        println!("Lion Optimizer Training Tips:");
        println!("1. Use 3-10x smaller learning rate than Adam (e.g., 1e-4 instead of 1e-3)");
        println!("2. Lion is more stable with larger batch sizes");
        println!("3. Weight decay can be more aggressive than with Adam");
        println!("4. Works particularly well for large models (>100M parameters)");
        println!("5. Monitor gradient norms - Lion handles large gradients well");
        println!("6. For fine-tuning, use even smaller LR (1e-5 to 3e-5)");
    }
}

/// Memory comparison helper
pub fn compare_optimizer_memory(num_params: usize) {
    let param_gb = (num_params * 4) as f64 / 1e9;
    
    println!("Optimizer Memory Usage for {}B parameters:", num_params as f64 / 1e9);
    println!("  Adam/AdamW:     {:.1} GB", param_gb * 3.0);  // params + m + v
    println!("  Lion:           {:.1} GB", param_gb * 2.0);  // params + m only
    println!("  SGD:            {:.1} GB", param_gb * 1.0);  // params only
    println!("  SGD+Momentum:   {:.1} GB", param_gb * 2.0);  // params + m
    println!();
    println!("Lion saves {:.1} GB compared to Adam!", param_gb * 1.0);
}

/// Helper to create Lion with preset
pub fn create_lion_preset(vars: Vec<Var>, preset: &str) -> Result<Lion> {
    let preset_config = match preset {
        "language" => LionPreset::Language.config(),
        "vision" => LionPreset::Vision.config(),
        "finetune" => LionPreset::FineTune.config(),
        _ => LionPreset::Default.config(),
    };
    
    Lion::new(vars, preset_config)
}

/// Helper to create Lion with scheduling
pub fn create_scheduled_lion(
    vars: Vec<Var>, 
    config: LionConfig,
    schedule: LionSchedule,
) -> Result<ScheduledLion> {
    let optimizer = Lion::new(vars, config)?;
    Ok(ScheduledLion {
        optimizer,
        schedule,
        base_lr: config.lr,
    })
}

/// Lion optimizer with learning rate scheduling
pub struct ScheduledLion {
    optimizer: Lion,
    schedule: LionSchedule,
    base_lr: f64,
}

impl ScheduledLion {
    pub fn step(&mut self, grads: &candle_core::backprop::GradStore) -> Result<()> {
        // Update learning rate based on schedule
        let new_lr = self.schedule.get_lr(self.base_lr, self.optimizer.step);
        self.optimizer.set_learning_rate(new_lr);
        
        // Perform optimization step
        self.optimizer.step(grads)
    }
    
    pub fn backward_step(&mut self, loss: &Tensor) -> Result<()> {
        let grads = loss.backward()?;
        self.step(&grads)
    }
    
    pub fn current_lr(&self) -> f64 {
        self.optimizer.learning_rate()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_lion_memory_efficiency() {
        // Lion uses 33% less memory than Adam
        compare_optimizer_memory(1_000_000_000);  // 1B params
    }
    
    #[test]
    fn test_lr_conversion() {
        assert_eq!(LionTrainingGuide::convert_adam_lr(1e-3), 1e-4);
        assert_eq!(LionTrainingGuide::convert_adam_lr(3e-3), 3e-4);
    }
}