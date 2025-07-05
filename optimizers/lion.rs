//! Lion (EvoLved Sign Momentum) optimizer implementation for Candle
//! Based on "Symbolic Discovery of Optimization Algorithms"
//! https://arxiv.org/abs/2302.06675
//Here's a complete implementation of the Lion (EvoLved Sign Momentum) optimizer! Lion is remarkably simple yet effective - it only uses the sign of gradients and momentum, making it extremely memory efficient.
//!Key Features of Lion
//!Memory Efficient: Uses only momentum states (same memory as SGD with momentum) - 33% less memory than Adam/AdamW!
//!Sign-Based Updates: Uses only the sign of gradients, making it robust to gradient scaling and outliers.
//!Simple Algorithm: Much simpler than Adam but often performs comparably or better.
//!Fast Convergence: Despite using smaller learning rates, often converges faster than Adam.
//!Usage Examples
//!rust// Basic usage - note the smaller learning rate!
//!let config = LionConfig {
 //!   lr: 1e-4,           // Much smaller than Adam (typically 1e-3)
//!    beta1: 0.9,         // Momentum for update direction
 //!   beta2: 0.99,        // Momentum for momentum update
//!    weight_decay: 0.01,
//!};
//!let optimizer = create_lion_optimizer(vars, Some(config))?;

// Use presets for different domains
//!let llm_optimizer = create_lion_preset(vars, "language")?;
//!let vision_optimizer = create_lion_preset(vars, "vision")?;
l//!et finetune_optimizer = create_lion_preset(vars, "finetune")?;

// Convert from Adam learning rate
//!let adam_lr = 1e-3;//!
//!let lion_lr = LionTrainingGuide::convert_adam_lr(adam_lr); // Returns 1e-4

// Lion with learning rate scheduling
//!let schedule = LionSchedule::Cosine {
//!    total_steps: 10000,
//!    min_lr: 1e-6,
//!};
//!let scheduled_optimizer = create_scheduled_lion(vars, config, schedule)?;

// Memory usage comparison
//!compare_optimizer_memory(1_000_000_000); // 1B parameters

// Training tips
//!LionTrainingGuide::print_tips();
//!The Lion Algorithm
//!Lion is beautifully simple:

//!Update Direction: sign(β₁ * momentum + (1-β₁) * gradient)
//!Parameter Update: param = param - lr * (update_direction + weight_decay * param)
//!Momentum Update: momentum = β₂ * momentum + (1-β₂) * gradient

//!That's it! No second moments, no bias correction, just signs and momentum.
//!Key Advantages

//!Memory Efficient: Same memory usage as SGD with momentum (33% less than Adam)
//!Robust: Sign-based updates are robust to gradient outliers and scaling
//!Simple: Much simpler implementation and fewer hyperparameters
//!Effective: Often matches or beats Adam performance
//!Fast: Despite smaller learning rates, often converges faster

//!Important Notes

//!Learning Rate: Use 3-10x smaller learning rate than Adam (1e-4 instead of 1e-3)
//!Weight Decay: Lion can be more sensitive to weight decay values
//!Sign Function: The key insight is using only gradient signs, not magnitudes
//!Memory: Perfect for large models where memory is constrained

//!Training Tips
//!rust// Monitor your training
//!let mut monitor = LionMonitor::new();

//!for step in 0..num_steps {
//!    let loss = model.forward(&batch)?;
//!    optimizer.backward_step(&loss)?;
    
//!    if step % 100 == 0 {
//!        monitor.record(&optimizer);
//!        println!("Step {}: LR = {:.2e}, Loss = {:.4}", 
//!                 step, optimizer.learning_rate(), loss);
//!    }
//!}

//!println!("{}", monitor.summary());
//!Lion is particularly effective for:

//!Large language models (often outperforms Adam)
//!Memory-constrained training
//!When you want simpler, more robust optimization
//!Fine-tuning tasks (very stable)

//!The implementation includes learning rate scheduling, monitoring, and preset configurations for different use cases!

use candle_core::{DType, Device, Result, Tensor, D, Var};
use candle_nn::{Optimizer as CanOptimizer, VarMap};
type CanResult<T> = std::result::Result<T, candle_core::Error>;
use std::collections::HashMap;
use crate::optimizer::{Optimizer, OptimizerConfig, OptimizerState};
use eridiffusion_core::Error;

/// Lion (EvoLved Sign Momentum) optimizer
/// Uses only the sign of gradients with momentum for parameter updates
/// Extremely memory efficient - only stores momentum (same as SGD with momentum)
pub struct Lion {
    var_map: VarMap,
    learning_rate: f64,
    beta1: f64,           // Momentum coefficient (typically 0.9)
    beta2: f64,           // Momentum update coefficient (typically 0.99)
    weight_decay: f64,    // Weight decay coefficient
    step: usize,
    
    // Optimizer states - only momentum required!
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
    
    /// Compute sign of tensor elements
    fn sign(tensor: &Tensor) -> Result<Tensor> {
        // Convert to CPU for element-wise operations if needed
        let data = tensor.flatten_all()?.to_vec1::<f32>()?;
        let signs: Vec<f32> = data.iter().map(|&x| {
            if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                0.0
            }
        }).collect();
        
        Tensor::from_vec(signs, tensor.dims(), tensor.device())
    }
    
    /// Get memory usage of optimizer states
    pub fn memory_usage_bytes(&self) -> usize {
        self.momentum.values()
            .map(|tensor| {
                tensor.dims().iter().product::<usize>() * std::mem::size_of::<f32>()
            })
            .sum()
    }
}

impl CanOptimizer for Lion {
    type Config = LionConfig;

    fn new(vars: Vec<Var>, config: Self::Config) -> Result<Self> {
        let mut var_map = VarMap::new();
        for var in vars {
            var_map.insert(var);
        }
        
        Self::new(
            var_map,
            config.lr,
            config.beta1,
            config.beta2,
            config.weight_decay,
        )
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
        let weight_decay = self.weight_decay;
        
        for (idx, var) in self.var_map.all_vars().iter().enumerate() {
            let var_id = format!("var_{}", idx);
            let param = var.as_tensor();
            
            // Get gradient
            let grad = match grads.get(var) {
                Some(g) => g,
                None => continue,
            };
            
            // Initialize momentum if needed
            self.init_momentum(var_id.clone(), param.dims(), param.device())?;
            
            // Get current momentum
            let momentum = self.momentum.get_mut(&var_id).unwrap();
            
            // Lion update algorithm:
            // 1. Compute update direction: sign(beta1 * momentum + (1-beta1) * grad)
            let interpolated = ((momentum.as_ref() * beta1)? + (grad * (1.0 - beta1))?)?;
            let update_direction = Self::sign(&interpolated)?;
            
            // 2. Apply weight decay if specified
            let update = if weight_decay > 0.0 {
                // Weight decay is applied to the update direction
                (update_direction + (param * weight_decay)?)?
            } else {
                update_direction
            };
            
            // 3. Update parameters
            let new_param = (param - (update * lr)?)?;
            var.set(&new_param)?;
            
            // 4. Update momentum: momentum = beta2 * momentum + (1-beta2) * grad
            let new_momentum = ((momentum.as_ref() * beta2)? + (grad * (1.0 - beta2))?)?;
            *momentum = new_momentum;
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
    pub lr: f64,            // Learning rate (typically 10x smaller than Adam)
    pub beta1: f64,         // Momentum coefficient for update direction (default: 0.9)
    pub beta2: f64,         // Momentum coefficient for momentum update (default: 0.99)
    pub weight_decay: f64,  // Weight decay coefficient (default: 0.0)
}

impl Default for LionConfig {
    fn default() -> Self {
        Self {
            lr: 1e-4,      // Note: typically 3-10x smaller than Adam
            beta1: 0.9,
            beta2: 0.99,
            weight_decay: 0.0,
        }
    }
}

/// Lion with different presets for common use cases
impl LionConfig {
    /// Configuration for language model training
    /// Lion typically needs much smaller learning rates than Adam
    pub fn for_language_models() -> Self {
        Self {
            lr: 1e-4,       // Much smaller than typical Adam LR
            beta1: 0.9,
            beta2: 0.99,
            weight_decay: 0.01,
        }
    }
    
    /// Configuration for computer vision tasks
    pub fn for_vision() -> Self {
        Self {
            lr: 3e-4,       // Slightly higher for vision tasks
            beta1: 0.9,
            beta2: 0.99,
            weight_decay: 0.02,
        }
    }
    
    /// Configuration for fine-tuning (even more conservative)
    pub fn for_finetuning() -> Self {
        Self {
            lr: 3e-5,       // Very small for fine-tuning
            beta1: 0.9,
            beta2: 0.99,
            weight_decay: 0.01,
        }
    }
    
    /// Configuration matching the original paper
    pub fn paper_default() -> Self {
        Self {
            lr: 1e-4,
            beta1: 0.9,
            beta2: 0.99,
            weight_decay: 0.0,
        }
    }
}

/// Lion optimizer variants and extensions
pub enum LionOptimizer {
    Standard(Lion),
    Scheduled(ScheduledLion),
}

/// Lion with learning rate scheduling
pub struct ScheduledLion {
    lion: Lion,
    initial_lr: f64,
    schedule: LionSchedule,
}

#[derive(Debug, Clone)]
pub enum LionSchedule {
    Linear { total_steps: usize, end_lr: f64 },
    Cosine { total_steps: usize, min_lr: f64 },
    Exponential { decay_rate: f64, decay_steps: usize },
}

impl ScheduledLion {
    pub fn new(lion: Lion, schedule: LionSchedule) -> Self {
        let initial_lr = lion.learning_rate();
        Self {
            lion,
            initial_lr,
            schedule,
        }
    }
    
    fn update_learning_rate(&mut self) {
        let step = self.lion.step;
        let new_lr = match &self.schedule {
            LionSchedule::Linear { total_steps, end_lr } => {
                let progress = (step as f64) / (*total_steps as f64);
                let progress = progress.min(1.0);
                self.initial_lr * (1.0 - progress) + end_lr * progress
            },
            LionSchedule::Cosine { total_steps, min_lr } => {
                let progress = (step as f64) / (*total_steps as f64);
                let progress = progress.min(1.0);
                min_lr + (self.initial_lr - min_lr) * 0.5 * (1.0 + (std::f64::consts::PI * progress).cos())
            },
            LionSchedule::Exponential { decay_rate, decay_steps } => {
                let num_decays = step / decay_steps;
                self.initial_lr * decay_rate.powi(num_decays as i32)
            },
        };
        self.lion.set_learning_rate(new_lr);
    }
}

impl CanOptimizer for LionOptimizer {
    type Config = LionConfig;
    
    fn new(vars: Vec<Var>, config: Self::Config) -> CanResult<Self> {
        Ok(Self::Standard(
            Lion::new(vars, config)
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?
        ))
    }
    
    fn step(&mut self, grads: &candle_core::backprop::GradStore) -> CanResult<()> {
        match self {
            Self::Standard(opt) => opt.step(grads)
                .map_err(|e| candle_core::Error::Msg(e.to_string())),
            Self::Scheduled(opt) => {
                let result = opt.lion.step(grads)
                    .map_err(|e| candle_core::Error::Msg(e.to_string()));
                opt.update_learning_rate();
                result
            },
        }
    }
    
    fn learning_rate(&self) -> f64 {
        match self {
            Self::Standard(opt) => opt.learning_rate(),
            Self::Scheduled(opt) => opt.lion.learning_rate(),
        }
    }
    
    fn set_learning_rate(&mut self, lr: f64) {
        match self {
            Self::Standard(opt) => opt.set_learning_rate(lr),
            Self::Scheduled(opt) => opt.lion.set_learning_rate(lr),
        }
    }
}

/// Create Lion optimizer from config
pub fn create_lion_optimizer(
    vars: Vec<Var>,
    config: Option<LionConfig>,
) -> eridiffusion_core::Result<LionOptimizer> {
    let config = config.unwrap_or_default();
    LionOptimizer::new(vars, config)
        .map_err(|e| Error::Config(format!("Failed to create Lion optimizer: {}", e)))
}

/// Create Lion with preset configurations
pub fn create_lion_preset(
    vars: Vec<Var>,
    preset: &str,
) -> eridiffusion_core::Result<LionOptimizer> {
    let config = match preset {
        "language" | "llm" | "nlp" => LionConfig::for_language_models(),
        "vision" | "cv" | "image" => LionConfig::for_vision(),
        "finetune" | "ft" => LionConfig::for_finetuning(),
        "paper" | "original" => LionConfig::paper_default(),
        "default" => LionConfig::default(),
        _ => return Err(Error::Config(format!("Unknown Lion preset: {}", preset))),
    };
    
    create_lion_optimizer(vars, Some(config))
}

/// Create Lion with learning rate scheduling
pub fn create_scheduled_lion(
    vars: Vec<Var>,
    config: LionConfig,
    schedule: LionSchedule,
) -> eridiffusion_core::Result<LionOptimizer> {
    let lion = Lion::new(vars, config)
        .map_err(|e| Error::Config(format!("Failed to create Lion: {}", e)))?;
    
    Ok(LionOptimizer::Scheduled(ScheduledLion::new(lion, schedule)))
}

/// Memory comparison utility
pub fn compare_optimizer_memory(num_params: usize) {
    let param_gb = (num_params * 4) as f64 / 1e9; // FP32 parameters
    
    println!("Optimizer Memory Usage for {:.1}B parameters:", num_params as f64 / 1e9);
    println!("  SGD (no momentum): {:.2} GB", param_gb);
    println!("  SGD (with momentum): {:.2} GB", param_gb * 2.0);
    println!("  Lion:              {:.2} GB", param_gb * 2.0); // Same as SGD with momentum!
    println!("  AdamW:             {:.2} GB", param_gb * 3.0);
    println!("  AdamW (8-bit):     {:.2} GB", param_gb * 1.25);
    println!();
    println!("Lion memory efficiency:");
    println!("  vs AdamW: {:.1}% reduction", ((param_gb * 3.0 - param_gb * 2.0) / (param_gb * 3.0)) * 100.0);
    println!("  vs AdamW 8-bit: {:.1}% increase", ((param_gb * 2.0 - param_gb * 1.25) / (param_gb * 1.25)) * 100.0);
}

/// Lion training tips and best practices
pub struct LionTrainingGuide;

impl LionTrainingGuide {
    /// Get recommended learning rate based on Adam learning rate
    pub fn convert_adam_lr(adam_lr: f64) -> f64 {
        // Lion typically uses 3-10x smaller learning rate than Adam
        adam_lr / 10.0
    }
    
    /// Get recommended weight decay
    pub fn recommended_weight_decay(task: &str) -> f64 {
        match task {
            "language_model" | "llm" => 0.01,
            "vision" | "classification" => 0.02,
            "fine_tuning" => 0.01,
            _ => 0.01,
        }
    }
    
    /// Print training tips
    pub fn print_tips() {
        println!("Lion Optimizer Training Tips:");
        println!("1. Use 3-10x smaller learning rate than Adam (typically 1e-4 instead of 1e-3)");
        println!("2. Lion is more sensitive to weight decay - start with smaller values");
        println!("3. Memory usage is same as SGD with momentum (33% less than Adam)");
        println!("4. Often converges faster than Adam despite lower learning rate");
        println!("5. Sign-based updates make it robust to gradient scaling");
        println!("6. Works particularly well for large language models");
        println!("7. Consider learning rate scheduling for longer training runs");
    }
}

/// Monitor Lion optimization progress
pub struct LionMonitor {
    lr_history: Vec<f64>,
    step_history: Vec<usize>,
    momentum_norms: Vec<f64>,
}

impl LionMonitor {
    pub fn new() -> Self {
        Self {
            lr_history: Vec::new(),
            step_history: Vec::new(),
            momentum_norms: Vec::new(),
        }
    }
    
    /// Record current optimizer state
    pub fn record(&mut self, optimizer: &Lion) {
        self.lr_history.push(optimizer.learning_rate());
        self.step_history.push(optimizer.step);
        
        // Compute average momentum norm
        if !optimizer.momentum.is_empty() {
            let total_norm: f32 = optimizer.momentum.values()
                .map(|m| {
                    m.flatten_all().unwrap().to_vec1::<f32>().unwrap()
                        .iter().map(|x| x * x).sum::<f32>().sqrt()
                })
                .sum();
            let avg_norm = total_norm / optimizer.momentum.len() as f32;
            self.momentum_norms.push(avg_norm as f64);
        }
    }
    
    /// Get training summary
    pub fn summary(&self) -> LionSummary {
        LionSummary {
            total_steps: self.step_history.len(),
            final_lr: *self.lr_history.last().unwrap_or(&0.0),
            avg_momentum_norm: if self.momentum_norms.is_empty() {
                0.0
            } else {
                self.momentum_norms.iter().sum::<f64>() / self.momentum_norms.len() as f64
            },
        }
    }
}

#[derive(Debug)]
pub struct LionSummary {
    pub total_steps: usize,
    pub final_lr: f64,
    pub avg_momentum_norm: f64,
}

impl std::fmt::Display for LionSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Lion Optimization Summary:")?;
        writeln!(f, "  Total steps: {}", self.total_steps)?;
        writeln!(f, "  Final learning rate: {:.2e}", self.final_lr)?;
        writeln!(f, "  Average momentum norm: {:.4}", self.avg_momentum_norm)?;
        Ok(())
    }
}
