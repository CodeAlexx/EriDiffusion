//! Learning rate schedulers

use eridiffusion_core::{Result, Error};
use serde::{Serialize, Deserialize};

/// Scheduler trait
pub trait Scheduler: Send + Sync {
    /// Get learning rate for step
    fn step(&mut self, step: usize) -> f64;
    
    /// Get scheduler name
    fn name(&self) -> &str;
    
    /// Get scheduler state
    fn state(&self) -> &SchedulerState;
    
    /// Set scheduler state
    fn set_state(&mut self, state: SchedulerState) -> Result<()>;
}

/// Scheduler state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerState {
    pub last_step: usize,
    pub last_lr: f64,
}

impl SchedulerState {
    pub fn new(lr: f64) -> Self {
        Self {
            last_step: 0,
            last_lr: lr,
        }
    }
}

impl Default for SchedulerState {
    fn default() -> Self {
        Self {
            last_step: 0,
            last_lr: 0.001,
        }
    }
}

/// Scheduler configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerConfig {
    pub scheduler_type: SchedulerType,
    pub base_lr: f64,
    pub warmup_steps: usize,
    pub num_training_steps: Option<usize>,
    pub num_cycles: f64,
    pub power: f64,
    pub min_lr: f64,
    pub gamma: f64,
    pub step_size: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            scheduler_type: SchedulerType::Cosine,
            base_lr: 1e-4,
            warmup_steps: 1000,
            num_training_steps: None,
            num_cycles: 0.5,
            power: 1.0,
            min_lr: 0.0,
            gamma: 0.1,
            step_size: 1000,
        }
    }
}

/// Scheduler type
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SchedulerType {
    Constant,
    Linear,
    Cosine,
    CosineWithRestarts,
    Polynomial,
    ExponentialDecay,
    StepLR,
}

/// Create scheduler
pub fn create_scheduler(config: SchedulerConfig) -> Result<Box<dyn Scheduler>> {
    match config.scheduler_type {
        SchedulerType::Constant => Ok(Box::new(ConstantScheduler::new(config))),
        SchedulerType::Linear => Ok(Box::new(LinearScheduler::new(config))),
        SchedulerType::Cosine => Ok(Box::new(CosineScheduler::new(config))),
        SchedulerType::CosineWithRestarts => Ok(Box::new(CosineWithRestartsScheduler::new(config))),
        SchedulerType::Polynomial => Ok(Box::new(PolynomialScheduler::new(config))),
        SchedulerType::ExponentialDecay => Ok(Box::new(ExponentialScheduler::new(config))),
        SchedulerType::StepLR => Ok(Box::new(StepLRScheduler::new(config))),
    }
}

/// Constant learning rate
pub struct ConstantScheduler {
    config: SchedulerConfig,
    state: SchedulerState,
}

impl ConstantScheduler {
    pub fn new(config: SchedulerConfig) -> Self {
        Self {
            state: SchedulerState::new(config.base_lr),
            config,
        }
    }
}

impl Scheduler for ConstantScheduler {
    fn step(&mut self, step: usize) -> f64 {
        self.state.last_step = step;
        self.state.last_lr = self.config.base_lr;
        self.config.base_lr
    }
    
    fn name(&self) -> &str {
        "Constant"
    }
    
    fn state(&self) -> &SchedulerState {
        &self.state
    }
    
    fn set_state(&mut self, state: SchedulerState) -> Result<()> {
        self.state = state;
        Ok(())
    }
}

/// Linear warmup and decay
pub struct LinearScheduler {
    config: SchedulerConfig,
    state: SchedulerState,
}

impl LinearScheduler {
    pub fn new(config: SchedulerConfig) -> Self {
        Self {
            state: SchedulerState::new(0.0),
            config,
        }
    }
}

impl Scheduler for LinearScheduler {
    fn step(&mut self, step: usize) -> f64 {
        self.state.last_step = step;
        
        let lr = if step < self.config.warmup_steps {
            // Warmup phase
            self.config.base_lr * (step as f64 / self.config.warmup_steps as f64)
        } else if let Some(total_steps) = self.config.num_training_steps {
            // Linear decay
            let progress = (step - self.config.warmup_steps) as f64 
                / (total_steps - self.config.warmup_steps) as f64;
            self.config.base_lr * (1.0 - progress).max(0.0)
        } else {
            self.config.base_lr
        };
        
        self.state.last_lr = lr;
        lr
    }
    
    fn name(&self) -> &str {
        "Linear"
    }
    
    fn state(&self) -> &SchedulerState {
        &self.state
    }
    
    fn set_state(&mut self, state: SchedulerState) -> Result<()> {
        self.state = state;
        Ok(())
    }
}

/// Cosine annealing
pub struct CosineScheduler {
    config: SchedulerConfig,
    state: SchedulerState,
}

impl CosineScheduler {
    pub fn new(config: SchedulerConfig) -> Self {
        Self {
            state: SchedulerState::new(0.0),
            config,
        }
    }
}

impl Scheduler for CosineScheduler {
    fn step(&mut self, step: usize) -> f64 {
        self.state.last_step = step;
        
        let lr = if step < self.config.warmup_steps {
            // Warmup phase
            self.config.base_lr * (step as f64 / self.config.warmup_steps as f64)
        } else if let Some(total_steps) = self.config.num_training_steps {
            // Cosine decay
            let progress = (step - self.config.warmup_steps) as f64 
                / (total_steps - self.config.warmup_steps) as f64;
            let cosine_decay = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
            self.config.min_lr + (self.config.base_lr - self.config.min_lr) * cosine_decay
        } else {
            self.config.base_lr
        };
        
        self.state.last_lr = lr;
        lr
    }
    
    fn name(&self) -> &str {
        "Cosine"
    }
    
    fn state(&self) -> &SchedulerState {
        &self.state
    }
    
    fn set_state(&mut self, state: SchedulerState) -> Result<()> {
        self.state = state;
        Ok(())
    }
}

/// Cosine with warm restarts
pub struct CosineWithRestartsScheduler {
    config: SchedulerConfig,
    state: SchedulerState,
}

impl CosineWithRestartsScheduler {
    pub fn new(config: SchedulerConfig) -> Self {
        Self {
            state: SchedulerState::new(0.0),
            config,
        }
    }
}

impl Scheduler for CosineWithRestartsScheduler {
    fn step(&mut self, step: usize) -> f64 {
        self.state.last_step = step;
        
        let lr = if step < self.config.warmup_steps {
            // Warmup phase
            self.config.base_lr * (step as f64 / self.config.warmup_steps as f64)
        } else if let Some(total_steps) = self.config.num_training_steps {
            // Cosine with restarts
            let progress = (step - self.config.warmup_steps) as f64 
                / (total_steps - self.config.warmup_steps) as f64;
            let cycles = self.config.num_cycles;
            let cosine_decay = 0.5 * (1.0 + (2.0 * std::f64::consts::PI * cycles * progress).cos());
            self.config.min_lr + (self.config.base_lr - self.config.min_lr) * cosine_decay
        } else {
            self.config.base_lr
        };
        
        self.state.last_lr = lr;
        lr
    }
    
    fn name(&self) -> &str {
        "CosineWithRestarts"
    }
    
    fn state(&self) -> &SchedulerState {
        &self.state
    }
    
    fn set_state(&mut self, state: SchedulerState) -> Result<()> {
        self.state = state;
        Ok(())
    }
}

/// Polynomial decay
pub struct PolynomialScheduler {
    config: SchedulerConfig,
    state: SchedulerState,
}

impl PolynomialScheduler {
    pub fn new(config: SchedulerConfig) -> Self {
        Self {
            state: SchedulerState::new(0.0),
            config,
        }
    }
}

impl Scheduler for PolynomialScheduler {
    fn step(&mut self, step: usize) -> f64 {
        self.state.last_step = step;
        
        let lr = if step < self.config.warmup_steps {
            // Warmup phase
            self.config.base_lr * (step as f64 / self.config.warmup_steps as f64)
        } else if let Some(total_steps) = self.config.num_training_steps {
            // Polynomial decay
            let remaining = 1.0 - (step - self.config.warmup_steps) as f64 
                / (total_steps - self.config.warmup_steps) as f64;
            self.config.base_lr * remaining.powf(self.config.power)
        } else {
            self.config.base_lr
        };
        
        self.state.last_lr = lr;
        lr
    }
    
    fn name(&self) -> &str {
        "Polynomial"
    }
    
    fn state(&self) -> &SchedulerState {
        &self.state
    }
    
    fn set_state(&mut self, state: SchedulerState) -> Result<()> {
        self.state = state;
        Ok(())
    }
}

/// Exponential decay
pub struct ExponentialScheduler {
    config: SchedulerConfig,
    state: SchedulerState,
}

impl ExponentialScheduler {
    pub fn new(config: SchedulerConfig) -> Self {
        Self {
            state: SchedulerState::new(config.base_lr),
            config,
        }
    }
}

impl Scheduler for ExponentialScheduler {
    fn step(&mut self, step: usize) -> f64 {
        self.state.last_step = step;
        
        let lr = if step < self.config.warmup_steps {
            // Warmup phase
            self.config.base_lr * (step as f64 / self.config.warmup_steps as f64)
        } else {
            // Exponential decay
            self.config.base_lr * self.config.gamma.powf((step - self.config.warmup_steps) as f64)
        };
        
        self.state.last_lr = lr;
        lr
    }
    
    fn name(&self) -> &str {
        "Exponential"
    }
    
    fn state(&self) -> &SchedulerState {
        &self.state
    }
    
    fn set_state(&mut self, state: SchedulerState) -> Result<()> {
        self.state = state;
        Ok(())
    }
}

/// Step learning rate
pub struct StepLRScheduler {
    config: SchedulerConfig,
    state: SchedulerState,
}

impl StepLRScheduler {
    pub fn new(config: SchedulerConfig) -> Self {
        Self {
            state: SchedulerState::new(config.base_lr),
            config,
        }
    }
}

impl Scheduler for StepLRScheduler {
    fn step(&mut self, step: usize) -> f64 {
        self.state.last_step = step;
        
        let lr = if step < self.config.warmup_steps {
            // Warmup phase
            self.config.base_lr * (step as f64 / self.config.warmup_steps as f64)
        } else {
            // Step decay
            let num_steps = (step - self.config.warmup_steps) / self.config.step_size;
            self.config.base_lr * self.config.gamma.powf(num_steps as f64)
        };
        
        self.state.last_lr = lr;
        lr
    }
    
    fn name(&self) -> &str {
        "StepLR"
    }
    
    fn state(&self) -> &SchedulerState {
        &self.state
    }
    
    fn set_state(&mut self, state: SchedulerState) -> Result<()> {
        self.state = state;
        Ok(())
    }
}

/// Register built-in schedulers
pub fn register_builtin_schedulers() -> Result<()> {
    // In a real implementation, would register with a global registry
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_linear_scheduler() {
        let config = SchedulerConfig {
            scheduler_type: SchedulerType::Linear,
            base_lr: 1e-3,
            warmup_steps: 100,
            num_training_steps: Some(1000),
            ..Default::default()
        };
        
        let mut scheduler = LinearScheduler::new(config);
        
        // Test warmup
        assert_eq!(scheduler.step(0), 0.0);
        assert_eq!(scheduler.step(50), 5e-4);
        assert_eq!(scheduler.step(100), 1e-3);
        
        // Test decay
        let lr_500 = scheduler.step(500);
        assert!(lr_500 < 1e-3);
        assert!(lr_500 > 0.0);
    }
    
    #[test]
    fn test_cosine_scheduler() {
        let config = SchedulerConfig {
            scheduler_type: SchedulerType::Cosine,
            base_lr: 1e-3,
            warmup_steps: 100,
            num_training_steps: Some(1000),
            min_lr: 1e-5,
            ..Default::default()
        };
        
        let mut scheduler = CosineScheduler::new(config);
        
        // Test warmup
        assert_eq!(scheduler.step(50), 5e-4);
        
        // Test cosine decay
        let lr_middle = scheduler.step(550);
        let lr_end = scheduler.step(999);
        
        assert!(lr_middle < 1e-3);
        assert!(lr_middle > lr_end);
        assert!(lr_end >= 1e-5);
    }
}