//! Learning rate schedulers

use eridiffusion_core::{Result, Error};
use serde::{Serialize, Deserialize};
use std::f32::consts::PI;

/// Scheduler configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerConfig {
    pub scheduler_type: SchedulerType,
    pub warmup_steps: Option<usize>,
    pub warmup_ratio: Option<f32>,
    pub num_cycles: f32,
    pub power: f32,
    pub min_lr: f32,
    pub max_lr: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SchedulerType {
    Constant,
    Linear,
    Cosine,
    CosineWithRestarts,
    Polynomial,
    Exponential,
    StepLR,
    MultiStepLR,
    OneCycleLR,
    ReduceLROnPlateau,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            scheduler_type: SchedulerType::Constant,
            warmup_steps: None,
            warmup_ratio: None,
            num_cycles: 0.5,
            power: 1.0,
            min_lr: 0.0,
            max_lr: 1e-3,
        }
    }
}

/// Base trait for learning rate schedulers
pub trait LRScheduler: Send + Sync {
    /// Get learning rate for current step
    fn get_lr(&self, step: usize) -> f32;
    
    /// Get last learning rate
    fn get_last_lr(&self) -> f32;
    
    /// Update internal state if needed
    fn step(&mut self, metrics: Option<f32>);
    
    /// Get scheduler config
    fn config(&self) -> &SchedulerConfig;
}

/// Constant learning rate scheduler
pub struct ConstantLR {
    config: SchedulerConfig,
    base_lr: f32,
}

impl ConstantLR {
    pub fn new(base_lr: f32, config: SchedulerConfig) -> Self {
        Self { config, base_lr }
    }
}

impl LRScheduler for ConstantLR {
    fn get_lr(&self, _step: usize) -> f32 {
        self.base_lr
    }
    
    fn get_last_lr(&self) -> f32 {
        self.base_lr
    }
    
    fn step(&mut self, _metrics: Option<f32>) {}
    
    fn config(&self) -> &SchedulerConfig {
        &self.config
    }
}

/// Linear learning rate scheduler
pub struct LinearLR {
    config: SchedulerConfig,
    base_lr: f32,
    total_steps: usize,
    last_lr: f32,
}

impl LinearLR {
    pub fn new(base_lr: f32, total_steps: usize, config: SchedulerConfig) -> Self {
        Self {
            config,
            base_lr,
            total_steps,
            last_lr: base_lr,
        }
    }
}

impl LRScheduler for LinearLR {
    fn get_lr(&self, step: usize) -> f32 {
        let warmup_steps = self.config.warmup_steps.unwrap_or(0);
        
        if step < warmup_steps {
            // Warmup phase
            self.base_lr * (step as f32 / warmup_steps as f32)
        } else {
            // Linear decay
            let progress = (step - warmup_steps) as f32 / 
                          (self.total_steps - warmup_steps) as f32;
            self.base_lr * (1.0 - progress).max(0.0) + self.config.min_lr
        }
    }
    
    fn get_last_lr(&self) -> f32 {
        self.last_lr
    }
    
    fn step(&mut self, _metrics: Option<f32>) {}
    
    fn config(&self) -> &SchedulerConfig {
        &self.config
    }
}

/// Cosine annealing scheduler
pub struct CosineAnnealingLR {
    config: SchedulerConfig,
    base_lr: f32,
    total_steps: usize,
    last_lr: f32,
}

impl CosineAnnealingLR {
    pub fn new(base_lr: f32, total_steps: usize, config: SchedulerConfig) -> Self {
        Self {
            config,
            base_lr,
            total_steps,
            last_lr: base_lr,
        }
    }
}

impl LRScheduler for CosineAnnealingLR {
    fn get_lr(&self, step: usize) -> f32 {
        let warmup_steps = self.config.warmup_steps.unwrap_or(0);
        
        if step < warmup_steps {
            // Warmup phase
            self.base_lr * (step as f32 / warmup_steps as f32)
        } else {
            // Cosine annealing
            let progress = (step - warmup_steps) as f32 / 
                          (self.total_steps - warmup_steps) as f32;
            let cosine_factor = 0.5 * (1.0 + (PI * progress).cos());
            self.config.min_lr + (self.base_lr - self.config.min_lr) * cosine_factor
        }
    }
    
    fn get_last_lr(&self) -> f32 {
        self.last_lr
    }
    
    fn step(&mut self, _metrics: Option<f32>) {}
    
    fn config(&self) -> &SchedulerConfig {
        &self.config
    }
}

/// Cosine annealing with warm restarts
pub struct CosineAnnealingWarmRestarts {
    config: SchedulerConfig,
    base_lr: f32,
    t_0: usize,
    t_mult: f32,
    last_lr: f32,
    current_cycle: usize,
    cycle_step: usize,
}

impl CosineAnnealingWarmRestarts {
    pub fn new(base_lr: f32, t_0: usize, t_mult: f32, config: SchedulerConfig) -> Self {
        Self {
            config,
            base_lr,
            t_0,
            t_mult,
            last_lr: base_lr,
            current_cycle: 0,
            cycle_step: 0,
        }
    }
}

impl LRScheduler for CosineAnnealingWarmRestarts {
    fn get_lr(&self, step: usize) -> f32 {
        let warmup_steps = self.config.warmup_steps.unwrap_or(0);
        
        if step < warmup_steps {
            // Warmup phase
            self.base_lr * (step as f32 / warmup_steps as f32)
        } else {
            // Find current cycle
            let mut cycle = 0;
            let mut cycle_length = self.t_0;
            let mut total_steps = warmup_steps;
            
            while total_steps + cycle_length <= step {
                total_steps += cycle_length;
                cycle += 1;
                cycle_length = (cycle_length as f32 * self.t_mult) as usize;
            }
            
            let cycle_progress = (step - total_steps) as f32 / cycle_length as f32;
            let cosine_factor = 0.5 * (1.0 + (PI * cycle_progress).cos());
            self.config.min_lr + (self.base_lr - self.config.min_lr) * cosine_factor
        }
    }
    
    fn get_last_lr(&self) -> f32 {
        self.last_lr
    }
    
    fn step(&mut self, _metrics: Option<f32>) {
        self.cycle_step += 1;
        let cycle_length = (self.t_0 as f32 * self.t_mult.powi(self.current_cycle as i32)) as usize;
        
        if self.cycle_step >= cycle_length {
            self.current_cycle += 1;
            self.cycle_step = 0;
        }
    }
    
    fn config(&self) -> &SchedulerConfig {
        &self.config
    }
}

/// One cycle learning rate scheduler
pub struct OneCycleLR {
    config: SchedulerConfig,
    max_lr: f32,
    total_steps: usize,
    pct_start: f32,
    anneal_strategy: String,
    last_lr: f32,
}

impl OneCycleLR {
    pub fn new(
        max_lr: f32,
        total_steps: usize,
        pct_start: f32,
        config: SchedulerConfig,
    ) -> Self {
        Self {
            config,
            max_lr,
            total_steps,
            pct_start,
            anneal_strategy: "cos".to_string(),
            last_lr: max_lr / 25.0, // Start at max_lr/25
        }
    }
}

impl LRScheduler for OneCycleLR {
    fn get_lr(&self, step: usize) -> f32 {
        let start_step = (self.pct_start * self.total_steps as f32) as usize;
        
        if step < start_step {
            // Increasing phase
            let progress = step as f32 / start_step as f32;
            let start_lr = self.max_lr / 25.0;
            start_lr + (self.max_lr - start_lr) * progress
        } else {
            // Decreasing phase
            let progress = (step - start_step) as f32 / 
                          (self.total_steps - start_step) as f32;
            
            match self.anneal_strategy.as_str() {
                "cos" => {
                    let cosine_factor = 0.5 * (1.0 + (PI * progress).cos());
                    self.config.min_lr + (self.max_lr - self.config.min_lr) * cosine_factor
                }
                "linear" => {
                    self.max_lr * (1.0 - progress).max(0.0) + self.config.min_lr
                }
                _ => self.max_lr
            }
        }
    }
    
    fn get_last_lr(&self) -> f32 {
        self.last_lr
    }
    
    fn step(&mut self, _metrics: Option<f32>) {}
    
    fn config(&self) -> &SchedulerConfig {
        &self.config
    }
}

/// Reduce LR on plateau
pub struct ReduceLROnPlateau {
    config: SchedulerConfig,
    base_lr: f32,
    factor: f32,
    patience: usize,
    threshold: f32,
    cooldown: usize,
    min_lr: f32,
    best_metric: Option<f32>,
    num_bad_epochs: usize,
    cooldown_counter: usize,
    last_lr: f32,
}

impl ReduceLROnPlateau {
    pub fn new(
        base_lr: f32,
        factor: f32,
        patience: usize,
        config: SchedulerConfig,
    ) -> Self {
        let min_lr = config.min_lr;
        Self {
            config,
            base_lr,
            factor,
            patience,
            threshold: 1e-4,
            cooldown: 0,
            min_lr,
            best_metric: None,
            num_bad_epochs: 0,
            cooldown_counter: 0,
            last_lr: base_lr,
        }
    }
}

impl LRScheduler for ReduceLROnPlateau {
    fn get_lr(&self, _step: usize) -> f32 {
        self.last_lr
    }
    
    fn get_last_lr(&self) -> f32 {
        self.last_lr
    }
    
    fn step(&mut self, metrics: Option<f32>) {
        if let Some(metric) = metrics {
            if self.cooldown_counter > 0 {
                self.cooldown_counter -= 1;
                return;
            }
            
            if let Some(best) = self.best_metric {
                if metric < best - self.threshold {
                    self.best_metric = Some(metric);
                    self.num_bad_epochs = 0;
                } else {
                    self.num_bad_epochs += 1;
                }
            } else {
                self.best_metric = Some(metric);
            }
            
            if self.num_bad_epochs >= self.patience {
                self.last_lr = (self.last_lr * self.factor).max(self.min_lr);
                self.cooldown_counter = self.cooldown;
                self.num_bad_epochs = 0;
            }
        }
    }
    
    fn config(&self) -> &SchedulerConfig {
        &self.config
    }
}

/// Create scheduler from config
pub fn create_scheduler(
    config: SchedulerConfig,
    base_lr: f32,
    total_steps: usize,
) -> Box<dyn LRScheduler> {
    match config.scheduler_type {
        SchedulerType::Constant => {
            Box::new(ConstantLR::new(base_lr, config))
        }
        SchedulerType::Linear => {
            Box::new(LinearLR::new(base_lr, total_steps, config))
        }
        SchedulerType::Cosine => {
            Box::new(CosineAnnealingLR::new(base_lr, total_steps, config))
        }
        SchedulerType::CosineWithRestarts => {
            Box::new(CosineAnnealingWarmRestarts::new(
                base_lr,
                total_steps / 4, // T_0
                1.5, // T_mult
                config,
            ))
        }
        SchedulerType::OneCycleLR => {
            Box::new(OneCycleLR::new(
                config.max_lr,
                total_steps,
                0.3, // pct_start
                config,
            ))
        }
        SchedulerType::ReduceLROnPlateau => {
            Box::new(ReduceLROnPlateau::new(
                base_lr,
                0.5, // factor
                10, // patience
                config,
            ))
        }
        _ => Box::new(ConstantLR::new(base_lr, config)),
    }
}

/// Scheduler utilities
pub mod utils {
    use super::*;
    
    /// Plot learning rate schedule
    pub fn plot_schedule(scheduler: &dyn LRScheduler, total_steps: usize) -> Vec<f32> {
        (0..total_steps)
            .map(|step| scheduler.get_lr(step))
            .collect()
    }
    
    /// Find optimal warmup steps
    pub fn find_optimal_warmup(base_lr: f32, total_steps: usize) -> usize {
        // Common heuristic: 5-10% of total steps
        (total_steps as f32 * 0.05) as usize
    }
    
    /// Create scheduler chain
    pub fn chain_schedulers(
        schedulers: Vec<(Box<dyn LRScheduler>, usize)>,
    ) -> Box<dyn LRScheduler> {
        // Would implement chained scheduler
        schedulers.into_iter().next().unwrap().0
    }
}