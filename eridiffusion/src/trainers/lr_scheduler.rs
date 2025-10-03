use anyhow::Context;
use flame_core::device::Device;
use flame_core::optimizers::{Adam, SGD};
use flame_core::{DType, Result, Shape, Tensor};

// Learning rate schedulers for training

/// Learning rate scheduler trait
pub trait LRScheduler {
    /// Get learning rate for current step
    fn get_lr(&self, step: usize) -> f32;

    /// Get scheduler name
    fn name(&self) -> &str;
}

/// Constant learning rate (no scheduling)
pub struct ConstantLR {
    base_lr: f32,
}

impl ConstantLR {
    pub fn new(base_lr: f32) -> Self {
        Self { base_lr }
    }
}

impl LRScheduler for ConstantLR {
    fn get_lr(&self, _step: usize) -> f32 {
        self.base_lr
    }

    fn name(&self) -> &str {
        "constant"
    }
}

/// Constant learning rate with warmup
#[derive(Clone)]
pub struct ConstantWithWarmupLR {
    base_lr: f32,
    warmup_steps: usize,
}

impl ConstantWithWarmupLR {
    pub fn new(base_lr: f32, warmup_steps: usize) -> Self {
        Self { base_lr, warmup_steps }
    }
}

impl LRScheduler for ConstantWithWarmupLR {
    fn get_lr(&self, step: usize) -> f32 {
        if step < self.warmup_steps {
            // Linear warmup
            self.base_lr * (step as f32 / self.warmup_steps as f32)
        } else {
            self.base_lr
        }
    }

    fn name(&self) -> &str {
        "constant_with_warmup"
    }
}

/// Cosine annealing learning rate scheduler
#[derive(Clone)]
pub struct CosineAnnealingLR {
    base_lr: f32,
    warmup_steps: usize,
    total_steps: usize,
    num_cycles: f32,
}

impl CosineAnnealingLR {
    pub fn new(base_lr: f32, warmup_steps: usize, total_steps: usize, num_cycles: f32) -> Self {
        Self { base_lr, warmup_steps, total_steps, num_cycles }
    }
}

impl LRScheduler for CosineAnnealingLR {
    fn get_lr(&self, step: usize) -> f32 {
        if step < self.warmup_steps {
            // Linear warmup
            self.base_lr * (step as f32 / self.warmup_steps as f32)
        } else {
            // Cosine annealing
            let progress =
                (step - self.warmup_steps) as f32 / (self.total_steps - self.warmup_steps) as f32;
            let cosine_value =
                0.5 * (1.0 + (std::f32::consts::PI * self.num_cycles * progress).cos());
            0.0 + (self.base_lr - 0.0) * cosine_value
        }
    }

    fn name(&self) -> &str {
        "cosine"
    }
}

/// Polynomial decay learning rate scheduler
#[derive(Clone)]
pub struct PolynomialLR {
    base_lr: f32,
    warmup_steps: usize,
    total_steps: usize,
    power: f32,
    end_lr: f32,
}

impl PolynomialLR {
    pub fn new(base_lr: f32, warmup_steps: usize, total_steps: usize, power: f32) -> Self {
        Self { base_lr, warmup_steps, total_steps, power, end_lr: 0.0 }
    }
}

impl LRScheduler for PolynomialLR {
    fn get_lr(&self, step: usize) -> f32 {
        if step < self.warmup_steps {
            // Linear warmup
            self.base_lr * (step as f32 / self.warmup_steps as f32)
        } else if step >= self.total_steps {
            self.end_lr
        } else {
            // Polynomial decay
            let remaining_progress =
                (self.total_steps - step) as f32 / (self.total_steps - self.warmup_steps) as f32;
            self.end_lr + (self.base_lr - self.end_lr) * remaining_progress.powf(self.power)
        }
    }

    fn name(&self) -> &str {
        "polynomial"
    }
}

/// Create scheduler from configuration
pub fn create_scheduler(
    scheduler_type: &str,
    base_lr: f32,
    warmup_steps: usize,
    total_steps: usize,
    num_cycles: Option<f32>,
    power: Option<f32>,
) -> flame_core::Result<Box<dyn LRScheduler>> {
    match scheduler_type {
        "constant" => Ok(Box::new(ConstantLR::new(base_lr))),
        "constant_with_warmup" => Ok(Box::new(ConstantWithWarmupLR::new(base_lr, warmup_steps))),
        "cosine" => Ok(Box::new(CosineAnnealingLR::new(
            base_lr,
            warmup_steps,
            total_steps,
            num_cycles.unwrap_or(1.0),
        ))),
        "polynomial" => Ok(Box::new(PolynomialLR::new(
            base_lr,
            warmup_steps,
            total_steps,
            power.unwrap_or(1.0),
        ))),
        _ => Err(flame_core::Error::InvalidOperation(format!(
            "Unknown scheduler type: {}",
            scheduler_type
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_lr() {
        let scheduler = ConstantLR::new(0.001);
        assert_eq!(scheduler.get_lr(0), 0.001);
        assert_eq!(scheduler.get_lr(1000), 0.001);
    }
}
