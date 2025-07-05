//! Gradient accumulation strategies

use eridiffusion_core::{Result, Error};
use candle_core::{Tensor, DType};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// Gradient accumulation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccumulationConfig {
    pub steps: usize,
    pub normalize: bool,
    pub dtype: AccumulationDType,
    pub sync_mode: SyncMode,
    pub reduction: ReductionMode,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum AccumulationDType {
    Float32,
    Float16,
    BFloat16,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SyncMode {
    Immediate,
    Deferred,
    Adaptive,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ReductionMode {
    Mean,
    Sum,
    WeightedMean(f32),
}

impl Default for AccumulationConfig {
    fn default() -> Self {
        Self {
            steps: 1,
            normalize: true,
            dtype: AccumulationDType::Float32,
            sync_mode: SyncMode::Immediate,
            reduction: ReductionMode::Mean,
        }
    }
}

/// Gradient accumulator
pub struct GradientAccumulator {
    config: AccumulationConfig,
    accumulated_grads: HashMap<String, AccumulatedGradient>,
    current_step: usize,
}

struct AccumulatedGradient {
    gradient: Tensor,
    num_accumulations: usize,
    requires_sync: bool,
}

impl GradientAccumulator {
    /// Create new gradient accumulator
    pub fn new(config: AccumulationConfig) -> Self {
        Self {
            config,
            accumulated_grads: HashMap::new(),
            current_step: 0,
        }
    }
    
    /// Accumulate gradients
    pub fn accumulate(
        &mut self,
        name: &str,
        gradient: &Tensor,
        scale: Option<f32>,
    ) -> Result<()> {
        let scale = scale.unwrap_or(1.0);
        
        if let Some(acc_grad) = self.accumulated_grads.get_mut(name) {
            // Add to existing accumulation
            let scaled = if scale != 1.0 {
                (gradient * scale as f64)?
            } else {
                gradient.clone()
            };
            
            acc_grad.gradient = (&acc_grad.gradient + scaled)?;
            acc_grad.num_accumulations += 1;
        } else {
            // First accumulation
            let gradient = if scale != 1.0 {
                (gradient * scale as f64)?
            } else {
                gradient.clone()
            };
            
            self.accumulated_grads.insert(name.to_string(), AccumulatedGradient {
                gradient,
                num_accumulations: 1,
                requires_sync: true,
            });
        }
        
        self.current_step += 1;
        Ok(())
    }
    
    /// Check if ready to sync
    pub fn should_sync(&self) -> bool {
        self.current_step >= self.config.steps
    }
    
    /// Get accumulated gradients
    pub fn get_gradients(&mut self) -> Result<HashMap<String, Tensor>> {
        if !self.should_sync() {
            return Err(Error::InvalidInput(
                "Accumulation not complete".to_string()
            ));
        }
        
        let mut gradients = HashMap::new();
        
        for (name, acc_grad) in &self.accumulated_grads {
            let grad = match self.config.reduction {
                ReductionMode::Mean => {
                    if self.config.normalize {
                        (&acc_grad.gradient / acc_grad.num_accumulations as f64)?
                    } else {
                        (&acc_grad.gradient / self.config.steps as f64)?
                    }
                }
                ReductionMode::Sum => acc_grad.gradient.clone(),
                ReductionMode::WeightedMean(weight) => {
                    let divisor = if self.config.normalize {
                        acc_grad.num_accumulations as f64
                    } else {
                        self.config.steps as f64
                    };
                    acc_grad.gradient.affine(weight as f64 / divisor, 0.0)?
                }
            };
            
            gradients.insert(name.clone(), grad);
        }
        
        Ok(gradients)
    }
    
    /// Reset accumulator
    pub fn reset(&mut self) {
        self.accumulated_grads.clear();
        self.current_step = 0;
    }
    
    /// Get memory usage
    pub fn memory_usage(&self) -> usize {
        self.accumulated_grads.values()
            .map(|acc| acc.gradient.elem_count() * 4) // Assuming f32
            .sum()
    }
}

/// Dynamic accumulation strategy
pub struct DynamicAccumulator {
    base_config: AccumulationConfig,
    memory_threshold: usize,
    performance_monitor: PerformanceMonitor,
}

struct PerformanceMonitor {
    step_times: Vec<f32>,
    memory_usage: Vec<usize>,
    loss_values: Vec<f32>,
}

impl DynamicAccumulator {
    pub fn new(base_config: AccumulationConfig, memory_threshold: usize) -> Self {
        Self {
            base_config,
            memory_threshold,
            performance_monitor: PerformanceMonitor {
                step_times: Vec::new(),
                memory_usage: Vec::new(),
                loss_values: Vec::new(),
            },
        }
    }
    
    /// Adjust accumulation steps dynamically
    pub fn adjust_steps(&mut self, current_memory: usize, step_time: f32) -> usize {
        self.performance_monitor.memory_usage.push(current_memory);
        self.performance_monitor.step_times.push(step_time);
        
        // Simple heuristic: increase steps if memory is high
        if current_memory > self.memory_threshold {
            self.base_config.steps = (self.base_config.steps * 2).min(64);
        } else if current_memory < self.memory_threshold / 2 {
            self.base_config.steps = (self.base_config.steps / 2).max(1);
        }
        
        self.base_config.steps
    }
    
    /// Get optimal configuration based on history
    pub fn get_optimal_config(&self) -> AccumulationConfig {
        // Analyze performance history
        if self.performance_monitor.step_times.len() < 10 {
            return self.base_config.clone();
        }
        
        // Find configuration that minimizes time per effective batch
        let avg_time: f32 = self.performance_monitor.step_times.iter()
            .rev()
            .take(10)
            .sum::<f32>() / 10.0;
        
        let avg_memory: usize = self.performance_monitor.memory_usage.iter()
            .rev()
            .take(10)
            .sum::<usize>() / 10;
        
        let mut config = self.base_config.clone();
        
        // Adjust sync mode based on performance
        if avg_time > 1.0 {
            config.sync_mode = SyncMode::Deferred;
        } else {
            config.sync_mode = SyncMode::Immediate;
        }
        
        config
    }
}

/// Gradient checkpointing
pub struct GradientCheckpointing {
    enabled: bool,
    segment_size: usize,
    checkpointed_layers: Vec<String>,
}

impl GradientCheckpointing {
    pub fn new(enabled: bool, segment_size: usize) -> Self {
        Self {
            enabled,
            segment_size,
            checkpointed_layers: Vec::new(),
        }
    }
    
    /// Mark layer for checkpointing
    pub fn checkpoint_layer(&mut self, layer_name: &str) {
        if self.enabled {
            self.checkpointed_layers.push(layer_name.to_string());
        }
    }
    
    /// Check if layer should be checkpointed
    pub fn should_checkpoint(&self, layer_name: &str) -> bool {
        self.enabled && self.checkpointed_layers.contains(&layer_name.to_string())
    }
    
    /// Estimate memory savings
    pub fn estimate_memory_savings(&self, total_layers: usize) -> f32 {
        if !self.enabled {
            return 0.0;
        }
        
        let checkpointed_ratio = self.checkpointed_layers.len() as f32 / total_layers as f32;
        
        // Rough estimate: checkpointing saves ~33% memory per layer
        checkpointed_ratio * 0.33
    }
}

/// Accumulation utilities
pub mod utils {
    use super::*;
    
    /// Calculate effective batch size
    pub fn effective_batch_size(
        batch_size: usize,
        accumulation_steps: usize,
        world_size: Option<usize>,
    ) -> usize {
        let local_effective = batch_size * accumulation_steps;
        
        if let Some(world_size) = world_size {
            local_effective * world_size
        } else {
            local_effective
        }
    }
    
    /// Estimate memory requirement
    pub fn estimate_memory_requirement(
        model_params: usize,
        batch_size: usize,
        accumulation_steps: usize,
        dtype_bytes: usize,
    ) -> usize {
        // Model parameters
        let param_memory = model_params * dtype_bytes;
        
        // Gradients (one per parameter)
        let gradient_memory = model_params * dtype_bytes;
        
        // Accumulated gradients
        let accumulated_memory = gradient_memory; // Only one copy needed
        
        // Activations (rough estimate: 4x parameters per batch)
        let activation_memory = model_params * 4 * batch_size * dtype_bytes;
        
        param_memory + gradient_memory + accumulated_memory + activation_memory
    }
    
    /// Create accumulation schedule
    pub fn create_accumulation_schedule(
        total_steps: usize,
        warmup_steps: usize,
        max_accumulation: usize,
    ) -> Vec<usize> {
        let mut schedule = Vec::new();
        
        for step in 0..total_steps {
            let accumulation = if step < warmup_steps {
                // Linear warmup
                1 + (step * (max_accumulation - 1)) / warmup_steps
            } else {
                max_accumulation
            };
            
            schedule.push(accumulation);
        }
        
        schedule
    }
}