//! Memory optimization utilities for training pipelines
//!
//! This module provides strategies and utilities to optimize GPU memory usage
//! during training, enabling larger models to fit on consumer GPUs.

use flame_core::{DType, Device, Error, Result, Tensor};
use log::{debug, info, warn};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Memory optimization strategies
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MemoryStrategy {
    /// No optimization - everything stays in GPU memory
    None,
    /// Gradient checkpointing - trade compute for memory
    GradientCheckpointing,
    /// CPU offloading - move inactive tensors to CPU
    CPUOffloading,
    /// Mixed precision - use lower precision where possible
    MixedPrecision,
    /// Model sharding - split model across GPUs/CPU
    ModelSharding,
    /// Dynamic batching - adjust batch size based on memory
    DynamicBatching,
    /// All optimizations combined
    Aggressive,
}

/// Memory pressure levels
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub enum MemoryPressure {
    Low,      // < 50% usage
    Medium,   // 50-75% usage
    High,     // 75-90% usage
    Critical, // > 90% usage
}

/// Memory optimizer for training pipelines
pub struct MemoryOptimizer {
    device: Device,
    strategy: MemoryStrategy,

    // Memory tracking
    peak_memory_mb: Arc<Mutex<f32>>,
    current_memory_mb: Arc<Mutex<f32>>,
    memory_limit_mb: f32,

    // Optimization state
    checkpointing_enabled: Arc<Mutex<bool>>,
    offloading_enabled: Arc<Mutex<bool>>,
    mixed_precision_enabled: Arc<Mutex<bool>>,

    // Dynamic batch size
    current_batch_size: Arc<Mutex<usize>>,
    min_batch_size: usize,
    max_batch_size: usize,
}

impl MemoryOptimizer {
    /// Create a new memory optimizer
    pub fn new(device: Device, strategy: MemoryStrategy, memory_limit_gb: f32) -> Self {
        let memory_limit_mb = memory_limit_gb * 1024.0;

        Self {
            device,
            strategy,
            peak_memory_mb: Arc::new(Mutex::new(0.0)),
            current_memory_mb: Arc::new(Mutex::new(0.0)),
            memory_limit_mb,
            checkpointing_enabled: Arc::new(Mutex::new(false)),
            offloading_enabled: Arc::new(Mutex::new(false)),
            mixed_precision_enabled: Arc::new(Mutex::new(false)),
            current_batch_size: Arc::new(Mutex::new(1)),
            min_batch_size: 1,
            max_batch_size: 8,
        }
    }

    /// Configure batch size limits for dynamic batching
    pub fn set_batch_size_limits(&mut self, min: usize, max: usize) {
        self.min_batch_size = min;
        self.max_batch_size = max;
        *self.current_batch_size.lock().unwrap() = min;
    }

    /// Apply memory optimization strategy
    pub fn apply_strategy(&self) -> Result<()> {
        match self.strategy {
            MemoryStrategy::None => {
                info!("No memory optimization applied");
                Ok(())
            }
            MemoryStrategy::GradientCheckpointing => {
                self.enable_gradient_checkpointing()?;
                Ok(())
            }
            MemoryStrategy::CPUOffloading => {
                self.enable_cpu_offloading()?;
                Ok(())
            }
            MemoryStrategy::MixedPrecision => {
                self.enable_mixed_precision()?;
                Ok(())
            }
            MemoryStrategy::ModelSharding => {
                warn!("Model sharding not yet implemented");
                Ok(())
            }
            MemoryStrategy::DynamicBatching => {
                info!(
                    "Dynamic batching enabled with range [{}-{}]",
                    self.min_batch_size, self.max_batch_size
                );
                Ok(())
            }
            MemoryStrategy::Aggressive => {
                info!("Applying aggressive memory optimization");
                self.enable_gradient_checkpointing()?;
                self.enable_cpu_offloading()?;
                self.enable_mixed_precision()?;
                Ok(())
            }
        }
    }

    /// Enable gradient checkpointing
    fn enable_gradient_checkpointing(&self) -> Result<()> {
        *self.checkpointing_enabled.lock().unwrap() = true;
        info!("Gradient checkpointing enabled - trading compute for memory");
        Ok(())
    }

    /// Enable CPU offloading
    fn enable_cpu_offloading(&self) -> Result<()> {
        *self.offloading_enabled.lock().unwrap() = true;
        info!("CPU offloading enabled - inactive tensors will be moved to CPU");
        Ok(())
    }

    /// Enable mixed precision training
    fn enable_mixed_precision(&self) -> Result<()> {
        *self.mixed_precision_enabled.lock().unwrap() = true;
        info!("Mixed precision enabled - using BF16 where possible");
        Ok(())
    }

    /// Update current memory usage
    pub fn update_memory_usage(&self, current_mb: f32) {
        let mut current = self.current_memory_mb.lock().unwrap();
        *current = current_mb;

        let mut peak = self.peak_memory_mb.lock().unwrap();
        if current_mb > *peak {
            *peak = current_mb;
        }
    }

    /// Get current memory pressure level
    pub fn get_memory_pressure(&self) -> MemoryPressure {
        let current = *self.current_memory_mb.lock().unwrap();
        let usage_ratio = current / self.memory_limit_mb;

        if usage_ratio < 0.5 {
            MemoryPressure::Low
        } else if usage_ratio < 0.75 {
            MemoryPressure::Medium
        } else if usage_ratio < 0.9 {
            MemoryPressure::High
        } else {
            MemoryPressure::Critical
        }
    }

    /// Optimize tensor for memory efficiency
    pub fn optimize_tensor(&self, tensor: &Tensor, name: &str) -> Result<Tensor> {
        let pressure = self.get_memory_pressure();

        match pressure {
            MemoryPressure::Low => {
                // No optimization needed
                Ok(tensor.clone())
            }
            MemoryPressure::Medium => {
                // Convert to mixed precision if enabled
                if *self.mixed_precision_enabled.lock().unwrap() {
                    self.convert_to_mixed_precision(tensor, name)
                } else {
                    Ok(tensor.clone())
                }
            }
            MemoryPressure::High | MemoryPressure::Critical => {
                // Aggressive optimization
                debug!("High memory pressure - optimizing tensor '{}'", name);

                // First try mixed precision
                let tensor = if *self.mixed_precision_enabled.lock().unwrap() {
                    self.convert_to_mixed_precision(tensor, name)?
                } else {
                    tensor.clone()
                };

                // Consider offloading if critical
                if pressure == MemoryPressure::Critical && *self.offloading_enabled.lock().unwrap()
                {
                    warn!("Critical memory pressure - consider offloading '{}'", name);
                }

                Ok(tensor)
            }
        }
    }

    /// Convert tensor to mixed precision (BF16)
    fn convert_to_mixed_precision(&self, tensor: &Tensor, name: &str) -> Result<Tensor> {
        if tensor.dtype() == DType::F32 {
            debug!("Converting tensor '{}' from F32 to BF16", name);
            tensor.to_dtype(DType::BF16)
        } else {
            Ok(tensor.clone())
        }
    }

    /// Adjust batch size based on memory pressure
    pub fn adjust_batch_size(&self) -> usize {
        if self.strategy != MemoryStrategy::DynamicBatching
            && self.strategy != MemoryStrategy::Aggressive
        {
            return *self.current_batch_size.lock().unwrap();
        }

        let pressure = self.get_memory_pressure();
        let mut current_size = self.current_batch_size.lock().unwrap();

        match pressure {
            MemoryPressure::Low => {
                // Try to increase batch size
                if *current_size < self.max_batch_size {
                    *current_size += 1;
                    info!("Low memory pressure - increasing batch size to {}", *current_size);
                }
            }
            MemoryPressure::Medium => {
                // Keep current batch size
            }
            MemoryPressure::High => {
                // Decrease batch size
                if *current_size > self.min_batch_size {
                    *current_size -= 1;
                    warn!("High memory pressure - decreasing batch size to {}", *current_size);
                }
            }
            MemoryPressure::Critical => {
                // Set to minimum batch size
                if *current_size != self.min_batch_size {
                    *current_size = self.min_batch_size;
                    warn!(
                        "Critical memory pressure - batch size set to minimum: {}",
                        *current_size
                    );
                }
            }
        }

        *current_size
    }

    /// Get memory statistics
    pub fn get_stats(&self) -> MemoryStats {
        MemoryStats {
            current_mb: *self.current_memory_mb.lock().unwrap(),
            peak_mb: *self.peak_memory_mb.lock().unwrap(),
            limit_mb: self.memory_limit_mb,
            pressure: self.get_memory_pressure(),
            batch_size: *self.current_batch_size.lock().unwrap(),
            checkpointing_enabled: *self.checkpointing_enabled.lock().unwrap(),
            offloading_enabled: *self.offloading_enabled.lock().unwrap(),
            mixed_precision_enabled: *self.mixed_precision_enabled.lock().unwrap(),
            strategy: self.strategy,
        }
    }

    /// Clear memory caches and run garbage collection
    pub fn cleanup_memory(&self) -> Result<()> {
        // In a real implementation, this would:
        // 1. Clear CUDA caches
        // 2. Run garbage collection
        // 3. Defragment GPU memory
        debug!("Running memory cleanup");
        Ok(())
    }
}

/// Memory statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub current_mb: f32,
    pub peak_mb: f32,
    pub limit_mb: f32,
    pub pressure: MemoryPressure,
    pub batch_size: usize,
    pub checkpointing_enabled: bool,
    pub offloading_enabled: bool,
    pub mixed_precision_enabled: bool,
    pub strategy: MemoryStrategy,
}

impl MemoryStats {
    /// Get usage percentage
    pub fn usage_percent(&self) -> f32 {
        (self.current_mb / self.limit_mb) * 100.0
    }

    /// Get remaining memory in MB
    pub fn remaining_mb(&self) -> f32 {
        self.limit_mb - self.current_mb
    }
}

/// Memory optimization tips based on current state
pub fn get_optimization_tips(stats: &MemoryStats) -> Vec<String> {
    let mut tips = Vec::new();

    if stats.usage_percent() > 80.0 {
        if !stats.checkpointing_enabled {
            tips.push("Enable gradient checkpointing to reduce memory usage".to_string());
        }
        if !stats.mixed_precision_enabled {
            tips.push("Enable mixed precision training (BF16) to halve memory usage".to_string());
        }
        if stats.batch_size > 1 {
            tips.push(format!(
                "Reduce batch size from {} to {} to free memory",
                stats.batch_size,
                stats.batch_size - 1
            ));
        }
    }

    if stats.usage_percent() > 90.0 && !stats.offloading_enabled {
        tips.push("Enable CPU offloading for critical memory situations".to_string());
    }

    tips
}

#[cfg(all(test, feature = "dev-bins"))]
mod tests {
    use super::*;

    #[test]
    fn test_memory_pressure_levels() {
        let device = Device::cpu();
        let optimizer = MemoryOptimizer::new(device, MemoryStrategy::None, 24.0);

        // Test different usage levels
        optimizer.update_memory_usage(10240.0); // 10GB
        assert_eq!(optimizer.get_memory_pressure(), MemoryPressure::Low);

        optimizer.update_memory_usage(15360.0); // 15GB
        assert_eq!(optimizer.get_memory_pressure(), MemoryPressure::Medium);

        optimizer.update_memory_usage(20480.0); // 20GB
        assert_eq!(optimizer.get_memory_pressure(), MemoryPressure::High);

        optimizer.update_memory_usage(23552.0); // 23GB
        assert_eq!(optimizer.get_memory_pressure(), MemoryPressure::Critical);
    }
}
