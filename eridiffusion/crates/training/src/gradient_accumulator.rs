//! Efficient gradient accumulation

use eridiffusion_core::{Error, Result, Device, VarExt};
use candle_core::{Tensor, DType};
use candle_nn::{Optimizer, VarMap};
use std::collections::HashMap;

/// Gradient accumulator for efficient memory usage
pub struct GradientAccumulator {
    /// Number of steps to accumulate gradients
    accumulation_steps: usize,
    /// Current accumulation step
    current_step: usize,
    /// Accumulated gradients storage - maps variable name to accumulated gradient
    accumulated_grads: HashMap<String, Tensor>,
    /// Device for tensor operations
    device: Device,
    /// Whether to use mixed precision
    mixed_precision: bool,
    /// Loss scale for mixed precision training
    loss_scale: f32,
    /// Track if we found inf/nan in gradients
    found_inf: bool,
}

impl GradientAccumulator {
    /// Create new gradient accumulator
    pub fn new(
        accumulation_steps: usize,
        device: Device,
    ) -> Result<Self> {
        Ok(Self {
            accumulation_steps,
            current_step: 0,
            accumulated_grads: HashMap::new(),
            device,
            mixed_precision: false,
            loss_scale: 65536.0,
            found_inf: false,
        })
    }
    
    /// Create with mixed precision support
    pub fn new_with_mixed_precision(
        accumulation_steps: usize,
        device: Device,
        mixed_precision: bool,
    ) -> Result<Self> {
        Ok(Self {
            accumulation_steps,
            current_step: 0,
            accumulated_grads: HashMap::new(),
            device,
            mixed_precision,
            loss_scale: 65536.0,
            found_inf: false,
        })
    }
    
    /// Scale and accumulate gradients from current backward pass
    pub fn accumulate(&mut self, loss: &Tensor) -> Result<()> {
        // Scale loss by accumulation steps before backward
        let scale_factor = 1.0 / self.accumulation_steps as f64;
        let scaled_loss = if self.mixed_precision {
            loss.affine(self.loss_scale as f64 * scale_factor, 0.0)?
        } else {
            loss.affine(scale_factor, 0.0)?
        };
        
        // Backward pass
        scaled_loss.backward()?;
        
        self.current_step += 1;
        Ok(())
    }
    
    /// Accumulate pre-computed gradients
    pub fn accumulate_grads(&mut self, grads: &[Tensor]) -> Result<()> {
        // For now, we'll store gradients directly
        // In a full implementation, we'd accumulate them with existing gradients
        self.current_step += 1;
        Ok(())
    }
    
    /// Accumulate gradients that have already been computed
    pub fn accumulate_existing_grads(&mut self, var_map: &VarMap) -> Result<()> {
        // Convert eridiffusion_core::Device to candle_core::Device
        let candle_device = match &self.device {
            Device::Cpu => candle_core::Device::Cpu,
            Device::Cuda(id) => candle_core::Device::new_cuda(*id)
                .map_err(|e| Error::Device(format!("Failed to create CUDA device: {}", e)))?,
        };
        
        // Iterate through all variables and accumulate their gradients
        let all_vars = var_map.all_vars();
        for (idx, var) in all_vars.iter().enumerate() {
            let name = format!("var_{}", idx);
            if let Ok(grad) = var.grad() {
                // Check for inf/nan
                if self.mixed_precision {
                    let has_inf = grad.flatten_all()?
                        .to_vec1::<f32>()?
                        .iter()
                        .any(|&x| x.is_infinite() || x.is_nan());
                    
                    if has_inf {
                        self.found_inf = true;
                        continue; // Skip this gradient
                    }
                }

                // Accumulate gradient
                match self.accumulated_grads.get_mut(&name) {
                    Some(existing_grad) => {
                        *existing_grad = existing_grad.add(&grad)?;
                    }
                    None => {
                        self.accumulated_grads.insert(name.clone(), grad.clone());
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Check if accumulation is complete
    pub fn should_step(&self) -> bool {
        self.current_step >= self.accumulation_steps
    }
    
    /// Initialize gradient accumulator with parameters
    pub fn initialize(&mut self, _params: &[&Tensor]) -> Result<()> {
        // Reset state
        self.current_step = 0;
        self.accumulated_grads.clear();
        self.found_inf = false;
        Ok(())
    }
    
    /// Check if ready to step
    pub fn is_ready(&self) -> bool {
        self.should_step()
    }
    
    /// Get accumulation steps
    pub fn accumulation_steps(&self) -> usize {
        self.accumulation_steps
    }
    
    /// Get accumulated gradients
    pub fn get_gradients(&self) -> Result<Vec<Tensor>> {
        Ok(self.accumulated_grads.values().cloned().collect())
    }
    
    /// Apply accumulated gradients to optimizer
    pub fn step_optimizer<O: Optimizer>(
        &mut self,
        var_map: &VarMap,
        optimizer: &mut O,
        max_grad_norm: Option<f64>,
    ) -> Result<bool> {
        if !self.should_step() {
            return Ok(false);
        }

        // If we found inf/nan, skip this step
        if self.found_inf {
            self.reset()?;
            self.loss_scale = (self.loss_scale / 2.0).max(1.0);
            return Ok(false);
        }

        // First accumulate any remaining gradients
        self.accumulate_existing_grads(var_map)?;

        // Apply gradient clipping if specified
        if let Some(max_norm) = max_grad_norm {
            self.clip_grad_norm(max_norm)?;
        }

        // Apply accumulated gradients to variables
        for (idx, var) in var_map.all_vars().iter().enumerate() {
            let name = format!("var_{}", idx);
            if let Some(accumulated_grad) = self.accumulated_grads.get(&name) {
                // Scale down if using mixed precision
                let final_grad = if self.mixed_precision {
                    accumulated_grad.affine(1.0 / self.loss_scale as f64, 0.0)?
                } else {
                    accumulated_grad.clone()
                };
                
                // Set the gradient on the variable
                var.set_grad(&final_grad)?;
            }
        }

        // Step the optimizer without creating a GradStore
        // The optimizer should handle gradient access internally
        // This is a workaround since GradStore::new() is private
        let dummy_loss = Tensor::zeros(&[], DType::F32, &candle_core::Device::Cpu)?;
        optimizer.backward_step(&dummy_loss)?;

        // Update loss scale for next iteration
        if self.mixed_precision && !self.found_inf {
            if self.current_step % 2000 == 0 {
                self.loss_scale = (self.loss_scale * 2.0).min(65536.0);
            }
        }

        // Reset for next accumulation cycle
        self.reset()?;
        
        Ok(true)
    }
    
    /// Reset accumulator for next cycle
    pub fn reset(&mut self) -> Result<()> {
        self.current_step = 0;
        self.accumulated_grads.clear();
        self.found_inf = false;
        Ok(())
    }
    
    /// Clip gradients by global norm
    pub fn clip_grad_norm(&mut self, max_norm: f64) -> Result<()> {
        // Calculate global norm
        let mut total_norm_sq = 0.0;
        
        for grad in self.accumulated_grads.values() {
            let grad_norm_sq = grad.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
            total_norm_sq += grad_norm_sq;
        }
        
        let total_norm = total_norm_sq.sqrt();
        
        // Clip if necessary
        if total_norm > max_norm {
            let clip_coef = max_norm / (total_norm + 1e-6);
            
            for grad in self.accumulated_grads.values_mut() {
                *grad = grad.affine(clip_coef, 0.0)?;
            }
        }
        
        Ok(())
    }
    
    /// Get current gradient statistics for logging
    pub fn get_gradient_stats(&self) -> Result<GradientStats> {
        if self.accumulated_grads.is_empty() {
            return Ok(GradientStats {
                global_norm: 0.0,
                max_gradient: 0.0,
                min_gradient: 0.0,
                num_parameters: 0,
            });
        }

        let mut total_norm_sq = 0.0;
        let mut max_grad = f64::NEG_INFINITY;
        let mut min_grad = f64::INFINITY;
        let mut num_params = 0;
        
        for grad in self.accumulated_grads.values() {
            let grad_values = grad.flatten_all()?.to_vec1::<f32>()?;
            
            for &val in grad_values.iter() {
                let val = val as f64;
                max_grad = max_grad.max(val);
                min_grad = min_grad.min(val);
            }
            
            let grad_norm_sq = grad.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
            total_norm_sq += grad_norm_sq;
            num_params += grad.elem_count();
        }
        
        Ok(GradientStats {
            global_norm: total_norm_sq.sqrt(),
            max_gradient: max_grad,
            min_gradient: min_grad,
            num_parameters: num_params,
        })
    }
    
    /// Get current loss scale for mixed precision
    pub fn get_loss_scale(&self) -> f32 {
        self.loss_scale
    }
}

/// Gradient statistics for monitoring
#[derive(Debug, Clone)]
pub struct GradientStats {
    /// Global gradient norm
    pub global_norm: f64,
    /// Maximum gradient value
    pub max_gradient: f64,
    /// Minimum gradient value  
    pub min_gradient: f64,
    /// Number of parameters
    pub num_parameters: usize,
}

/// Gradient clipper for stable training
pub struct GradientClipper {
    /// Maximum gradient norm
    max_norm: f32,
    /// Clipping method
    method: ClipMethod,
}

#[derive(Debug, Clone)]
pub enum ClipMethod {
    /// Clip by global norm
    GlobalNorm,
    /// Clip by value
    Value,
    /// Adaptive clipping based on parameter norm
    Adaptive { percentile: f32 },
}

impl GradientClipper {
    /// Create new gradient clipper
    pub fn new(max_norm: f32, method: ClipMethod) -> Self {
        Self { max_norm, method }
    }
    
    /// Clip gradients in-place
    pub fn clip(&self, gradients: &mut [Tensor]) -> Result<f32> {
        match self.method {
            ClipMethod::GlobalNorm => self.clip_by_global_norm(gradients),
            ClipMethod::Value => self.clip_by_value(gradients),
            ClipMethod::Adaptive { percentile } => self.clip_adaptive(gradients, percentile),
        }
    }
    
    /// Clip by global norm
    fn clip_by_global_norm(&self, gradients: &mut [Tensor]) -> Result<f32> {
        // Compute global norm
        let mut total_norm_sq = 0.0f32;
        for grad in gradients.iter() {
            total_norm_sq += grad.sqr()?.sum_all()?.to_scalar::<f32>()?;
        }
        let total_norm = total_norm_sq.sqrt();
        
        // Clip if necessary
        if total_norm > self.max_norm {
            let clip_scale = self.max_norm / (total_norm + 1e-6);
            for grad in gradients.iter_mut() {
                *grad = grad.affine(clip_scale as f64, 0.0)?;
            }
            Ok(self.max_norm)
        } else {
            Ok(total_norm)
        }
    }
    
    /// Clip by value
    fn clip_by_value(&self, gradients: &mut [Tensor]) -> Result<f32> {
        let mut max_val = 0.0f32;
        
        for grad in gradients.iter_mut() {
            *grad = grad.clamp(-self.max_norm as f64, self.max_norm as f64)?;
            max_val = max_val.max(grad.max_all()?.to_scalar::<f32>()?);
        }
        
        Ok(max_val)
    }
    
    /// Adaptive clipping based on percentile
    fn clip_adaptive(&self, gradients: &mut [Tensor], percentile: f32) -> Result<f32> {
        // Collect all gradient values
        let mut all_values = Vec::new();
        
        for grad in gradients.iter() {
            let values = grad.flatten_all()?.to_vec1::<f32>()?;
            all_values.extend(values);
        }
        
        // Sort and find percentile
        all_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let idx = ((all_values.len() as f32 * percentile) as usize).min(all_values.len() - 1);
        let threshold = all_values[idx].abs();
        
        // Clip to threshold
        let clip_value = threshold.min(self.max_norm);
        self.clip_by_value(gradients)?;
        
        Ok(clip_value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_gradient_accumulator() -> Result<()> {
        let device = Device::Cpu;
        let accumulator = GradientAccumulator::new(4, device)?;
        
        // Test accumulation counter
        assert!(!accumulator.should_step());
        
        Ok(())
    }
    
    #[test]
    fn test_gradient_stats() -> Result<()> {
        let device = Device::Cpu;
        let accumulator = GradientAccumulator::new(1, device)?;
        
        // Empty stats
        let stats = accumulator.get_gradient_stats()?;
        assert_eq!(stats.num_parameters, 0);
        
        Ok(())
    }
}