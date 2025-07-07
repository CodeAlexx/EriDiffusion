//! CPU offloading for optimizer states
//! 
//! This module implements CPU offloading for AdamW optimizer states,
//! which can save significant GPU memory during training.

use anyhow::{Result, Context};
use candle_core::{Device, DType, Tensor, Var};
use candle_nn::{Optimizer, ParamsAdamW};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// CPU-offloaded AdamW optimizer
pub struct OffloadedAdamW {
    /// Learning rate
    lr: f64,
    /// Beta1 parameter
    beta1: f64,
    /// Beta2 parameter  
    beta2: f64,
    /// Epsilon for numerical stability
    eps: f64,
    /// Weight decay
    weight_decay: f64,
    /// Current step
    step: usize,
    /// Parameters being optimized
    params: Vec<Var>,
    /// First moment estimates (on CPU)
    m: Arc<Mutex<HashMap<String, Tensor>>>,
    /// Second moment estimates (on CPU)
    v: Arc<Mutex<HashMap<String, Tensor>>>,
    /// GPU device for computation
    device: Device,
}

impl OffloadedAdamW {
    pub fn new(params: Vec<Var>, config: ParamsAdamW) -> Result<Self> {
        // Detect device from first parameter
        let device = if let Some(first_param) = params.first() {
            first_param.device().clone()
        } else {
            Device::Cpu
        };
        
        println!("Creating CPU-offloaded AdamW optimizer for {} parameters", params.len());
        
        Ok(Self {
            lr: config.lr,
            beta1: config.beta1,
            beta2: config.beta2,
            eps: config.eps,
            weight_decay: config.weight_decay,
            step: 0,
            params,
            m: Arc::new(Mutex::new(HashMap::new())),
            v: Arc::new(Mutex::new(HashMap::new())),
            device,
        })
    }
    
    /// Initialize moment buffers on CPU
    fn init_moments(&self) -> Result<()> {
        let mut m = self.m.lock().unwrap();
        let mut v = self.v.lock().unwrap();
        
        for (i, param) in self.params.iter().enumerate() {
            let shape = param.shape();
            let zeros_cpu = Tensor::zeros(shape, DType::F32, &Device::Cpu)?;
            
            m.insert(format!("param_{}", i), zeros_cpu.clone());
            v.insert(format!("param_{}", i), zeros_cpu);
        }
        
        println!("Initialized moment buffers on CPU");
        Ok(())
    }
    
    /// Perform optimization step with CPU offloading
    pub fn step(&mut self, gradients: &[Option<Tensor>]) -> Result<()> {
        self.step += 1;
        let step = self.step;
        
        // Initialize moments on first step
        if step == 1 {
            self.init_moments()?;
        }
        
        // Bias correction
        let bias_correction1 = 1.0 - self.beta1.powi(step as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(step as i32);
        let step_size = self.lr * (bias_correction2.sqrt() / bias_correction1);
        
        let mut m = self.m.lock().unwrap();
        let mut v = self.v.lock().unwrap();
        
        for (i, (param, grad_opt)) in self.params.iter().zip(gradients.iter()).enumerate() {
            if let Some(grad) = grad_opt {
                let param_name = format!("param_{}", i);
                
                // Move gradient to CPU for moment computation
                let grad_cpu = grad.to_device(&Device::Cpu)?;
                
                // Get moment buffers
                let m_t = m.get_mut(&param_name)
                    .ok_or_else(|| anyhow::anyhow!("Missing first moment for {}", param_name))?;
                let v_t = v.get_mut(&param_name)
                    .ok_or_else(|| anyhow::anyhow!("Missing second moment for {}", param_name))?;
                
                // Update biased first moment estimate
                // m_t = beta1 * m_t + (1 - beta1) * grad
                *m_t = (m_t.affine(self.beta1, 0.0)? + grad_cpu.affine(1.0 - self.beta1, 0.0)?)?;
                
                // Update biased second raw moment estimate
                // v_t = beta2 * v_t + (1 - beta2) * grad^2
                let grad_sq = grad_cpu.powf(2.0)?;
                *v_t = (v_t.affine(self.beta2, 0.0)? + grad_sq.affine(1.0 - self.beta2, 0.0)?)?;
                
                // Compute update on CPU
                // update = step_size * m_t / (sqrt(v_t) + eps)
                let v_sqrt = v_t.sqrt()?;
                let denom = v_sqrt.affine(1.0, self.eps)?;
                let update_cpu = (m_t.affine(step_size, 0.0)? / denom)?;
                
                // Apply weight decay if needed
                let update_cpu = if self.weight_decay > 0.0 {
                    let param_cpu = param.as_tensor().to_device(&Device::Cpu)?;
                    (update_cpu + param_cpu.affine(self.weight_decay * self.lr, 0.0)?)?
                } else {
                    update_cpu
                };
                
                // Move update back to GPU and apply
                let update_gpu = update_cpu.to_device(&self.device)?;
                
                // Update parameter: param = param - update
                let new_value = (param.as_tensor() - update_gpu)?;
                param.set(&new_value)?;
            }
        }
        
        Ok(())
    }
    
    /// Zero gradients
    pub fn zero_grad(&self) -> Result<()> {
        // In Candle, gradients are typically managed differently
        // This is a placeholder for compatibility
        Ok(())
    }
    
    /// Get current learning rate
    pub fn learning_rate(&self) -> f64 {
        self.lr
    }
    
    /// Set learning rate
    pub fn set_learning_rate(&mut self, lr: f64) {
        self.lr = lr;
    }
    
    /// Get memory usage statistics
    pub fn memory_stats(&self) -> Result<(usize, usize)> {
        let m = self.m.lock().unwrap();
        let v = self.v.lock().unwrap();
        
        let mut cpu_bytes = 0;
        let mut gpu_bytes = 0;
        
        // CPU memory (moment buffers)
        for (_, tensor) in m.iter() {
            cpu_bytes += tensor.elem_count() * 4; // F32 = 4 bytes
        }
        for (_, tensor) in v.iter() {
            cpu_bytes += tensor.elem_count() * 4;
        }
        
        // GPU memory (parameters only)
        for param in &self.params {
            gpu_bytes += param.elem_count() * 2; // Assuming F16 = 2 bytes
        }
        
        Ok((cpu_bytes, gpu_bytes))
    }
}

/// Create an offloaded optimizer from regular AdamW parameters
pub fn create_offloaded_optimizer(
    params: Vec<Var>,
    lr: f64,
) -> Result<OffloadedAdamW> {
    let config = ParamsAdamW {
        lr,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: 0.01,
    };
    
    OffloadedAdamW::new(params, config)
}

/// Memory usage comparison
pub fn print_optimizer_memory_comparison(param_count: usize) {
    println!("\n=== Optimizer Memory Comparison ===");
    
    let param_bytes = param_count * 2; // F16 parameters
    let moment_bytes = param_count * 4 * 2; // F32 first and second moments
    
    println!("Standard AdamW (all on GPU):");
    println!("  - Parameters: {:.1} MB", param_bytes as f32 / 1e6);
    println!("  - Moments (m, v): {:.1} MB", moment_bytes as f32 / 1e6);
    println!("  - Total GPU: {:.1} MB", (param_bytes + moment_bytes) as f32 / 1e6);
    
    println!("\nCPU-Offloaded AdamW:");
    println!("  - Parameters: {:.1} MB (GPU)", param_bytes as f32 / 1e6);
    println!("  - Moments (m, v): {:.1} MB (CPU)", moment_bytes as f32 / 1e6);
    println!("  - Total GPU: {:.1} MB", param_bytes as f32 / 1e6);
    println!("  - GPU savings: {:.1} MB", moment_bytes as f32 / 1e6);
    
    println!("\nFor Flux LoRA with 50M parameters:");
    println!("  - Saves ~400MB GPU memory");
    println!("  - Overhead: ~10-15% slower due to PCIe transfers");
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    
    #[test]
    fn test_offloaded_optimizer() -> Result<()> {
        // Create dummy parameters
        let device = Device::Cpu;
        let param1 = Var::from_tensor(&Tensor::randn(0f32, 1.0, (100, 100), &device)?)?;
        let param2 = Var::from_tensor(&Tensor::randn(0f32, 1.0, (50, 50), &device)?)?;
        
        let params = vec![param1, param2];
        
        // Create optimizer
        let mut optimizer = create_offloaded_optimizer(params, 0.001)?;
        
        // Test memory stats
        let (cpu_mem, gpu_mem) = optimizer.memory_stats()?;
        assert!(cpu_mem == 0); // No moments initialized yet
        assert!(gpu_mem > 0); // Parameters should be counted
        
        Ok(())
    }
}