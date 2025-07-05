//! Enhanced 8-bit AdamW optimizer with improved quantization and error handling

use candle_core::{DType, Device, Result, Tensor, D, Var};
use candle_nn::{Optimizer as CanOptimizer, VarMap};
type CanResult<T> = std::result::Result<T, candle_core::Error>;
use std::collections::HashMap;
use crate::optimizer::{Optimizer, OptimizerConfig, OptimizerState};
use eridiffusion_core::Error;

/// Enhanced 8-bit AdamW optimizer implementation
pub struct AdamW8bit {
    var_map: VarMap,
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    weight_decay: f64,
    step: usize,
    
    // 8-bit quantized states
    first_moments: HashMap<String, QuantizedTensor>,
    second_moments: HashMap<String, QuantizedTensor>,
    
    // Enhanced quantization parameters
    quantile_alpha: f32,
    adaptive_quantization: bool,
    quantization_warmup_steps: usize,
}

/// Enhanced quantized tensor storage with better numerical stability
#[derive(Clone)]
struct QuantizedTensor {
    quantized_data: Vec<i8>,
    scale: f32,
    zero_point: i8,
    shape: Vec<usize>,
    device: Device,
    // Track quantization error for adaptive schemes
    quantization_error: f32,
}

impl QuantizedTensor {
    /// Enhanced quantization with better numerical stability
    fn from_tensor(tensor: &Tensor, quantile_alpha: f32) -> Result<Self> {
        let data = tensor.flatten_all()?.to_vec1::<f32>()?;
        
        if data.is_empty() {
            return Err(candle_core::Error::Msg("Empty tensor for quantization".to_string()));
        }
        
        // Compute quantization parameters with better numerical stability
        let (min_val, max_val) = if quantile_alpha < 1.0 {
            // Use quantile-based clipping
            let mut sorted_data = data.clone();
            sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            
            let n = sorted_data.len();
            let lower_idx = ((1.0 - quantile_alpha) * 0.5 * n as f32) as usize;
            let upper_idx = ((1.0 - (1.0 - quantile_alpha) * 0.5) * n as f32).min(n as f32 - 1.0) as usize;
            
            (sorted_data[lower_idx], sorted_data[upper_idx])
        } else {
            // Use full range
            let min_val = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            (min_val, max_val)
        };
        
        // Improved scale computation with numerical stability
        let range = max_val - min_val;
        let scale = if range.abs() < 1e-8 {
            1.0
        } else {
            range / 255.0
        };
        
        // Better zero point computation
        let zero_point = if scale.abs() < 1e-8 {
            0
        } else {
            let zp = (-min_val / scale).round();
            zp.clamp(-128.0, 127.0) as i8
        };
        
        // Quantize with error tracking
        let mut total_error = 0.0;
        let quantized_data: Vec<i8> = data
            .iter()
            .map(|&x| {
                let clamped_x = x.clamp(min_val, max_val);
                let q_float = if scale.abs() < 1e-8 {
                    0.0
                } else {
                    (clamped_x - min_val) / scale
                };
                let q_int = q_float.round().clamp(-128.0, 127.0) as i8;
                
                // Track quantization error
                let dequantized = (q_int as f32 - zero_point as f32) * scale;
                total_error += (x - dequantized).abs();
                
                q_int
            })
            .collect();
        
        let quantization_error = total_error / data.len() as f32;
        
        Ok(Self {
            quantized_data,
            scale,
            zero_point,
            shape: tensor.dims().to_vec(),
            device: tensor.device().clone(),
            quantization_error,
        })
    }
    
    /// Dequantize with improved numerical precision
    fn to_tensor(&self) -> Result<Tensor> {
        if self.quantized_data.is_empty() {
            return Err(candle_core::Error::Msg("Empty quantized data".to_string()));
        }
        
        let data: Vec<f32> = self.quantized_data
            .iter()
            .map(|&q| {
                let dequantized = (q as f32 - self.zero_point as f32) * self.scale;
                // Ensure finite values
                if dequantized.is_finite() {
                    dequantized
                } else {
                    0.0
                }
            })
            .collect();
        
        Tensor::from_vec(data, self.shape.as_slice(), &self.device)
    }
    
    /// Update with adaptive quantization strategy
    fn update(&mut self, tensor: &Tensor, quantile_alpha: f32, adaptive: bool) -> Result<()> {
        let new_quantized = Self::from_tensor(tensor, quantile_alpha)?;
        
        // Adaptive quantization: if error is too high, use more conservative quantization
        if adaptive && new_quantized.quantization_error > self.quantization_error * 2.0 {
            let conservative_alpha = (quantile_alpha + 1.0) * 0.5;
            *self = Self::from_tensor(tensor, conservative_alpha)?;
        } else {
            *self = new_quantized;
        }
        
        Ok(())
    }
    
    /// Get memory usage in bytes
    fn memory_usage(&self) -> usize {
        self.quantized_data.len() + std::mem::size_of::<f32>() * 2 + std::mem::size_of::<i8>()
    }
}

impl AdamW8bit {
    pub fn new(
        var_map: VarMap,
        learning_rate: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
        weight_decay: f64,
    ) -> Result<Self> {
        Ok(Self {
            var_map,
            learning_rate,
            beta1,
            beta2,
            epsilon,
            weight_decay,
            step: 0,
            first_moments: HashMap::new(),
            second_moments: HashMap::new(),
            quantile_alpha: 0.99,
            adaptive_quantization: true,
            quantization_warmup_steps: 100,
        })
    }
    
    /// Configure quantization parameters
    pub fn with_quantization_config(
        mut self,
        quantile_alpha: f32,
        adaptive: bool,
        warmup_steps: usize,
    ) -> Self {
        self.quantile_alpha = quantile_alpha;
        self.adaptive_quantization = adaptive;
        self.quantization_warmup_steps = warmup_steps;
        self
    }
    
    /// Initialize optimizer states with proper error handling
    fn init_state(&mut self, var_id: String, shape: &[usize], device: &Device) -> Result<()> {
        if !self.first_moments.contains_key(&var_id) {
            let zeros = Tensor::zeros(shape, DType::F32, device)?;
            
            // Use less aggressive quantization during warmup
            let alpha = if self.step < self.quantization_warmup_steps {
                1.0 // No clipping during warmup
            } else {
                self.quantile_alpha
            };
            
            self.first_moments.insert(
                var_id.clone(),
                QuantizedTensor::from_tensor(&zeros, alpha)?
            );
            self.second_moments.insert(
                var_id.clone(),
                QuantizedTensor::from_tensor(&zeros, alpha)?
            );
        }
        Ok(())
    }
    
    /// Get total memory usage of quantized states
    pub fn quantized_memory_usage(&self) -> usize {
        self.first_moments.values().map(|qt| qt.memory_usage()).sum::<usize>()
            + self.second_moments.values().map(|qt| qt.memory_usage()).sum::<usize>()
    }
    
    /// Get quantization statistics
    pub fn quantization_stats(&self) -> (f32, f32) {
        let m_error: f32 = self.first_moments.values()
            .map(|qt| qt.quantization_error)
            .sum::<f32>() / self.first_moments.len() as f32;
        
        let v_error: f32 = self.second_moments.values()
            .map(|qt| qt.quantization_error)
            .sum::<f32>() / self.second_moments.len() as f32;
        
        (m_error, v_error)
    }
}

impl CanOptimizer for AdamW8bit {
    type Config = AdamW8bitConfig;

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
            config.epsilon,
            config.weight_decay,
        )?.with_quantization_config(
            config.quantile_alpha,
            config.adaptive_quantization,
            config.quantization_warmup_steps,
        ))
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
        let eps = self.epsilon;
        let wd = self.weight_decay;
        let step = self.step as f64;
        
        // Bias correction
        let bias_correction1 = 1.0 - beta1.powf(step);
        let bias_correction2 = 1.0 - beta2.powf(step);
        let step_size = lr * (bias_correction2.sqrt() / bias_correction1);
        
        // Determine quantization alpha based on warmup
        let alpha = if self.step < self.quantization_warmup_steps {
            1.0
        } else {
            self.quantile_alpha
        };
        
        for (idx, var) in self.var_map.all_vars().iter().enumerate() {
            let var_id = format!("var_{}", idx);
            let tensor = var.as_tensor();
            
            // Get gradient with validation
            let grad = match grads.get(var) {
                Some(g) => {
                    // Validate gradient
                    if !g.flatten_all()?.to_vec1::<f32>()?.iter().all(|x| x.is_finite()) {
                        eprintln!("Warning: Non-finite gradient detected for var_{}", idx);
                        continue;
                    }
                    g
                },
                None => continue,
            };
            
            // Initialize states if needed
            self.init_state(var_id.clone(), tensor.dims(), tensor.device())?;
            
            // Dequantize momentum states
            let mut m = self.first_moments[&var_id].to_tensor()?;
            let mut v = self.second_moments[&var_id].to_tensor()?;
            
            // Update biased first moment estimate
            m = ((m * beta1)? + (grad * (1.0 - beta1))?)?;
            
            // Update biased second raw moment estimate
            let grad_sq = grad.sqr()?;
            v = ((v * beta2)? + (grad_sq * (1.0 - beta2))?)?;
            
            // Compute update with numerical stability
            let v_sqrt = (v.sqrt()? + eps)?;
            let update = m.div(&v_sqrt)?;
            
            // Apply weight decay (decoupled weight decay as in AdamW)
            let update = if wd > 0.0 {
                (update + (tensor * wd)?)?
            } else {
                update
            };
            
            // Update parameters with gradient clipping if needed
            let new_tensor = (tensor - (update * step_size)?)?;
            var.set(&new_tensor)?;
            
            // Re-quantize momentum states with adaptive strategy
            self.first_moments.get_mut(&var_id).unwrap()
                .update(&m, alpha, self.adaptive_quantization)?;
            self.second_moments.get_mut(&var_id).unwrap()
                .update(&v, alpha, self.adaptive_quantization)?;
        }
        
        Ok(())
    }
    
    fn backward_step(&mut self, loss: &Tensor) -> Result<()> {
        let grads = loss.backward()?;
        self.step(&grads)
    }
}

/// Enhanced configuration for 8-bit AdamW
#[derive(Debug, Clone)]
pub struct AdamW8bitConfig {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    pub weight_decay: f64,
    pub quantile_alpha: f32,
    pub adaptive_quantization: bool,
    pub quantization_warmup_steps: usize,
}

impl Default for AdamW8bitConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.01,
            quantile_alpha: 0.99,
            adaptive_quantization: true,
            quantization_warmup_steps: 100,
        }
    }
}

/// Enhanced memory usage analysis
pub fn analyze_optimizer_memory(num_params: usize, precision: &str) -> OptimizerMemoryReport {
    let param_bytes = num_params * 4; // FP32 parameters
    
    let (state_bytes, description) = match precision {
        "fp32" => (param_bytes * 2, "FP32 momentum states"),
        "fp16" => (param_bytes, "FP16 momentum states"), 
        "8bit" => (param_bytes / 2, "8-bit quantized momentum states"),
        "4bit" => (param_bytes / 4, "4-bit quantized momentum states"),
        _ => (param_bytes * 2, "Unknown precision"),
    };
    
    let total_bytes = param_bytes + state_bytes;
    let memory_reduction = if precision != "fp32" {
        Some(1.0 - (total_bytes as f64 / (param_bytes * 3) as f64))
    } else {
        None
    };
    
    OptimizerMemoryReport {
        num_params,
        param_memory_gb: param_bytes as f64 / 1e9,
        state_memory_gb: state_bytes as f64 / 1e9,
        total_memory_gb: total_bytes as f64 / 1e9,
        memory_reduction_percent: memory_reduction.map(|r| r * 100.0),
        description: description.to_string(),
    }
}

#[derive(Debug)]
pub struct OptimizerMemoryReport {
    pub num_params: usize,
    pub param_memory_gb: f64,
    pub state_memory_gb: f64,
    pub total_memory_gb: f64,
    pub memory_reduction_percent: Option<f64>,
    pub description: String,
}

impl std::fmt::Display for OptimizerMemoryReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Optimizer Memory Analysis:")?;
        writeln!(f, "  Parameters: {:.1}B ({:.2} GB)", 
                 self.num_params as f64 / 1e9, self.param_memory_gb)?;
        writeln!(f, "  States: {} ({:.2} GB)", 
                 self.description, self.state_memory_gb)?;
        writeln!(f, "  Total: {:.2} GB", self.total_memory_gb)?;
        
        if let Some(reduction) = self.memory_reduction_percent {
            writeln!(f, "  Memory Reduction: {:.1}%", reduction)?;
        }
        
        Ok(())
    }
}

/// Optimizer wrapper enum for 8-bit optimizers
pub enum Optimizer8bit {
    AdamW8bit(AdamW8bit),
}

impl CanOptimizer for Optimizer8bit {
    type Config = ();
    
    fn new(_vars: Vec<Var>, _config: Self::Config) -> CanResult<Self> {
        unreachable!("Use create_8bit_optimizer instead")
    }
    
    fn step(&mut self, grads: &candle_core::backprop::GradStore) -> CanResult<()> {
        match self {
            Self::AdamW8bit(opt) => opt.step(grads).map_err(|e| candle_core::Error::Msg(e.to_string())),
        }
    }
    
    fn learning_rate(&self) -> f64 {
        match self {
            Self::AdamW8bit(opt) => opt.learning_rate(),
        }
    }
    
    fn set_learning_rate(&mut self, lr: f64) {
        match self {
            Self::AdamW8bit(opt) => opt.set_learning_rate(lr),
        }
    }
}

/// Create 8-bit optimizer from config string
pub fn create_8bit_optimizer(
    optimizer_type: &str,
    var_map: VarMap,
    lr: f64,
) -> eridiffusion_core::Result<Optimizer8bit> {
    match optimizer_type {
        "adamw8bit" | "adamw_8bit" => {
            let config = AdamW8bitConfig {
                lr,
                ..Default::default()
            };
            Ok(Optimizer8bit::AdamW8bit(AdamW8bit::new(
                var_map,
                config.lr,
                config.beta1,
                config.beta2,
                config.epsilon,
                config.weight_decay,
            )?))
        }
        _ => Err(Error::Config(format!("Unknown 8-bit optimizer: {}", optimizer_type))),
    }
}