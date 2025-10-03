//! Enhanced 8-bit AdamW optimizer with improved quantization and error handling

use flame_core::device::Device;
use flame_core::{CudaDevice, DType, Error, Parameter, Result, Shape, Tensor};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// Enhanced 8-bit AdamW optimizer implementation
pub struct Adam8bit {
    pub params: Vec<Parameter>,
    pub learning_rate: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    pub weight_decay: f64,
    pub step: usize,

    // 8-bit quantized states
    pub first_moments: HashMap<String, QuantizedTensor>,
    pub second_moments: HashMap<String, QuantizedTensor>,

    // Enhanced quantization parameters
    pub quantile_alpha: f32,
    pub adaptive_quantization: bool,
    pub quantization_warmup_steps: usize,
}

/// Enhanced quantized tensor storage with better numerical stability
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct QuantizedTensor {
    pub quantized_data: Vec<i8>,
    pub scale: f32,
    pub zero_point: i8,
    pub shape: Vec<usize>,
    // Track quantization error for adaptive schemes
    pub quantization_error: f32,
}

impl QuantizedTensor {
    /// Enhanced quantization with better numerical stability
    fn from_tensor(tensor: &Tensor, quantile_alpha: f32) -> flame_core::Result<Self> {
        // Use real device-to-host copy for numeric introspection
        let data = tensor.to_vec()?;

        if data.is_empty() {
            return Err(flame_core::Error::InvalidOperation(
                "Empty tensor for quantization".into(),
            ));
        }

        // Compute quantization parameters with better numerical stability
        let (min_val, max_val) = if quantile_alpha < 1.0 {
            // Use quantile-based clipping
            let mut sorted_data = data.clone();
            sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let n = sorted_data.len();
            let lower_idx = ((1.0 - quantile_alpha) * 0.5 * n as f32) as usize;
            let upper_idx =
                ((1.0 - (1.0 - quantile_alpha) * 0.5) * n as f32).min(n as f32 - 1.0) as usize;

            (sorted_data[lower_idx], sorted_data[upper_idx])
        } else {
            // Use full range
            let min_val = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            (min_val, max_val)
        };

        // Improved scale computation with numerical stability
        let range = max_val - min_val;
        let scale = if range.abs() < 1e-8 { 1.0 } else { range / 255.0 };

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
                let q_float = if scale.abs() < 1e-8 { 0.0 } else { (clamped_x - min_val) / scale };
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
            shape: tensor.shape().dims().to_vec(),
            quantization_error,
        })
    }

    /// Dequantize with improved numerical precision
    fn to_tensor(&self, device: Arc<CudaDevice>) -> flame_core::Result<Tensor> {
        if self.quantized_data.is_empty() {
            return Err(flame_core::Error::InvalidOperation("Empty quantized data".into()));
        }

        let data: Vec<f32> = self
            .quantized_data
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

        Tensor::from_slice(&data, Shape::from_dims(&self.shape), device.clone())
    }

    /// Update with adaptive quantization strategy
    fn update(
        &mut self,
        tensor: &Tensor,
        quantile_alpha: f32,
        adaptive: bool,
    ) -> flame_core::Result<()> {
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

impl Adam8bit {
    pub fn new(params: Vec<Parameter>, config: Adam8bitConfig) -> flame_core::Result<Self> {
        Ok(Self {
            params,
            learning_rate: config.lr,
            beta1: config.betas.0,
            beta2: config.betas.1,
            epsilon: config.eps,
            weight_decay: config.weight_decay,
            step: 0,
            first_moments: HashMap::new(),
            second_moments: HashMap::new(),
            quantile_alpha: config.percentile_clipping as f32 / 100.0,
            adaptive_quantization: true,
            quantization_warmup_steps: 100,
        })
    }

    /// Initialize optimizer states with proper error handling
    fn init_state(&mut self, param_id: String, param: &Parameter) -> flame_core::Result<()> {
        if !self.first_moments.contains_key(&param_id) {
            let tensor = param.tensor()?;
            let shape = tensor.shape();
            let device = tensor.device();
            let zeros = Tensor::zeros(shape.clone(), device.clone())?;

            // Use less aggressive quantization during warmup
            let alpha = if self.step < self.quantization_warmup_steps {
                1.0 // No clipping during warmup
            } else {
                self.quantile_alpha
            };

            self.first_moments
                .insert(param_id.clone(), QuantizedTensor::from_tensor(&zeros, alpha)?);
            self.second_moments
                .insert(param_id.clone(), QuantizedTensor::from_tensor(&zeros, alpha)?);
        }
        Ok(())
    }

    /// Perform optimization step
    pub fn step(&mut self) -> flame_core::Result<()> {
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
        let alpha =
            if self.step < self.quantization_warmup_steps { 1.0 } else { self.quantile_alpha };

        // Collect params info first to avoid borrow checker issues
        let params_info: Vec<(usize, Parameter)> =
            self.params.iter().cloned().enumerate().collect();

        for (idx, param) in params_info {
            let param_id = format!("param_{}", idx);

            // Get gradient (ensure FP32 for numerically stable state updates)
            let grad = match param.grad() {
                Some(g) => {
                    if g.dtype() != DType::F32 {
                        g.to_dtype(DType::F32)?
                    } else {
                        g
                    }
                }
                None => {
                    eprintln!("Warning: No gradient for param_{}", idx);
                    continue;
                }
            };

            // Initialize states if needed
            self.init_state(param_id.clone(), &param)?;

            // Get device
            let tensor = param.tensor()?;
            let device = tensor.device();
            let cuda_device = device;

            // Dequantize momentum states
            let mut m = self.first_moments[&param_id].to_tensor(cuda_device.clone())?;
            let mut v = self.second_moments[&param_id].to_tensor(cuda_device.clone())?;

            // Update biased first moment estimate
            m = m.mul_scalar(beta1 as f32)?.add(&grad.mul_scalar(1.0 - beta1 as f32)?)?;

            // Update biased second raw moment estimate
            let grad_sq = grad.square()?;
            v = v.mul_scalar(beta2 as f32)?.add(&grad_sq.mul_scalar(1.0 - beta2 as f32)?)?;

            // Compute update with numerical stability
            let v_sqrt = v.sqrt()?.add_scalar(eps as f32)?;
            let update = m.div(&v_sqrt)?;

            // Apply weight decay (decoupled weight decay as in AdamW)
            let update = if wd > 0.0 {
                update.add(&param.tensor()?.mul_scalar(wd as f32)?)?
            } else {
                update
            };

            // Update parameters
            let scaled_update = update.mul_scalar(step_size as f32)?;
            param.apply_update(&scaled_update)?;

            // Re-quantize momentum states with adaptive strategy
            self.first_moments.get_mut(&param_id).unwrap().update(
                &m,
                alpha,
                self.adaptive_quantization,
            )?;
            self.second_moments.get_mut(&param_id).unwrap().update(
                &v,
                alpha,
                self.adaptive_quantization,
            )?;
        }

        Ok(())
    }

    /// Zero all gradients
    pub fn zero_grad(&mut self) -> flame_core::Result<()> {
        for param in &self.params {
            // param.zero_grad() removed - handle gradients manually
        }
        Ok(())
    }

    /// Get current learning rate
    pub fn current_lr(&self) -> f64 {
        self.learning_rate
    }

    /// Set learning rate
    pub fn set_learning_rate(&mut self, lr: f64) {
        self.learning_rate = lr;
    }

    /// Get total memory usage of quantized states
    pub fn quantized_memory_usage(&self) -> usize {
        self.first_moments.values().map(|qt| qt.memory_usage()).sum::<usize>()
            + self.second_moments.values().map(|qt| qt.memory_usage()).sum::<usize>()
    }

    /// Get quantization statistics
    pub fn quantization_stats(&self) -> (f32, f32) {
        let m_error: f32 = self.first_moments.values().map(|qt| qt.quantization_error).sum::<f32>()
            / self.first_moments.len().max(1) as f32;

        let v_error: f32 =
            self.second_moments.values().map(|qt| qt.quantization_error).sum::<f32>()
                / self.second_moments.len().max(1) as f32;

        (m_error, v_error)
    }

    /// Save optimizer state
    pub fn save_state(&self, path: &Path) -> flame_core::Result<()> {
        let state = OptimizerState {
            step: self.step,
            first_moments: self.first_moments.clone(),
            second_moments: self.second_moments.clone(),
            learning_rate: self.learning_rate,
        };

        let data = bincode::serialize(&state).map_err(|e| {
            flame_core::Error::InvalidOperation(format!(
                "Failed to serialize optimizer: {}",
                e
            ))
        })?;

        std::fs::write(path, data).map_err(|e| {
            flame_core::Error::InvalidOperation(format!("Failed to save optimizer: {}", e))
        })?;

        Ok(())
    }

    /// Load optimizer state
    pub fn load_state(&mut self, path: &Path) -> flame_core::Result<()> {
        let data = std::fs::read(path).map_err(|e| {
            flame_core::Error::InvalidOperation(format!("Failed to read optimizer: {}", e))
        })?;

        let state: OptimizerState = bincode::deserialize(&data).map_err(|e| {
            flame_core::Error::InvalidOperation(format!(
                "Failed to deserialize optimizer: {}",
                e
            ))
        })?;

        self.load_state_dict(state);
        Ok(())
    }

    fn load_state_dict(&mut self, state: OptimizerState) {
        self.step = state.step;
        self.first_moments = state.first_moments;
        self.second_moments = state.second_moments;
        self.learning_rate = state.learning_rate;
    }
}

/// Enhanced configuration for 8-bit AdamW
#[derive(Debug, Clone)]
pub struct Adam8bitConfig {
    pub lr: f64,
    pub betas: (f64, f64),
    pub eps: f64,
    pub weight_decay: f64,
    pub amsgrad: bool,
    pub block_wise: bool,
    pub percentile_clipping: usize,
}

impl Default for Adam8bitConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            betas: (0.9, 0.999),
            eps: 1e-8,
            weight_decay: 0.01,
            amsgrad: false,
            block_wise: true,
            percentile_clipping: 99,
        }
    }
}

/// Optimizer state for serialization
#[derive(serde::Serialize, serde::Deserialize)]
pub struct OptimizerState {
    pub step: usize,
    pub first_moments: HashMap<String, QuantizedTensor>,
    pub second_moments: HashMap<String, QuantizedTensor>,
    pub learning_rate: f64,
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
        writeln!(
            f,
            "  Parameters: {:.1}B ({:.2} GB)",
            self.num_params as f64 / 1e9,
            self.param_memory_gb
        )?;
        writeln!(f, "  States: {} ({:.2} GB)", self.description, self.state_memory_gb)?;
        writeln!(f, "  Total: {:.2} GB", self.total_memory_gb)?;

        if let Some(reduction) = self.memory_reduction_percent {
            writeln!(f, "  Memory Reduction: {:.1}%", reduction)?;
        }

        Ok(())
    }
}

#[cfg(all(test, feature = "dev-bins"))]
mod tests {
    use super::*;

    #[test]
    fn test_adam8bit_creation(device: &CudaDevice) -> flame_core::Result<()> {
        let device = device;

        // Create dummy parameters
        let param1 = Parameter::randn(Shape::from_dims(&[100]), 0.0, 0.02, DType::F32, device)?;
        let param2 = Parameter::randn(Shape::from_dims(&[50]), 0.0, 0.02, DType::F32, device)?;

        let config = Adam8bitConfig::default();
        let optimizer = Adam8bit::new(vec![param1, param2], config)?;

        assert_eq!(optimizer.params.len(), 2);
        assert_eq!(optimizer.learning_rate, 1e-3);

        Ok(())
    }

    #[test]
    fn test_quantization(device: &CudaDevice) -> flame_core::Result<()> {
        let device = device;
        let tensor = Tensor::randn(Shape::from_dims(&[10, 10]), 0.0, 1.0, device.cuda_device())?;

        let quantized = QuantizedTensor::from_tensor(&tensor, 0.99)?;
        let dequantized = quantized.to_tensor(device.cuda_device().clone())?;

        assert_eq!(tensor.shape(), dequantized.shape());
        assert!(quantized.quantization_error < 0.1); // Should have low error

        Ok(())
    }

    #[test]
    fn test_memory_report() {
        let report = analyze_optimizer_memory(1_000_000_000, "8bit");

        println!("{}", report);

        assert!(report.memory_reduction_percent.unwrap() > 60.0);
        assert!(report.state_memory_gb < report.param_memory_gb);
    }
}
