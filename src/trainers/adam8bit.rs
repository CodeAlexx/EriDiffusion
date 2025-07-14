//! 8-bit Adam optimizer for memory-efficient training
//! Stores optimizer states in 8-bit format to reduce memory usage

use anyhow::Result;
use candle_core::{Device, DType, Tensor, Var};
use std::collections::HashMap;

/// Quantization constants for 8-bit storage
const QMIN: i8 = -128;  // Use full i8 range
const QMAX: i8 = 127;

pub struct Adam8bit {
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
    
    // 8-bit quantized states
    m_quantized: HashMap<String, QuantizedTensor>,
    v_quantized: HashMap<String, QuantizedTensor>,
    
    // Step counter
    step: usize,
}

/// Quantized tensor with scale factor
#[derive(Clone)]
pub struct QuantizedTensor {
    pub data: Tensor, // i8 tensor
    pub scale: f32,   // Scale factor for dequantization
}

/// State for checkpoint saving/loading
#[derive(Clone)]
pub struct Adam8bitState {
    pub m_quantized: HashMap<String, QuantizedTensor>,
    pub v_quantized: HashMap<String, QuantizedTensor>,
    pub step: usize,
    pub learning_rate: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
}

impl Adam8bit {
    pub fn new(learning_rate: f64) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            m_quantized: HashMap::new(),
            v_quantized: HashMap::new(),
            step: 0,
        }
    }
    
    pub fn with_params(learning_rate: f64, beta1: f64, beta2: f64, eps: f64, weight_decay: f64) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            eps,
            weight_decay,
            m_quantized: HashMap::new(),
            v_quantized: HashMap::new(),
            step: 0,
        }
    }
    
    /// Increment step counter - should be called once per optimization step
    pub fn step(&mut self) {
        self.step += 1;
    }
    
    /// Update learning rate
    pub fn set_lr(&mut self, lr: f32) {
        self.learning_rate = lr as f64;
    }
    
    /// Update a single parameter
    pub fn update(&mut self, name: &str, param: &Var, grad: &Tensor) -> Result<()> {
        // Note: step counter should be incremented separately via step() method
        
        // Convert gradient to F32 for optimizer computations
        let grad_f32 = grad.to_dtype(DType::F32)?;
        
        // Apply weight decay if configured
        let grad_f32 = if self.weight_decay > 0.0 {
            let param_f32 = param.as_tensor().to_dtype(DType::F32)?;
            (grad_f32 + param_f32 * self.weight_decay)?
        } else {
            grad_f32
        };
        
        // Initialize states if needed
        if !self.m_quantized.contains_key(name) {
            let zeros = Tensor::zeros_like(&grad_f32)?;
            self.m_quantized.insert(name.to_string(), Self::quantize(&zeros)?);
            self.v_quantized.insert(name.to_string(), Self::quantize(&zeros)?);
        }
        
        // Dequantize current states (returns F32)
        let m_quant = &self.m_quantized[name];
        let v_quant = &self.v_quantized[name];
        let m = Self::dequantize(m_quant)?;
        let v = Self::dequantize(v_quant)?;
        
        // Update biased first moment estimate
        let m_new = ((m * self.beta1)? + (grad_f32.clone() * (1.0 - self.beta1))?)?;
        
        // Update biased second raw moment estimate
        let v_new = ((v * self.beta2)? + (grad_f32.sqr()? * (1.0 - self.beta2))?)?;
        
        // Quantize and store updated states
        self.m_quantized.insert(name.to_string(), Self::quantize(&m_new)?);
        self.v_quantized.insert(name.to_string(), Self::quantize(&v_new)?);
        
        // Compute bias-corrected first moment estimate
        // Use max(step, 1) to avoid division by zero on first update
        let step = self.step.max(1);
        let m_hat = (m_new / (1.0 - self.beta1.powi(step as i32)))?;
        
        // Compute bias-corrected second raw moment estimate
        let v_hat = (v_new / (1.0 - self.beta2.powi(step as i32)))?;
        
        // Ensure v_hat is non-negative for numerical stability
        let v_hat = v_hat.clamp(0.0, f64::INFINITY)?;
        
        // Update parameters
        let update = (m_hat / (v_hat.sqrt()? + self.eps)?)?;
        
        // Convert update back to param dtype before applying
        let update = update.to_dtype(param.dtype())?;
        let new_value = (param.as_tensor() - (update * self.learning_rate)?)?;
        
        // Apply update
        param.set(&new_value)?;
        
        Ok(())
    }
    
    /// Quantize tensor to 8-bit
    fn quantize(tensor: &Tensor) -> Result<QuantizedTensor> {
        // Convert to F32 first if needed (handles BF16 inputs)
        let tensor_f32 = if tensor.dtype() != DType::F32 {
            tensor.to_dtype(DType::F32)?
        } else {
            tensor.clone()
        };
        
        // Find absolute max for scaling
        let abs_max = tensor_f32.abs()?.max_all()?.to_scalar::<f32>()?;
        
        // Avoid division by zero
        // Use 127.5 to better utilize the full i8 range [-128, 127]
        let scale = if abs_max > 0.0 {
            abs_max / 127.5
        } else {
            1.0
        };
        
        // Quantize: round(tensor / scale) clamped to [QMIN, QMAX]
        let scaled = (tensor_f32 / scale as f64)?;
        let rounded = scaled.round()?;
        let clamped = rounded.clamp(QMIN as f64, QMAX as f64)?;
        let quantized = clamped.to_dtype(DType::U8)?;
        
        Ok(QuantizedTensor {
            data: quantized,
            scale,
        })
    }
    
    /// Dequantize 8-bit tensor back to float
    fn dequantize(quant: &QuantizedTensor) -> Result<Tensor> {
        let float_data = quant.data.to_dtype(DType::F32)?;
        let result = (float_data * quant.scale as f64)?;
        Ok(result)
    }
    
    /// Get memory usage statistics
    pub fn memory_stats(&self) -> (usize, usize) {
        let num_params = self.m_quantized.len();
        let mut total_elements = 0;
        
        for (_, m) in &self.m_quantized {
            total_elements += m.data.elem_count();
        }
        
        // Each element is 1 byte (i8) + scale factor (4 bytes per tensor)
        // We have 2 tensors per parameter (m and v), so 8 bytes of scale factors per param
        let memory_bytes = total_elements + num_params * 8;
        
        (num_params, memory_bytes)
    }
    
    /// Get optimizer state for checkpoint saving
    pub fn get_state(&self) -> Adam8bitState {
        Adam8bitState {
            m_quantized: self.m_quantized.clone(),
            v_quantized: self.v_quantized.clone(),
            step: self.step,
            learning_rate: self.learning_rate,
            beta1: self.beta1,
            beta2: self.beta2,
            eps: self.eps,
            weight_decay: self.weight_decay,
        }
    }
    
    /// Load optimizer state from checkpoint
    pub fn load_state(&mut self, state: Adam8bitState) {
        self.m_quantized = state.m_quantized;
        self.v_quantized = state.v_quantized;
        self.step = state.step;
        self.learning_rate = state.learning_rate;
        self.beta1 = state.beta1;
        self.beta2 = state.beta2;
        self.eps = state.eps;
        self.weight_decay = state.weight_decay;
    }
    
    /// Get state as unquantized tensors for saving
    pub fn get_state_tensors(&self) -> Result<HashMap<String, (Tensor, Tensor)>> {
        let mut state = HashMap::new();
        
        for (name, m_quant) in &self.m_quantized {
            if let Some(v_quant) = self.v_quantized.get(name) {
                let m = Self::dequantize(m_quant)?;
                let v = Self::dequantize(v_quant)?;
                state.insert(name.clone(), (m, v));
            }
        }
        
        Ok(state)
    }
    
    /// Get current step count
    pub fn get_step(&self) -> usize {
        self.step
    }
    
    /// Load state from unquantized tensors
    pub fn load_state_tensors(&mut self, state: HashMap<String, (Tensor, Tensor)>) -> Result<()> {
        self.m_quantized.clear();
        self.v_quantized.clear();
        
        for (name, (m, v)) in state {
            self.m_quantized.insert(name.clone(), Self::quantize(&m)?);
            self.v_quantized.insert(name.clone(), Self::quantize(&v)?);
        }
        
        Ok(())
    }
}

/// Batch update for multiple parameters
pub fn update_params_8bit(
    optimizer: &mut Adam8bit,
    params: &[(String, &Var)],
    grads: &candle_core::backprop::GradStore,
) -> Result<()> {
    for (name, param) in params {
        if let Some(grad) = grads.get(param.as_tensor()) {
            optimizer.update(name, param, grad)?;
        }
    }
    Ok(())
}