use flame_core::device::Device;
use flame_core::optimizers::{Adam, SGD};
use flame_core::{DType, Parameter, Result, Shape, Tensor};
use std::collections::HashMap;

// 8-bit Adam optimizer for memory-efficient training
// Stores optimizer states in 8-bit format to reduce memory usage

// FLAME uses flame_core::device::Device instead of Device

/// Quantization constants for 8-bit storage
const QMIN: i8 = -128; // Use full i8 range
const QMAX: i8 = 127;

pub struct Adam8bit {
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,

    // 8-bit quantized states
    m_quantized: std::collections::HashMap<String, QuantizedTensor>,
    v_quantized: std::collections::HashMap<String, QuantizedTensor>,

    // Step counter
    step: usize,
}

/// Quantized tensor with scale factor
#[derive(Clone)]
pub struct QuantizedTensor {
    pub data: Tensor, // i8 tensor,
    pub scale: f32,   // Scale factor for dequantization,
}

/// State for checkpoint saving/loading
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
    pub fn new(
        parameters: Vec<&Parameter>,
        learning_rate: f32,
        betas: (f32, f32),
        eps: f32,
        weight_decay: f32,
    ) -> flame_core::Result<Self> {
        Ok(Self {
            learning_rate: learning_rate as f64,
            beta1: betas.0 as f64,
            beta2: betas.1 as f64,
            eps: eps as f64,
            weight_decay: weight_decay as f64,
            m_quantized: HashMap::new(),
            v_quantized: HashMap::new(),
            step: 0,
        })
    }

    pub fn with_params(
        learning_rate: f64,
        beta1: f64,
        beta2: f64,
        eps: f64,
        weight_decay: f64,
    ) -> Self {
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
    pub fn step(&mut self) -> flame_core::Result<()> {
        self.step += 1;
        Ok(())
    }

    /// Update learning rate
    pub fn set_lr(&mut self, lr: f32) -> flame_core::Result<()> {
        self.learning_rate = lr as f64;
        Ok(())
    }

    /// Update a single parameter and return the updated value
    pub fn update(
        &mut self,
        name: &str,
        param: &Tensor,
        grad: &Tensor,
    ) -> flame_core::Result<Tensor> {
        // Note: step counter should be incremented separately via step() method

        // Convert gradient to F32 for optimizer computations
        let grad_f32 = grad.to_dtype(DType::F32)?;

        // Apply weight decay if configured
        let grad_f32 = if self.weight_decay > 0.0 {
            let param_f32 = param.to_dtype(DType::F32)?;
            let weight_decay_scalar = Tensor::full(
                param_f32.shape().clone(),
                self.weight_decay as f32,
                param_f32.device().clone(),
            )?;
            grad_f32.add(&param_f32.mul(&weight_decay_scalar)?)?
        } else {
            grad_f32
        };

        // Initialize states if needed
        if !self.m_quantized.contains_key(name) {
            let zeros = Tensor::zeros(grad_f32.shape().clone(), grad_f32.device().clone())?;
            self.m_quantized.insert(name.to_string(), Self::quantize(&zeros)?);
            self.v_quantized.insert(name.to_string(), Self::quantize(&zeros)?);
        }

        // Dequantize current states (returns F32)
        let m_quant = &self.m_quantized[name];
        let v_quant = &self.v_quantized[name];
        let m = Self::dequantize(m_quant)?;
        let v = Self::dequantize(v_quant)?;

        // Update biased first moment estimate
        let beta1_scalar = Tensor::full(m.shape().clone(), self.beta1 as f32, m.device().clone())?;
        let one_minus_beta1_scalar = Tensor::full(
            grad_f32.shape().clone(),
            (1.0 - self.beta1) as f32,
            grad_f32.device().clone(),
        )?;
        let m_new = m.mul(&beta1_scalar)?.add(&grad_f32.clone().mul(&one_minus_beta1_scalar)?)?;

        // Update biased second raw moment estimate
        let beta2_scalar = Tensor::full(v.shape().clone(), self.beta2 as f32, v.device().clone())?;
        let one_minus_beta2_scalar = Tensor::full(
            grad_f32.shape().clone(),
            (1.0 - self.beta2) as f32,
            grad_f32.device().clone(),
        )?;
        let grad_squared = grad_f32.square()?;
        let grad_squared_scaled = grad_squared.mul(&one_minus_beta2_scalar)?;
        let v_new = v.mul(&beta2_scalar)?.add(&grad_squared_scaled)?;

        // Quantize and store updated states
        self.m_quantized.insert(name.to_string(), Self::quantize(&m_new)?);
        self.v_quantized.insert(name.to_string(), Self::quantize(&v_new)?);

        // Compute bias-corrected first moment estimate
        // Use max(step, 1) to avoid division by zero on first update
        let step = self.step.max(1);
        let beta1_pow = Tensor::full(
            m_new.shape().clone(),
            (1.0 - self.beta1.powi(step as i32)) as f32,
            m_new.device().clone(),
        )?;
        let m_hat = m_new.div(&beta1_pow)?;

        // Compute bias-corrected second raw moment estimate
        let beta2_pow = Tensor::full(
            v_new.shape().clone(),
            (1.0 - self.beta2.powi(step as i32)) as f32,
            v_new.device().clone(),
        )?;
        let v_hat = v_new.div(&beta2_pow)?;

        // Ensure v_hat is non-negative for numerical stability
        let v_hat =
            v_hat.maximum(&Tensor::zeros(v_hat.shape().clone(), v_hat.device().clone())?)?;

        // Update parameters
        let eps_scalar =
            Tensor::full(v_hat.shape().clone(), self.eps as f32, v_hat.device().clone())?;
        let update = m_hat.div(&v_hat.sqrt()?.add(&eps_scalar)?)?;

        // Convert update to match parameter dtype
        let param_dtype = param.dtype();
        let lr_scalar = Tensor::full(
            update.shape().clone(),
            self.learning_rate as f32,
            update.device().clone(),
        )?;
        let update_scaled = update.mul(&lr_scalar)?;
        let update_final = if param_dtype != DType::F32 {
            update_scaled.to_dtype(param_dtype)?
        } else {
            update_scaled
        };

        // Note: Since param is now a Tensor, we can't update it in-place
        // The caller needs to handle the updated value
        let new_value = param.sub(&update_final)?;

        Ok(new_value)
    }

    /// Quantize tensor to 8-bit
    fn quantize(tensor: &Tensor) -> flame_core::Result<QuantizedTensor> {
        // Convert to F32 first if needed (handles BF16 inputs)
        let tensor_f32 = if tensor.dtype() != DType::F32 {
            tensor.to_dtype(DType::F32)?
        } else {
            tensor.clone()
        };

        // Find absolute max for scaling
        let abs_tensor = tensor_f32.abs()?;
        let abs_max = abs_tensor.max_all()?;

        // Avoid division by zero
        // Use 127.5 to better utilize the full i8 range [-128, 127]
        let scale = if abs_max > 0.0 { abs_max / 127.5 } else { 1.0 };

        // Quantize: round(tensor / scale) clamped to [QMIN, QMAX]
        let scale_tensor =
            Tensor::full(tensor_f32.shape().clone(), scale, tensor_f32.device().clone())?;
        let scaled = tensor_f32.div(&scale_tensor)?;
        // Use FLAME's round operation
        let rounded = scaled.round()?;
        let min_tensor =
            Tensor::full(rounded.shape().clone(), QMIN as f32, rounded.device().clone())?;
        let max_tensor =
            Tensor::full(rounded.shape().clone(), QMAX as f32, rounded.device().clone())?;
        let clamped = rounded.minimum(&max_tensor)?.maximum(&min_tensor)?;
        let quantized = clamped.to_dtype(DType::U8)?;

        Ok(QuantizedTensor { data: quantized, scale })
    }

    /// Dequantize 8-bit tensor back to float
    fn dequantize(quant: &QuantizedTensor) -> flame_core::Result<Tensor> {
        let float_data = quant.data.to_dtype(DType::F32)?;
        let scale_tensor =
            Tensor::full(float_data.shape().clone(), quant.scale, float_data.device().clone())?;
        let result = float_data.mul(&scale_tensor)?;
        Ok(result)
    }

    /// Get memory usage statistics
    pub fn memory_stats(&self) -> (usize, usize) {
        let num_params = self.m_quantized.len();
        let mut total_elements = 0;

        for (_, m) in &self.m_quantized {
            total_elements += m.data.shape().dims().iter().product::<usize>();
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
    pub fn load_state(&mut self, state: Adam8bitState) -> flame_core::Result<()> {
        self.m_quantized = state.m_quantized;
        self.v_quantized = state.v_quantized;
        self.step = state.step;
        self.learning_rate = state.learning_rate;
        self.beta1 = state.beta1;
        self.beta2 = state.beta2;
        self.eps = state.eps;
        self.weight_decay = state.weight_decay;
        Ok(())
    }

    /// Get state as unquantized tensors for saving
    pub fn get_state_as_tensors(&self) -> flame_core::Result<HashMap<String, (Tensor, Tensor)>> {
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
    pub fn load_state_from_tensors(
        &mut self,
        state: HashMap<String, (Tensor, Tensor)>,
    ) -> flame_core::Result<()> {
        self.m_quantized.clear();
        self.v_quantized.clear();

        for (name, (m, v)) in state {
            self.m_quantized.insert(name.clone(), Self::quantize(&m)?);
            self.v_quantized.insert(name.clone(), Self::quantize(&v)?);
        }

        Ok(())
    }
} // Close impl Adam8bit

/// Batch update for multiple parameters
pub fn update_params_8bit(
    optimizer: &mut Adam8bit,
    params: &mut HashMap<String, Tensor>,
    grads: &HashMap<String, Tensor>,
) -> flame_core::Result<()> {
    for (name, param) in params.iter_mut() {
        if let Some(grad) = grads.get(name) {
            let new_value = optimizer.update(name, param, grad)?;
            *param = new_value;
        }
    }
    Ok(())
}
