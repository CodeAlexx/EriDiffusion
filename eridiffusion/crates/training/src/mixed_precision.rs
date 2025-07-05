//! Mixed precision training support

use eridiffusion_core::{Result, Error};
use candle_core::{Tensor, DType, Device};
use serde::{Serialize, Deserialize};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Mixed precision configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MixedPrecisionConfig {
    pub enabled: bool,
    pub dtype: MixedPrecisionDType,
    pub loss_scale: LossScaleConfig,
    pub gradient_clipping: Option<f32>,
    pub opt_level: OptLevel,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum MixedPrecisionDType {
    Float16,
    BFloat16,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LossScaleConfig {
    Static(f32),
    Dynamic {
        init_scale: f32,
        growth_factor: f32,
        backoff_factor: f32,
        growth_interval: usize,
    },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum OptLevel {
    O0, // FP32 training
    O1, // Mixed precision (conservative)
    O2, // Mixed precision (aggressive)
    O3, // FP16 training
}

impl Default for MixedPrecisionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            dtype: MixedPrecisionDType::Float16,
            loss_scale: LossScaleConfig::Dynamic {
                init_scale: 2048.0,
                growth_factor: 2.0,
                backoff_factor: 0.5,
                growth_interval: 2000,
            },
            gradient_clipping: Some(1.0),
            opt_level: OptLevel::O1,
        }
    }
}

/// Mixed precision scaler
pub struct GradScaler {
    config: MixedPrecisionConfig,
    current_scale: Arc<RwLock<f32>>,
    growth_tracker: Arc<RwLock<usize>>,
    found_inf: Arc<RwLock<bool>>,
}

impl GradScaler {
    pub fn new(config: MixedPrecisionConfig) -> Self {
        let init_scale = match &config.loss_scale {
            LossScaleConfig::Static(scale) => *scale,
            LossScaleConfig::Dynamic { init_scale, .. } => *init_scale,
        };
        
        Self {
            config,
            current_scale: Arc::new(RwLock::new(init_scale)),
            growth_tracker: Arc::new(RwLock::new(0)),
            found_inf: Arc::new(RwLock::new(false)),
        }
    }
    
    /// Scale loss for backward pass
    pub async fn scale(&self, loss: &Tensor) -> Result<Tensor> {
        if !self.config.enabled {
            return Ok(loss.clone());
        }
        
        let scale = *self.current_scale.read().await;
        Ok((loss * scale as f64)?)
    }
    
    /// Unscale gradients
    pub async fn unscale_gradients(&self, gradients: &mut [Tensor]) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }
        
        let scale = *self.current_scale.read().await;
        let inv_scale = 1.0 / scale;
        
        for grad in gradients {
            *grad = (grad.as_ref() * inv_scale as f64)?;
        }
        
        Ok(())
    }
    
    /// Check for inf/nan in gradients
    pub async fn check_gradients(&self, gradients: &[Tensor]) -> Result<bool> {
        let mut has_inf_nan = false;
        
        for grad in gradients {
            if self.has_inf_nan(grad)? {
                has_inf_nan = true;
                break;
            }
        }
        
        *self.found_inf.write().await = has_inf_nan;
        Ok(!has_inf_nan)
    }
    
    /// Update scale based on gradient check
    pub async fn update_scale(&self) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }
        
        let found_inf = *self.found_inf.read().await;
        
        match &self.config.loss_scale {
            LossScaleConfig::Static(_) => {
                // Static scale doesn't update
            }
            LossScaleConfig::Dynamic {
                growth_factor,
                backoff_factor,
                growth_interval,
                ..
            } => {
                if found_inf {
                    // Decrease scale
                    let mut scale = self.current_scale.write().await;
                    *scale *= backoff_factor;
                    
                    // Reset growth tracker
                    *self.growth_tracker.write().await = 0;
                } else {
                    // Increase growth tracker
                    let mut tracker = self.growth_tracker.write().await;
                    *tracker += 1;
                    
                    // Check if we should increase scale
                    if *tracker >= *growth_interval {
                        let mut scale = self.current_scale.write().await;
                        *scale *= growth_factor;
                        *tracker = 0;
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Get current scale
    pub async fn get_scale(&self) -> f32 {
        *self.current_scale.read().await
    }
    
    /// Check if tensor has inf/nan
    fn has_inf_nan(&self, tensor: &Tensor) -> Result<bool> {
        let flat = tensor.flatten_all()?;
        let max_val = flat.max_all()?.to_scalar::<f32>()?;
        let min_val = flat.min_all()?.to_scalar::<f32>()?;
        
        Ok(max_val.is_infinite() || max_val.is_nan() || 
           min_val.is_infinite() || min_val.is_nan())
    }
    
    /// Unscale gradients (synchronous version for compatibility)
    pub fn unscale(&self, grads: &Vec<Tensor>) -> Result<Vec<Tensor>> {
        if !self.config.enabled {
            return Ok(grads.clone());
        }
        
        let scale = tokio::runtime::Handle::current()
            .block_on(async { *self.current_scale.read().await });
        
        grads.iter()
            .map(|g| g.affine(1.0 / scale as f64, 0.0).map_err(|e| Error::Training(e.to_string())))
            .collect::<Result<Vec<_>>>()
    }
    
    /// Update scaler state (synchronous version for compatibility)
    pub fn update(&mut self) {
        // This is a no-op for compatibility - actual update happens in update_scale
        tokio::runtime::Handle::current()
            .block_on(async { self.update_scale().await })
            .unwrap_or_else(|e| eprintln!("Failed to update scale: {}", e));
    }
}

/// Mixed precision optimizer wrapper
pub struct MixedPrecisionOptimizer<O> {
    optimizer: O,
    scaler: GradScaler,
    master_params: Arc<RwLock<Vec<Tensor>>>,
    model_params: Arc<RwLock<Vec<Tensor>>>,
}

impl<O> MixedPrecisionOptimizer<O> {
    pub fn new(
        optimizer: O,
        scaler: GradScaler,
        model_params: Vec<Tensor>,
    ) -> Result<Self> {
        // Create master parameters in FP32
        let mut master_params = Vec::new();
        for param in &model_params {
            let master = param.to_dtype(DType::F32)?;
            master_params.push(master);
        }
        
        Ok(Self {
            optimizer,
            scaler,
            master_params: Arc::new(RwLock::new(master_params)),
            model_params: Arc::new(RwLock::new(model_params)),
        })
    }
    
    /// Step optimizer with mixed precision
    pub async fn step(&mut self, gradients: &[Tensor]) -> Result<bool> {
        // Unscale gradients
        let mut unscaled_grads = gradients.to_vec();
        self.scaler.unscale_gradients(&mut unscaled_grads).await?;
        
        // Check for inf/nan
        if !self.scaler.check_gradients(&unscaled_grads).await? {
            // Skip this step
            self.scaler.update_scale().await?;
            return Ok(false);
        }
        
        // Clip gradients if configured
        if let Some(max_norm) = self.scaler.config.gradient_clipping {
            clip_grad_norm(&mut unscaled_grads, max_norm)?;
        }
        
        // Update master parameters
        let mut master_params = self.master_params.write().await;
        for (master, grad) in master_params.iter_mut().zip(&unscaled_grads) {
            // Would apply optimizer update here
            *master = (master.as_ref() - grad * 0.001)?; // Simplified
        }
        
        // Copy back to model parameters
        let model_params = self.model_params.read().await;
        for (model, master) in model_params.iter().zip(master_params.iter()) {
            // Convert back to model dtype
            let dtype = match self.scaler.config.dtype {
                MixedPrecisionDType::Float16 => DType::F16,
                MixedPrecisionDType::BFloat16 => DType::BF16,
            };
            
            // Would copy master to model param with dtype conversion
        }
        
        // Update scale for next iteration
        self.scaler.update_scale().await?;
        
        Ok(true)
    }
}

/// Clip gradient norm
fn clip_grad_norm(gradients: &mut [Tensor], max_norm: f32) -> Result<()> {
    // Calculate total norm
    let mut total_norm = 0.0f32;
    
    for grad in gradients.iter() {
        let grad_norm = grad.flatten_all()?.sqr()?.sum_all()?.to_scalar::<f32>()?;
        total_norm += grad_norm;
    }
    
    total_norm = total_norm.sqrt();
    
    // Clip if needed
    if total_norm > max_norm {
        let clip_coef = max_norm / (total_norm + 1e-6);
        
        for grad in gradients {
            *grad = (grad.as_ref() * clip_coef as f64)?;
        }
    }
    
    Ok(())
}

/// Automatic mixed precision context
pub struct AMPContext {
    enabled: bool,
    dtype: DType,
    device: Device,
    whitelist: Vec<String>, // Ops to run in low precision
    blacklist: Vec<String>, // Ops to run in full precision
}

impl AMPContext {
    pub fn new(config: &MixedPrecisionConfig, device: Device) -> Self {
        let dtype = match config.dtype {
            MixedPrecisionDType::Float16 => DType::F16,
            MixedPrecisionDType::BFloat16 => DType::BF16,
        };
        
        // Default whitelisted ops for mixed precision
        let whitelist = vec![
            "matmul".to_string(),
            "conv2d".to_string(),
            "linear".to_string(),
        ];
        
        // Default blacklisted ops (should stay in FP32)
        let blacklist = vec![
            "softmax".to_string(),
            "layer_norm".to_string(),
            "batch_norm".to_string(),
            "loss".to_string(),
        ];
        
        Self {
            enabled: config.enabled,
            dtype,
            device,
            whitelist,
            blacklist,
        }
    }
    
    /// Cast tensor based on operation type
    pub fn autocast(&self, tensor: &Tensor, op: &str) -> Result<Tensor> {
        if !self.enabled {
            return Ok(tensor.clone());
        }
        
        if self.whitelist.contains(&op.to_string()) {
            // Cast to low precision
            tensor.to_dtype(self.dtype).map_err(|e| Error::Training(e.to_string()))
        } else if self.blacklist.contains(&op.to_string()) {
            // Keep in FP32
            tensor.to_dtype(DType::F32).map_err(|e| Error::Training(e.to_string()))
        } else {
            // Default behavior based on opt level
            Ok(tensor.clone())
        }
    }
}

/// Mixed precision utilities
pub mod utils {
    use super::*;
    
    /// Convert model to mixed precision
    pub fn convert_model_to_mp(
        model_params: &[Tensor],
        config: &MixedPrecisionConfig,
    ) -> Result<Vec<Tensor>> {
        let target_dtype = match config.dtype {
            MixedPrecisionDType::Float16 => DType::F16,
            MixedPrecisionDType::BFloat16 => DType::BF16,
        };
        
        let mut converted = Vec::new();
        for param in model_params {
            converted.push(param.to_dtype(target_dtype)?);
        }
        
        Ok(converted)
    }
    
    /// Profile memory usage with mixed precision
    pub fn estimate_memory_savings(
        model_size_bytes: usize,
        batch_size: usize,
        config: &MixedPrecisionConfig,
    ) -> MemoryEstimate {
        let fp32_bytes_per_param = 4;
        let mp_bytes_per_param = match config.dtype {
            MixedPrecisionDType::Float16 => 2,
            MixedPrecisionDType::BFloat16 => 2,
        };
        
        let fp32_memory = model_size_bytes;
        let mp_memory = (model_size_bytes / fp32_bytes_per_param) * mp_bytes_per_param;
        
        // Estimate activation memory (rough approximation)
        let activation_factor = match config.opt_level {
            OptLevel::O0 => 1.0,
            OptLevel::O1 => 0.7,
            OptLevel::O2 => 0.5,
            OptLevel::O3 => 0.4,
        };
        
        let fp32_activation = model_size_bytes * batch_size * 4;
        let mp_activation = (fp32_activation as f32 * activation_factor) as usize;
        
        MemoryEstimate {
            fp32_total: fp32_memory + fp32_activation,
            mp_total: mp_memory + mp_activation,
            savings_percent: ((fp32_memory + fp32_activation - mp_memory - mp_activation) as f32 / 
                            (fp32_memory + fp32_activation) as f32 * 100.0),
        }
    }
}

#[derive(Debug)]
pub struct MemoryEstimate {
    pub fp32_total: usize,
    pub mp_total: usize,
    pub savings_percent: f32,
}