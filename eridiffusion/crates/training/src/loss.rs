//! Loss functions for training

use eridiffusion_core::{Result, Error};
use candle_core::{Tensor, DType};
use candle_nn::Module;
use serde::{Serialize, Deserialize};

/// Loss function trait
pub trait Loss: Send + Sync {
    /// Compute loss
    fn compute(&self, pred: &Tensor, target: &Tensor) -> Result<Tensor>;
    
    /// Get loss name
    fn name(&self) -> &str;
}

/// Loss type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum LossType {
    MSE,
    MAE,
    Huber,
    LPIPS,
    Perceptual,
    FlowMatching,
    VLoss,
}

/// Create loss function
pub fn create_loss(loss_type: LossType, config: LossConfig) -> Result<Box<dyn Loss>> {
    match loss_type {
        LossType::MSE => Ok(Box::new(MSELoss::new(config))),
        LossType::MAE => Ok(Box::new(MAELoss::new(config))),
        LossType::Huber => Ok(Box::new(HuberLoss::new(config))),
        LossType::LPIPS => Ok(Box::new(LPIPSLoss::new(config)?)),
        LossType::Perceptual => Ok(Box::new(PerceptualLoss::new(config)?)),
        LossType::FlowMatching => Ok(Box::new(FlowMatchingLoss::new(config))),
        LossType::VLoss => Ok(Box::new(VLoss::new(config))),
    }
}

/// Loss configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LossConfig {
    pub weight: f32,
    pub reduction: Reduction,
    pub huber_delta: Option<f32>,
    pub lpips_net: Option<String>,
    pub perceptual_layers: Option<Vec<String>>,
}

impl Default for LossConfig {
    fn default() -> Self {
        Self {
            weight: 1.0,
            reduction: Reduction::Mean,
            huber_delta: Some(1.0),
            lpips_net: Some("vgg".to_string()),
            perceptual_layers: None,
        }
    }
}

/// Reduction type
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Reduction {
    None,
    Mean,
    Sum,
}

/// MSE loss
pub struct MSELoss {
    config: LossConfig,
}

impl MSELoss {
    pub fn new(config: LossConfig) -> Self {
        Self { config }
    }
}

impl Loss for MSELoss {
    fn compute(&self, pred: &Tensor, target: &Tensor) -> Result<Tensor> {
        let diff = (pred - target)?;
        let squared = diff.sqr()?;
        
        let loss = match self.config.reduction {
            Reduction::None => squared,
            Reduction::Mean => squared.mean_all()?,
            Reduction::Sum => squared.sum_all()?,
        };
        
        // Safely convert weight to f64 with validation
        let weight = self.config.weight as f64;
        if !weight.is_finite() || weight < 0.0 {
            return Err(Error::Training(format!("Invalid loss weight: {}", weight)));
        }
        Ok((loss * weight)?)
    }
    
    fn name(&self) -> &str {
        "MSE"
    }
}

/// MAE loss
pub struct MAELoss {
    config: LossConfig,
}

impl MAELoss {
    pub fn new(config: LossConfig) -> Self {
        Self { config }
    }
}

impl Loss for MAELoss {
    fn compute(&self, pred: &Tensor, target: &Tensor) -> Result<Tensor> {
        let diff = (pred - target)?;
        let abs_diff = diff.abs()?;
        
        let loss = match self.config.reduction {
            Reduction::None => abs_diff,
            Reduction::Mean => abs_diff.mean_all()?,
            Reduction::Sum => abs_diff.sum_all()?,
        };
        
        // Safely convert weight to f64 with validation
        let weight = self.config.weight as f64;
        if !weight.is_finite() || weight < 0.0 {
            return Err(Error::Training(format!("Invalid loss weight: {}", weight)));
        }
        Ok((loss * weight)?)
    }
    
    fn name(&self) -> &str {
        "MAE"
    }
}

/// Huber loss
pub struct HuberLoss {
    config: LossConfig,
    delta: f32,
}

impl HuberLoss {
    pub fn new(config: LossConfig) -> Self {
        let delta = config.huber_delta.unwrap_or(1.0);
        Self { config, delta }
    }
}

impl Loss for HuberLoss {
    fn compute(&self, pred: &Tensor, target: &Tensor) -> Result<Tensor> {
        let diff = (pred - target)?;
        let abs_diff = diff.abs()?;
        
        // Huber loss: 0.5 * x^2 if |x| <= delta, else delta * |x| - 0.5 * delta^2
        let delta_tensor = Tensor::new(&[self.delta], pred.device())?
            .broadcast_as(abs_diff.shape())?;
        
        let is_small = abs_diff.le(&delta_tensor)?;
        
        let squared_loss = diff.sqr()?.affine(0.5, 0.0)?;
        let abs_diff_scaled = abs_diff.affine(self.delta as f64, 0.0)?;
        let linear_loss = abs_diff_scaled.affine(1.0, -(0.5 * self.delta * self.delta) as f64)?;
        
        let loss = is_small.where_cond(&squared_loss, &linear_loss)?;
        
        let final_loss = match self.config.reduction {
            Reduction::None => loss,
            Reduction::Mean => loss.mean_all()?,
            Reduction::Sum => loss.sum_all()?,
        };
        
        Ok((final_loss * self.config.weight as f64)?)
    }
    
    fn name(&self) -> &str {
        "Huber"
    }
}

/// LPIPS loss (simplified version)
pub struct LPIPSLoss {
    config: LossConfig,
    // In practice, would load pretrained VGG/AlexNet features
}

impl LPIPSLoss {
    pub fn new(config: LossConfig) -> Result<Self> {
        // Validate config
        if config.weight < 0.0 || !config.weight.is_finite() {
            return Err(Error::Training(format!("Invalid LPIPS weight: {}", config.weight)));
        }
        Ok(Self { config })
    }
    
    fn downsample(&self, x: &Tensor, factor: usize) -> Result<Tensor> {
        let (b, c, h, w) = x.dims4()?;
        let new_h = h / factor;
        let new_w = w / factor;
        
        // Simple average pooling by reshaping and mean
        let x_reshaped = x.reshape(&[b, c, new_h, factor, new_w, factor])?;
        let pooled = x_reshaped.mean_keepdim(3)?.mean_keepdim(5)?;
        Ok(pooled.reshape(&[b, c, new_h, new_w])?)
    }
    
    fn extract_features(&self, x: &Tensor) -> Result<Vec<Tensor>> {
        // Multi-scale feature extraction (simplified)
        let mut features = Vec::new();
        
        // Scale 1: Original resolution
        features.push(x.clone());
        
        // Scale 2: Downsampled by 2
        let scale2 = self.downsample(x, 2)?;
        features.push(scale2);
        
        // Scale 3: Downsampled by 4  
        let scale3 = self.downsample(x, 4)?;
        features.push(scale3);
        
        Ok(features)
    }
}

impl Loss for LPIPSLoss {
    fn compute(&self, pred: &Tensor, target: &Tensor) -> Result<Tensor> {
        // Validate inputs
        if pred.shape() != target.shape() {
            return Err(Error::Training("LPIPS: pred and target shapes must match".to_string()));
        }
        
        // Extract multi-scale features
        let features_pred = self.extract_features(pred)?;
        let features_target = self.extract_features(target)?;
        
        // Compute perceptual distance at each scale
        let mut total_loss = Tensor::zeros(&[], candle_core::DType::F32, &pred.device())?;
        
        for (fp, ft) in features_pred.iter().zip(features_target.iter()) {
            let diff = (fp - ft)?;
            let scale_loss = diff.sqr()?.mean_all()?;
            total_loss = (total_loss + scale_loss)?;
        }
        
        // Average across scales and apply weight
        let loss = total_loss.affine(1.0 / features_pred.len() as f64, 0.0)?;
        
        // Safely convert weight to f64 with validation
        let weight = self.config.weight as f64;
        if !weight.is_finite() || weight < 0.0 {
            return Err(Error::Training(format!("Invalid loss weight: {}", weight)));
        }
        Ok((loss * weight)?)
    }
    
    fn name(&self) -> &str {
        "LPIPS"
    }
}

/// Perceptual loss
pub struct PerceptualLoss {
    config: LossConfig,
    // Would contain feature extractor
}

impl PerceptualLoss {
    pub fn new(config: LossConfig) -> Result<Self> {
        Ok(Self { config })
    }
}

impl Loss for PerceptualLoss {
    fn compute(&self, pred: &Tensor, target: &Tensor) -> Result<Tensor> {
        // Simplified perceptual loss
        // In practice would compute features at multiple layers
        
        let diff = (pred - target)?;
        let loss = diff.sqr()?.mean_all()?;
        
        // Safely convert weight to f64 with validation
        let weight = self.config.weight as f64;
        if !weight.is_finite() || weight < 0.0 {
            return Err(Error::Training(format!("Invalid loss weight: {}", weight)));
        }
        Ok((loss * weight)?)
    }
    
    fn name(&self) -> &str {
        "Perceptual"
    }
}

/// Flow matching loss
pub struct FlowMatchingLoss {
    config: LossConfig,
}

impl FlowMatchingLoss {
    pub fn new(config: LossConfig) -> Self {
        Self { config }
    }
}

impl Loss for FlowMatchingLoss {
    fn compute(&self, pred: &Tensor, target: &Tensor) -> Result<Tensor> {
        // Flow matching: minimize ||v_θ(x_t, t) - (x_1 - x_0)||²
        let velocity_error = (pred - target)?;
        let loss = velocity_error.sqr()?.mean_all()?;
        
        // Safely convert weight to f64 with validation
        let weight = self.config.weight as f64;
        if !weight.is_finite() || weight < 0.0 {
            return Err(Error::Training(format!("Invalid loss weight: {}", weight)));
        }
        Ok((loss * weight)?)
    }
    
    fn name(&self) -> &str {
        "FlowMatching"
    }
}

/// V-parameterization loss
pub struct VLoss {
    config: LossConfig,
}

impl VLoss {
    pub fn new(config: LossConfig) -> Self {
        Self { config }
    }
}

impl Loss for VLoss {
    fn compute(&self, pred: &Tensor, target: &Tensor) -> Result<Tensor> {
        // V-parameterization loss
        let diff = (pred - target)?;
        let loss = diff.sqr()?.mean_all()?;
        
        // Safely convert weight to f64 with validation
        let weight = self.config.weight as f64;
        if !weight.is_finite() || weight < 0.0 {
            return Err(Error::Training(format!("Invalid loss weight: {}", weight)));
        }
        Ok((loss * weight)?)
    }
    
    fn name(&self) -> &str {
        "V-Loss"
    }
}

/// Compute combined loss
pub fn compute_loss(
    losses: &[(Box<dyn Loss>, f32)],
    pred: &Tensor,
    target: &Tensor,
) -> Result<Tensor> {
    let mut total_loss = Tensor::zeros(&[], DType::F32, pred.device())?;
    
    for (loss_fn, weight) in losses {
        let loss = loss_fn.compute(pred, target)?;
        total_loss = (total_loss + (loss * *weight as f64)?)?;
    }
    
    Ok(total_loss)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    
    #[test]
    fn test_mse_loss() -> Result<()> {
        let device = Device::Cpu;
        let pred = Tensor::randn(0.0f32, 1.0, (2, 3, 4, 4), &device)?;
        let target = Tensor::randn(0.0f32, 1.0, (2, 3, 4, 4), &device)?;
        
        let loss = MSELoss::new(LossConfig::default());
        let result = loss.compute(&pred, &target)?;
        
        assert_eq!(result.dims(), &[]);
        Ok(())
    }
    
    #[test]
    fn test_huber_loss() -> Result<()> {
        let device = Device::Cpu;
        let pred = Tensor::randn(0.0f32, 1.0, (2, 3, 4, 4), &device)?;
        let target = Tensor::randn(0.0f32, 1.0, (2, 3, 4, 4), &device)?;
        
        let config = LossConfig {
            huber_delta: Some(1.0),
            ..Default::default()
        };
        
        let loss = HuberLoss::new(config);
        let result = loss.compute(&pred, &target)?;
        
        assert_eq!(result.dims(), &[]);
        Ok(())
    }
}