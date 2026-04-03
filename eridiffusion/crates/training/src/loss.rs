//! Loss functions for training

use std::sync::Arc;

use eridiffusion_core::{Error, Result};
use flame_core::{CudaDevice, DType, Shape, Tensor};
use serde::{Deserialize, Serialize};

use crate::policy;

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
        let diff = pred.sub(target)?;
        let squared = diff.square()?;

        let loss = match self.config.reduction {
            Reduction::None => squared,
            Reduction::Mean => crate::policy::reduce_mean_fp32_keepdim(&squared)?,
            Reduction::Sum => squared.sum()?,
        };

        // Safely convert weight to f64 with validation
        let weight = self.config.weight as f32;
        if !weight.is_finite() || weight < 0.0 {
            return Err(Error::Training(format!("Invalid loss weight: {}", weight)));
        }
        Ok(loss.affine(weight, 0.0f32)?)
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
        let diff = pred.sub(target)?;
        let abs_diff = diff.abs()?;

        let loss = match self.config.reduction {
            Reduction::None => abs_diff,
            Reduction::Mean => crate::policy::reduce_mean_fp32_keepdim(&abs_diff)?,
            Reduction::Sum => abs_diff.sum()?,
        };

        // Safely convert weight to f64 with validation
        let weight = self.config.weight as f32;
        if !weight.is_finite() || weight < 0.0 {
            return Err(Error::Training(format!("Invalid loss weight: {}", weight)));
        }
        Ok(loss.affine(weight, 0.0f32)?)
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
        let diff = pred.sub(target)?;
        let abs_diff = diff.abs()?;

        // Huber loss: 0.5 * x^2 if |x| <= delta, else delta * |x| - 0.5 * delta^2
        let one_shape = Shape::from_dims(&[1]);
        let delta_tensor = Tensor::from_slice(&[self.delta], one_shape, pred.device().clone())?
            .broadcast_to(abs_diff.shape())?;

        let is_small = abs_diff.le(&delta_tensor)?;

        let squared_loss = diff.square()?.affine(0.5f32, 0.0f32)?;
        let abs_diff_scaled = abs_diff.affine(self.delta as f32, 0.0f32)?;
        let linear_loss = abs_diff_scaled.affine(1.0f32, -(0.5f32 * self.delta * self.delta))?;

        let loss = Tensor::where_mask(&is_small, &squared_loss, &linear_loss)?;

        let final_loss = match self.config.reduction {
            Reduction::None => loss,
            Reduction::Mean => crate::policy::reduce_mean_fp32_keepdim(&loss)?,
            Reduction::Sum => loss.sum()?,
        };

        Ok(final_loss.affine(self.config.weight as f32, 0.0f32)?)
    }

    fn name(&self) -> &str {
        "Huber"
    }
}

/// LPIPS (Learned Perceptual Image Patch Similarity) loss
pub struct LPIPSLoss {
    config: LossConfig,
    // Feature extraction layers (mimicking VGG-like architecture)
    conv1_weight: Tensor,
    conv2_weight: Tensor,
    conv3_weight: Tensor,
}

impl LPIPSLoss {
    pub fn new(config: LossConfig) -> Result<Self> {
        // Validate config
        if config.weight < 0.0 || !config.weight.is_finite() {
            return Err(Error::Training(format!("Invalid LPIPS weight: {}", config.weight)));
        }

        // Initialize simple feature extraction kernels
        // In production, these would be loaded from pretrained VGG/AlexNet
        let dev: Arc<CudaDevice> =
            CudaDevice::new(0).map_err(|e| Error::Device(format!("CUDA device: {}", e)))?;

        // Edge detection kernel for conv1 (3x3)
        let conv1_weight = Tensor::from_slice(
            &[-1.0, -1.0, -1.0, -1.0, 8.0, -1.0, -1.0, -1.0, -1.0],
            Shape::from_dims(&[1, 1, 3, 3]),
            dev.clone(),
        )?;

        // Gabor-like kernel for conv2 (3x3)
        let conv2_weight = Tensor::from_slice(
            &[0.0, -1.0, 0.0, -1.0, 4.0, -1.0, 0.0, -1.0, 0.0],
            Shape::from_dims(&[1, 1, 3, 3]),
            dev.clone(),
        )?;

        // Texture detection kernel for conv3 (3x3)
        let conv3_weight = Tensor::from_slice(
            &[1.0, -2.0, 1.0, -2.0, 4.0, -2.0, 1.0, -2.0, 1.0],
            Shape::from_dims(&[1, 1, 3, 3]),
            dev,
        )?;

        Ok(Self { config, conv1_weight, conv2_weight, conv3_weight })
    }

    fn downsample(&self, x: &Tensor, factor: usize) -> Result<Tensor> {
        let [b, c, h, w] = x.dims4();
        let new_h = h / factor;
        let new_w = w / factor;

        // Simple average pooling by reshaping and mean
        let x_reshaped = x.reshape(&[b, c, new_h, factor, new_w, factor])?;
        let pooled = x_reshaped.mean_along_dims(&[3], true)?.mean_along_dims(&[5], true)?;
        Ok(pooled.reshape(&[b, c, new_h, new_w])?)
    }

    fn extract_features(&self, x: &Tensor) -> Result<Vec<Tensor>> {
        // Multi-scale perceptual feature extraction
        let mut features = Vec::new();

        // Convert to grayscale if needed (average across channels)
        let gray = if x.dims()[1] == 3 { x.mean_along_dims(&[1], true)? } else { x.clone() };

        // Feature 1: Edge detection at original resolution
        let edges = gray.conv2d(
            &self.conv1_weight,
            None,
            1, // stride
            1, // padding
        )?;
        features.push(edges.abs()?);

        // Feature 2: Mid-level features at half resolution
        let scale2 = self.downsample(&gray, 2)?;
        let mid_features = scale2.conv2d(&self.conv2_weight, None, 1, 1)?;
        features.push(mid_features.abs()?);

        // Feature 3: Texture features at quarter resolution
        let scale3 = self.downsample(&gray, 4)?;
        let texture = scale3.conv2d(&self.conv3_weight, None, 1, 1)?;
        features.push(texture.abs()?);

        // Feature 4: Color statistics (if input is color)
        if x.dims()[1] == 3 {
            // Compute color channel statistics
            let color_mean = x.mean_along_dims(&[2, 3], true)?;
            let color_var = x.var(&[2, 3], false, true)?;
            let color_std = color_var.sqrt()?;
            features.push(color_mean);
            features.push(color_std);
        }

        Ok(features)
    }
}

impl Loss for LPIPSLoss {
    fn compute(&self, pred: &Tensor, target: &Tensor) -> Result<Tensor> {
        // Validate inputs
        if pred.shape() != target.shape() {
            return Err(Error::Training("LPIPS: pred and target shapes must match".into()));
        }

        // Extract multi-scale features
        let features_pred = self.extract_features(pred)?;
        let features_target = self.extract_features(target)?;

        // Compute perceptual distance at each scale
        let mut total_loss = Tensor::zeros(Shape::from_dims(&[]), pred.device().clone())?;

        for (fp, ft) in features_pred.iter().zip(features_target.iter()) {
            let diff = fp.sub(ft)?;
            let scale_loss = crate::policy::reduce_mean_fp32_keepdim(&diff.square()?)?;
            total_loss = total_loss.add(&scale_loss)?;
        }

        // Average across scales and apply weight
        let loss = total_loss.affine(1.0f32 / features_pred.len() as f32, 0.0f32)?;

        // Safely convert weight to f64 with validation
        let weight = self.config.weight as f32;
        if !weight.is_finite() || weight < 0.0 {
            return Err(Error::Training(format!("Invalid loss weight: {}", weight)));
        }
        Ok(loss.affine(weight, 0.0f32)?)
    }

    fn name(&self) -> &str {
        "LPIPS"
    }
}

// -----------------------------------------------------------------------------
// Helpers wired to policy.rs — FP32 reductions and explicit ops
// -----------------------------------------------------------------------------

/// ε-loss with optional latent mask; FP32 reduction via policy
pub fn masked_eps_loss(
    pred: &Tensor,
    eps: &Tensor,
    mask: Option<&Tensor>,
) -> anyhow::Result<Tensor> {
    let diff = pred.sub(eps)?; // pred - eps
    let sq = diff.mul(&diff)?; // (pred - eps)^2
    let out = if let Some(m) = mask { sq.mul(m)? } else { sq };
    policy::reduce_mean_fp32_keepdim(&out).map_err(|e| anyhow::anyhow!(e))
}

/// v-pred loss with optional mask
pub fn masked_v_loss(
    pred_v: &Tensor,
    target_v: &Tensor,
    mask: Option<&Tensor>,
) -> anyhow::Result<Tensor> {
    let diff = pred_v.sub(target_v)?;
    let sq = diff.mul(&diff)?;
    let out = if let Some(m) = mask { sq.mul(m)? } else { sq };
    policy::reduce_mean_fp32_keepdim(&out).map_err(|e| anyhow::anyhow!(e))
}

/// L1 variant (debugging)
pub fn masked_l1_loss(
    pred: &Tensor,
    target: &Tensor,
    mask: Option<&Tensor>,
) -> anyhow::Result<Tensor> {
    let diff = pred.sub(target)?.abs()?;
    let out = if let Some(m) = mask { diff.mul(m)? } else { diff };
    policy::reduce_mean_fp32_keepdim(&out).map_err(|e| anyhow::anyhow!(e))
}

/// Broadcast a per-sample sigma (B,) to [B,1,1,1] and scale x
pub fn scale_by_sigma(x: &Tensor, sigma_b: &Tensor) -> Result<Tensor> {
    let b = sigma_b.shape().dims()[0];
    let s = sigma_b.reshape(&[b, 1, 1, 1])?;
    Ok(x.mul(&s)?)
}

/// Ensure FP32 numerics for targets
#[inline]
pub fn to_fp32(x: &Tensor) -> Result<Tensor> {
    Ok(x.to_dtype(DType::F32)?)
}

/// Perceptual loss
pub struct PerceptualLoss {
    config: LossConfig,
    // Would contain feature extractor
}

impl PerceptualLoss {
    pub fn new(config: LossConfig) -> anyhow::Result<Self> {
        Ok(Self { config })
    }
}

impl Loss for PerceptualLoss {
    fn compute(&self, pred: &Tensor, target: &Tensor) -> Result<Tensor> {
        // Simplified perceptual loss
        // In practice would compute features at multiple layers

        let diff = pred.sub(target)?;
        let loss = crate::policy::reduce_mean_fp32_keepdim(&diff.square()?)?;

        // Safely convert weight to f64 with validation
        let weight = self.config.weight as f32;
        if !weight.is_finite() || weight < 0.0 {
            return Err(Error::Training(format!("Invalid loss weight: {}", weight)));
        }
        Ok(loss.mul_scalar(weight)?)
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
        let velocity_error = pred.sub(target)?;
        let loss = crate::policy::reduce_mean_fp32_keepdim(&velocity_error.square()?)?;

        // Safely convert weight to f64 with validation
        let weight = self.config.weight as f32;
        if !weight.is_finite() || weight < 0.0 {
            return Err(Error::Training(format!("Invalid loss weight: {}", weight)));
        }
        Ok(loss.mul_scalar(weight)?)
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
        let diff = pred.sub(target)?;
        let loss = crate::policy::reduce_mean_fp32_keepdim(&diff.square()?)?;

        // Safely convert weight to f64 with validation
        let weight = self.config.weight as f32;
        if !weight.is_finite() || weight < 0.0 {
            return Err(Error::Training(format!("Invalid loss weight: {}", weight)));
        }
        Ok(loss.mul_scalar(weight)?)
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
    let mut total_loss = Tensor::zeros(Shape::from_dims(&[]), pred.device().clone())?;

    for (loss_fn, weight) in losses {
        let loss = loss_fn.compute(pred, target)?;
        total_loss = total_loss.add(&loss.mul_scalar(*weight as f32)?)?;
    }

    Ok(total_loss)
}
