use crate::loaders::WeightLoader;
use crate::ops::{Conv2d, GroupNorm, LayerNorm, Linear};
use flame_core::device::Device;
use flame_core::{DType, Error, Result, Shape, Tensor};
// Optimizer trait is in optimizers module
// Module trait is in tensor module
use flame_core::optimizers::{Adam, SGD};
use std::sync::Arc;

// Extension methods for Tensor dimensions
use anyhow;
pub trait TensorDims {
    fn dims4(&self) -> Result<(usize, usize, usize, usize)>;
    fn dims3(&self) -> Result<(usize, usize, usize)>;
}

impl TensorDims for Tensor {
    fn dims4(&self) -> Result<(usize, usize, usize, usize)> {
        let shape = self.shape();
        if shape.rank() != 4 {
            return Err(flame_core::Error::InvalidOperation(format!(
                "Expected 4D tensor, got {}D",
                shape.rank()
            )));
        }
        let dims = shape.dims();
        Ok((dims[0], dims[1], dims[2], dims[3]))
    }

    fn dims3(&self) -> Result<(usize, usize, usize)> {
        let shape = self.shape();
        if shape.rank() != 3 {
            return Err(flame_core::Error::InvalidOperation(format!(
                "Expected 3D tensor, got {}D",
                shape.rank()
            )));
        }
        let dims = shape.dims();
        Ok((dims[0], dims[1], dims[2]))
    }
}

// ResNet blocks implemented in FLAME for diffusion models

// FLAME uses flame_core::device::Device instead of Device

/// Group normalization for FLAME

// Extension trait for Tensor to add missing methods

#[derive(Clone)]
pub struct Conv2dConfig {
    pub padding: usize,
    pub stride: usize,
    pub dilation: usize,
    pub groups: usize,
}

// GroupNorm implementation is in flame_core::group_norm::GroupNorm

/// Conv2D with optional bias
pub struct ResNetConv2d {
    weight: Tensor,
    bias: Option<Tensor>,
    stride: usize,
    padding: usize,
}

impl ResNetConv2d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        device: Device,
    ) -> Result<Self> {
        let fan_in = in_channels * kernel_size * kernel_size;
        let std = (1.0 / fan_in as f32).sqrt();

        let weight = Tensor::randn(
            Shape::from_dims(&[out_channels, in_channels, kernel_size, kernel_size]),
            0.0,
            0.02,
            device.cuda_device().clone(),
        )?;

        let bias = Tensor::zeros(Shape::from_dims(&[out_channels]), device.cuda_device().clone())?;

        Ok(Self { weight, bias: Some(bias), stride, padding })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.conv2d(&self.weight, self.bias.as_ref(), self.stride, self.padding)
    }
}

/// ResNet block for diffusion models
pub struct ResnetBlock2D {
    norm1: GroupNorm,
    conv1: Conv2d,
    norm2: GroupNorm,
    conv2: Conv2d,
    conv_shortcut: Option<Conv2d>,
    time_emb_proj: Option<Linear>,
    dropout: f32,
}

impl ResnetBlock2D {
    pub fn load(weights: &WeightLoader, prefix: &str, channels: usize) -> Result<Self> {
        let prefixed_weights = weights.pp(prefix);

        let norm1_weight = weights.tensor("norm1.weight", &[channels])?;
        let norm1_bias = weights.tensor("norm1.bias", &[channels])?;
        let mut norm1 = GroupNorm::new(32, channels, 1e-6, true, norm1_weight.device().clone())?;
        norm1.weight = Some(norm1_weight);
        norm1.bias = Some(norm1_bias);

        let conv1_weight = weights.tensor("conv1.weight", &[channels, channels, 3, 3])?;
        let conv1_bias = weights.tensor("conv1.bias", &[channels]).ok();
        let conv1 = {
            let weight_tensor = conv1_weight;
            let bias_tensor = conv1_bias;
            let in_channels = weight_tensor.shape().dims()[1];
            let out_channels = weight_tensor.shape().dims()[0];
            let kernel_size = weight_tensor.shape().dims()[2];
            let mut conv = Conv2d::new(
                in_channels,
                out_channels,
                kernel_size,
                1,
                1,
                weight_tensor.device().clone(),
            )?;
            conv.weight = weight_tensor;
            conv.bias = bias_tensor;
            conv
        };

        let norm2_weight = prefixed_weights.tensor("norm2.weight", &[channels])?;
        let norm2_bias = prefixed_weights.tensor("norm2.bias", &[channels])?;
        let mut norm2 = GroupNorm::new(32, channels, 1e-6, true, norm2_weight.device().clone())?;
        norm2.weight = Some(norm2_weight);
        norm2.bias = Some(norm2_bias);

        let conv2_weight = prefixed_weights.tensor("conv2.weight", &[channels, channels, 3, 3])?;
        let conv2_bias = prefixed_weights.tensor("conv2.bias", &[channels]).ok();
        let conv2 = {
            let weight_tensor = conv2_weight;
            let bias_tensor = conv2_bias;
            let in_channels = weight_tensor.shape().dims()[1];
            let out_channels = weight_tensor.shape().dims()[0];
            let kernel_size = weight_tensor.shape().dims()[2];
            let mut conv = Conv2d::new(
                in_channels,
                out_channels,
                kernel_size,
                1,
                1,
                weight_tensor.device().clone(),
            )?;
            conv.weight = weight_tensor;
            conv.bias = bias_tensor;
            conv
        };

        // Skip connection if exists
        let conv_shortcut = if prefixed_weights.get("conv_shortcut.weight").is_ok() {
            let weight =
                prefixed_weights.tensor("conv_shortcut.weight", &[channels, channels, 1, 1])?;
            let bias = prefixed_weights.tensor("conv_shortcut.bias", &[channels]).ok();
            Some({
                let weight_tensor = weight;
                let bias_tensor = bias;
                let in_channels = weight_tensor.shape().dims()[1];
                let out_channels = weight_tensor.shape().dims()[0];
                let kernel_size = weight_tensor.shape().dims()[2];
                let mut conv = Conv2d::new(
                    in_channels,
                    out_channels,
                    kernel_size,
                    1,
                    0,
                    weight_tensor.device().clone(),
                )?;
                conv.weight = weight_tensor;
                conv.bias = bias_tensor;
                conv
            })
        } else {
            None
        };

        // Time projection if exists
        let time_emb_proj = if weights.get("time_emb_proj.weight").is_ok() {
            let weight = weights.tensor("time_emb_proj.weight", &[channels, 1280])?;
            let bias = weights.tensor("time_emb_proj.bias", &[channels]).ok();
            Some({
                let weight_tensor = weight;
                let bias_tensor = bias;
                let in_features = weight_tensor.shape().dims()[1];
                let out_features = weight_tensor.shape().dims()[0];
                let mut linear = Linear::new(
                    in_features,
                    out_features,
                    bias_tensor.is_some(),
                    &weight_tensor.device().clone(),
                )?;
                linear.weight = weight_tensor;
                linear.bias = bias_tensor;
                linear
            })
        } else {
            None
        };

        Ok(Self { norm1, conv1, norm2, conv2, conv_shortcut, time_emb_proj, dropout: 0.0 })
    }

    pub fn new(
        in_channels: usize,
        out_channels: usize,
        temb_channels: Option<usize>,
        groups: usize,
        device: Device,
    ) -> Result<Self> {
        let norm1 = GroupNorm::new(groups, in_channels, 1e-6, true, device.cuda_device().clone())?;
        let conv1 = Conv2d::new(in_channels, out_channels, 3, 1, 1, device.cuda_device().clone())?;

        let norm2 = GroupNorm::new(groups, out_channels, 1e-6, true, device.cuda_device().clone())?;
        let conv2 = Conv2d::new(out_channels, out_channels, 3, 1, 1, device.cuda_device().clone())?;

        let conv_shortcut = if in_channels != out_channels {
            Some(Conv2d::new(in_channels, out_channels, 1, 1, 0, device.cuda_device().clone())?)
        } else {
            None
        };

        let time_emb_proj = temb_channels.map(|temb_ch| {
            Linear::new(temb_ch, out_channels, true, &device.cuda_device()).unwrap()
        });

        Ok(Self { norm1, conv1, norm2, conv2, conv_shortcut, time_emb_proj, dropout: 0.0 })
    }

    pub fn forward(&self, hidden_states: &Tensor, temb: Option<&Tensor>) -> Result<Tensor> {
        let residual = hidden_states;

        // First conv block
        let hidden_states = self.norm1.forward(hidden_states)?;
        let hidden_states = hidden_states.silu()?;
        let hidden_states = self.conv1.forward(&hidden_states)?;

        // Add time embedding if present
        let hidden_states = if let (Some(temb), Some(proj)) = (temb, &self.time_emb_proj) {
            let temb_out = proj.forward(temb)?.silu()?;
            // Reshape temb from [batch, channels] to [batch, channels, 1, 1]
            let temb_out = temb_out.unsqueeze(2)?.unsqueeze(3)?;
            hidden_states.add(&temb_out)?
        } else {
            hidden_states
        };

        // Second conv block
        let hidden_states = self.norm2.forward(&hidden_states)?;
        let hidden_states = hidden_states.silu()?;

        // Dropout would go here (skipped for inference)

        let hidden_states = self.conv2.forward(&hidden_states)?;

        // Skip connection
        let residual = if let Some(conv) = &self.conv_shortcut {
            conv.forward(residual)?
        } else {
            residual.clone()
        };

        hidden_states.add(&residual)
    }
}

/// Downsample block
pub struct Downsample2D {
    conv: Conv2d,
}

impl Downsample2D {
    pub fn load(weights: &WeightLoader, prefix: &str) -> Result<Self> {
        let prefixed_weights = weights.pp(prefix);

        // TODO: Fix hardcoded channel sizes
        let weight = prefixed_weights.tensor("conv.weight", &[320, 320, 3, 3])?; // channels will vary
        let conv = Conv2d::new(320, 320, 3, 2, 1, weight.device().clone())?;
        // TODO: Load weight and bias after creating Conv2d

        Ok(Self { conv })
    }

    pub fn new(channels: usize, device: Device) -> Result<Self> {
        // Downsample with stride 2
        let conv = Conv2d::new(channels, channels, 3, 2, 1, device.cuda_device().clone())?;
        Ok(Self { conv })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.conv.forward(x)
    }
}

/// Upsample block
pub struct Upsample2D {
    conv: Conv2d,
}

impl Upsample2D {
    pub fn load(weights: &WeightLoader, prefix: &str) -> Result<Self> {
        let prefixed_weights = weights.pp(prefix);

        // TODO: Fix hardcoded channel sizes
        let weight = prefixed_weights.tensor("conv.weight", &[320, 320, 3, 3])?; // channels will vary
        let conv = Conv2d::new(320, 320, 3, 1, 1, weight.device().clone())?;
        // TODO: Load weight and bias after creating Conv2d

        Ok(Self { conv })
    }

    pub fn new(channels: usize, device: Device) -> Result<Self> {
        let conv = Conv2d::new(channels, channels, 3, 1, 1, device.cuda_device().clone())?;
        Ok(Self { conv })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Nearest neighbor upsampling by 2x
        let dims = x.shape().dims();
        let batch = dims[0];
        let channels = dims[1];
        let height = dims[2];
        let width = dims[3];

        // Simple nearest neighbor upsampling
        let x_up = x.repeat_interleave(2, 2)?.repeat_interleave(2, 3)?;

        // Apply convolution
        self.conv.forward(&x_up)
    }
}

// Helper extension methods for Tensor

// Tensor extension methods for ResNet operations
pub trait ResNetTensorExt {
    fn unsqueeze(&self, dim: usize) -> Result<Tensor>;
    fn mean_dim(&self, dim: usize) -> Result<Tensor>;
    fn div(&self, other: &Tensor) -> Result<Tensor>;
    fn recip(&self) -> Result<Tensor>;
    fn sqrt(&self) -> Result<Tensor>;
    fn repeat_interleave(&self, repeats: usize, dim: usize) -> Result<Tensor>;
}

impl ResNetTensorExt for Tensor {
    /// Add dimension at specified position
    fn unsqueeze(&self, dim: usize) -> Result<Tensor> {
        let mut new_dims = self.shape().dims().to_vec();
        new_dims.insert(dim, 1);
        self.reshape(&new_dims)
    }

    /// Calculate mean along dimension
    fn mean_dim(&self, dim: usize) -> Result<Tensor> {
        // Use FLAME's mean_dim operation with keepdim=true
        self.mean_dim(&[dim], true)
    }

    /// Element-wise division
    fn div(&self, other: &Tensor) -> Result<Tensor> {
        // Implement as multiplication by reciprocal
        let recip = other.reciprocal()?;
        self.mul(&recip)
    }

    /// Reciprocal (1/x)
    fn recip(&self) -> Result<Tensor> {
        // This would need a CUDA kernel implementation
        let data = self.to_vec()?;
        let recip_data: Vec<f32> = data.iter().map(|&x| 1.0 / x).collect();
        Ok(Tensor::from_vec(recip_data, self.shape().clone(), self.device().clone())?)
    }

    /// Square root
    fn sqrt(&self) -> Result<Tensor> {
        // This would need a CUDA kernel implementation
        let data = self.to_vec()?;
        let sqrt_data: Vec<f32> = data.iter().map(|&x| x.sqrt()).collect();
        Ok(Tensor::from_vec(sqrt_data, self.shape().clone(), self.device().clone())?)
    }

    /// Repeat tensor along dimension
    fn repeat_interleave(&self, repeats: usize, dim: usize) -> Result<Tensor> {
        // Simplified implementation
        let dims = self.shape().dims();
        let mut new_dims = dims.to_vec();
        new_dims[dim] *= repeats;

        // This would need proper CUDA implementation
        let data = self.to_vec()?;
        let mut repeated_data = Vec::new();

        // Simple CPU implementation for now
        // Would be replaced with CUDA kernel
        match dim {
            2 => {
                // Repeat along height
                for b in 0..dims[0] {
                    for c in 0..dims[1] {
                        for h in 0..dims[2] {
                            for _ in 0..repeats {
                                for w in 0..dims[3] {
                                    let idx = b * dims[1] * dims[2] * dims[3]
                                        + c * dims[2] * dims[3]
                                        + h * dims[3]
                                        + w;
                                    repeated_data.push(data[idx]);
                                }
                            }
                        }
                    }
                }
            }
            3 => {
                // Repeat along width
                for i in 0..data.len() {
                    for _ in 0..repeats {
                        repeated_data.push(data[i]);
                    }
                }
            }
            _ => {
                return Err(flame_core::Error::InvalidOperation(
                    "Unsupported dimension for repeat_interleave".to_string(),
                ));
            }
        }

        Ok(Tensor::from_vec(repeated_data, Shape::from_dims(&new_dims), self.device().clone())?)
    }
}
