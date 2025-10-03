use crate::loaders::WeightLoader;
use anyhow::{anyhow, Context};
use flame_core::device::Device;
use flame_core::optimizers::{Adam, SGD};
use flame_core::Result;
use flame_core::{DType, Shape, Tensor};
use std::collections::HashMap;

pub struct PrefixedWeightLoader<'a> {
    loader: &'a WeightLoader,
    prefix: String,
}
pub struct RmsNorm {
    weight: Tensor,
    eps: f64,
}
pub struct RMSNorm(pub RmsNorm);

impl RMSNorm {
    pub fn new(dim: usize, eps: f64, wl: &WeightLoader) -> flame_core::Result<Self> {
        let weight = wl.tensor("weight", &[dim])?;
        let rms_norm = RmsNorm { weight, eps };
        Ok(RMSNorm(rms_norm))
    }

    pub fn forward(&self, xs: &Tensor, device: &Device) -> flame_core::Result<Tensor> {
        self.0.forward(xs)
    }
}

// FLAME uses flame_core::device::Device instead of Device

/// RMS Norm implementation that works on GPU by using native FLAME operations
/// This avoids the "no cuda implementation" error by using operations that do have CUDA support

// WeightLoader implementation is in crate::loaders::WeightLoader

// Clone implementation removed - defined elsewhere

impl<'a> PrefixedWeightLoader<'a> {
    pub fn get(&self, key: &str) -> flame_core::Result<&Tensor> {
        let full_key = format!("{}.{}", self.prefix, key);
        self.loader.get(&full_key)
    }

    pub fn tensor(&self, key: &str, shape: &[usize]) -> flame_core::Result<Tensor> {
        let full_key = format!("{}.{}", self.prefix, key);
        self.loader.tensor(&full_key, shape)
    }

    pub fn pp(&self, prefix: &str) -> PrefixedWeightLoader<'a> {
        PrefixedWeightLoader { loader: self.loader, prefix: format!("{}.{}", self.prefix, prefix) }
    }
}

impl RmsNorm {
    pub fn new(weight: Tensor, eps: f64) -> Self {
        Self { weight, eps }
    }

    pub fn load(weights: &WeightLoader, dim: usize, eps: f64) -> flame_core::Result<Self> {
        let weight = weights.tensor("weight", &[dim])?;
        Ok(Self::new(weight, eps))
    }

    pub fn forward(&self, xs: &Tensor) -> flame_core::Result<Tensor> {
        // This implementation uses only operations that have CUDA support
        let x_dtype = xs.dtype();

        // Use F32 for internal calculations for stability
        let xs = match x_dtype {
            DType::F16 | DType::BF16 => xs.to_dtype(DType::F32)?,
            _ => xs.clone(),
        };

        // Get the last dimension size
        let hidden_size = xs.shape().dims()[xs.shape().rank() - 1] as f64;

        // Compute x^2
        let x_squared = xs.square()?;

        // Mean of x^2 along last dimension
        let mean = x_squared
            .sum_dim_keepdim(xs.shape().rank() - 1)?
            .mul_scalar(1.0 / hidden_size as f32)?;

        // Add epsilon and compute rsqrt
        let rsqrt_val = mean.add_scalar(self.eps as f32)?.sqrt()?;
        let one = Tensor::full(rsqrt_val.shape().clone(), 1.0, rsqrt_val.device().clone())?;
        let rsqrt = one.div(&rsqrt_val)?;

        // Normalize: x * rsqrt
        let normalized = xs.mul(&rsqrt)?;

        // Convert back to original dtype
        let normalized = match x_dtype {
            DType::F16 => normalized.to_dtype(DType::F16)?,
            DType::BF16 => normalized.to_dtype(DType::BF16)?,
            _ => normalized,
        };

        // Apply weight
        Ok(normalized.mul(&self.weight)?)
    }
}

/// Fast GPU-friendly RMS norm function that can be used as drop-in replacement
pub fn rms_norm(xs: &Tensor, weight: &Tensor, eps: f32) -> flame_core::Result<Tensor> {
    let norm = RmsNorm::new(weight.clone(), eps as f64);
    norm.forward(xs)
}
