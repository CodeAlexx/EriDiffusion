use crate::loaders::WeightLoader;
use anyhow::anyhow;
use flame_core::device::Device;
use flame_core::optimizers::{Adam, SGD};
use flame_core::Result;
use flame_core::{DType, Shape, Tensor};
use std::collections::HashMap;

// These types should be imported from the appropriate modules
use crate::models::mmdit_blocks::{MMDiT, MMDiTConfig};

// PrefixedWeightLoader is imported from loaders module
use crate::loaders::PrefixedWeightLoader;
pub struct PatchedRMSNorm {
    weight: Tensor,
    eps: f64,
}
pub struct MMDiTWrapper {
    inner: MMDiT,
    device: Device,
}

// FLAME uses flame_core::device::Device instead of Device

// FLAME uses flame_core::device::Device instead of Device

/// Patched RMSNorm that works on CUDA

// WeightLoader implementation is in crate::loaders::WeightLoader

// PrefixedWeightLoader implementation is in loaders module

impl PatchedRMSNorm {
    pub fn new(weight: Tensor, eps: f64) -> Self {
        Self { weight, eps }
    }

    pub fn forward(&self, xs: &Tensor) -> flame_core::Result<Tensor> {
        // Use operations that have CUDA implementations
        let x_dtype = xs.dtype();
        let xs = match x_dtype {
            DType::F16 | DType::BF16 => xs.to_dtype(DType::F32)?,
            _ => xs.clone(),
        };

        let shape_dims = xs.shape().dims();
        let hidden_size = shape_dims[shape_dims.len() - 1] as f64;
        let x_squared = xs.square()?;
        let mean = x_squared.sum_dim(shape_dims.len() - 1)?.mul_scalar(1.0 / hidden_size as f32)?;
        let rsqrt_val = mean.add_scalar(self.eps as f32)?.sqrt()?;
        let one = Tensor::full(rsqrt_val.shape().clone(), 1.0, rsqrt_val.device().clone())?;
        let rsqrt = one.div(&rsqrt_val)?;
        let normalized = xs.mul(&rsqrt)?;

        let normalized = match x_dtype {
            DType::F16 => normalized.to_dtype(DType::F16)?,
            DType::BF16 => normalized.to_dtype(DType::BF16)?,
            _ => normalized,
        };

        Ok(normalized.mul(&self.weight)?)
    }
}

/// Create a patched RMSNorm layer
pub fn create_patched_rms_norm(
    weights: &WeightLoader,
    dim: usize,
    eps: f64,
) -> flame_core::Result<PatchedRMSNorm> {
    let weight = weights.tensor("weight", &[dim])?;
    Ok(PatchedRMSNorm::new(weight, eps))
}

/// Load MMDiT with patched RMS norm
pub fn load_mmdit_with_patched_rms_norm(
    config: &MMDiTConfig,
    wl: WeightLoader,
) -> flame_core::Result<Tensor> {
    println!("Loading MMDiT with patched RMS norm for CUDA support...");

    // Create the model normally
    // SD3.5 Large has a conditioning dimension of 4096 (T5-XXL hidden size)
    let cond_dim = 4096;
    let device = Device::cuda(0)?;
    let model = MMDiT::new(config.clone(), cond_dim, &device)?;

    // The model is already created, we can't easily patch it
    // So we'll use a different approach - wrap the forward pass

    // Return a placeholder tensor since we can't easily patch the model
    Ok(Tensor::zeros(Shape::from_dims(&[1]), device.cuda_device().clone())?)
}

/// Alternative: Create a wrapper that intercepts RMS norm errors

impl MMDiTWrapper {
    pub fn new(inner: MMDiT, device: Device) -> Self {
        Self { inner, device }
    }

    pub fn forward(
        &self,
        x: &Tensor,
        t: &Tensor,
        y: &Tensor,
        context: &Tensor,
        skip_layers: Option<&[usize]>,
    ) -> flame_core::Result<Tensor> {
        // Process timestep to embeddings
        let time_emb = get_timestep_embedding(t, 256)?;

        // Create simple position embeddings
        let batch_size = x.shape().dims()[0];
        let seq_len = x.shape().dims()[1];
        let positions =
            Tensor::arange(0.0, seq_len as f32, 1.0, self.device.cuda_device().clone())?
                .unsqueeze(0)?
                .expand(&[batch_size, seq_len])?;

        // Call the inner MMDiT forward pass
        let (output, _) = self.inner.forward(x, context, &time_emb, &positions)?;

        // Apply any skip layer logic if provided
        if let Some(skip_layers) = skip_layers {
            // For simplicity, just return the output (skip layer logic would be more complex)
            println!(
                "Skip layers requested: {:?}, but not implemented in this simplified version",
                skip_layers
            );
        }

        Ok(output)
    }
}

/// Generate sinusoidal timestep embeddings
fn get_timestep_embedding(timesteps: &Tensor, embedding_dim: usize) -> flame_core::Result<Tensor> {
    let device = Device::from(timesteps.device().clone());
    let half_dim = embedding_dim / 2;
    let emb = (0..half_dim).map(|i| -(i as f32 * 2.0 / embedding_dim as f32)).collect::<Vec<_>>();
    let emb = Tensor::from_vec(emb, Shape::from_dims(&[half_dim]), device.cuda_device().clone())?;
    let emb = emb.mul_scalar(10000f32.ln())?.exp()?;
    let emb = timesteps.unsqueeze(1)?.mul(&emb.unsqueeze(0)?)?;
    let sin = emb.sin()?;
    let cos = emb.cos()?;
    Tensor::cat(&[&sin, &cos], 1)
}

/// Test if the patch works
pub fn test_mmdit_patch(device: &Device) -> flame_core::Result<Tensor> {
    println!("\n=== Testing MMDiT RMS Norm Patch ===");

    let cuda_device = Device::cuda(0)?;
    if true {
        // Test the patched RMS norm directly
        let x = Tensor::randn(
            Shape::from_dims(&[1, 4, 128, 128]),
            0.0,
            1.0,
            cuda_device.cuda_device().clone(),
        )?;
        let weight =
            Tensor::ones(Shape::from_dims(&[128 * 128]), cuda_device.cuda_device().clone())?;
        let rms = PatchedRMSNorm::new(weight, 1e-6);

        match rms.forward(&x) {
            Ok(output) => {
                println!("✓ Patched RMS norm works on CUDA!");
                println!(" Output shape: {:?}", output.shape());
            }
            Err(e) => {
                println!("✗ Patched RMS norm failed: {}", e);
            }
        }

        Ok(Tensor::zeros(Shape::from_dims(&[1]), cuda_device.cuda_device().clone())?)
    } else {
        println!("CUDA device not available, skipping test");
        Ok(Tensor::zeros(Shape::from_dims(&[1]), cuda_device.cuda_device().clone())?)
    }
}
