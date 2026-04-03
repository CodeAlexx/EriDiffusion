//! Streaming RMSNorm implementation for Flux model
//!
//! This module provides memory-efficient RMSNorm operations for the streaming
//! Flux model, which uses RMSNorm instead of LayerNorm.

use flame_core::{DType, Device, Result, Tensor};
use std::collections::HashMap;

// Remove cudnn check for now since we're using FLAME's RMSNorm workaround

/// Streaming RMSNorm implementation
pub struct StreamingRMSNorm {
    pub normalized_shape: Vec<usize>,
    pub eps: f64,
}

impl StreamingRMSNorm {
    pub fn new(normalized_shape: Vec<usize>, eps: f64) -> Self {
        Self { normalized_shape, eps }
    }

    /// Forward pass using the RMSNorm workaround from rms_norm_fix.rs
    pub fn forward(&self, input: &Tensor, weight: Option<&Tensor>) -> Result<Tensor> {
        // Use the workaround implementation that's known to work
        let x_dtype = input.dtype();

        // Use F32 for internal calculations for stability
        let x_f32 = match x_dtype {
            DType::F16 | DType::BF16 => input.to_dtype(DType::F32)?,
            _ => input.clone(),
        };

        let hidden_size = x_f32.shape().dims()[x_f32.shape().rank() - 1] as f64;

        // All these operations have CUDA implementations
        let x_squared = x_f32.square()?;
        let mean = x_squared
            .sum_dim_keepdim(x_squared.shape().rank() - 1)?
            .mul_scalar(1.0 / hidden_size as f32)?;
        let rsqrt_val = mean.add_scalar(self.eps as f32)?;
        let rsqrt_val = rsqrt_val.sqrt()?;
        let one = Tensor::full(rsqrt_val.shape().clone(), 1.0, rsqrt_val.device().clone())?;
        let rsqrt = one.div(&rsqrt_val)?;
        let normalized = x_f32.mul(&rsqrt)?;

        // Convert back to original dtype
        let normalized = match x_dtype {
            DType::F16 => normalized.to_dtype(DType::F16)?,
            DType::BF16 => normalized.to_dtype(DType::BF16)?,
            _ => normalized,
        };

        // Apply weight if provided
        if let Some(w) = weight {
            normalized.mul(w)
        } else {
            Ok(normalized)
        }
    }
}

/// Apply RMSNorm to double stream blocks (img and txt)
pub fn apply_double_stream_rms_norm(
    img: &Tensor,
    txt: &Tensor,
    norm_weights: &HashMap<String, &Tensor>,
    eps: f64,
) -> Result<(Tensor, Tensor)> {
    println!("🔄 Applying streaming RMSNorm to double stream block");

    // Apply RMSNorm to img stream
    let img_normed = if let Some(&weight) = norm_weights.get("img_norm.weight") {
        let norm = StreamingRMSNorm::new(vec![img.shape().dims()[img.shape().rank() - 1]], eps);
        norm.forward(img, Some(weight))?
    } else {
        img.clone()
    };

    // Apply RMSNorm to txt stream
    let txt_normed = if let Some(&weight) = norm_weights.get("txt_norm.weight") {
        let norm = StreamingRMSNorm::new(vec![txt.shape().dims()[txt.shape().rank() - 1]], eps);
        norm.forward(txt, Some(weight))?
    } else {
        txt.clone()
    };

    Ok((img_normed, txt_normed))
}

/// Apply RMSNorm to single stream blocks
pub fn apply_single_stream_rms_norm(
    x: &Tensor,
    norm_weights: &HashMap<String, &Tensor>,
    eps: f64,
) -> Result<Tensor> {
    println!("🔄 Applying streaming RMSNorm to single stream block");

    if let Some(&weight) = norm_weights.get("norm.weight") {
        let norm = StreamingRMSNorm::new(vec![x.shape().dims()[x.shape().rank() - 1]], eps);
        norm.forward(x, Some(weight))
    } else {
        Ok(x.clone())
    }
}

/// Extract RMSNorm weights from block weights
pub fn extract_rms_norm_weights<'a>(
    weights: &'a HashMap<String, Tensor>,
    block_name: &str,
) -> HashMap<String, &'a Tensor> {
    let mut norm_weights = HashMap::new();

    // For double blocks, look for img_norm and txt_norm
    if block_name.contains("double_blocks") {
        // Check for various possible naming patterns
        let patterns = [
            "img_norm.weight",
            "txt_norm.weight",
            "img_norm1.weight",
            "txt_norm1.weight",
            "img_norm2.weight",
            "txt_norm2.weight",
            // QK-Norm weights for attention
            "img_attn.norm.query_norm.scale",
            "img_attn.norm.key_norm.scale",
            "txt_attn.norm.query_norm.scale",
            "txt_attn.norm.key_norm.scale",
        ];

        for pattern in &patterns {
            if let Some(weight) = weights.get(*pattern) {
                norm_weights.insert(pattern.to_string(), weight);
            }
        }
    }

    // For single blocks, look for norm weights
    if block_name.contains("single_blocks") {
        let patterns = [
            "norm.weight",
            "norm1.weight",
            "norm2.weight",
            // QK-Norm weights for attention
            "norm.query_norm.scale",
            "norm.key_norm.scale",
        ];

        for pattern in &patterns {
            if let Some(weight) = weights.get(*pattern) {
                norm_weights.insert(pattern.to_string(), weight);
            }
        }
    }

    if !norm_weights.is_empty() {
        println!("📊 Found {} RMSNorm weights for {}", norm_weights.len(), block_name);
        for (key, _) in &norm_weights {
            println!("  - {}", key);
        }
    }

    norm_weights
}

#[cfg(all(test, feature = "dev-bins"))]
mod tests {
    use super::*;
    use flame_core::{CudaDevice, Device, Shape};

    #[test]
    fn test_streaming_rms_norm() -> Result<()> {
        let device = Device::Cpu;
        let x = Tensor::randn(Shape::from_dims(&[2, 4, 768]), 0.0, 1.0, device)?;
        let weight = Tensor::ones(Shape::from_dims(&[768]), device)?;

        let norm = StreamingRMSNorm::new(vec![768], 1e-6);
        let output = norm.forward(&x, Some(&weight))?;

        assert_eq!(output.shape().dims(), &[2, 4, 768]);
        println!("✅ RMSNorm test passed!");

        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_streaming_rms_norm_cuda() -> Result<()> {
        if let Ok(device) = CudaDevice::new(0) {
            let x = Tensor::randn(Shape::from_dims(&[2, 4, 768]), 0.0, 1.0, device.clone())?;
            let weight = Tensor::ones(Shape::from_dims(&[768]), device.clone())?;

            let norm = StreamingRMSNorm::new(vec![768], 1e-6);
            let output = norm.forward(&x, Some(&weight))?;

            assert_eq!(output.shape().dims(), &[2, 4, 768]);
            println!("✅ CUDA RMSNorm test passed!");
        }

        Ok(())
    }
}
