//! Fixed streaming RMSNorm implementation for Flux model
//!
//! This module provides memory-efficient RMSNorm operations for the streaming
//! Flux model, which uses RMSNorm instead of LayerNorm.
//!
//! FIXED: Properly handles QK-Norm weight extraction with block prefixes

use flame_core::{DType, Device, Result, Tensor};
use std::collections::HashMap;

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

/// Extract RMSNorm weights from block weights with proper handling of block prefixes
pub fn extract_rms_norm_weights<'a>(
    weights: &'a HashMap<String, Tensor>,
    block_name: &str,
) -> HashMap<String, &'a Tensor> {
    let mut norm_weights = HashMap::new();

    // For double blocks, look for img_norm and txt_norm
    if block_name.contains("double_blocks") {
        // Extract block number if present
        let block_prefix = if block_name.contains('.') {
            // e.g., "double_blocks.0" -> we need to check for "double_blocks.0.img_attn.norm..."
            format!("{}.", block_name)
        } else {
            String::new()
        };

        // Check for various possible naming patterns WITH the block prefix
        let patterns = [
            // Block-level normalization
            format!("{}img_norm.weight", block_prefix),
            format!("{}txt_norm.weight", block_prefix),
            format!("{}img_norm1.weight", block_prefix),
            format!("{}txt_norm1.weight", block_prefix),
            format!("{}img_norm2.weight", block_prefix),
            format!("{}txt_norm2.weight", block_prefix),
            // QK-Norm weights for attention - THESE ARE THE CRITICAL ONES!
            format!("{}img_attn.norm.query_norm.scale", block_prefix),
            format!("{}img_attn.norm.key_norm.scale", block_prefix),
            format!("{}txt_attn.norm.query_norm.scale", block_prefix),
            format!("{}txt_attn.norm.key_norm.scale", block_prefix),
        ];

        for pattern in &patterns {
            if let Some(weight) = weights.get(pattern) {
                // Store WITHOUT the block prefix for compatibility with forward_with_norm
                let key_without_prefix = pattern.strip_prefix(&block_prefix).unwrap_or(pattern);
                norm_weights.insert(key_without_prefix.to_string(), weight);
                println!("  ✓ Found QK-Norm weight: {} -> {}", pattern, key_without_prefix);
            }
        }
    }

    // For single blocks, look for norm weights
    if block_name.contains("single_blocks") {
        let block_prefix =
            if block_name.contains('.') { format!("{}.", block_name) } else { String::new() };

        let patterns = [
            format!("{}norm.weight", block_prefix),
            format!("{}norm1.weight", block_prefix),
            format!("{}norm2.weight", block_prefix),
            // QK-Norm weights for attention in single blocks
            format!("{}norm.query_norm.scale", block_prefix),
            format!("{}norm.key_norm.scale", block_prefix),
        ];

        for pattern in &patterns {
            if let Some(weight) = weights.get(pattern) {
                let key_without_prefix = pattern.strip_prefix(&block_prefix).unwrap_or(pattern);
                norm_weights.insert(key_without_prefix.to_string(), weight);
                println!("  ✓ Found norm weight: {} -> {}", pattern, key_without_prefix);
            }
        }
    }

    if !norm_weights.is_empty() {
        println!("📊 Found {} RMSNorm weights for {}", norm_weights.len(), block_name);
        for (key, tensor) in &norm_weights {
            let shape = tensor.shape().dims();
            println!("  - {} (shape: {:?})", key, shape);
        }
    } else {
        println!(
            "⚠️  No RMSNorm weights found for {} (checked {} tensors)",
            block_name,
            weights.len()
        );
        // Debug: show what weights ARE available
        println!("  Available weights:");
        for (key, tensor) in weights.iter().take(10) {
            println!("    - {} (shape: {:?})", key, tensor.shape().dims());
        }
        if weights.len() > 10 {
            println!("    ... and {} more", weights.len() - 10);
        }
    }

    norm_weights
}

#[cfg(all(test, feature = "dev-bins"))]
mod tests {
    use super::*;
    use flame_core::{CudaDevice, Device, Shape};

    #[test]
    fn test_qk_norm_extraction() -> Result<()> {
        let device = Device::Cpu;

        // Simulate loaded weights for double_blocks.0
        let mut weights = HashMap::new();
        weights.insert(
            "double_blocks.0.img_attn.norm.query_norm.scale".to_string(),
            Tensor::ones(Shape::from_dims(&[64]), device.clone())?,
        );
        weights.insert(
            "double_blocks.0.img_attn.norm.key_norm.scale".to_string(),
            Tensor::ones(Shape::from_dims(&[64]), device.clone())?,
        );

        // Extract with block name
        let norm_weights = extract_rms_norm_weights(&weights, "double_blocks.0");

        // Should find the QK-Norm weights and strip the prefix
        assert!(norm_weights.contains_key("img_attn.norm.query_norm.scale"));
        assert!(norm_weights.contains_key("img_attn.norm.key_norm.scale"));
        assert_eq!(norm_weights.len(), 2);

        println!("✅ QK-Norm extraction test passed!");

        Ok(())
    }
}
