//! QK-Norm implementation for Flux attention mechanism
//!
//! This module implements Query-Key normalization as used in Flux models.
//! QK-Norm normalizes query and key vectors before computing attention scores
//! to prevent activation explosion.

use flame_core::{DType, Result, Shape, Tensor};

/// Apply QK normalization to query and key tensors
///
/// # Arguments
/// * `q` - Query tensor of shape [batch, seq_len, num_heads, head_dim]
/// * `k` - Key tensor of shape [batch, seq_len, num_heads, head_dim]
/// * `query_norm_scale` - Normalization scale for queries
/// * `key_norm_scale` - Normalization scale for keys
/// * `eps` - Small epsilon for numerical stability
pub fn apply_qk_norm(
    q: &Tensor,
    k: &Tensor,
    query_norm_scale: Option<&Tensor>,
    key_norm_scale: Option<&Tensor>,
    eps: f64,
) -> Result<(Tensor, Tensor)> {
    // Normalize queries
    let q_normed =
        if let Some(scale) = query_norm_scale { rms_norm_qk(q, scale, eps)? } else { q.clone() };

    // Normalize keys
    let k_normed =
        if let Some(scale) = key_norm_scale { rms_norm_qk(k, scale, eps)? } else { k.clone() };

    Ok((q_normed, k_normed))
}

/// RMS normalization specifically for QK vectors
/// This normalizes over the head dimension (last dimension)
fn rms_norm_qk(x: &Tensor, scale: &Tensor, eps: f64) -> Result<Tensor> {
    // Debug shape
    println!("🔍 rms_norm_qk - input shape: {:?}", x.shape().dims());
    println!("🔍 rms_norm_qk - scale shape: {:?}", scale.shape().dims());

    let x_dtype = x.dtype();

    // Convert to F32 for stability
    let x_f32 = match x_dtype {
        DType::F16 | DType::BF16 => x.to_dtype(DType::F32)?,
        _ => x.clone(),
    };

    // Get the head dimension (last dimension)
    let shape_dims = x_f32.shape().dims();
    let head_dim = shape_dims[shape_dims.len() - 1] as f64;
    println!("🔍 rms_norm_qk - computed head_dim from shape: {}", head_dim);

    // Compute RMS norm over head dimension
    // RMS = sqrt(mean(x^2))
    let x_squared = x_f32.square()?;
    let rank = x_squared.shape().rank();
    let sum_dim = rank - 1;
    println!(
        "🔍 rms_norm_qk - x_squared shape: {:?}, rank: {}, summing over dimension: {}",
        x_squared.shape().dims(),
        rank,
        sum_dim
    );

    // Workaround for FLAME's sum_dim_keepdim bug with 4D tensors
    // Instead of sum_dim_keepdim, use mean_dim which should work
    let mean = if rank == 4 {
        // For 4D tensors, manually compute mean over last dimension
        // This is a temporary workaround for the FLAME bug
        x_squared.mean_dim(&[sum_dim], true)?
    } else {
        // For other ranks, use the original approach
        x_squared.sum_dim_keepdim(sum_dim)?.mul_scalar(1.0 / head_dim as f32)?
    };

    // Add epsilon and take sqrt
    let rms = mean.add_scalar(eps as f32)?.sqrt()?;

    // Normalize: x / rms
    let normalized = x_f32.div(&rms)?;

    // Convert back to original dtype
    let normalized = match x_dtype {
        DType::F16 => normalized.to_dtype(DType::F16)?,
        DType::BF16 => normalized.to_dtype(DType::BF16)?,
        _ => normalized,
    };

    // Apply scale
    println!(
        "🔍 rms_norm_qk - normalized shape: {:?}, scale shape: {:?}",
        normalized.shape().dims(),
        scale.shape().dims()
    );

    let result = normalized.mul(scale)?;

    // 🔒 Assert norm output is finite and reasonable
    debug_assert!(
        {
            // Check a sample of values for efficiency
            let data = result.to_vec1::<f32>().unwrap_or_default();
            let sample_size = data.len().min(1000);
            let sample = &data[..sample_size];

            let all_finite = sample.iter().all(|&x| x.is_finite());
            let max_val = sample.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let min_val = sample.iter().fold(f32::INFINITY, |a, &b| a.min(b));

            if !all_finite {
                eprintln!("❌ QK Norm exploded: found NaN or Inf");
                false
            } else if max_val.abs() > 1e6 || min_val.abs() > 1e6 {
                eprintln!("❌ QK Norm output too large: [{:.2e}, {:.2e}]", min_val, max_val);
                false
            } else {
                true
            }
        },
        "QK normalization produced invalid output"
    );

    Ok(result)
}

/// Split QKV tensor into separate Q, K, V tensors
///
/// # Arguments
/// * `qkv` - Combined QKV tensor of shape [batch, seq_len, 3 * num_heads * head_dim]
/// * `num_heads` - Number of attention heads
/// * `head_dim` - Dimension per head
pub fn split_qkv(
    qkv: &Tensor,
    num_heads: usize,
    head_dim: usize,
) -> Result<(Tensor, Tensor, Tensor)> {
    let shape = qkv.shape().dims();
    let batch = shape[0];
    let seq_len = shape[1];
    let hidden_dim = num_heads * head_dim;

    // Instead of reshaping to 5D and using narrow, split directly using narrow on the last dimension
    // qkv is [batch, seq_len, 3 * hidden_dim]

    // Split into Q, K, V by slicing the last dimension
    let q = qkv.narrow(2, 0, hidden_dim)?.reshape(&[batch, seq_len, num_heads, head_dim])?;

    let k =
        qkv.narrow(2, hidden_dim, hidden_dim)?.reshape(&[batch, seq_len, num_heads, head_dim])?;

    let v = qkv
        .narrow(2, 2 * hidden_dim, hidden_dim)?
        .reshape(&[batch, seq_len, num_heads, head_dim])?;

    Ok((q, k, v))
}

/// Compute scaled dot-product attention with QK normalization
///
/// # Arguments
/// * `q` - Query tensor [batch, seq_len, num_heads, head_dim]
/// * `k` - Key tensor [batch, seq_len, num_heads, head_dim]
/// * `v` - Value tensor [batch, seq_len, num_heads, head_dim]
/// * `scale` - Scale factor (typically 1/sqrt(head_dim))
pub fn scaled_dot_product_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
) -> Result<Tensor> {
    let shape = q.shape().dims();
    let batch = shape[0];
    let seq_len = shape[1];
    let num_heads = shape[2];
    let head_dim = shape[3];

    // CRITICAL: Use Flash Attention for memory efficiency during training
    // This reduces memory usage by 50-70% compared to standard attention
    let use_flash_attention = true; // ALWAYS USE FLASH ATTENTION - WE HAVE CUDNN!

    if use_flash_attention {
        println!("⚡ Using Flash Attention for memory-efficient computation");

        // Flash attention expects [batch, num_heads, seq_len, head_dim]
        let q_flash = q.transpose_dims(1, 2)?;
        let k_flash = k.transpose_dims(1, 2)?;
        let v_flash = v.transpose_dims(1, 2)?;

        // Use Flash Attention for memory-efficient computation
        let output = flame_core::flash_attention_forward(
            &q_flash,
            &k_flash,
            &v_flash,
            None, // No attention mask
            Some(scale),
            false, // Not causal
        )?;

        // Transpose back: [batch, seq_len, num_heads, head_dim]
        let output = output.transpose_dims(1, 2)?;

        // Reshape to [batch, seq_len, num_heads * head_dim]
        output.reshape(&[batch, seq_len, num_heads * head_dim])
    } else {
        // Fallback to standard attention (for debugging only)
        // Transpose for batched matmul: [batch, num_heads, seq_len, head_dim]
        let q = q.transpose_dims(1, 2)?;
        let k = k.transpose_dims(1, 2)?;
        let v = v.transpose_dims(1, 2)?;

        // Compute attention scores: Q @ K^T
        // [batch, num_heads, seq_len, head_dim] @ [batch, num_heads, head_dim, seq_len]
        let k_t = k.transpose_dims(2, 3)?;
        let scores = q.matmul(&k_t)?;

        // Scale scores
        let scores = scores.mul_scalar(scale)?;

        // Apply softmax along last dimension
        let attn_weights = scores.softmax((scores.shape().rank() - 1) as isize)?;

        // Apply attention weights to values
        // [batch, num_heads, seq_len, seq_len] @ [batch, num_heads, seq_len, head_dim]
        let output = attn_weights.matmul(&v)?;

        // Transpose back: [batch, seq_len, num_heads, head_dim]
        let output = output.transpose_dims(1, 2)?;

        // Reshape to [batch, seq_len, num_heads * head_dim]
        output.reshape(&[batch, seq_len, num_heads * head_dim])
    }
}

#[cfg(all(test, feature = "dev-bins"))]
mod tests {
    use super::*;

    // Tests are disabled for now due to Device API changes
    // TODO: Re-enable when we figure out the correct Device construction
}
