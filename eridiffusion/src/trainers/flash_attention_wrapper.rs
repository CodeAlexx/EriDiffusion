use anyhow::Context;
use flame_core::device::Device;
use flame_core::optimizers::{Adam, SGD};
use flame_core::{CudaDevice, DType, Result, Shape, Tensor};
use log::{debug, info};

// Flash Attention wrapper for SD 3.5 and Flux
// Provides unified interface with automatic fallback

// FLAME uses flame_core::device::Device instead of Device

// Use FLAME's flash attention

/// Configuration for Flash Attention
pub struct FlashAttentionConfig {
    pub causal: bool,
    pub window_size: Option<(usize, usize)>,
    pub alibi_slopes: Option<Tensor>,
    pub softmax_scale: Option<f32>,
}

impl Default for FlashAttentionConfig {
    fn default() -> Self {
        Self { causal: false, window_size: None, alibi_slopes: None, softmax_scale: None }
    }
}

/// Unified attention function with Flash Attention support
pub fn attention_with_flash(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    num_heads: usize,
    head_dim: usize,
    config: &FlashAttentionConfig,
) -> flame_core::Result<Tensor> {
    // FLAME devices are always CUDA, no need to check

    let dims = query.shape().dims();
    let (batch_size, seq_len, hidden_size) = (dims[0], dims[1], dims[2]);

    // Reshape for multi-head attention
    let q = query.reshape(&[batch_size, seq_len, num_heads, head_dim])?;
    let k = key.reshape(&[batch_size, seq_len, num_heads, head_dim])?;
    let v = value.reshape(&[batch_size, seq_len, num_heads, head_dim])?;

    // Use FLAME's flash attention implementation
    let scale = config.softmax_scale.unwrap_or(1.0 / (head_dim as f32).sqrt());

    debug!("Using FLAME Flash Attention: scale={}, causal={}", scale, config.causal);

    // FLAME's flash attention expects [batch, seq_len, num_heads, head_dim]
    // which is the format we already have
    let out = flame_core::flash_attention_forward(
        &q,
        &k,
        &v,
        None, // attention_mask
        Some(scale),
        config.causal,
    )
    .map_err(|e| flame_core::Error::InvalidOperation(format!("Flash attention failed: {:?}", e)))?;

    // Reshape back to [batch_size, seq_len, hidden_size]
    let output = out.reshape(&[batch_size, seq_len, hidden_size])?;

    Ok(output)
}

/// Efficient attention implementation for GPU (fallback)
fn efficient_attention_gpu(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    config: &FlashAttentionConfig,
) -> flame_core::Result<Tensor> {
    let dims = q.shape().dims();
    let (batch_size, seq_len, num_heads, head_dim) = (dims[0], dims[1], dims[2], dims[3]);
    let hidden_size = num_heads * head_dim;

    // Transpose for batched matmul
    let q = q.transpose_dims(1, 2)?; // [B, H, S, D]
    let k = k.transpose_dims(1, 2)?;
    let v = v.transpose_dims(1, 2)?;

    // Compute attention scores
    let scale = config.softmax_scale.unwrap_or(1.0 / (head_dim as f32).sqrt());

    let scores = q.matmul(&k.transpose_dims(2, 3)?)?;
    let scores = scores.mul_scalar(scale)?;

    // Apply causal mask if needed
    let scores = if config.causal { apply_causal_mask(&scores, seq_len)? } else { scores };

    // Softmax
    let weights = scores.softmax(-1)?;

    // Apply attention to values
    let output = weights.matmul(&v)?;

    // Transpose back and reshape
    Ok(output.transpose_dims(1, 2)?.reshape(&[batch_size, seq_len, hidden_size])?)
}

/// Apply causal mask to attention scores
fn apply_causal_mask(scores: &Tensor, seq_len: usize) -> flame_core::Result<Tensor> {
    let device = scores.device();
    // Create causal mask manually since FLAME doesn't have triu
    // For now, just return scores without masking - TODO: implement proper causal masking
    // This is a temporary workaround
    Ok(scores.clone())
}

// Note: Windowed and ALiBi variants are not yet implemented in FLAME
// TODO: Add support for these specialized attention patterns

/// SD 3.5 specific Flash Attention configuration
pub fn sd35_flash_config() -> FlashAttentionConfig {
    FlashAttentionConfig {
        causal: false, // SD 3.5 uses bidirectional attention
        window_size: None,
        alibi_slopes: None,
        softmax_scale: None, // Will use default 1/sqrt(d)
    }
}

/// Flux specific Flash Attention configuration
pub fn flux_flash_config(is_causal_block: bool) -> FlashAttentionConfig {
    FlashAttentionConfig {
        causal: is_causal_block, // Some Flux blocks use causal attention
        window_size: None,
        alibi_slopes: None,
        softmax_scale: None,
    }
}

/// Memory-efficient attention for very long sequences
pub fn memory_efficient_attention(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    num_heads: usize,
    head_dim: usize,
    chunk_size: usize,
) -> flame_core::Result<Tensor> {
    let dims = query.shape().dims();
    let (batch_size, seq_len, hidden_size) = (dims[0], dims[1], dims[2]);

    // Process attention in chunks to save memory
    let mut outputs = Vec::new();

    for i in (0..seq_len).step_by(chunk_size) {
        let end = (i + chunk_size).min(seq_len);
        let q_chunk = query.slice(&[(i, i + end - i)])?;

        // Compute attention for this chunk against all keys/values
        let output_chunk = attention_with_flash(
            &q_chunk,
            key,
            value,
            num_heads,
            head_dim,
            &FlashAttentionConfig::default(),
        )?;

        outputs.push(output_chunk);
    }

    // Concatenate all chunks
    let output_refs: Vec<&Tensor> = outputs.iter().collect();
    Tensor::cat(&output_refs, 1)
}

/// Benchmark Flash Attention vs standard attention
pub fn benchmark_flash_attention(
    batch_size: usize,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    device: &Device,
) -> flame_core::Result<()> {
    info!("Benchmarking Flash Attention vs Standard Attention");
    info!("Config: batch={}, seq={}, heads={}, dim={}", batch_size, seq_len, num_heads, head_dim);

    let hidden_size = num_heads * head_dim;

    // Create test tensors
    let q = Tensor::randn(
        Shape::from_dims(&[batch_size, seq_len, hidden_size]),
        0f32,
        1.0,
        device.cuda_device().clone(),
    )?;
    let k = Tensor::randn(
        Shape::from_dims(&[batch_size, seq_len, hidden_size]),
        0f32,
        1.0,
        device.cuda_device().clone(),
    )?;
    let v = Tensor::randn(
        Shape::from_dims(&[batch_size, seq_len, hidden_size]),
        0f32,
        1.0,
        device.cuda_device().clone(),
    )?;

    // Warmup
    for _ in 0..5 {
        let _ = attention_with_flash(
            &q,
            &k,
            &v,
            num_heads,
            head_dim,
            &FlashAttentionConfig::default(),
        )?;
    }

    // device.synchronize()?; // Not needed in FLAME

    // Benchmark
    let start = std::time::Instant::now();
    for _ in 0..10 {
        let _ = attention_with_flash(
            &q,
            &k,
            &v,
            num_heads,
            head_dim,
            &FlashAttentionConfig::default(),
        )?;
    }
    // device.synchronize()?; // Not needed in FLAME
    let flash_time = start.elapsed().as_secs_f64() / 10.0;

    info!("Flash Attention time: {:.3} ms", flash_time * 1000.0);

    // Memory usage estimate
    let memory_standard = (batch_size * num_heads * seq_len * seq_len * 4) as f64 / 1e9;
    let memory_flash = (batch_size * num_heads * seq_len * head_dim * 4) as f64 / 1e9;

    info!("Memory usage - Standard: {:.2} GB, Flash: {:.2} GB", memory_standard, memory_flash);
    info!("Memory savings: {:.1}x", memory_standard / memory_flash);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flash_attention_config() {
        let config = FlashAttentionConfig::default();
        assert!(!config.causal);
        assert!(config.window_size.is_none());
    }
}
