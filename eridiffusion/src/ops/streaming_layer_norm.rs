use flame_core::device::Device;
use flame_core::{DType, Error, Result, Shape, Tensor};
use std::collections::HashMap;

/// Streaming LayerNorm implementation for Flux model
/// Optimized for memory efficiency with cuDNN acceleration
pub struct StreamingLayerNorm {
    pub normalized_shape: Vec<usize>,
    pub eps: f32,
}

impl StreamingLayerNorm {
    pub fn new(normalized_shape: Vec<usize>, eps: f32) -> Self {
        Self { normalized_shape, eps }
    }

    /// Apply layer normalization with optional weight and bias
    /// Uses cuDNN when available for hardware acceleration
    pub fn forward(
        &self,
        input: &Tensor,
        weight: Option<&Tensor>,
        bias: Option<&Tensor>,
    ) -> Result<Tensor> {
        // Check if cuDNN is available and use it
        #[cfg(feature = "cudnn")]
        {
            use flame_core::cudnn::{cudnn_layer_norm, is_cudnn_norm_compatible};
            if is_cudnn_norm_compatible(input, "layer") {
                println!("🚀 Using cuDNN-accelerated LayerNorm for Flux streaming");
                return cudnn_layer_norm(
                    input,
                    &self.normalized_shape,
                    weight,
                    bias,
                    self.eps as f64,
                );
            }
        }

        // Otherwise use FLAME's optimized LayerNorm
        use flame_core::layer_norm::layer_norm;
        let result = layer_norm(input, &self.normalized_shape, weight, bias, self.eps)?;

        // 🔒 Assert layer norm output is finite and reasonable
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
                    eprintln!("❌ LayerNorm exploded: found NaN or Inf");
                    false
                } else if max_val.abs() > 1e6 || min_val.abs() > 1e6 {
                    eprintln!("❌ LayerNorm output too large: [{:.2e}, {:.2e}]", min_val, max_val);
                    false
                } else {
                    true
                }
            },
            "Layer normalization produced invalid output"
        );

        Ok(result)
    }
}

/// Apply LayerNorm to Flux double stream blocks
pub fn apply_double_stream_norm(
    img: &Tensor,
    txt: &Tensor,
    norm_weights: &HashMap<String, &Tensor>,
    eps: f32,
) -> Result<(Tensor, Tensor)> {
    // Get normalized shape from the last dimension
    let img_shape = img.shape().dims();
    let txt_shape = txt.shape().dims();
    let img_norm_shape = vec![img_shape[img_shape.len() - 1]];
    let txt_norm_shape = vec![txt_shape[txt_shape.len() - 1]];

    // Create LayerNorm instances
    let img_ln = StreamingLayerNorm::new(img_norm_shape, eps);
    let txt_ln = StreamingLayerNorm::new(txt_norm_shape, eps);

    // Apply normalization to img stream
    let img_normed = img_ln.forward(
        img,
        norm_weights.get("img_norm.weight").copied(),
        norm_weights.get("img_norm.bias").copied(),
    )?;

    // Apply normalization to txt stream
    let txt_normed = txt_ln.forward(
        txt,
        norm_weights.get("txt_norm.weight").copied(),
        norm_weights.get("txt_norm.bias").copied(),
    )?;

    Ok((img_normed, txt_normed))
}

/// Apply LayerNorm for single stream blocks
pub fn apply_single_stream_norm(
    x: &Tensor,
    norm_weights: &HashMap<String, &Tensor>,
    eps: f32,
) -> Result<Tensor> {
    // Get normalized shape from the last dimension
    let x_shape = x.shape().dims();
    let norm_shape = vec![x_shape[x_shape.len() - 1]];

    // Create LayerNorm instance
    let ln = StreamingLayerNorm::new(norm_shape, eps);

    // Apply normalization
    ln.forward(x, norm_weights.get("norm.weight").copied(), norm_weights.get("norm.bias").copied())
}

/// Memory-efficient streaming LayerNorm for very large sequences
/// Processes the sequence in chunks to reduce peak memory usage
pub fn streaming_layer_norm_chunked(
    input: &Tensor,
    normalized_shape: &[usize],
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    eps: f32,
    chunk_size: usize,
) -> Result<Tensor> {
    let input_shape = input.shape().dims();
    let batch_size = input_shape[0];
    let seq_len = input_shape[1];
    let hidden_size = input_shape[2];

    // If sequence is small enough, use regular LayerNorm
    if seq_len <= chunk_size {
        let ln = StreamingLayerNorm::new(normalized_shape.to_vec(), eps);
        return ln.forward(input, weight, bias);
    }

    // Process in chunks
    let mut output_chunks = Vec::new();

    for chunk_start in (0..seq_len).step_by(chunk_size) {
        let chunk_end = (chunk_start + chunk_size).min(seq_len);
        let chunk_len = chunk_end - chunk_start;

        // Extract chunk using narrow
        let chunk = input.narrow(1, chunk_start, chunk_len)?;

        // Apply LayerNorm to chunk
        let ln = StreamingLayerNorm::new(normalized_shape.to_vec(), eps);
        let chunk_normed = ln.forward(&chunk, weight, bias)?;

        output_chunks.push(chunk_normed);
    }

    // Concatenate chunks back together
    let chunk_refs: Vec<&Tensor> = output_chunks.iter().collect();
    Tensor::cat(&chunk_refs, 1)
}

/// Helper to extract normalization weights for a specific block
/// Returns references to avoid cloning tensors
pub fn extract_norm_weights<'a>(
    block_weights: &'a HashMap<String, Tensor>,
    block_prefix: &str,
) -> HashMap<String, &'a Tensor> {
    let mut norm_weights = HashMap::new();

    // Keys to look for in double stream blocks
    let double_keys = vec![
        "img_norm.weight",
        "img_norm.bias",
        "txt_norm.weight",
        "txt_norm.bias",
        "img_norm1.weight",
        "img_norm1.bias",
        "txt_norm1.weight",
        "txt_norm1.bias",
        "img_norm2.weight",
        "img_norm2.bias",
        "txt_norm2.weight",
        "txt_norm2.bias",
    ];

    // Keys for single stream blocks
    let single_keys = vec![
        "norm.weight",
        "norm.bias",
        "norm1.weight",
        "norm1.bias",
        "norm2.weight",
        "norm2.bias",
    ];

    // Check for both patterns
    for key in double_keys.iter().chain(single_keys.iter()) {
        let full_key = format!("{}.{}", block_prefix, key);
        if let Some(tensor) = block_weights.get(&full_key) {
            // Just store the reference
            norm_weights.insert(key.to_string(), tensor);
        }
    }

    norm_weights
}

#[cfg(all(test, feature = "dev-bins"))]
mod tests {
    use super::*;
    use flame_core::device::Device;

    #[test]
    fn test_streaming_layer_norm() -> Result<()> {
        let device = Device::cuda_if_available(0)?;

        // Create test input
        let batch_size = 2;
        let seq_len = 1024;
        let hidden_size = 3072;
        let input = Tensor::randn(
            Shape::from_dims(&[batch_size, seq_len, hidden_size]),
            DType::F32,
            device.cuda_device().clone(),
        )?;

        // Create LayerNorm
        let ln = StreamingLayerNorm::new(vec![hidden_size], 1e-6);

        // Test forward pass
        let output = ln.forward(&input, None, None)?;
        assert_eq!(output.shape(), input.shape());

        // Verify normalization worked
        let mean = output.mean_keepdim(2)?;
        let var = output.var_keepdim(2)?;

        // Mean should be close to 0
        let mean_abs = mean.abs()?.max(0)?.max(0)?;
        let mean_val: f32 = mean_abs.to_scalar()?;
        assert!(mean_val < 1e-5, "Mean not close to 0: {}", mean_val);

        // Variance should be close to 1
        let var_mean = var.mean(0)?.mean(0)?;
        let var_val: f32 = var_mean.to_scalar()?;
        assert!((var_val - 1.0).abs() < 0.1, "Variance not close to 1: {}", var_val);

        Ok(())
    }
}
