use flame_core::device::Device;
use flame_core::optimizers::{Adam, SGD};
use flame_core::{DType, Result, Shape, Tensor};

// Efficient attention that actually works on 24GB

/// Memory efficient attention that processes in slices
pub fn efficient_attention(
    q: &Tensor, // [batch * num_heads, seq_len, head_dim]
    k: &Tensor,
    v: &Tensor,
    head_dim: usize,
) -> flame_core::Result<Tensor> {
    let dims = q.shape().dims();
    let (batch_heads, seq_len, _) = (dims[0], dims[1], dims[2]);
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    // Process attention in slices to avoid OOM
    let slice_size = 64; // Process 64 queries at a time
    let num_slices = (seq_len + slice_size - 1) / slice_size;

    let mut outputs = Vec::new();

    for i in 0..num_slices {
        let start = i * slice_size;
        let end = ((i + 1) * slice_size).min(seq_len);
        let slice_len = end - start;

        // Get query slice
        let q_slice = q.slice(&[(start, start + slice_len)])?;

        // Compute attention for this slice
        // Transpose k from [batch, seq_len, heads, head_dim] to [batch, heads, head_dim, seq_len]
        let k_t = k.transpose_dims(k.shape().rank() - 2, k.shape().rank() - 1)?;
        let scores = q_slice.matmul(&k_t)?;
        let scores = scores.mul_scalar(scale)?;
        let weights = scores.softmax((scores.shape().rank() - 1) as isize)?;
        let output_slice = weights.matmul(v)?;

        outputs.push(output_slice);
    }

    // Concatenate all slices
    let output_refs: Vec<&Tensor> = outputs.iter().collect();
    Ok(Tensor::cat(&output_refs, 1)?)
}
