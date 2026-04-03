use flame_core::device::Device;
use flame_core::{DType, Error, Result, Shape, Tensor};
use std::f32::consts::PI;
use std::sync::Arc;

// Rotary Position Embedding (RoPE) with CUDA acceleration
//
// Provides GPU-accelerated rotary embeddings for Flux attention mechanisms,
// supporting both 1D (text) and 2D (image) position encodings.

pub struct RotaryEmbedding {
    max_seq_len: usize,
    dim: usize,
    base: f32,
    cached_cos: Option<Tensor>,
    cached_sin: Option<Tensor>,
}

impl RotaryEmbedding {
    pub fn new(
        dim: usize,
        max_seq_len: usize,
        base: f32,
        device: &Device,
    ) -> flame_core::Result<Self> {
        // Ensure dim is even
        if dim % 2 != 0 {
            return Err(flame_core::Error::InvalidOperation(format!(
                "RoPE dimension must be even, got {}",
                dim
            )));
        }

        let mut rope = Self { max_seq_len, dim, base, cached_cos: None, cached_sin: None };

        // Always precompute cache on GPU
        rope.precompute_cache(device)?;

        Ok(rope)
    }

    fn precompute_cache(&mut self, device: &Device) -> flame_core::Result<()> {
        // Compute frequencies for each dimension pair
        let half_dim = self.dim / 2;
        let mut freqs = vec![0.0f32; half_dim];
        for i in 0..half_dim {
            freqs[i] = 1.0 / self.base.powf(2.0 * i as f32 / self.dim as f32);
        }

        // Precompute cos and sin values for all positions
        let mut cos_values = vec![0.0f32; self.max_seq_len * half_dim];
        let mut sin_values = vec![0.0f32; self.max_seq_len * half_dim];

        for pos in 0..self.max_seq_len {
            for i in 0..half_dim {
                let angle = pos as f32 * freqs[i];
                let idx = pos * half_dim + i;
                cos_values[idx] = angle.cos();
                sin_values[idx] = angle.sin();
            }
        }

        // Create tensors from computed values
        self.cached_cos = Some(Tensor::from_vec(
            cos_values,
            Shape::from_dims(&[self.max_seq_len, half_dim]),
            device.cuda_device().clone(),
        )?);
        self.cached_sin = Some(Tensor::from_vec(
            sin_values,
            Shape::from_dims(&[self.max_seq_len, half_dim]),
            device.cuda_device().clone(),
        )?);

        Ok(())
    }

    pub fn forward(
        &self,
        _x: &Tensor,
        _positions: &Tensor,
        _is_2d: bool,
    ) -> flame_core::Result<Tensor> {
        // Strict GPU-only invariant: no CPU fallback or simplified rotary.
        // A proper GPU kernel-backed RoPE is not wired here yet.
        Err(Error::InvalidOperation(
            "Unsupported path: RotaryEmbedding.forward requires a GPU kernel; no placeholder/CPU path allowed".to_string(),
        ))
    }

    /// Helper for Flux 2D positions
    pub fn get_2d_positions(
        height: usize,
        width: usize,
        device: &Device,
    ) -> flame_core::Result<Tensor> {
        let mut positions = vec![0u32; height * width * 2];

        for h in 0..height {
            for w in 0..width {
                let idx = h * width + w;
                positions[idx * 2] = h as u32;
                positions[idx * 2 + 1] = w as u32;
            }
        }

        let positions_f32: Vec<f32> = positions.iter().map(|&x| x as f32).collect();
        Ok(Tensor::from_vec(
            positions_f32,
            Shape::from_dims(&[1, height * width, 2]),
            device.cuda_device().clone(),
        )?)
    }

    /// Helper for 1D positions
    pub fn get_1d_positions(seq_len: usize, device: &Device) -> flame_core::Result<Tensor> {
        let positions: Vec<f32> = (0..seq_len as u32).map(|x| x as f32).collect();
        Ok(Tensor::from_vec(
            positions,
            Shape::from_dims(&[1, seq_len]),
            device.cuda_device().clone(),
        )?)
    }
}

/// Apply RoPE to attention queries and keys
pub fn apply_rotary_emb(
    q: &Tensor,
    k: &Tensor,
    positions: &Tensor,
    rope: &RotaryEmbedding,
    is_2d: bool,
) -> flame_core::Result<(Tensor, Tensor)> {
    // Do not attempt a fake rotation if kernel is unavailable.
    let q_rot = rope.forward(q, positions, is_2d)?;
    let k_rot = rope.forward(k, positions, is_2d)?;
    Ok((q_rot, k_rot))
}
