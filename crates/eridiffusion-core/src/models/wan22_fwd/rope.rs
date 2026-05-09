//! Pure-GPU 3-axis RoPE for Wan 2.2.
//!
//! Verbatim port of `wan-trainer/src/forward_impl/rope.rs` — no EDv2-specific
//! changes (the archive code only depends on `flame_core`).
//!
//! ## Axis split for head_dim=128
//!
//! From `wan/modules/model.py::rope_params`:
//!
//! ```text
//! d6  = head_dim / 6 = 21
//! ax0 = head_dim - 4 * d6 = 44   (temporal)
//! ax1 = 2 * d6          = 42     (height)
//! ax2 = 2 * d6          = 42     (width)
//! ```
//!
//! Within each axis the `ax_dim` channels are paired as `(2i, 2i+1)`
//! = `(re, im)` and rotated as a 2-D complex number — the "interleaved"
//! (complex) RoPE convention that `flame_core::bf16_ops::rope_fused_bf16`
//! implements.
//!
//! ## Fused kernel shape contract
//!
//! `rope_fused_bf16` expects:
//!   - x:   `[B, H, N, D]` BF16 (interleaved pairs)
//!   - cos: `[1, 1, N, D/2]` BF16
//!   - sin: `[1, 1, N, D/2]` BF16

use flame_core::{bf16_ops::rope_fused_bf16, DType, Error, Result, Shape, Tensor};
use std::sync::Arc;

/// Precomputed cos/sin tables for one `(F, H, W)` grid.
#[derive(Clone)]
pub struct WanRope {
    /// Cos table shaped `[1, 1, seq, head_dim/2]` BF16.
    pub cos: Tensor,
    /// Sin table shaped `[1, 1, seq, head_dim/2]` BF16.
    pub sin: Tensor,
    /// `head_dim / 2` — the paired-dim count.
    pub half_dim: usize,
}

impl WanRope {
    pub fn new(
        grid: (usize, usize, usize),
        head_dim: usize,
        theta: f64,
        device: Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        if head_dim % 2 != 0 {
            return Err(Error::InvalidInput(format!(
                "WanRope: head_dim must be even, got {head_dim}"
            )));
        }
        let d6 = head_dim / 6;
        let ax0 = head_dim - 4 * d6; // temporal
        let ax1 = 2 * d6; // height
        let ax2 = 2 * d6; // width
        if ax0 + ax1 + ax2 != head_dim {
            return Err(Error::InvalidInput(format!(
                "WanRope: axis split {ax0}+{ax1}+{ax2} != head_dim {head_dim}"
            )));
        }
        if ax0 % 2 != 0 || ax1 % 2 != 0 || ax2 % 2 != 0 {
            return Err(Error::InvalidInput(format!(
                "WanRope: every axis must be even (pairs), got [{ax0}, {ax1}, {ax2}]"
            )));
        }
        let half_ax0 = ax0 / 2;
        let half_ax1 = ax1 / 2;
        let half_ax2 = ax2 / 2;
        let half_dim = half_ax0 + half_ax1 + half_ax2;

        let (grid_f, grid_h, grid_w) = grid;
        let seq = grid_f * grid_h * grid_w;

        // Precompute 1-D freq tables per axis once — same formula as
        // `Wan22Dit::load_with_config`:
        //   freq[i] = 1 / theta^(2 * i / axis_dim)
        let freqs_for_axis = |axis_dim: usize| -> Vec<f64> {
            let half = axis_dim / 2;
            (0..half)
                .map(|i| 1.0 / theta.powf(2.0 * i as f64 / axis_dim as f64))
                .collect()
        };
        let fq0 = freqs_for_axis(ax0);
        let fq1 = freqs_for_axis(ax1);
        let fq2 = freqs_for_axis(ax2);

        // Build [seq, half_dim] cos / sin tables on host, upload once.
        let mut cos_host = vec![0.0f32; seq * half_dim];
        let mut sin_host = vec![0.0f32; seq * half_dim];

        for fi in 0..grid_f {
            for hi in 0..grid_h {
                for wi in 0..grid_w {
                    let si = fi * grid_h * grid_w + hi * grid_w + wi;
                    let row = si * half_dim;

                    for i in 0..half_ax0 {
                        let angle = fi as f64 * fq0[i];
                        cos_host[row + i] = angle.cos() as f32;
                        sin_host[row + i] = angle.sin() as f32;
                    }
                    for i in 0..half_ax1 {
                        let angle = hi as f64 * fq1[i];
                        cos_host[row + half_ax0 + i] = angle.cos() as f32;
                        sin_host[row + half_ax0 + i] = angle.sin() as f32;
                    }
                    for i in 0..half_ax2 {
                        let angle = wi as f64 * fq2[i];
                        cos_host[row + half_ax0 + half_ax1 + i] = angle.cos() as f32;
                        sin_host[row + half_ax0 + half_ax1 + i] = angle.sin() as f32;
                    }
                }
            }
        }

        let cos = Tensor::from_vec(
            cos_host,
            Shape::from_dims(&[1, 1, seq, half_dim]),
            device.clone(),
        )?
        .to_dtype(DType::BF16)?;
        let sin = Tensor::from_vec(
            sin_host,
            Shape::from_dims(&[1, 1, seq, half_dim]),
            device,
        )?
        .to_dtype(DType::BF16)?;

        Ok(Self { cos, sin, half_dim })
    }

    /// Apply RoPE to an already-permuted `[B, H, N, D]` BF16 tensor.
    ///
    /// `flame_core::bf16_ops::rope_fused_bf16` itself records
    /// `Op::RoPePrecomputed` when its input has `requires_grad`, so this
    /// is a thin shape-check wrapper. (2026-05-09 audit M3: a previous
    /// version manually re-recorded the op here, which both
    /// double-counted on the tape and saved with the wrong cos/sin
    /// reshaped IDs. Klein and chroma both use rope_fused_bf16 directly
    /// without a wrapper for the same reason.)
    pub fn apply(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.shape().dims();
        if dims.len() != 4 {
            return Err(Error::InvalidInput(format!(
                "WanRope::apply expects [B, H, N, D], got {dims:?}"
            )));
        }
        let hd = dims[3];
        if hd != self.half_dim * 2 {
            return Err(Error::InvalidInput(format!(
                "WanRope::apply head_dim mismatch: tensor has {hd}, rope was built for {}",
                self.half_dim * 2
            )));
        }
        rope_fused_bf16(x, &self.cos, &self.sin)
    }
}
