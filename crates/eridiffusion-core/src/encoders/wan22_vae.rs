//! Wan2.2 high-compression VAE — encoder.
//!
//! Port of `wan/modules/vae2_2.py::Wan2_2_VAE` (Apache-2.0), encode path.
//!
//! ## High-level architecture
//! - Input: `[B, 3, T, H, W]` RGB video in `[-1, 1]`.
//! - `patchify(patch_size=2)`: `[B, 3, T, H, W]` → `[B, 12, T, H/2, W/2]`.
//! - `Encoder3d`:
//!     - `conv1`: CausalConv3d(12→dims[0]=160, 3×3×3, pad=1).
//!     - 4 `Down_ResidualBlock` groups:
//!         dims = `[160, 160, 320, 640, 640]` (`[dim*u for u in [1]+dim_mult]`).
//!         Each: `num_res_blocks=2` `ResidualBlock`s + optional `Resample`
//!         (downsample2d or downsample3d) + `AvgDown3D` shortcut.
//!         `temperal_downsample = [False, True, True]` — first group spatial
//!         only, groups 1 and 2 temporal+spatial, group 3 no downsample.
//!     - middle: ResBlock(640) → AttnBlock(640) → ResBlock(640).
//!     - head: `RMS_norm(640) + SiLU + CausalConv3d(640→96, 3×3×3, pad=1)`.
//! - `conv1` (top-level): CausalConv3d(96→96, 1×1×1) → chunk → mu (48ch).
//! - Normalize: `z = (mu - mean) / std` per-channel.
//! - Output: `[B, 48, T', H', W']` normalized latents.
//!
//! ## Weight key format
//! Keys use Python module hierarchy:
//!   `encoder.conv1.weight/bias`
//!   `encoder.downsamples.{i}.downsamples.{j}.residual.{0,3}.gamma`
//!   `encoder.downsamples.{i}.downsamples.{j}.residual.{2,6}.weight/bias`
//!   `encoder.downsamples.{i}.downsamples.{j}.shortcut.weight/bias`
//!   `encoder.downsamples.{i}.downsamples.{N}.resample.{0,1}.*`
//!   `encoder.downsamples.{i}.downsamples.{N}.time_conv.weight/bias`
//!   `encoder.middle.{0,2}.residual.{...}`
//!   `encoder.middle.1.norm.gamma, to_qkv.weight/bias, proj.weight/bias`
//!   `encoder.head.{0}.gamma, encoder.head.{2}.weight/bias`
//!   `conv1.weight/bias` (top-level 1×1 conv)

use flame_core::conv::Conv2d;
use flame_core::conv3d_bf16::Conv3dBF16;
use flame_core::sdpa::forward as sdpa_forward;
use flame_core::serialization::load_file;
use flame_core::{DType, Error, Result, Shape, Tensor};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

type Weights = HashMap<String, Tensor>;

// ---------------------------------------------------------------------------
// Normalisation constants — same as decoder, from `Wan2_2_VAE.__init__`.
// ---------------------------------------------------------------------------

#[rustfmt::skip]
const MEAN: [f32; 48] = [
    -0.2289, -0.0052, -0.1323, -0.2339, -0.2799,  0.0174,  0.1838,  0.1557,
    -0.1382,  0.0542,  0.2813,  0.0891,  0.1570, -0.0098,  0.0375, -0.1825,
    -0.2246, -0.1207, -0.0698,  0.5109,  0.2665, -0.2108, -0.2158,  0.2502,
    -0.2055, -0.0322,  0.1109,  0.1567, -0.0729,  0.0899, -0.2799, -0.1230,
    -0.0313, -0.1649,  0.0117,  0.0723, -0.2839, -0.2083, -0.0520,  0.3748,
     0.0152,  0.1957,  0.1433, -0.2944,  0.3573, -0.0548, -0.1681, -0.0667,
];

#[rustfmt::skip]
const STD: [f32; 48] = [
    0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.4990, 0.4818, 0.5013,
    0.8158, 1.0344, 0.5894, 1.0901, 0.6885, 0.6165, 0.8454, 0.4978,
    0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
    0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093,
    0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887,
    0.3971, 1.0600, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744,
];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn get(weights: &Weights, key: &str) -> Result<Tensor> {
    weights
        .get(key)
        .cloned()
        .ok_or_else(|| Error::InvalidOperation(format!("Wan2.2 VAE encoder: missing weight: {key}")))
}
fn get_bf16(weights: &Weights, key: &str) -> Result<Tensor> {
    get(weights, key)?.to_dtype(DType::BF16)
}

// ---------------------------------------------------------------------------
// CausalConv3d — BF16-native (Conv3dBF16), mirrored from vae.rs decoder.
// ---------------------------------------------------------------------------

/// Number of cache slots needed by the encoder (all CausalConv3d that
/// participate in cached forward — encoder.conv1, residual-path convs,
/// downsample3d time_convs, head conv, but NOT shortcut convs which are
/// kernel-1 and don't use temporal cache).  count_conv3d(encoder) in Python
/// gives 26, but only 24 slots are actually written/read. We allocate 26
/// to match Python's indexing exactly.
const NUM_ENCODER_CACHE_SLOTS: usize = 26;

/// Python CACHE_T = 2.
const CACHE_T: usize = 2;

struct CausalConv3d {
    conv: Conv3dBF16,
    time_pad: usize,
}

impl CausalConv3d {
    fn load(
        weights: &Weights,
        prefix: &str,
        _in_ch: usize,
        _out_ch: usize,
        _kernel: (usize, usize, usize),
        stride: (usize, usize, usize),
        pad: (usize, usize, usize),
        _device: &Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        let time_pad = 2 * pad.0;
        let weight = get_bf16(weights, &format!("{prefix}.weight"))?;
        let bias = Some(get_bf16(weights, &format!("{prefix}.bias"))?);
        let conv = Conv3dBF16::from_weights(
            weight,
            bias,
            stride,
            (0, pad.1, pad.2), // temporal padding handled manually
        );
        Ok(Self { conv, time_pad })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Causal zero-pad along temporal axis (dim 2).
        // Python `CausalConv3d` uses `F.pad(x, padding)` which is constant
        // zero padding, left-padding dim 2 by `2 * padding[0]`.
        let x_padded = if self.time_pad > 0 {
            let d = x.shape().dims().to_vec();
            debug_assert_eq!(d.len(), 5);
            let zero_shape = [d[0], d[1], self.time_pad, d[3], d[4]];
            let pad = Tensor::zeros_dtype(
                Shape::from_dims(&zero_shape),
                DType::BF16,
                x.device().clone(),
            )?;
            let x_bf16 = if x.dtype() == DType::BF16 { x.clone() } else { x.to_dtype(DType::BF16)? };
            Tensor::cat(&[&pad, &x_bf16], 2)?
        } else if x.dtype() != DType::BF16 {
            x.to_dtype(DType::BF16)?
        } else {
            x.clone()
        };

        // Temporal chunking for memory — same budget logic as the decoder.
        let dims_in = x_padded.shape().dims();
        let d_in = dims_in[2];
        let (kd, _kh, _kw) = self.conv.kernel_size;
        let (sd, _sh, _sw) = self.conv.stride;
        let (pd, _ph, _pw) = self.conv.padding;
        let d_out_total = (d_in + 2 * pd - kd) / sd + 1;

        const COL_BYTE_BUDGET: usize = 1usize << 30; // 1 GB
        let hw = dims_in[3] * dims_in[4];
        let col_rows = dims_in[1] * kd * self.conv.kernel_size.1 * self.conv.kernel_size.2;
        let cols_per_out_frame = hw;
        let max_frames_per_chunk = (COL_BYTE_BUDGET / 2 / col_rows / cols_per_out_frame.max(1)).max(1);

        if max_frames_per_chunk >= d_out_total || d_out_total == 1 {
            return self.conv.forward(&x_padded);
        }

        let mut out_chunks: Vec<Tensor> = Vec::new();
        let mut t_out = 0usize;
        while t_out < d_out_total {
            let chunk = max_frames_per_chunk.min(d_out_total - t_out);
            let in_start = t_out * sd;
            let in_len = (chunk - 1) * sd + kd;
            let in_slice = x_padded.narrow(2, in_start, in_len)?;
            let piece = self.conv.forward(&in_slice)?;
            out_chunks.push(piece);
            t_out += chunk;
        }
        let refs: Vec<&Tensor> = out_chunks.iter().collect();
        Tensor::cat(&refs, 2)
    }

    /// Cached forward matching Python's `CausalConv3d.forward(x, cache_x)`.
    ///
    /// When `cache_x` is Some, it is prepended along dim-2 instead of zero-pad.
    /// If cache_x.T < time_pad, remaining left-padding is zeros.
    /// When `cache_x` is None, same as `forward` (full zero-pad).
    fn forward_cached(&self, x: &Tensor, cache_x: Option<&Tensor>) -> Result<Tensor> {
        let x_bf16 = if x.dtype() != DType::BF16 { x.to_dtype(DType::BF16)? } else { x.clone() };

        let x_padded = if self.time_pad > 0 {
            match cache_x {
                Some(cx) => {
                    let cx_bf16 = if cx.dtype() != DType::BF16 { cx.to_dtype(DType::BF16)? } else { cx.clone() };
                    let cx_t = cx_bf16.shape().dims()[2];
                    if cx_t >= self.time_pad {
                        // Cache provides enough frames — just prepend
                        let trimmed = cx_bf16.narrow(2, cx_t - self.time_pad, self.time_pad)?;
                        Tensor::cat(&[&trimmed, &x_bf16], 2)?
                    } else {
                        // Cache provides fewer frames than time_pad — zero-pad the rest
                        let remaining = self.time_pad - cx_t;
                        let d = x_bf16.shape().dims().to_vec();
                        let zeros = Tensor::zeros_dtype(
                            Shape::from_dims(&[d[0], d[1], remaining, d[3], d[4]]),
                            DType::BF16,
                            x.device().clone(),
                        )?;
                        Tensor::cat(&[&zeros, &cx_bf16, &x_bf16], 2)?
                    }
                }
                None => {
                    // No cache — zero-pad (same as forward)
                    let d = x_bf16.shape().dims().to_vec();
                    let zeros = Tensor::zeros_dtype(
                        Shape::from_dims(&[d[0], d[1], self.time_pad, d[3], d[4]]),
                        DType::BF16,
                        x.device().clone(),
                    )?;
                    Tensor::cat(&[&zeros, &x_bf16], 2)?
                }
            }
        } else {
            x_bf16
        };

        self.conv.forward(&x_padded)
    }
}

// ---------------------------------------------------------------------------
// RMS_norm variants (same as decoder)
// ---------------------------------------------------------------------------

struct RmsNorm5d {
    gamma: Tensor, // [C, 1, 1, 1]
    scale: f32,
}

impl RmsNorm5d {
    fn load(weights: &Weights, prefix: &str, dim: usize) -> Result<Self> {
        Ok(Self {
            gamma: get_bf16(weights, &format!("{prefix}.gamma"))?,
            scale: (dim as f32).sqrt(),
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_f32 = x.to_dtype(DType::F32)?;
        let sum_sq = x_f32.mul(&x_f32)?.sum_dim(1)?.unsqueeze(1)?;
        let norm = sum_sq.sqrt()?.add_scalar(1e-12)?;
        let n = x_f32.div(&norm)?;
        let s = n.mul_scalar(self.scale)?.to_dtype(DType::BF16)?;
        s.mul(&self.gamma)
    }
}

struct RmsNorm4d {
    gamma: Tensor, // [C, 1, 1]
    scale: f32,
}

impl RmsNorm4d {
    fn load(weights: &Weights, prefix: &str, dim: usize) -> Result<Self> {
        Ok(Self {
            gamma: get_bf16(weights, &format!("{prefix}.gamma"))?,
            scale: (dim as f32).sqrt(),
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_f32 = x.to_dtype(DType::F32)?;
        let sum_sq = x_f32.mul(&x_f32)?.sum_dim(1)?.unsqueeze(1)?;
        let norm = sum_sq.sqrt()?.add_scalar(1e-12)?;
        let n = x_f32.div(&norm)?;
        let s = n.mul_scalar(self.scale)?.to_dtype(DType::BF16)?;
        s.mul(&self.gamma)
    }
}

// ---------------------------------------------------------------------------
// ResidualBlock (same layout as decoder)
// ---------------------------------------------------------------------------

struct ResidualBlock {
    norm1: RmsNorm5d,
    conv1: CausalConv3d,
    norm2: RmsNorm5d,
    conv2: CausalConv3d,
    shortcut: Option<CausalConv3d>,
}

impl ResidualBlock {
    fn load(
        weights: &Weights,
        prefix: &str,
        in_dim: usize,
        out_dim: usize,
        device: &Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        // residual Sequential layout:
        //   0: RMS_norm(in)
        //   1: SiLU
        //   2: CausalConv3d(in, out, 3, pad=1)
        //   3: RMS_norm(out)
        //   4: SiLU
        //   5: Dropout (no params)
        //   6: CausalConv3d(out, out, 3, pad=1)
        let norm1 = RmsNorm5d::load(weights, &format!("{prefix}.residual.0"), in_dim)?;
        let conv1 = CausalConv3d::load(
            weights,
            &format!("{prefix}.residual.2"),
            in_dim, out_dim,
            (3, 3, 3), (1, 1, 1), (1, 1, 1),
            device,
        )?;
        let norm2 = RmsNorm5d::load(weights, &format!("{prefix}.residual.3"), out_dim)?;
        let conv2 = CausalConv3d::load(
            weights,
            &format!("{prefix}.residual.6"),
            out_dim, out_dim,
            (3, 3, 3), (1, 1, 1), (1, 1, 1),
            device,
        )?;
        let shortcut = if in_dim != out_dim {
            Some(CausalConv3d::load(
                weights,
                &format!("{prefix}.shortcut"),
                in_dim, out_dim,
                (1, 1, 1), (1, 1, 1), (0, 0, 0),
                device,
            )?)
        } else {
            None
        };
        Ok(Self { norm1, conv1, norm2, conv2, shortcut })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = if let Some(ref s) = self.shortcut { s.forward(x)? } else { x.clone() };
        let mut out = self.norm1.forward(x)?;
        out = out.silu()?;
        out = self.conv1.forward(&out)?;
        out = self.norm2.forward(&out)?;
        out = out.silu()?;
        out = self.conv2.forward(&out)?;
        out.add(&h)
    }

    /// Cached forward matching Python's `ResidualBlock.forward(x, feat_cache, feat_idx)`.
    ///
    /// The residual path's two CausalConv3d layers each get a cache slot.
    /// The shortcut (kernel=1, time_pad=0) doesn't use cache.
    fn forward_cached(
        &self,
        x: &Tensor,
        cache: &mut Vec<Option<Tensor>>,
        idx: &mut usize,
    ) -> Result<Tensor> {
        // Shortcut — no cache (kernel=1, no temporal padding)
        let h = if let Some(ref s) = self.shortcut { s.forward(x)? } else { x.clone() };

        // --- conv1 through residual path ---
        // Python: for each CausalConv3d in self.residual:
        //   cache_x = x[:, :, -CACHE_T:].clone()
        //   if cache_x.T < 2 and feat_cache[idx] is not None:
        //       cache_x = cat([feat_cache[idx][:,:,-1,:,:].unsqueeze(2), cache_x], 2)
        //   x = layer(x, feat_cache[idx])
        //   feat_cache[idx] = cache_x
        //   feat_idx[0] += 1
        //
        // Note: 'x' here is the input to the CURRENT layer (after norm/silu),
        // but cache_x is built from 'x' (the variable being threaded through
        // the sequential). Let me re-read Python more carefully...
        //
        // In Python, `x` is threaded through ALL layers in residual:
        //   for layer in self.residual:
        //     if isinstance(layer, CausalConv3d) and feat_cache is not None:
        //       cache_x = x[:, :, -CACHE_T:].clone()
        //       ...
        //       x = layer(x, feat_cache[idx])
        //       feat_cache[idx] = cache_x
        //     else:
        //       x = layer(x)
        //
        // So the cache is built from the input to the CausalConv3d (which includes
        // norm+silu preprocessing). Let me trace:
        //   x = norm1(x)      → x = layer(x) [RMS_norm]
        //   x = silu(x)       → x = layer(x) [SiLU]
        //   CONV1: cache_x = x[:,:,-2:].clone()
        //          ... pad cache_x if needed ...
        //          x = conv1(x, old_cache)
        //          feat_cache[idx] = cache_x
        //   x = norm2(x)
        //   x = silu(x)
        //   x = dropout(x)    [no-op at inference]
        //   CONV2: cache_x = x[:,:,-2:].clone()
        //          ...
        //          x = conv2(x, old_cache)
        //          feat_cache[idx] = cache_x

        let mut out = self.norm1.forward(x)?;
        out = out.silu()?;

        // Conv1 cache
        {
            let i = *idx;
            let mut cache_x = out.narrow(2, out.shape().dims()[2].saturating_sub(CACHE_T), out.shape().dims()[2].min(CACHE_T))?;
            if cache_x.shape().dims()[2] < CACHE_T {
                if let Some(ref old) = cache[i] {
                    // Prepend last frame from old cache
                    let old_last = old.narrow(2, old.shape().dims()[2] - 1, 1)?;
                    cache_x = Tensor::cat(&[&old_last, &cache_x], 2)?;
                }
            }
            out = self.conv1.forward_cached(&out, cache[i].as_ref())?;
            cache[i] = Some(cache_x);
            *idx += 1;
        }

        out = self.norm2.forward(&out)?;
        out = out.silu()?;
        // dropout is no-op at inference

        // Conv2 cache
        {
            let i = *idx;
            let mut cache_x = out.narrow(2, out.shape().dims()[2].saturating_sub(CACHE_T), out.shape().dims()[2].min(CACHE_T))?;
            if cache_x.shape().dims()[2] < CACHE_T {
                if let Some(ref old) = cache[i] {
                    let old_last = old.narrow(2, old.shape().dims()[2] - 1, 1)?;
                    cache_x = Tensor::cat(&[&old_last, &cache_x], 2)?;
                }
            }
            out = self.conv2.forward_cached(&out, cache[i].as_ref())?;
            cache[i] = Some(cache_x);
            *idx += 1;
        }

        out.add(&h)
    }
}

// ---------------------------------------------------------------------------
// AttentionBlock — single-head per-frame self-attention (same as decoder).
// ---------------------------------------------------------------------------

struct AttentionBlock {
    norm: RmsNorm4d,
    to_qkv: Conv2d,
    proj: Conv2d,
}

impl AttentionBlock {
    fn load(
        weights: &Weights,
        prefix: &str,
        dim: usize,
        device: &Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        let norm = RmsNorm4d::load(weights, &format!("{prefix}.norm"), dim)?;

        let mut to_qkv = Conv2d::new_with_bias(dim, dim * 3, 1, 1, 0, device.clone(), true)?;
        to_qkv.copy_weight_from(&get_bf16(weights, &format!("{prefix}.to_qkv.weight"))?)?;
        to_qkv.copy_bias_from(&get_bf16(weights, &format!("{prefix}.to_qkv.bias"))?)?;

        let mut proj = Conv2d::new_with_bias(dim, dim, 1, 1, 0, device.clone(), true)?;
        proj.copy_weight_from(&get_bf16(weights, &format!("{prefix}.proj.weight"))?)?;
        proj.copy_bias_from(&get_bf16(weights, &format!("{prefix}.proj.bias"))?)?;

        Ok(Self { norm, to_qkv, proj })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let (b, c, t, h, w) = (dims[0], dims[1], dims[2], dims[3], dims[4]);
        let identity = x.clone();

        // [B, C, T, H, W] -> [B*T, C, H, W]
        let x_4d = x.permute(&[0, 2, 1, 3, 4])?.reshape(&[b * t, c, h, w])?;
        let x_normed = self.norm.forward(&x_4d)?;

        let qkv = self.to_qkv.forward(&x_normed)?;
        let n = h * w;
        let qkv_flat = qkv
            .reshape(&[b * t, c * 3, n])?
            .permute(&[0, 2, 1])?
            .reshape(&[b * t, 1, n, c * 3])?;

        let q = qkv_flat.narrow(3, 0, c)?;
        let k = qkv_flat.narrow(3, c, c)?;
        let v = qkv_flat.narrow(3, c * 2, c)?;

        let attn = sdpa_forward(&q, &k, &v, None)?;
        let attn = attn
            .squeeze(Some(1))?
            .permute(&[0, 2, 1])?
            .reshape(&[b * t, c, h, w])?;

        let projected = self.proj.forward(&attn)?;
        let out = projected
            .reshape(&[b, t, c, h, w])?
            .permute(&[0, 2, 1, 3, 4])?;
        identity.add(&out)
    }
}

// ---------------------------------------------------------------------------
// Downsample Resample — mirror of decoder's Upsample Resample.
//
// downsample2d: ZeroPad2d(0,1,0,1) + Conv2d(dim, dim, 3, stride=2)
// downsample3d: same spatial + CausalConv3d(dim, dim, (3,1,1), stride=(2,1,1))
//               temporal downsample comes AFTER spatial in the forward pass.
// ---------------------------------------------------------------------------

enum DownResample {
    Downsample2d { conv: Conv2d },
    Downsample3d { conv: Conv2d, time_conv: CausalConv3d },
}

impl DownResample {
    fn load_2d(
        weights: &Weights,
        prefix: &str,
        dim: usize,
        device: &Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        // resample = nn.Sequential(ZeroPad2d(0,1,0,1), Conv2d(dim, dim, 3, stride=2))
        // resample.0 = ZeroPad2d (no weights)
        // resample.1 = Conv2d
        let mut conv = Conv2d::new_with_bias(dim, dim, 3, 2, 0, device.clone(), true)?;
        conv.copy_weight_from(&get_bf16(weights, &format!("{prefix}.resample.1.weight"))?)?;
        conv.copy_bias_from(&get_bf16(weights, &format!("{prefix}.resample.1.bias"))?)?;
        Ok(DownResample::Downsample2d { conv })
    }

    fn load_3d(
        weights: &Weights,
        prefix: &str,
        dim: usize,
        device: &Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        // Same spatial conv as 2d
        let mut conv = Conv2d::new_with_bias(dim, dim, 3, 2, 0, device.clone(), true)?;
        conv.copy_weight_from(&get_bf16(weights, &format!("{prefix}.resample.1.weight"))?)?;
        conv.copy_bias_from(&get_bf16(weights, &format!("{prefix}.resample.1.bias"))?)?;

        // time_conv: CausalConv3d(dim, dim, (3,1,1), stride=(2,1,1), padding=(0,0,0))
        //
        // Python uses padding=(0,0,0) — time_pad=0. The chunked cache
        // mechanism provides temporal context explicitly. The full-pass
        // code bypasses CausalConv3d.forward() and calls conv.forward()
        // directly, so time_pad=0 is fine for both paths.
        let time_conv = CausalConv3d::load(
            weights,
            &format!("{prefix}.time_conv"),
            dim, dim,
            (3, 1, 1), (2, 1, 1), (0, 0, 0),
            device,
        )?;

        Ok(DownResample::Downsample3d { conv, time_conv })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            DownResample::Downsample2d { conv } => {
                let dims = x.shape().dims().to_vec();
                let (b, c, t, h, w) = (dims[0], dims[1], dims[2], dims[3], dims[4]);

                // [B, C, T, H, W] -> [B*T, C, H, W]
                let x_4d = x.permute(&[0, 2, 1, 3, 4])?.reshape(&[b * t, c, h, w])?;

                // ZeroPad2d(0, 1, 0, 1) -> pad right by 1, pad bottom by 1
                let x_padded = Self::zero_pad2d_right_bottom(&x_4d)?;

                // Conv2d with stride 2
                let x_conv = conv.forward(&x_padded)?;
                let c_out = x_conv.shape().dims()[1];
                let h_out = x_conv.shape().dims()[2];
                let w_out = x_conv.shape().dims()[3];
                x_conv
                    .reshape(&[b, t, c_out, h_out, w_out])?
                    .permute(&[0, 2, 1, 3, 4])
            }
            DownResample::Downsample3d { conv, time_conv } => {
                let dims = x.shape().dims().to_vec();
                let (b, c, t, h, w) = (dims[0], dims[1], dims[2], dims[3], dims[4]);

                // Spatial downsample first (same as 2d)
                let x_4d = x.permute(&[0, 2, 1, 3, 4])?.reshape(&[b * t, c, h, w])?;
                let x_padded = Self::zero_pad2d_right_bottom(&x_4d)?;
                let x_conv = conv.forward(&x_padded)?;
                let c_out = x_conv.shape().dims()[1];
                let h_out = x_conv.shape().dims()[2];
                let w_out = x_conv.shape().dims()[3];
                let x_5d = x_conv
                    .reshape(&[b, t, c_out, h_out, w_out])?
                    .permute(&[0, 2, 1, 3, 4])?;

                // Temporal downsample — replicate Python's chunked cache behavior:
                //
                // The Python encoder processes video in chunks. On the first chunk
                // (frame 0), the Resample(downsample3d) SKIPS time_conv entirely
                // and passes the spatially-downsampled frame through directly.
                // On subsequent chunks, it concatenates the cached last frame from
                // the previous chunk and applies time_conv.
                //
                // To replicate in a single pass:
                //   - Frame 0: pass through without time_conv
                //   - Frames 1+: apply time_conv to [frame0, frame1, ..., frameN]
                //     with NO zero padding (the Python cache provides frame0 as
                //     context, not zeros).
                let first_frame = x_5d.narrow(2, 0, 1)?; // [B, C, 1, H', W']

                if t == 1 {
                    // Only 1 frame: just return it (no temporal downsample)
                    Ok(first_frame)
                } else {
                    // Apply time_conv to [frame0, frame1, ..., frameN] without
                    // causal zero padding. We call the raw conv directly to bypass
                    // CausalConv3d's padding logic.
                    //
                    // time_conv kernel=(3,1,1), stride=(2,1,1), no padding needed.
                    // Input: all t frames → output: (t-3)/2 + 1 = (t-1)/2 frames
                    // For t=5: (5-3)/2+1 = 2 frames. Total = 1+2 = 3. ✓
                    let x_bf16 = if x_5d.dtype() == DType::BF16 { x_5d.clone() } else { x_5d.to_dtype(DType::BF16)? };
                    let rest_frames = time_conv.conv.forward(&x_bf16)?;
                    // Concatenate: [first_frame, rest_frames]
                    Tensor::cat(&[&first_frame, &rest_frames], 2)
                }
            }
        }
    }

    /// Cached forward for downsample.
    ///
    /// - `Downsample2d`: no cache interaction — just spatial downsample.
    /// - `Downsample3d`: special cache handling per Python's `Resample.forward`:
    ///   - First chunk (cache is None): store full spatially-downsampled output
    ///     in cache. Skip time_conv. Return spatial result.
    ///   - Subsequent chunks: prepend last frame from cache, run time_conv
    ///     (stride=2), store last frame as new cache.
    fn forward_cached(
        &self,
        x: &Tensor,
        cache: &mut Vec<Option<Tensor>>,
        idx: &mut usize,
    ) -> Result<Tensor> {
        match self {
            DownResample::Downsample2d { conv } => {
                // No cache interaction — same as forward()
                let dims = x.shape().dims().to_vec();
                let (b, c, t, h, w) = (dims[0], dims[1], dims[2], dims[3], dims[4]);
                let x_4d = x.permute(&[0, 2, 1, 3, 4])?.reshape(&[b * t, c, h, w])?;
                let x_padded = Self::zero_pad2d_right_bottom(&x_4d)?;
                let x_conv = conv.forward(&x_padded)?;
                let c_out = x_conv.shape().dims()[1];
                let h_out = x_conv.shape().dims()[2];
                let w_out = x_conv.shape().dims()[3];
                x_conv
                    .reshape(&[b, t, c_out, h_out, w_out])?
                    .permute(&[0, 2, 1, 3, 4])
            }
            DownResample::Downsample3d { conv, time_conv } => {
                let dims = x.shape().dims().to_vec();
                let (b, c, t, h, w) = (dims[0], dims[1], dims[2], dims[3], dims[4]);

                // Spatial downsample (same as non-cached)
                let x_4d = x.permute(&[0, 2, 1, 3, 4])?.reshape(&[b * t, c, h, w])?;
                let x_padded = Self::zero_pad2d_right_bottom(&x_4d)?;
                let x_conv = conv.forward(&x_padded)?;
                let c_out = x_conv.shape().dims()[1];
                let h_out = x_conv.shape().dims()[2];
                let w_out = x_conv.shape().dims()[3];
                let x_5d = x_conv
                    .reshape(&[b, t, c_out, h_out, w_out])?
                    .permute(&[0, 2, 1, 3, 4])?;

                // Downsample3d cache handling
                let i = *idx;
                let result = if cache[i].is_none() {
                    // First chunk: store full spatial result, skip time_conv
                    cache[i] = Some(x_5d.clone());
                    x_5d
                } else {
                    // Subsequent chunks: prepend last frame from cache, run time_conv
                    let old_cache = cache[i].as_ref().unwrap();
                    let last_frame = old_cache.narrow(2, old_cache.shape().dims()[2] - 1, 1)?;
                    let with_cache = Tensor::cat(&[&last_frame, &x_5d], 2)?;

                    // Run time_conv WITHOUT cache_x (Python calls self.time_conv(...) directly)
                    // time_conv has time_pad=0 (padding=(0,0,0) in Python for downsample3d)
                    // so forward_cached with None is same as forward
                    let result = time_conv.forward_cached(&with_cache, None)?;

                    // Store last frame as new cache
                    let new_cache = x_5d.narrow(2, x_5d.shape().dims()[2] - 1, 1)?;
                    cache[i] = Some(new_cache);
                    result
                };
                *idx += 1;

                Ok(result)
            }
        }
    }

    /// ZeroPad2d(0, 1, 0, 1) — pad right by 1, pad bottom by 1.
    /// Input: [N, C, H, W] -> Output: [N, C, H+1, W+1]
    fn zero_pad2d_right_bottom(x: &Tensor) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let (n, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);

        // Pad width: cat with zeros column [N, C, H, 1]
        let col_pad = Tensor::zeros_dtype(
            Shape::from_dims(&[n, c, h, 1]),
            DType::BF16,
            x.device().clone(),
        )?;
        let x_wpad = Tensor::cat(&[x, &col_pad], 3)?;

        // Pad height: cat with zeros row [N, C, 1, W+1]
        let row_pad = Tensor::zeros_dtype(
            Shape::from_dims(&[n, c, 1, w + 1]),
            DType::BF16,
            x.device().clone(),
        )?;
        Tensor::cat(&[&x_wpad, &row_pad], 2)
    }
}

// ---------------------------------------------------------------------------
// AvgDown3D — parameter-free avg-pool shortcut (mirror of DupUp3D).
//
// Rearranges spatial/temporal into channel groups and means them.
// ---------------------------------------------------------------------------

struct AvgDown3D {
    in_channels: usize,
    out_channels: usize,
    factor_t: usize,
    factor_s: usize,
    group_size: usize,
}

impl AvgDown3D {
    fn new(in_channels: usize, out_channels: usize, factor_t: usize, factor_s: usize) -> Self {
        let factor = factor_t * factor_s * factor_s;
        debug_assert_eq!((in_channels * factor) % out_channels, 0);
        let group_size = in_channels * factor / out_channels;
        Self { in_channels, out_channels, factor_t, factor_s, group_size }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let (b, _c, t, h, w) = (dims[0], dims[1], dims[2], dims[3], dims[4]);

        // Pad temporal dim if needed so it's divisible by factor_t
        let pad_t = (self.factor_t - t % self.factor_t) % self.factor_t;
        let x = if pad_t > 0 {
            let zero_shape = [b, self.in_channels, pad_t, h, w];
            let pad = Tensor::zeros_dtype(
                Shape::from_dims(&zero_shape),
                DType::BF16,
                x.device().clone(),
            )?;
            // Pad on the LEFT (prepend), matching Python's F.pad(..., (0,0,0,0,pad_t,0))
            // Actually Python pads (pad_t, 0) on dim 2 which is left-padding.
            Tensor::cat(&[&pad, x], 2)?
        } else {
            x.clone()
        };

        let dims = x.shape().dims().to_vec();
        let t = dims[2]; // updated after padding

        // Reshape: [B, C, T/ft, ft, H/fs, fs, W/fs, fs]
        let x = x.reshape(&[
            b,
            self.in_channels,
            t / self.factor_t,
            self.factor_t,
            h / self.factor_s,
            self.factor_s,
            w / self.factor_s,
            self.factor_s,
        ])?;

        // Permute: (0, 1, 3, 5, 7, 2, 4, 6) -> [B, C, ft, fs, fs, T/ft, H/fs, W/fs]
        let x = x.permute(&[0, 1, 3, 5, 7, 2, 4, 6])?;

        // View: [B, C*factor, T/ft, H/fs, W/fs]
        let factor = self.factor_t * self.factor_s * self.factor_s;
        let t_out = t / self.factor_t;
        let h_out = h / self.factor_s;
        let w_out = w / self.factor_s;
        let x = x.reshape(&[b, self.in_channels * factor, t_out, h_out, w_out])?;

        // View: [B, out_channels, group_size, T/ft, H/fs, W/fs]
        let x = x.reshape(&[b, self.out_channels, self.group_size, t_out, h_out, w_out])?;

        // Mean over group_size dim (dim=2)
        // sum_dim removes the dim → [B, out_channels, T_out, H_out, W_out]
        let x_f32 = x.to_dtype(DType::F32)?;
        let sum = x_f32.sum_dim(2)?;
        let mean = sum.mul_scalar(1.0 / self.group_size as f32)?;
        mean.to_dtype(DType::BF16)
    }
}

// ---------------------------------------------------------------------------
// Down_ResidualBlock: (N ResidualBlocks + optional Resample) + AvgDown3D shortcut.
// ---------------------------------------------------------------------------

struct DownResidualBlock {
    residuals: Vec<ResidualBlock>,
    resample: Option<DownResample>,
    avg_shortcut: AvgDown3D,
}

impl DownResidualBlock {
    fn load(
        weights: &Weights,
        prefix: &str,
        in_dim: usize,
        out_dim: usize,
        mult: usize,           // num_res_blocks
        temperal_downsample: bool,
        down_flag: bool,
        device: &Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        // Main path Sequential: `downsamples` in Python
        //   [0..mult]: ResidualBlock (first with in_dim→out_dim, rest out_dim→out_dim)
        //   [mult]:    Resample (only if down_flag)
        let mut residuals: Vec<ResidualBlock> = Vec::with_capacity(mult);
        let mut cur_in = in_dim;
        for i in 0..mult {
            let rb_prefix = format!("{prefix}.downsamples.{i}");
            let rb = ResidualBlock::load(weights, &rb_prefix, cur_in, out_dim, device)?;
            residuals.push(rb);
            cur_in = out_dim;
        }

        let resample = if down_flag {
            let idx = mult;
            let res_prefix = format!("{prefix}.downsamples.{idx}");
            let r = if temperal_downsample {
                DownResample::load_3d(weights, &res_prefix, out_dim, device)?
            } else {
                DownResample::load_2d(weights, &res_prefix, out_dim, device)?
            };
            Some(r)
        } else {
            None
        };

        // AvgDown3D shortcut (parameter-free, always present)
        let avg_shortcut = AvgDown3D::new(
            in_dim,
            out_dim,
            if temperal_downsample { 2 } else { 1 },
            if down_flag { 2 } else { 1 },
        );

        Ok(Self { residuals, resample, avg_shortcut })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Main path: ResBlocks → optional Resample.
        let mut main = x.clone();
        for rb in &self.residuals {
            main = rb.forward(&main)?;
        }
        if let Some(ref r) = self.resample {
            main = r.forward(&main)?;
        }

        // Shortcut: avg-pool of the original input x
        let short = self.avg_shortcut.forward(x)?;
        main.add(&short)
    }

    /// Cached forward matching Python's `Down_ResidualBlock.forward(x, feat_cache, feat_idx)`.
    fn forward_cached(
        &self,
        x: &Tensor,
        cache: &mut Vec<Option<Tensor>>,
        idx: &mut usize,
    ) -> Result<Tensor> {
        let x_copy = x.clone();

        let mut main = x.clone();
        for rb in &self.residuals {
            main = rb.forward_cached(&main, cache, idx)?;
        }
        if let Some(ref r) = self.resample {
            main = r.forward_cached(&main, cache, idx)?;
        }

        let short = self.avg_shortcut.forward(&x_copy)?;
        main.add(&short)
    }
}

// ---------------------------------------------------------------------------
// Top-level Wan2.2 VAE encoder.
// ---------------------------------------------------------------------------

pub struct Wan22VaeEncoder {
    device: Arc<cudarc::driver::CudaDevice>,

    /// Per-channel mean `[1, 48, 1, 1, 1]` BF16.
    mean: Tensor,
    /// Per-channel 1/std `[1, 48, 1, 1, 1]` BF16 (multiply to normalize).
    inv_std: Tensor,

    /// `Encoder3d.conv1`: CausalConv3d(12→dims[0]=160, 3×3×3, pad=1).
    encoder_conv1: CausalConv3d,

    /// 4 Down_ResidualBlock groups.
    down_groups: Vec<DownResidualBlock>,

    /// Middle: ResBlock → Attn → ResBlock.
    mid_res0: ResidualBlock,
    mid_attn: AttentionBlock,
    mid_res1: ResidualBlock,

    /// Encoder head: RMS_norm + SiLU + CausalConv3d(→z_dim*2=96).
    head_norm: RmsNorm5d,
    head_conv: CausalConv3d,

    /// Top-level conv1: CausalConv3d(96→96, 1×1×1).
    conv1: CausalConv3d,
}

impl Wan22VaeEncoder {
    /// Load encoder from the same safetensors file as the decoder.
    pub fn load(path: &Path, device: &Arc<cudarc::driver::CudaDevice>) -> Result<Self> {
        let weights = load_file(path, device)?;
        log::info!(
            "[Wan22 VAE Encoder] Loaded {} tensors from {}",
            weights.len(),
            path.display()
        );
        Self::from_weights(&weights, device)
    }

    fn from_weights(
        weights: &Weights,
        device: &Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        let z_dim: usize = 48;
        let enc_dim: usize = 160; // c_dim in Python
        let dim_mult: [usize; 4] = [1, 2, 4, 4];
        let num_res_blocks: usize = 2;
        let temperal_downsample: [bool; 3] = [false, true, true];

        // mean & inv_std broadcast tensors.
        let mean = Tensor::from_vec(
            MEAN.to_vec(),
            Shape::from_dims(&[1, z_dim, 1, 1, 1]),
            device.clone(),
        )?
        .to_dtype(DType::BF16)?;

        let inv_std_vals: Vec<f32> = STD.iter().map(|s| 1.0 / s).collect();
        let inv_std = Tensor::from_vec(
            inv_std_vals,
            Shape::from_dims(&[1, z_dim, 1, 1, 1]),
            device.clone(),
        )?
        .to_dtype(DType::BF16)?;

        // Encoder3d dims: [dim*u for u in [1] + dim_mult] = [160, 160, 320, 640, 640]
        let dims: Vec<usize> = {
            let mut d = vec![enc_dim]; // [1] prefix
            for &m in &dim_mult {
                d.push(enc_dim * m);
            }
            d
        };

        // encoder.conv1: CausalConv3d(12, dims[0]=160, 3, pad=1)
        let encoder_conv1 = CausalConv3d::load(
            weights, "encoder.conv1",
            12, dims[0],
            (3, 3, 3), (1, 1, 1), (1, 1, 1),
            device,
        )?;

        // Down groups: zip(dims[:-1], dims[1:]) → 4 groups
        // i=0: 160→160 (downsample2d, down_flag=True)
        // i=1: 160→320 (downsample3d, down_flag=True)
        // i=2: 320→640 (downsample3d, down_flag=True)
        // i=3: 640→640 (no downsample, down_flag=False)
        let num_groups = dims.len() - 1; // 4
        let mut down_groups: Vec<DownResidualBlock> = Vec::with_capacity(num_groups);
        for i in 0..num_groups {
            let in_d = dims[i];
            let out_d = dims[i + 1];
            let t_down = if i < temperal_downsample.len() {
                temperal_downsample[i]
            } else {
                false
            };
            let down_flag = i != dim_mult.len() - 1; // i != 3
            let prefix = format!("encoder.downsamples.{i}");
            down_groups.push(DownResidualBlock::load(
                weights, &prefix, in_d, out_d, num_res_blocks, t_down, down_flag, device,
            )?);
        }

        // Middle block: dims[-1] = 640
        let top_dim = *dims.last().unwrap();
        let mid_res0 = ResidualBlock::load(weights, "encoder.middle.0", top_dim, top_dim, device)?;
        let mid_attn = AttentionBlock::load(weights, "encoder.middle.1", top_dim, device)?;
        let mid_res1 = ResidualBlock::load(weights, "encoder.middle.2", top_dim, top_dim, device)?;

        // Head: RMS_norm(top_dim) + SiLU + CausalConv3d(top_dim, z_dim*2=96, 3, pad=1)
        let head_norm = RmsNorm5d::load(weights, "encoder.head.0", top_dim)?;
        let head_conv = CausalConv3d::load(
            weights, "encoder.head.2",
            top_dim, z_dim * 2,
            (3, 3, 3), (1, 1, 1), (1, 1, 1),
            device,
        )?;

        // Top-level conv1: CausalConv3d(z_dim*2=96, z_dim*2=96, 1)
        let conv1 = CausalConv3d::load(
            weights, "conv1",
            z_dim * 2, z_dim * 2,
            (1, 1, 1), (1, 1, 1), (0, 0, 0),
            device,
        )?;

        Ok(Self {
            device: device.clone(),
            mean, inv_std,
            encoder_conv1,
            down_groups,
            mid_res0, mid_attn, mid_res1,
            head_norm, head_conv,
            conv1,
        })
    }

    /// Encode video to normalized latents (full single-pass, no chunking).
    ///
    /// Input: `[B, 3, T, H, W]` BF16 RGB in `[-1, 1]`.
    /// Output: `[B, 48, T', H', W']` BF16 normalized latents.
    pub fn encode_full_pass(&self, video: &Tensor) -> Result<Tensor> {
        let dbg = std::env::var("WAN_VAE_DBG").is_ok();

        // 1. Patchify: [B, 3, T, H, W] → [B, 12, T, H/2, W/2]
        let x = Self::patchify2(video)?;
        if dbg { Self::dbg_stats("after patchify", &x); }

        // 2. Encoder conv1
        let mut x = self.encoder_conv1.forward(&x)?;
        if dbg { Self::dbg_stats("after encoder_conv1", &x); }

        // 3. Downsample groups
        for (i, g) in self.down_groups.iter().enumerate() {
            x = g.forward(&x)?;
            if dbg { Self::dbg_stats(&format!("after down_group[{i}]"), &x); }
        }

        // 4. Middle block
        x = self.mid_res0.forward(&x)?;
        if dbg { Self::dbg_stats("after mid_res0", &x); }
        x = self.mid_attn.forward(&x)?;
        if dbg { Self::dbg_stats("after mid_attn", &x); }
        x = self.mid_res1.forward(&x)?;
        if dbg { Self::dbg_stats("after mid_res1", &x); }

        // 5. Head: RMS_norm + SiLU + CausalConv3d → [B, 96, T', H', W']
        x = self.head_norm.forward(&x)?;
        if dbg { Self::dbg_stats("after head_norm", &x); }
        x = x.silu()?;
        if dbg { Self::dbg_stats("after head_silu", &x); }
        x = self.head_conv.forward(&x)?;
        if dbg { Self::dbg_stats("after head_conv", &x); }

        // 6. conv1 (top-level 1×1) → chunk into mu/log_var
        let out = self.conv1.forward(&x)?;
        if dbg { Self::dbg_stats("after conv1 (top-level)", &out); }

        // Chunk along channel dim: first 48 = mu, second 48 = log_var (discarded)
        let mu = out.narrow(1, 0, 48)?;
        if dbg { Self::dbg_stats("mu (before normalize)", &mu); }

        // 7. Normalize: z = (mu - mean) * inv_std = (mu - mean) / std
        //    Compute in F32 for numerical stability, matching PyTorch behavior.
        let mu_f32 = mu.to_dtype(DType::F32)?;
        let mean_f32 = self.mean.to_dtype(DType::F32)?;
        let inv_std_f32 = self.inv_std.to_dtype(DType::F32)?;
        let z = mu_f32.sub(&mean_f32)?.mul(&inv_std_f32)?.to_dtype(DType::BF16)?;

        Ok(z)
    }

    /// Encode video to normalized latents (chunked, matching Python's cache mechanism).
    ///
    /// Input: `[B, 3, T, H, W]` BF16 RGB in `[-1, 1]`.
    /// Output: `[B, 48, T', H', W']` BF16 normalized latents.
    pub fn encode(&self, video: &Tensor) -> Result<Tensor> {
        let dbg = std::env::var("WAN_VAE_DBG").is_ok();

        // 1. Patchify: [B, 3, T, H, W] → [B, 12, T, H/2, W/2]
        let x = Self::patchify2(video)?;
        if dbg { Self::dbg_stats("after patchify", &x); }

        let t = x.shape().dims()[2];
        let num_iters = 1 + (t.saturating_sub(1)) / 4;

        // Initialize cache (matching Python's clear_cache → _enc_feat_map)
        let mut cache: Vec<Option<Tensor>> = vec![None; NUM_ENCODER_CACHE_SLOTS];

        let mut out: Option<Tensor> = None;
        for i in 0..num_iters {
            let mut idx = 0usize;

            // Build chunk: iter 0 = frame 0 alone, iter i≥1 = 4 frames
            let chunk = if i == 0 {
                x.narrow(2, 0, 1)?
            } else {
                let start = 1 + 4 * (i - 1);
                let len = (4).min(t - start);
                x.narrow(2, start, len)?
            };
            if dbg { Self::dbg_stats(&format!("chunk {i} input"), &chunk); }

            let chunk_out = self.encode_chunk(&chunk, &mut cache, &mut idx)?;
            if dbg { Self::dbg_stats(&format!("chunk {i} output"), &chunk_out); }

            out = match out {
                None => Some(chunk_out),
                Some(prev) => Some(Tensor::cat(&[&prev, &chunk_out], 2)?),
            };
        }

        let out = out.unwrap();
        if dbg { Self::dbg_stats("after encoder (all chunks)", &out); }

        // conv1 (top-level 1×1, after all chunks — NOT inside chunked loop)
        let out = self.conv1.forward(&out)?;
        if dbg { Self::dbg_stats("after conv1 (top-level)", &out); }

        // Chunk into mu (first 48 channels)
        let mu = out.narrow(1, 0, 48)?;
        if dbg { Self::dbg_stats("mu (before normalize)", &mu); }

        // Normalize in F32
        let mu_f32 = mu.to_dtype(DType::F32)?;
        let mean_f32 = self.mean.to_dtype(DType::F32)?;
        let inv_std_f32 = self.inv_std.to_dtype(DType::F32)?;
        let z = mu_f32.sub(&mean_f32)?.mul(&inv_std_f32)?.to_dtype(DType::BF16)?;

        Ok(z)
    }

    /// Encode one temporal chunk through the encoder (conv1 → downsamples →
    /// middle → head), using the feat_cache mechanism.
    fn encode_chunk(
        &self,
        x: &Tensor,
        cache: &mut Vec<Option<Tensor>>,
        idx: &mut usize,
    ) -> Result<Tensor> {
        // encoder.conv1 with cache (same pattern as ResidualBlock's internal convs)
        let mut x = {
            let i = *idx;
            let t_in = x.shape().dims()[2];
            let mut cache_x = x.narrow(2, t_in.saturating_sub(CACHE_T), t_in.min(CACHE_T))?;
            if cache_x.shape().dims()[2] < CACHE_T {
                if let Some(ref old) = cache[i] {
                    let old_last = old.narrow(2, old.shape().dims()[2] - 1, 1)?;
                    cache_x = Tensor::cat(&[&old_last, &cache_x], 2)?;
                }
            }
            let out = self.encoder_conv1.forward_cached(x, cache[i].as_ref())?;
            cache[i] = Some(cache_x);
            *idx += 1;
            out
        };

        // Downsample groups
        for g in &self.down_groups {
            x = g.forward_cached(&x, cache, idx)?;
        }

        // Middle block: ResBlock → Attn → ResBlock
        // Python: if isinstance(layer, ResidualBlock) and feat_cache is not None:
        //             x = layer(x, feat_cache, feat_idx)
        //         else: x = layer(x)  # AttentionBlock — no cache
        x = self.mid_res0.forward_cached(&x, cache, idx)?;
        x = self.mid_attn.forward(&x)?;
        x = self.mid_res1.forward_cached(&x, cache, idx)?;

        // Head: RMS_norm + SiLU + CausalConv3d (with cache)
        // Python: for layer in self.head:
        //   if isinstance(layer, CausalConv3d) and feat_cache is not None:
        //       ... same cache pattern ...
        //   else: x = layer(x)
        x = self.head_norm.forward(&x)?;
        x = x.silu()?;
        {
            let i = *idx;
            let t_in = x.shape().dims()[2];
            let mut cache_x = x.narrow(2, t_in.saturating_sub(CACHE_T), t_in.min(CACHE_T))?;
            if cache_x.shape().dims()[2] < CACHE_T {
                if let Some(ref old) = cache[i] {
                    let old_last = old.narrow(2, old.shape().dims()[2] - 1, 1)?;
                    cache_x = Tensor::cat(&[&old_last, &cache_x], 2)?;
                }
            }
            x = self.head_conv.forward_cached(&x, cache[i].as_ref())?;
            cache[i] = Some(cache_x);
            *idx += 1;
        }

        Ok(x)
    }

    fn dbg_stats(name: &str, t: &Tensor) {
        if let Ok(data) = t.to_dtype(DType::F32).and_then(|t| t.to_vec1::<f32>()) {
            let n = data.len();
            let mean: f32 = data.iter().copied().sum::<f32>() / n as f32;
            let abs_mean: f32 = data.iter().map(|v| v.abs()).sum::<f32>() / n as f32;
            let lo = data.iter().copied().filter(|v| v.is_finite()).fold(f32::INFINITY, f32::min);
            let hi = data.iter().copied().filter(|v| v.is_finite()).fold(f32::NEG_INFINITY, f32::max);
            let nan = data.iter().filter(|v| v.is_nan()).count();
            eprintln!(
                "  [DBG {name}] shape={:?} mean={mean:.4} |mean|={abs_mean:.4} range=[{lo:.4}, {hi:.4}] nan={nan}",
                t.shape().dims()
            );
        }
    }

    /// Patchify with patch_size=2 on spatial dims.
    ///
    /// Input `[B, 3, T, H, W]`, output `[B, 12, T, H/2, W/2]`.
    /// `rearrange(x, 'b c f (h q) (w r) -> b (c r q) f h w', q=2, r=2)`
    fn patchify2(x: &Tensor) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        assert_eq!(dims.len(), 5, "patchify2 expects [B,C,T,H,W]");
        let (b, c, t, h, w) = (dims[0], dims[1], dims[2], dims[3], dims[4]);
        assert!(h % 2 == 0 && w % 2 == 0, "H and W must be even for patchify");

        // Reshape: [B, C, T, H/2, 2, W/2, 2]
        let v = x.reshape(&[b, c, t, h / 2, 2, w / 2, 2])?;
        // We need: 'b c f (h q) (w r) -> b (c r q) f h w'
        // Indices: [b=0, c=1, f=2, h=3, q=4, w=5, r=6]
        // Target:  [b=0, (c=1,r=6,q=4), f=2, h=3, w=5]
        // Permute: [0, 1, 6, 4, 2, 3, 5]
        let p = v.permute(&[0, 1, 6, 4, 2, 3, 5])?;
        // Reshape: [B, C*2*2, T, H/2, W/2]
        p.reshape(&[b, c * 4, t, h / 2, w / 2])
    }

    pub fn device(&self) -> &Arc<cudarc::driver::CudaDevice> { &self.device }
}
