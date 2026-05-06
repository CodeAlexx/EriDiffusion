//! Wan 2.1 / Qwen-Image 3D VAE encoder — pure flame_core.
//!
//! Ported from `inference-flame/src/vae/wan21_encoder.rs` (2026-05-05).
//! Encode-only; the decoder lives in `wan21_decoder.rs` (port deferred).
//!
//! On-disk weights `qwen_image_vae.safetensors` (shipped under
//! `circlestone-labs/Anima/split_files/vae/` and the Z-Image checkpoint
//! distribution) ARE Wan-VAE in wan21 internal-key format
//! (`encoder.conv1.weight`, `encoder.downsamples.{n}.residual.{0,3}.gamma`, ...).
//! Anima + Z-Image use the **zero-pad** CausalConv3d variant (QwenImage
//! distillation convention), not Wan 2.1 frame-replicate.
//!
//! Architecture (dim=96, z_dim=16, dim_mult=[1,2,4,4]):
//!   encoder.conv1: CausalConv3d(3, 96, 3x3x3)
//!   encoder.downsamples: 11 flat blocks (8 ResBlocks + 3 Resamples)
//!   encoder.middle: ResBlock(384) + AttentionBlock(384) + ResBlock(384)
//!   encoder.head: RMS_norm(384) + SiLU + CausalConv3d(384, 32, 3x3x3)
//!   conv1 (top-level): CausalConv3d(32, 32, 1x1x1)
//!   -> first 16 channels = mu (posterior mode)
//!
//! Two image-mode entry points:
//!   - [`Wan21VaeEncoder::encode_image_raw`] — `[B,3,H,W] -> [B,16,H/8,W/8]`
//!     raw VAE z (posterior mode). Use this when downstream applies its own
//!     shift/scale (e.g. Z-Image trainer with `(raw - 0.1159) * 0.3611`).
//!   - [`Wan21VaeEncoder::encode_image_normalized`] — same shape, but with
//!     canonical per-channel `(z - MEAN) / STD` applied. Use this when the
//!     trainer expects pre-normalized latents (Anima).

use flame_core::conv::Conv2d;
use flame_core::conv3d_simple::Conv3d;
use flame_core::sdpa::forward as sdpa_forward;
use flame_core::serialization::load_file;
use flame_core::{DType, Error, Result, Shape, Tensor};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

type Weights = HashMap<String, Tensor>;

// ---------------------------------------------------------------------------
// Per-channel normalization constants (Wan-VAE posterior statistics).
// Used by encode_image_normalized; ignored by encode_image_raw.
// ---------------------------------------------------------------------------

const MEAN: [f32; 16] = [
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
    0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
];

const STD: [f32; 16] = [
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
];

// ---------------------------------------------------------------------------
// Weight loading helpers
// ---------------------------------------------------------------------------

fn get(weights: &Weights, key: &str) -> Result<Tensor> {
    weights
        .get(key)
        .cloned()
        .ok_or_else(|| Error::InvalidOperation(format!("Wan21 VAE encoder: missing weight: {key}")))
}

fn get_bf16(weights: &Weights, key: &str) -> Result<Tensor> {
    get(weights, key)?.to_dtype(DType::BF16)
}

// ---------------------------------------------------------------------------
// CausalConv3d — Conv3d (F32 internal) + temporal left-pad. zero_pad=true
// matches QwenImage / Anima convention; false matches Wan 2.1 originals.
// ---------------------------------------------------------------------------

struct CausalConv3d {
    conv: Conv3d,
    time_pad: usize,
    zero_pad: bool,
}

impl CausalConv3d {
    fn load(
        weights: &Weights,
        prefix: &str,
        in_ch: usize,
        out_ch: usize,
        kernel: (usize, usize, usize),
        stride: (usize, usize, usize),
        pad: (usize, usize, usize),
        device: &Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        let time_pad = 2 * pad.0;
        let mut conv = Conv3d::new(
            in_ch,
            out_ch,
            kernel,
            Some(stride),
            Some((0, pad.1, pad.2)),
            None,
            None,
            true,
            device.clone(),
        )?;
        conv.weight = get(weights, &format!("{prefix}.weight"))?.to_dtype(DType::F32)?;
        conv.bias_tensor = Some(get(weights, &format!("{prefix}.bias"))?.to_dtype(DType::F32)?);
        Ok(Self { conv, time_pad, zero_pad: false })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_padded = if self.time_pad > 0 {
            if self.zero_pad {
                let dims = x.shape().dims().to_vec();
                let pad_shape = Shape::from_dims(&[dims[0], dims[1], self.time_pad, dims[3], dims[4]]);
                let zeros = Tensor::zeros_dtype(pad_shape, x.dtype(), x.device().clone())?;
                Tensor::cat(&[&zeros, x], 2)?
            } else {
                let first_frame = x.narrow(2, 0, 1)?;
                let repeated = first_frame.repeat_axis_device(2, self.time_pad)?;
                Tensor::cat(&[&repeated, x], 2)?
            }
        } else {
            x.clone()
        };
        let is_bf16 = x_padded.dtype() == DType::BF16;
        let input = if is_bf16 { x_padded.to_dtype(DType::F32)? } else { x_padded };
        let out = self.conv.forward(&input)?;
        if is_bf16 { out.to_dtype(DType::BF16) } else { Ok(out) }
    }

    fn forward_raw(&self, x: &Tensor) -> Result<Tensor> {
        let is_bf16 = x.dtype() == DType::BF16;
        let input = if is_bf16 { x.to_dtype(DType::F32)? } else { x.clone() };
        let out = self.conv.forward(&input)?;
        if is_bf16 { out.to_dtype(DType::BF16) } else { Ok(out) }
    }
}

// ---------------------------------------------------------------------------
// RMS_norm — channel_first=True. 5D for video tensors, 4D for the 2D
// per-frame attention path.
// ---------------------------------------------------------------------------

struct RmsNorm5d { gamma: Tensor, scale: f32 }

impl RmsNorm5d {
    fn load(weights: &Weights, prefix: &str, dim: usize) -> Result<Self> {
        let gamma = get_bf16(weights, &format!("{prefix}.gamma"))?;
        Ok(Self { gamma, scale: (dim as f32).sqrt() })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_f32 = x.to_dtype(DType::F32)?;
        let x_sq = x_f32.mul(&x_f32)?;
        let sum_sq = x_sq.sum_dim(1)?.unsqueeze(1)?;
        let norm = sum_sq.sqrt()?.add_scalar(1e-12)?;
        let normalized = x_f32.div(&norm)?;
        let scaled = normalized.mul_scalar(self.scale)?.to_dtype(DType::BF16)?;
        scaled.mul(&self.gamma)
    }
}

struct RmsNorm4d { gamma: Tensor, scale: f32 }

impl RmsNorm4d {
    fn load(weights: &Weights, prefix: &str, dim: usize) -> Result<Self> {
        let gamma = get_bf16(weights, &format!("{prefix}.gamma"))?;
        Ok(Self { gamma, scale: (dim as f32).sqrt() })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_f32 = x.to_dtype(DType::F32)?;
        let x_sq = x_f32.mul(&x_f32)?;
        let sum_sq = x_sq.sum_dim(1)?.unsqueeze(1)?;
        let norm = sum_sq.sqrt()?.add_scalar(1e-12)?;
        let normalized = x_f32.div(&norm)?;
        let scaled = normalized.mul_scalar(self.scale)?.to_dtype(DType::BF16)?;
        scaled.mul(&self.gamma)
    }
}

// ---------------------------------------------------------------------------
// ResidualBlock
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
        let norm1 = RmsNorm5d::load(weights, &format!("{prefix}.residual.0"), in_dim)?;
        let conv1 = CausalConv3d::load(
            weights, &format!("{prefix}.residual.2"), in_dim, out_dim,
            (3, 3, 3), (1, 1, 1), (1, 1, 1), device,
        )?;
        let norm2 = RmsNorm5d::load(weights, &format!("{prefix}.residual.3"), out_dim)?;
        let conv2 = CausalConv3d::load(
            weights, &format!("{prefix}.residual.6"), out_dim, out_dim,
            (3, 3, 3), (1, 1, 1), (1, 1, 1), device,
        )?;
        let shortcut = if in_dim != out_dim {
            Some(CausalConv3d::load(
                weights, &format!("{prefix}.shortcut"), in_dim, out_dim,
                (1, 1, 1), (1, 1, 1), (0, 0, 0), device,
            )?)
        } else {
            None
        };
        Ok(Self { norm1, conv1, norm2, conv2, shortcut })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = if let Some(ref s) = self.shortcut {
            s.forward(x)?
        } else {
            x.clone()
        };
        let mut out = self.norm1.forward(x)?;
        out = out.silu()?;
        out = self.conv1.forward(&out)?;
        out = self.norm2.forward(&out)?;
        out = out.silu()?;
        out = self.conv2.forward(&out)?;
        out.add(&h)
    }
}

// ---------------------------------------------------------------------------
// AttentionBlock — per-frame 2D self-attention, single head.
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
        let identity = x;

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

        let attn_out = sdpa_forward(&q, &k, &v, None)
            .map_err(|e| Error::InvalidOperation(format!("Wan21 VAE encoder sdpa: {e}")))?;

        let attn_out = attn_out
            .squeeze(Some(1))?
            .permute(&[0, 2, 1])?
            .reshape(&[b * t, c, h, w])?;

        let out = self.proj.forward(&attn_out)?;

        let out = out
            .reshape(&[b, t, c, h, w])?
            .permute(&[0, 2, 1, 3, 4])?;

        identity.add(&out)
    }
}

// ---------------------------------------------------------------------------
// DownResample — downsample2d or downsample3d.
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
        let mut conv = Conv2d::new_with_bias(dim, dim, 3, 2, 0, device.clone(), true)?;
        conv.copy_weight_from(&get_bf16(weights, &format!("{prefix}.resample.1.weight"))?)?;
        conv.copy_bias_from(&get_bf16(weights, &format!("{prefix}.resample.1.bias"))?)?;

        let time_conv = CausalConv3d::load(
            weights, &format!("{prefix}.time_conv"), dim, dim,
            (3, 1, 1), (2, 1, 1), (0, 0, 0), device,
        )?;

        Ok(DownResample::Downsample3d { conv, time_conv })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            DownResample::Downsample2d { conv } => {
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

                let x_4d = x.permute(&[0, 2, 1, 3, 4])?.reshape(&[b * t, c, h, w])?;
                let x_padded = Self::zero_pad2d_right_bottom(&x_4d)?;
                let x_conv = conv.forward(&x_padded)?;
                let c_out = x_conv.shape().dims()[1];
                let h_out = x_conv.shape().dims()[2];
                let w_out = x_conv.shape().dims()[3];
                let x_5d = x_conv
                    .reshape(&[b, t, c_out, h_out, w_out])?
                    .permute(&[0, 2, 1, 3, 4])?;

                // Match Python chunked-cache behavior:
                //   Frame 0: pass through without time_conv.
                //   Frames 1+: apply time_conv to all frames (no zero pad).
                let first_frame = x_5d.narrow(2, 0, 1)?;

                if t == 1 {
                    Ok(first_frame)
                } else {
                    let rest_frames = time_conv.forward_raw(&x_5d)?;
                    Tensor::cat(&[&first_frame, &rest_frames], 2)
                }
            }
        }
    }

    fn zero_pad2d_right_bottom(x: &Tensor) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let (n, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);

        let col_pad = Tensor::zeros_dtype(
            Shape::from_dims(&[n, c, h, 1]),
            DType::BF16,
            x.device().clone(),
        )?;
        let x_wpad = Tensor::cat(&[x, &col_pad], 3)?;

        let row_pad = Tensor::zeros_dtype(
            Shape::from_dims(&[n, c, 1, w + 1]),
            DType::BF16,
            x.device().clone(),
        )?;
        Tensor::cat(&[&x_wpad, &row_pad], 2)
    }
}

enum EncoderBlock {
    Res(ResidualBlock),
    Downsample(DownResample),
}

impl EncoderBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            EncoderBlock::Res(r) => r.forward(x),
            EncoderBlock::Downsample(d) => d.forward(x),
        }
    }
}

// ---------------------------------------------------------------------------
// Wan21VaeEncoder
// ---------------------------------------------------------------------------

pub struct Wan21VaeEncoder {
    /// Per-channel mean `[1, 16, 1, 1]` BF16 (4D — sized for image use).
    mean_4d: Tensor,
    /// Per-channel 1/std `[1, 16, 1, 1]` BF16.
    inv_std_4d: Tensor,

    encoder_conv1: CausalConv3d,
    downsamples: Vec<EncoderBlock>,
    mid_res0: ResidualBlock,
    mid_attn: AttentionBlock,
    mid_res1: ResidualBlock,
    head_norm: RmsNorm5d,
    head_conv: CausalConv3d,
    /// Top-level conv1: CausalConv3d(32, 32, 1x1x1).
    conv1_top: CausalConv3d,
}

impl Wan21VaeEncoder {
    /// Load a `qwen_image_vae.safetensors` (zero-pad CausalConv3d variant —
    /// QwenImage / Anima / Z-Image distribution convention).
    pub fn from_safetensors(
        path: &str,
        device: &Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        let weights = load_file(Path::new(path), device)?;
        log::info!(
            "[Wan21VaeEncoder] Loaded {} weight tensors from {}",
            weights.len(),
            path
        );
        let mut enc = Self::from_weights(&weights, device)?;
        enc.set_zero_pad();
        Ok(enc)
    }

    fn from_weights(
        weights: &Weights,
        device: &Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        let z_dim: usize = 16;

        let mean_4d = Tensor::from_vec(
            MEAN.to_vec(),
            Shape::from_dims(&[1, z_dim, 1, 1]),
            device.clone(),
        )?
        .to_dtype(DType::BF16)?;

        let inv_std_vals: Vec<f32> = STD.iter().map(|s| 1.0 / s).collect();
        let inv_std_4d = Tensor::from_vec(
            inv_std_vals,
            Shape::from_dims(&[1, z_dim, 1, 1]),
            device.clone(),
        )?
        .to_dtype(DType::BF16)?;

        let encoder_conv1 = CausalConv3d::load(
            weights, "encoder.conv1", 3, 96, (3, 3, 3), (1, 1, 1), (1, 1, 1), device,
        )?;

        // dim_mult=[1,2,4,4], num_res_blocks=2, temporal_downsample=[F,T,T].
        // Flat blocks 0..=10:
        //   0,1 ResBlock(96,96)         2 Resample 2d
        //   3 ResBlock(96,192) 4 Res(192,192)   5 Resample 3d
        //   6 ResBlock(192,384) 7 Res(384,384)  8 Resample 3d
        //   9,10 ResBlock(384,384)
        let dim_mult: [usize; 4] = [1, 2, 4, 4];
        let enc_dim: usize = 96;
        let dims: Vec<usize> = {
            let mut d = vec![enc_dim];
            for &m in &dim_mult { d.push(enc_dim * m); }
            d
        };
        let temporal_downsample = [false, true, true];

        let mut blocks: Vec<EncoderBlock> = Vec::new();
        let mut idx = 0usize;

        for i in 0..4 {
            let in_dim = dims[i];
            let out_dim = dims[i + 1];

            blocks.push(EncoderBlock::Res(ResidualBlock::load(
                weights, &format!("encoder.downsamples.{idx}"), in_dim, out_dim, device,
            )?));
            idx += 1;

            blocks.push(EncoderBlock::Res(ResidualBlock::load(
                weights, &format!("encoder.downsamples.{idx}"), out_dim, out_dim, device,
            )?));
            idx += 1;

            if i != dim_mult.len() - 1 {
                let t_down = temporal_downsample[i];
                if t_down {
                    blocks.push(EncoderBlock::Downsample(DownResample::load_3d(
                        weights, &format!("encoder.downsamples.{idx}"), out_dim, device,
                    )?));
                } else {
                    blocks.push(EncoderBlock::Downsample(DownResample::load_2d(
                        weights, &format!("encoder.downsamples.{idx}"), out_dim, device,
                    )?));
                }
                idx += 1;
            }
        }

        let top_dim = *dims.last().unwrap();
        let mid_res0 = ResidualBlock::load(weights, "encoder.middle.0", top_dim, top_dim, device)?;
        let mid_attn = AttentionBlock::load(weights, "encoder.middle.1", top_dim, device)?;
        let mid_res1 = ResidualBlock::load(weights, "encoder.middle.2", top_dim, top_dim, device)?;

        let head_norm = RmsNorm5d::load(weights, "encoder.head.0", top_dim)?;
        let head_conv = CausalConv3d::load(
            weights, "encoder.head.2", top_dim, z_dim * 2, (3, 3, 3), (1, 1, 1), (1, 1, 1), device,
        )?;

        let conv1_top = CausalConv3d::load(
            weights, "conv1", z_dim * 2, z_dim * 2, (1, 1, 1), (1, 1, 1), (0, 0, 0), device,
        )?;

        Ok(Self {
            mean_4d,
            inv_std_4d,
            encoder_conv1,
            downsamples: blocks,
            mid_res0,
            mid_attn,
            mid_res1,
            head_norm,
            head_conv,
            conv1_top,
        })
    }

    /// Patch every CausalConv3d to use zero-padding (QwenImage / Anima / Z-Image).
    fn set_zero_pad(&mut self) {
        self.encoder_conv1.zero_pad = true;
        for block in &mut self.downsamples {
            match block {
                EncoderBlock::Res(r) => {
                    r.conv1.zero_pad = true;
                    r.conv2.zero_pad = true;
                    if let Some(ref mut s) = r.shortcut { s.zero_pad = true; }
                }
                EncoderBlock::Downsample(d) => {
                    if let DownResample::Downsample3d { time_conv, .. } = d {
                        time_conv.zero_pad = true;
                    }
                }
            }
        }
        self.mid_res0.conv1.zero_pad = true;
        self.mid_res0.conv2.zero_pad = true;
        self.mid_res1.conv1.zero_pad = true;
        self.mid_res1.conv2.zero_pad = true;
        self.head_conv.zero_pad = true;
        self.conv1_top.zero_pad = true;
    }

    /// Internal 5D forward: `[B,3,T,H,W] -> [B,16,T',H/8,W/8]` raw mu (no norm).
    fn encode_video_raw(&self, video: &Tensor) -> Result<Tensor> {
        let mut x = self.encoder_conv1.forward(video)?;
        for block in &self.downsamples { x = block.forward(&x)?; }
        x = self.mid_res0.forward(&x)?;
        x = self.mid_attn.forward(&x)?;
        x = self.mid_res1.forward(&x)?;
        x = self.head_norm.forward(&x)?;
        x = x.silu()?;
        x = self.head_conv.forward(&x)?;
        let out = self.conv1_top.forward(&x)?;
        out.narrow(1, 0, 16) // mu = first 16 channels of the 32-ch posterior
    }

    /// Encode a 4D image batch `[B,3,H,W]` to raw VAE z `[B,16,H/8,W/8]`.
    /// **No normalization applied** — caller does its own (e.g. Z-Image's
    /// scalar `(z - 0.1159) * 0.3611` shift+scale).
    pub fn encode_image_raw(&self, image: &Tensor) -> Result<Tensor> {
        let dims = image.shape().dims();
        if dims.len() != 4 || dims[1] != 3 {
            return Err(Error::InvalidOperation(format!(
                "Wan21VaeEncoder::encode_image_raw: expected [B,3,H,W], got {:?}",
                dims
            )));
        }
        let b = dims[0];
        let h = dims[2];
        let w = dims[3];
        let video = image.unsqueeze(2)?; // [B,3,1,H,W]
        let z5 = self.encode_video_raw(&video)?; // [B,16,1,h/8,w/8]
        let z5_dims = z5.shape().dims().to_vec();
        let h_lat = z5_dims[3];
        let w_lat = z5_dims[4];
        z5.reshape(&[b, 16, h_lat, w_lat]).map_err(|e| {
            // Defensive: T>1 shouldn't happen for image input, but surface a
            // clear error if some future encoder change breaks the invariant.
            let _ = (h, w);
            e
        })
    }

    /// Encode a 4D image batch `[B,3,H,W]` to per-channel-normalized
    /// `[B,16,H/8,W/8]`. Applies `(z - MEAN[c]) / STD[c]` using the canonical
    /// Wan-VAE constants (matches Anima's expected input distribution).
    pub fn encode_image_normalized(&self, image: &Tensor) -> Result<Tensor> {
        let raw = self.encode_image_raw(image)?;
        let raw_f32 = raw.to_dtype(DType::F32)?;
        let mean_f32 = self.mean_4d.to_dtype(DType::F32)?;
        let inv_std_f32 = self.inv_std_4d.to_dtype(DType::F32)?;
        raw_f32.sub(&mean_f32)?.mul(&inv_std_f32)?.to_dtype(DType::BF16)
    }
}
