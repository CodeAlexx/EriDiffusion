//! Wan 2.1 / Qwen-Image 3D VAE decoder — pure flame_core.
//!
//! Ported from `inference-flame/src/vae/wan21_vae.rs` (decoder side, 2026-05-05).
//! Self-contained: shares no types with `wan21_encoder.rs` (matches the
//! inference-flame source's structural separation).
//!
//! Anima/Z-Image use the **zero-pad** CausalConv3d variant (QwenImage
//! distillation convention), and image-mode decode (`decode_image`) which
//! skips the temporal doubling inside every `upsample3d` block so
//! `T_out == T_in == 1` for single-frame T2I.
//!
//! Architecture (dim=96, z_dim=16, dim_mult=[1,2,4,4]):
//!   conv2 (top-level): CausalConv3d(16, 16, 1x1x1)
//!   decoder.conv1: CausalConv3d(16, 384, 3x3x3)
//!   decoder.middle: ResBlock(384) + AttentionBlock(384) + ResBlock(384)
//!   decoder.upsamples: 15 flat blocks (12 ResBlocks + 3 Upsamples)
//!   decoder.head: RMS_norm(96) + SiLU + CausalConv3d(96, 3, 3x3x3)
//!
//! Public image-mode API:
//!   - [`Wan21VaeDecoder::decode_image_normalized`] — `[B,16,h,w] -> [B,3,H,W]`
//!     (per-channel normalized latent input; unnormalizes internally then
//!     decodes, output clamped to [-1, 1]). Use this when the latent came
//!     from `Wan21VaeEncoder::encode_image_normalized` (Anima's case).
//!   - [`Wan21VaeDecoder::decode_image_raw`] — same shape, but assumes the
//!     input is already raw VAE z (skips the unnormalize step). For Z-Image
//!     style where the trainer applies its own scalar shift+scale outside.

use flame_core::conv::Conv2d;
use flame_core::conv3d_simple::Conv3d;
use flame_core::cuda_ops::GpuOps;
use flame_core::sdpa::forward as sdpa_forward;
use flame_core::serialization::load_file;
use flame_core::{DType, Error, Result, Shape, Tensor};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

type Weights = HashMap<String, Tensor>;

const MEAN: [f32; 16] = [
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
    0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
];

const STD: [f32; 16] = [
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
];

fn get(weights: &Weights, key: &str) -> Result<Tensor> {
    weights
        .get(key)
        .cloned()
        .ok_or_else(|| Error::InvalidOperation(format!("Wan21 VAE decoder: missing weight: {key}")))
}

fn get_bf16(weights: &Weights, key: &str) -> Result<Tensor> {
    get(weights, key)?.to_dtype(DType::BF16)
}

// ---------------------------------------------------------------------------
// CausalConv3d. zero_pad=true matches QwenImage / Anima (Z-Image too).
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
            let pad = if self.zero_pad {
                let dims = x.shape().dims().to_vec();
                let pad_shape = Shape::from_dims(&[dims[0], dims[1], self.time_pad, dims[3], dims[4]]);
                Tensor::zeros_dtype(pad_shape, x.dtype(), x.device().clone())?
            } else {
                let first_frame = x.narrow(2, 0, 1)?;
                first_frame.repeat_axis_device(2, self.time_pad)?
            };
            Tensor::cat(&[&pad, x], 2)?
        } else {
            x.clone()
        };
        let is_bf16 = x_padded.dtype() == DType::BF16;
        let input = if is_bf16 { x_padded.to_dtype(DType::F32)? } else { x_padded };
        let out = self.conv.forward(&input)?;
        if is_bf16 { out.to_dtype(DType::BF16) } else { Ok(out) }
    }
}

// ---------------------------------------------------------------------------
// RMSNorm 5D (channel_first images=False) and 4D (per-frame attn path).
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
            .map_err(|e| Error::InvalidOperation(format!("Wan21 VAE decoder sdpa: {e}")))?;

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
// UpsampleBlock — upsample2d (nearest 2× + Conv2d) or upsample3d
// (CausalConv3d temporal 2× then nearest 2× spatial + Conv2d).
// ---------------------------------------------------------------------------

enum UpsampleBlock {
    Upsample2d { conv: Conv2d },
    Upsample3d { conv: Conv2d, time_conv: CausalConv3d },
}

impl UpsampleBlock {
    fn load_2d(
        weights: &Weights,
        prefix: &str,
        dim: usize,
        device: &Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        let mut conv = Conv2d::new_with_bias(dim, dim / 2, 3, 1, 1, device.clone(), true)?;
        conv.copy_weight_from(&get_bf16(weights, &format!("{prefix}.resample.1.weight"))?)?;
        conv.copy_bias_from(&get_bf16(weights, &format!("{prefix}.resample.1.bias"))?)?;
        Ok(UpsampleBlock::Upsample2d { conv })
    }

    fn load_3d(
        weights: &Weights,
        prefix: &str,
        dim: usize,
        device: &Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        let mut conv = Conv2d::new_with_bias(dim, dim / 2, 3, 1, 1, device.clone(), true)?;
        conv.copy_weight_from(&get_bf16(weights, &format!("{prefix}.resample.1.weight"))?)?;
        conv.copy_bias_from(&get_bf16(weights, &format!("{prefix}.resample.1.bias"))?)?;

        let time_conv = CausalConv3d::load(
            weights, &format!("{prefix}.time_conv"), dim, dim * 2,
            (3, 1, 1), (1, 1, 1), (1, 0, 0), device,
        )?;

        Ok(UpsampleBlock::Upsample3d { conv, time_conv })
    }

    fn forward(&self, x: &Tensor, image_mode: bool) -> Result<Tensor> {
        match self {
            UpsampleBlock::Upsample2d { conv } => {
                let dims = x.shape().dims().to_vec();
                let (b, c, t, h, w) = (dims[0], dims[1], dims[2], dims[3], dims[4]);

                let x_4d = x.permute(&[0, 2, 1, 3, 4])?.reshape(&[b * t, c, h, w])?;
                // upsample2d_nearest kernel is F32-only — same constraint as the encoder.
                let x_f32 = x_4d.to_dtype(DType::F32)?;
                let x_up = GpuOps::upsample2d_nearest(&x_f32, (h * 2, w * 2))?;
                let x_up = x_up.to_dtype(DType::BF16)?;
                let x_conv = conv.forward(&x_up)?;
                let c_out = x_conv.shape().dims()[1];
                x_conv
                    .reshape(&[b, t, c_out, h * 2, w * 2])?
                    .permute(&[0, 2, 1, 3, 4])
            }
            UpsampleBlock::Upsample3d { conv, time_conv } => {
                let dims = x.shape().dims().to_vec();
                let (b, c, t, h, w) = (dims[0], dims[1], dims[2], dims[3], dims[4]);

                // Image-mode (feat_cache=None in diffusers): the temporal
                // doubling inside upsample3d is skipped — this block degenerates
                // to a pure spatial upsample. T_out == T_in.
                let (x_t, t_out) = if image_mode {
                    (x.clone(), t)
                } else {
                    let tc_out = time_conv.forward(x)?; // [B, dim*2, T, H, W]
                    let tc_out = tc_out.reshape(&[b, 2, c, t, h, w])?;
                    let x0 = tc_out.narrow(1, 0, 1)?.squeeze(Some(1))?;
                    let x1 = tc_out.narrow(1, 1, 1)?.squeeze(Some(1))?;
                    let stacked = Tensor::cat(&[
                        &x0.unsqueeze(3)?,
                        &x1.unsqueeze(3)?,
                    ], 3)?;
                    (stacked.reshape(&[b, c, t * 2, h, w])?, t * 2)
                };

                let x_4d = x_t.permute(&[0, 2, 1, 3, 4])?.reshape(&[b * t_out, c, h, w])?;
                let x_f32 = x_4d.to_dtype(DType::F32)?;
                let x_up = GpuOps::upsample2d_nearest(&x_f32, (h * 2, w * 2))?;
                let x_up = x_up.to_dtype(DType::BF16)?;
                let x_conv = conv.forward(&x_up)?;
                let c_out = x_conv.shape().dims()[1];
                x_conv
                    .reshape(&[b, t_out, c_out, h * 2, w * 2])?
                    .permute(&[0, 2, 1, 3, 4])
            }
        }
    }
}

enum DecoderBlock {
    Res(ResidualBlock),
    Upsample(UpsampleBlock),
}

impl DecoderBlock {
    fn forward(&self, x: &Tensor, image_mode: bool) -> Result<Tensor> {
        match self {
            DecoderBlock::Res(r) => r.forward(x),
            DecoderBlock::Upsample(u) => u.forward(x, image_mode),
        }
    }
}

// ---------------------------------------------------------------------------
// Wan21VaeDecoder
// ---------------------------------------------------------------------------

pub struct Wan21VaeDecoder {
    /// Per-channel mean `[1, 16, 1, 1]` BF16 (4D — image use).
    mean_4d: Tensor,
    /// Per-channel std `[1, 16, 1, 1]` BF16 (NOT inv_std — decode multiplies).
    std_4d: Tensor,

    /// Top-level conv2: CausalConv3d(16, 16, 1x1x1).
    conv2_top: CausalConv3d,
    /// decoder.conv1: CausalConv3d(16, 384, 3x3x3).
    decoder_conv1: CausalConv3d,
    mid_res0: ResidualBlock,
    mid_attn: AttentionBlock,
    mid_res1: ResidualBlock,
    upsamples: Vec<DecoderBlock>,
    head_norm: RmsNorm5d,
    head_conv: CausalConv3d,
}

impl Wan21VaeDecoder {
    /// Load `qwen_image_vae.safetensors` (zero-pad CausalConv3d, QwenImage /
    /// Anima / Z-Image distribution).
    pub fn from_safetensors(
        path: &str,
        device: &Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        let weights = load_file(Path::new(path), device)?;
        log::info!(
            "[Wan21VaeDecoder] Loaded {} weight tensors from {}",
            weights.len(),
            path
        );
        let mut dec = Self::from_weights(&weights, device)?;
        dec.set_zero_pad();
        Ok(dec)
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

        let std_4d = Tensor::from_vec(
            STD.to_vec(),
            Shape::from_dims(&[1, z_dim, 1, 1]),
            device.clone(),
        )?
        .to_dtype(DType::BF16)?;

        let conv2_top = CausalConv3d::load(
            weights, "conv2", z_dim, z_dim, (1, 1, 1), (1, 1, 1), (0, 0, 0), device,
        )?;

        let decoder_conv1 = CausalConv3d::load(
            weights, "decoder.conv1", z_dim, 384, (3, 3, 3), (1, 1, 1), (1, 1, 1), device,
        )?;

        let mid_res0 = ResidualBlock::load(weights, "decoder.middle.0", 384, 384, device)?;
        let mid_attn = AttentionBlock::load(weights, "decoder.middle.1", 384, device)?;
        let mid_res1 = ResidualBlock::load(weights, "decoder.middle.2", 384, 384, device)?;

        // Upsample structure: 12 ResBlocks interleaved with 3 Resamples at
        // flat indices 3, 7, 11. dims = [384, 384, 384, 192, 96],
        // temperal_upsample = [False, True, True].
        let block_spec = [
            // i=0: upsamples 0,1,2 (then upsample2d at idx=3)
            (384, 384), (384, 384), (384, 384),
            // i=1: upsamples 4,5,6 (then upsample3d at idx=7)
            (192, 384), (384, 384), (384, 384),
            // i=2: upsamples 8,9,10 (then upsample3d at idx=11)
            (192, 192), (192, 192), (192, 192),
            // i=3: upsamples 12,13,14 (no resample after — final group)
            (96, 96),   (96, 96),   (96, 96),
        ];

        let mut blocks: Vec<DecoderBlock> = Vec::new();
        let mut idx = 0usize;
        for &(in_ch, out_ch) in &block_spec {
            blocks.push(DecoderBlock::Res(ResidualBlock::load(
                weights, &format!("decoder.upsamples.{idx}"), in_ch, out_ch, device,
            )?));
            idx += 1;

            if idx == 3 || idx == 7 || idx == 11 {
                let dim = if idx == 11 { 192 } else { 384 };
                let has_time_conv = weights.contains_key(
                    &format!("decoder.upsamples.{idx}.time_conv.weight"),
                );
                if has_time_conv {
                    blocks.push(DecoderBlock::Upsample(UpsampleBlock::load_3d(
                        weights, &format!("decoder.upsamples.{idx}"), dim, device,
                    )?));
                } else {
                    blocks.push(DecoderBlock::Upsample(UpsampleBlock::load_2d(
                        weights, &format!("decoder.upsamples.{idx}"), dim, device,
                    )?));
                }
                idx += 1;
            }
        }

        let head_norm = RmsNorm5d::load(weights, "decoder.head.0", 96)?;
        let head_conv = CausalConv3d::load(
            weights, "decoder.head.2", 96, 3, (3, 3, 3), (1, 1, 1), (1, 1, 1), device,
        )?;

        Ok(Self {
            mean_4d,
            std_4d,
            conv2_top,
            decoder_conv1,
            mid_res0,
            mid_attn,
            mid_res1,
            upsamples: blocks,
            head_norm,
            head_conv,
        })
    }

    fn set_zero_pad(&mut self) {
        self.conv2_top.zero_pad = true;
        self.decoder_conv1.zero_pad = true;
        self.mid_res0.conv1.zero_pad = true;
        self.mid_res0.conv2.zero_pad = true;
        self.mid_res1.conv1.zero_pad = true;
        self.mid_res1.conv2.zero_pad = true;
        for block in &mut self.upsamples {
            match block {
                DecoderBlock::Res(r) => {
                    r.conv1.zero_pad = true;
                    r.conv2.zero_pad = true;
                    if let Some(ref mut s) = r.shortcut { s.zero_pad = true; }
                }
                DecoderBlock::Upsample(u) => {
                    if let UpsampleBlock::Upsample3d { time_conv, .. } = u {
                        time_conv.zero_pad = true;
                    }
                }
            }
        }
        self.head_conv.zero_pad = true;
    }

    /// 5D forward (raw z, no unnormalize). `image_mode=true` skips temporal
    /// doubling in upsample3d so `T_out == T_in`.
    fn decode_video(&self, z5: &Tensor, image_mode: bool) -> Result<Tensor> {
        let mut x = self.conv2_top.forward(z5)?;
        x = self.decoder_conv1.forward(&x)?;
        x = self.mid_res0.forward(&x)?;
        x = self.mid_attn.forward(&x)?;
        x = self.mid_res1.forward(&x)?;
        for block in &self.upsamples {
            x = block.forward(&x, image_mode)?;
        }
        x = self.head_norm.forward(&x)?;
        x = x.silu()?;
        x = self.head_conv.forward(&x)?;
        x.clamp(-1.0, 1.0)
    }

    /// Decode a 4D image latent `[B,16,h,w]` (per-channel **normalized**) to
    /// RGB `[B,3,H,W]` in `[-1, 1]` BF16. Unnormalizes (`z * STD + MEAN`)
    /// before forward — use this when the latent came out of
    /// `Wan21VaeEncoder::encode_image_normalized` (Anima case).
    pub fn decode_image_normalized(&self, latent: &Tensor) -> Result<Tensor> {
        let dims = latent.shape().dims();
        if dims.len() != 4 || dims[1] != 16 {
            return Err(Error::InvalidOperation(format!(
                "Wan21VaeDecoder::decode_image_normalized: expected [B,16,h,w], got {:?}",
                dims
            )));
        }
        let z_f32 = latent.to_dtype(DType::F32)?;
        let mean_f32 = self.mean_4d.to_dtype(DType::F32)?;
        let std_f32 = self.std_4d.to_dtype(DType::F32)?;
        let raw = z_f32.mul(&std_f32)?.add(&mean_f32)?.to_dtype(DType::BF16)?;
        self.decode_image_raw(&raw)
    }

    /// Decode a 4D **raw** image latent `[B,16,h,w]` (no normalization) to
    /// RGB `[B,3,H,W]` in `[-1, 1]` BF16. Use when the upstream produced raw
    /// VAE z (e.g. Z-Image's scalar shift+scale path is applied externally).
    pub fn decode_image_raw(&self, latent: &Tensor) -> Result<Tensor> {
        let dims = latent.shape().dims();
        if dims.len() != 4 || dims[1] != 16 {
            return Err(Error::InvalidOperation(format!(
                "Wan21VaeDecoder::decode_image_raw: expected [B,16,h,w], got {:?}",
                dims
            )));
        }
        let b = dims[0];
        let z5 = latent.unsqueeze(2)?; // [B,16,1,h,w]
        let img5 = self.decode_video(&z5, true)?; // [B,3,1,H,W]
        let i5_dims = img5.shape().dims().to_vec();
        let h_out = i5_dims[3];
        let w_out = i5_dims[4];
        img5.reshape(&[b, 3, h_out, w_out])
    }
}
