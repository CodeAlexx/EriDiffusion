//! SDXL VAE — 4-channel AutoencoderKL encoder + decoder, pure Rust.
//!
//! Distinct from Flux/Z-Image/SD3 16-ch VAEs. Keep the SDXL-specific
//! constants here so callers don't have to remember scale=0.13025 (which
//! differs from SD 1.5's 0.18215 and from Flux's BFL convention).
//!
//! # Encoder
//! Reuses the generic `LdmVAEEncoder` with `latent_channels=4`. Same down
//! block / mid block layout as every LDM VAE; the channel split at the end
//! takes the first 4 channels (deterministic mean) of the 8-ch output.
//!
//! # Decoder
//! Standalone port from `inference-flame/src/vae/ldm_decoder.rs`, kept in
//! this module rather than a generic `ldm_decoder.rs` because (a) only SDXL
//! needs a 4-ch LDM decoder in ED-v2 right now and (b) SDXL has the
//! `post_quant_conv` 1x1 the Z-Image VAE does not, so the construction
//! path differs.

use std::collections::HashMap;
use std::sync::Arc;

use cudarc::driver::CudaDevice;
use flame_core::conv::Conv2d;
use flame_core::cuda_kernels::CudaKernels;
use flame_core::group_norm::group_norm;
use flame_core::sdpa::forward as sdpa_forward;
use flame_core::serialization::load_file_filtered;
use flame_core::{DType, Error, Result, Shape, Tensor};

use crate::encoders::ldm_vae::LdmVAEEncoder;

// SDXL VAE scale/shift constants. Source: diffusers AutoencoderKL config
// "stabilityai/sdxl-vae" (vae_scale_factor=8, scaling_factor=0.13025,
// shift=0.0). NOT the SD 1.5 0.18215 — using that produces black images
// from a correct LoRA.
pub const SDXL_LATENT_CHANNELS: usize = 4;
pub const SDXL_VAE_SCALE: f32 = 0.13025;
pub const SDXL_VAE_SHIFT: f32 = 0.0;

// ---------------------------------------------------------------------------
// Encoder wrapper
// ---------------------------------------------------------------------------

/// SDXL VAE encoder (4-channel). Just hands off to `LdmVAEEncoder` with
/// SDXL-pinned constants.
pub struct SdxlVaeEncoder {
    inner: LdmVAEEncoder,
}

impl SdxlVaeEncoder {
    pub fn from_safetensors(path: &str, device: &Arc<CudaDevice>) -> Result<Self> {
        Ok(Self {
            inner: LdmVAEEncoder::from_safetensors(path, SDXL_LATENT_CHANNELS, device)?,
        })
    }

    /// Encode `[B, 3, H, W]` pixels in `[-1, 1]` → `[B, 4, H/8, W/8]` scaled latent.
    /// Matches the SDXL `latent = (z_mean - shift) * scale` convention.
    pub fn encode(&self, image: &Tensor) -> Result<Tensor> {
        self.inner.encode_scaled(image, SDXL_VAE_SCALE, SDXL_VAE_SHIFT)
    }
}

// ---------------------------------------------------------------------------
// Decoder
// ---------------------------------------------------------------------------
//
// Layout helpers (NCHW <-> NHWC). The naive `Tensor::permute([0,2,3,1])`
// returns a strided view that GroupNorm/Conv2d kernels misread; use the
// dedicated GPU permute kernels.

fn to_nhwc(x: &Tensor) -> Result<Tensor> {
    flame_core::cuda_ops::GpuOps::permute_nchw_to_nhwc(x)
}

fn to_nchw(x: &Tensor) -> Result<Tensor> {
    flame_core::cuda_ops::GpuOps::permute_nhwc_to_nchw(x)
}

fn group_norm_nchw(
    x: &Tensor,
    num_groups: usize,
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    eps: f32,
) -> Result<Tensor> {
    let nhwc = to_nhwc(x)?;
    let out_nhwc = group_norm(&nhwc, num_groups, weight, bias, eps)?;
    to_nchw(&out_nhwc)
}

fn squeeze_1x1(t: &Tensor) -> Result<Tensor> {
    let dims = t.shape().dims();
    if dims.len() == 4 && dims[2] == 1 && dims[3] == 1 {
        t.reshape(&[dims[0], dims[1]])
    } else {
        t.clone_result()
    }
}

fn linear_3d(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    let (b, n, c) = (dims[0], dims[1], dims[2]);
    let out_features = weight.shape().dims()[0];
    let x_2d = x.reshape(&[b * n, c])?;
    let wt = weight.permute(&[1, 0])?;
    let out_2d = x_2d.matmul(&wt)?;
    let bias_row = bias.reshape(&[1, out_features])?;
    let out_2d = out_2d.add(&bias_row)?;
    out_2d.reshape(&[b, n, out_features])
}

// --- ResBlock ------------------------------------------------------------

struct ResBlock {
    norm1_w: Tensor,
    norm1_b: Tensor,
    conv1: Conv2d,
    norm2_w: Tensor,
    norm2_b: Tensor,
    conv2: Conv2d,
    shortcut: Option<Conv2d>,
}

impl ResBlock {
    fn from_weights(
        w: &HashMap<String, Tensor>,
        prefix: &str,
        in_ch: usize,
        out_ch: usize,
        device: &Arc<CudaDevice>,
    ) -> Result<Self> {
        let get = |key: &str| -> Result<&Tensor> {
            w.get(key).ok_or_else(|| Error::InvalidInput(format!("Missing key: {key}")))
        };
        let mut conv1 = Conv2d::new_with_bias(in_ch, out_ch, 3, 1, 1, device.clone(), true)?;
        conv1.copy_weight_from(get(&format!("{prefix}.conv1.weight"))?)?;
        conv1.copy_bias_from(get(&format!("{prefix}.conv1.bias"))?)?;
        let mut conv2 = Conv2d::new_with_bias(out_ch, out_ch, 3, 1, 1, device.clone(), true)?;
        conv2.copy_weight_from(get(&format!("{prefix}.conv2.weight"))?)?;
        conv2.copy_bias_from(get(&format!("{prefix}.conv2.bias"))?)?;
        let shortcut = if in_ch != out_ch {
            let mut s = Conv2d::new_with_bias(in_ch, out_ch, 1, 1, 0, device.clone(), true)?;
            s.copy_weight_from(get(&format!("{prefix}.nin_shortcut.weight"))?)?;
            s.copy_bias_from(get(&format!("{prefix}.nin_shortcut.bias"))?)?;
            Some(s)
        } else {
            None
        };
        Ok(Self {
            norm1_w: get(&format!("{prefix}.norm1.weight"))?.clone_result()?,
            norm1_b: get(&format!("{prefix}.norm1.bias"))?.clone_result()?,
            conv1,
            norm2_w: get(&format!("{prefix}.norm2.weight"))?.clone_result()?,
            norm2_b: get(&format!("{prefix}.norm2.bias"))?.clone_result()?,
            conv2,
            shortcut,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = group_norm_nchw(x, 32, Some(&self.norm1_w), Some(&self.norm1_b), 1e-6)?;
        let h = h.silu()?;
        let h = self.conv1.forward(&h)?;
        let h = group_norm_nchw(&h, 32, Some(&self.norm2_w), Some(&self.norm2_b), 1e-6)?;
        let h = h.silu()?;
        let h = self.conv2.forward(&h)?;
        let residual = if let Some(ref s) = self.shortcut {
            s.forward(x)?
        } else {
            x.clone_result()?
        };
        residual.add(&h)
    }
}

// --- AttnBlock (1x1 self-attention, single-head, head_dim=512) -----------

struct AttnBlock {
    norm_w: Tensor,
    norm_b: Tensor,
    q_w: Tensor, q_b: Tensor,
    k_w: Tensor, k_b: Tensor,
    v_w: Tensor, v_b: Tensor,
    proj_out_w: Tensor, proj_out_b: Tensor,
}

impl AttnBlock {
    fn from_weights_ldm(w: &HashMap<String, Tensor>, prefix: &str) -> Result<Self> {
        let get = |key: &str| -> Result<&Tensor> {
            w.get(key).ok_or_else(|| Error::InvalidInput(format!("Missing key: {key}")))
        };
        Ok(Self {
            norm_w: get(&format!("{prefix}.norm.weight"))?.clone_result()?,
            norm_b: get(&format!("{prefix}.norm.bias"))?.clone_result()?,
            q_w: squeeze_1x1(get(&format!("{prefix}.q.weight"))?)?,
            q_b: get(&format!("{prefix}.q.bias"))?.clone_result()?,
            k_w: squeeze_1x1(get(&format!("{prefix}.k.weight"))?)?,
            k_b: get(&format!("{prefix}.k.bias"))?.clone_result()?,
            v_w: squeeze_1x1(get(&format!("{prefix}.v.weight"))?)?,
            v_b: get(&format!("{prefix}.v.bias"))?.clone_result()?,
            proj_out_w: squeeze_1x1(get(&format!("{prefix}.proj_out.weight"))?)?,
            proj_out_b: get(&format!("{prefix}.proj_out.bias"))?.clone_result()?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let (b, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);
        let n = h * w;
        let h_norm = group_norm_nchw(x, 32, Some(&self.norm_w), Some(&self.norm_b), 1e-6)?;
        let h_flat = h_norm.permute(&[0, 2, 3, 1])?.reshape(&[b, n, c])?;
        let q = linear_3d(&h_flat, &self.q_w, &self.q_b)?;
        let k = linear_3d(&h_flat, &self.k_w, &self.k_b)?;
        let v = linear_3d(&h_flat, &self.v_w, &self.v_b)?;
        let q = q.unsqueeze(1)?; let k = k.unsqueeze(1)?; let v = v.unsqueeze(1)?;

        // Tiled SDPA — single-head head_dim=512 falls back to the materialized
        // path (head_dim 64/96/128 only hit flash). Without tiling, an
        // [1, 1, N, N] F32 scores matrix at N=16384 is 1 GB.
        const ATTN_TILE: usize = 1024;
        let out = if n <= ATTN_TILE {
            sdpa_forward(&q, &k, &v, None)?
        } else {
            let mut tiles: Vec<Tensor> = Vec::with_capacity(n.div_ceil(ATTN_TILE));
            let mut start = 0;
            while start < n {
                let len = (n - start).min(ATTN_TILE);
                let q_tile = q.narrow(2, start, len)?;
                tiles.push(sdpa_forward(&q_tile, &k, &v, None)?);
                start += len;
            }
            let refs: Vec<&Tensor> = tiles.iter().collect();
            Tensor::cat(&refs, 2)?
        };
        let out = out.squeeze(Some(1))?;
        let out = linear_3d(&out, &self.proj_out_w, &self.proj_out_b)?;
        let out = out.reshape(&[b, h, w, c])?;
        let out = flame_core::cuda_ops::GpuOps::permute_nhwc_to_nchw(&out)?;
        x.add(&out)
    }
}

// --- Mid + Up blocks ------------------------------------------------------

struct MidBlock { resnet0: ResBlock, attn: AttnBlock, resnet1: ResBlock }

impl MidBlock {
    fn from_weights(
        w: &HashMap<String, Tensor>, prefix: &str, channels: usize, device: &Arc<CudaDevice>,
    ) -> Result<Self> {
        Ok(Self {
            resnet0: ResBlock::from_weights(w, &format!("{prefix}.block_1"), channels, channels, device)?,
            attn: AttnBlock::from_weights_ldm(w, &format!("{prefix}.attn_1"))?,
            resnet1: ResBlock::from_weights(w, &format!("{prefix}.block_2"), channels, channels, device)?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.resnet0.forward(x)?;
        let x = self.attn.forward(&x)?;
        self.resnet1.forward(&x)
    }
}

struct UpBlock { resnets: Vec<ResBlock>, upsample_conv: Option<Conv2d> }

impl UpBlock {
    fn from_weights(
        w: &HashMap<String, Tensor>, prefix: &str, in_ch: usize, out_ch: usize,
        num_resnets: usize, has_upsample: bool, device: &Arc<CudaDevice>,
    ) -> Result<Self> {
        let get = |key: &str| -> Result<&Tensor> {
            w.get(key).ok_or_else(|| Error::InvalidInput(format!("Missing key: {key}")))
        };
        let mut resnets = Vec::new();
        let mut ch = in_ch;
        for m in 0..num_resnets {
            resnets.push(ResBlock::from_weights(w, &format!("{prefix}.block.{m}"), ch, out_ch, device)?);
            ch = out_ch;
        }
        let upsample_conv = if has_upsample {
            let mut conv = Conv2d::new_with_bias(out_ch, out_ch, 3, 1, 1, device.clone(), true)?;
            conv.copy_weight_from(get(&format!("{prefix}.upsample.conv.weight"))?)?;
            conv.copy_bias_from(get(&format!("{prefix}.upsample.conv.bias"))?)?;
            Some(conv)
        } else { None };
        Ok(Self { resnets, upsample_conv })
    }
    fn forward(&self, x: &Tensor, kernels: &CudaKernels) -> Result<Tensor> {
        let mut x = x.clone_result()?;
        for r in &self.resnets { x = r.forward(&x)?; }
        if let Some(ref conv) = self.upsample_conv {
            let dims = x.shape().dims();
            let h_out = dims[2] * 2;
            let w_out = dims[3] * 2;
            x = kernels.upsample2d_nearest(&x, (h_out, w_out))?;
            x = conv.forward(&x)?;
        }
        Ok(x)
    }
}

// --- Diffusers→LDM key remap (so SDXL VAEs in either format load) --------

fn remap_diffusers_to_ldm(w: HashMap<String, Tensor>) -> HashMap<String, Tensor> {
    let is_diffusers = w.keys().any(|k| k.contains("mid_block") || k.contains("up_blocks"));
    if !is_diffusers { return w; }
    let mut out = HashMap::with_capacity(w.len());
    for (k, v) in w { out.insert(remap_one(&k), v); }
    out
}

fn remap_one(key: &str) -> String {
    let k = key.to_string();
    if k.starts_with("decoder.conv_norm_out.") {
        return k.replace("decoder.conv_norm_out.", "decoder.norm_out.");
    }
    if k.starts_with("decoder.mid_block.resnets.") {
        let rest = &k["decoder.mid_block.resnets.".len()..];
        if let Some(d) = rest.find('.') {
            let idx: usize = rest[..d].parse().unwrap_or(0);
            let suffix = &rest[d+1..];
            return format!("decoder.mid.block_{}.{suffix}", idx + 1);
        }
    }
    if let Some(s) = k.strip_prefix("decoder.mid_block.attentions.0.group_norm.") {
        return format!("decoder.mid.attn_1.norm.{s}");
    }
    if let Some(s) = k.strip_prefix("decoder.mid_block.attentions.0.to_q.") {
        return format!("decoder.mid.attn_1.q.{s}");
    }
    if let Some(s) = k.strip_prefix("decoder.mid_block.attentions.0.to_k.") {
        return format!("decoder.mid.attn_1.k.{s}");
    }
    if let Some(s) = k.strip_prefix("decoder.mid_block.attentions.0.to_v.") {
        return format!("decoder.mid.attn_1.v.{s}");
    }
    if let Some(s) = k.strip_prefix("decoder.mid_block.attentions.0.to_out.0.") {
        return format!("decoder.mid.attn_1.proj_out.{s}");
    }
    if k.starts_with("decoder.up_blocks.") {
        let rest = &k["decoder.up_blocks.".len()..];
        if let Some(d) = rest.find('.') {
            let diff_idx: usize = rest[..d].parse().unwrap_or(0);
            let ldm_idx = 3 - diff_idx;
            let inner = &rest[d+1..];
            if inner.starts_with("resnets.") {
                let rr = &inner["resnets.".len()..];
                if let Some(d2) = rr.find('.') {
                    let ridx = &rr[..d2];
                    let suf = &rr[d2+1..];
                    let suf = suf.replace("conv_shortcut.", "nin_shortcut.");
                    return format!("decoder.up.{ldm_idx}.block.{ridx}.{suf}");
                }
            }
            if let Some(s) = inner.strip_prefix("upsamplers.0.conv.") {
                return format!("decoder.up.{ldm_idx}.upsample.conv.{s}");
            }
        }
    }
    k
}

// --- Full decoder ---------------------------------------------------------

/// SDXL VAE decoder (4-channel). Standalone — does NOT share state with
/// `inference-flame`. Auto-detects diffusers-format weights and remaps to
/// LDM-format internally. Includes the SDXL `post_quant_conv`.
pub struct SdxlVaeDecoder {
    post_quant_conv: Option<Conv2d>,
    conv_in: Conv2d,
    mid_block: MidBlock,
    up_blocks: Vec<UpBlock>, // processing order: up.3 first, up.0 last
    norm_out_w: Tensor,
    norm_out_b: Tensor,
    conv_out: Conv2d,
    kernels: CudaKernels,
}

impl SdxlVaeDecoder {
    pub fn from_safetensors(path: &str, device: &Arc<CudaDevice>) -> Result<Self> {
        let raw = load_file_filtered(path, device, |k| {
            k.starts_with("decoder.")
                || k.starts_with("first_stage_model.decoder.")
                || k == "post_quant_conv.weight"
                || k == "post_quant_conv.bias"
                || k == "first_stage_model.post_quant_conv.weight"
                || k == "first_stage_model.post_quant_conv.bias"
        })?;
        let fsm = "first_stage_model.";
        let mut w = HashMap::with_capacity(raw.len());
        for (key, val) in raw {
            let k = key.strip_prefix(fsm).unwrap_or(&key).to_string();
            let val = if val.dtype() == DType::BF16 { val } else { val.to_dtype(DType::BF16)? };
            w.insert(k, val);
        }
        let w = remap_diffusers_to_ldm(w);
        Self::from_weights(w, device)
    }

    pub fn from_weights(w: HashMap<String, Tensor>, device: &Arc<CudaDevice>) -> Result<Self> {
        let get = |key: &str| -> Result<&Tensor> {
            w.get(key).ok_or_else(|| Error::InvalidInput(format!("Missing key: {key}")))
        };

        let ch: usize = 128;
        let ch_mult: [usize; 4] = [1, 2, 4, 4];
        let num_resnets: usize = 3; // layers_per_block + 1
        let top_ch = ch * ch_mult[3]; // 512

        // post_quant_conv (SDXL has it; Z-Image-style VAEs do not).
        let post_quant_conv = if w.contains_key("post_quant_conv.weight") {
            let mut c = Conv2d::new_with_bias(
                SDXL_LATENT_CHANNELS, SDXL_LATENT_CHANNELS, 1, 1, 0, device.clone(), true,
            )?;
            c.copy_weight_from(get("post_quant_conv.weight")?)?;
            c.copy_bias_from(get("post_quant_conv.bias")?)?;
            Some(c)
        } else { None };

        let mut conv_in = Conv2d::new_with_bias(SDXL_LATENT_CHANNELS, top_ch, 3, 1, 1, device.clone(), true)?;
        conv_in.copy_weight_from(get("decoder.conv_in.weight")?)?;
        conv_in.copy_bias_from(get("decoder.conv_in.bias")?)?;

        let mid_block = MidBlock::from_weights(&w, "decoder.mid", top_ch, device)?;

        let mut up_blocks = Vec::new();
        let mut prev_ch = top_ch;
        for ldm_idx in [3usize, 2, 1, 0] {
            let out_ch = ch * ch_mult[ldm_idx];
            let has_up = ldm_idx > 0;
            up_blocks.push(UpBlock::from_weights(
                &w, &format!("decoder.up.{ldm_idx}"), prev_ch, out_ch, num_resnets, has_up, device,
            )?);
            prev_ch = out_ch;
        }

        let mut conv_out = Conv2d::new_with_bias(ch, 3, 3, 1, 1, device.clone(), true)?;
        conv_out.copy_weight_from(get("decoder.conv_out.weight")?)?;
        conv_out.copy_bias_from(get("decoder.conv_out.bias")?)?;

        Ok(Self {
            post_quant_conv,
            conv_in,
            mid_block,
            up_blocks,
            norm_out_w: get("decoder.norm_out.weight")?.clone_result()?,
            norm_out_b: get("decoder.norm_out.bias")?.clone_result()?,
            conv_out,
            kernels: CudaKernels::new(device.clone())?,
        })
    }

    /// Decode SDXL latent `[B, 4, H, W]` → RGB `[B, 3, H*8, W*8]`.
    /// Undoes the `(z - shift) * scale` encode normalization first.
    pub fn decode(&self, z: &Tensor) -> Result<Tensor> {
        // Undo SDXL latent normalization: z = z / scale + shift
        let z = z.mul_scalar(1.0 / SDXL_VAE_SCALE)?.add_scalar(SDXL_VAE_SHIFT)?;
        let z = if let Some(ref pqc) = self.post_quant_conv {
            pqc.forward(&z)?
        } else {
            z
        };
        let mut h = self.conv_in.forward(&z)?;
        h = self.mid_block.forward(&h)?;
        for block in &self.up_blocks {
            h = block.forward(&h, &self.kernels)?;
        }
        h = group_norm_nchw(&h, 32, Some(&self.norm_out_w), Some(&self.norm_out_b), 1e-6)?;
        h = h.silu()?;
        self.conv_out.forward(&h)
    }
}

// Silence unused-shape import in some toolchains.
#[allow(dead_code)]
fn _shape_marker(_: Shape) {}
