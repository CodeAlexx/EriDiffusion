//! Z-Image VAE decoder — pure Rust, LDM-format weight keys.
//!
//! Architecture: Standard LDM AutoencoderKL decoder with 16 latent channels.
//! - block_out_channels = (128, 256, 512, 512)
//! - 3 resnets per up block (layers_per_block + 1)
//! - Mid block: ResBlock + SelfAttention + ResBlock
//! - No post_quant_conv (Z-Image disables it)
//! - Encode normalization: z_scaled = (z_raw - 0.1159) * 0.3611
//! - Decode therefore inverts to: z_raw = z_scaled / 0.3611 + 0.1159
//!   (BUG FIX 2026-05-05: previously applied the encode formula here,
//!   which double-distorted the latent and produced gray-stripe garbage.)
//!
//! LDM key format:
//!   decoder.conv_in.weight/bias
//!   decoder.mid.block_{1,2}.norm1/conv1/norm2/conv2.weight/bias
//!   decoder.mid.attn_1.norm/q/k/v/proj_out.weight/bias  (Conv2d 1x1)
//!   decoder.up.{0-3}.block.{0-2}.norm1/conv1/norm2/conv2.weight/bias
//!   decoder.up.{n}.block.{m}.nin_shortcut.weight/bias
//!   decoder.up.{1-3}.upsample.conv.weight/bias
//!   decoder.norm_out.weight/bias
//!   decoder.conv_out.weight/bias
//!
//! Up block ordering is REVERSED from processing order:
//!   up.3 (512->512, has upsample) processed FIRST
//!   up.0 (256->128, no upsample) processed LAST

use flame_core::sdpa::forward as sdpa_forward;
use flame_core::conv::Conv2d;
use flame_core::cuda_kernels::CudaKernels;
use flame_core::group_norm::group_norm;
use flame_core::serialization::load_file_filtered;
use flame_core::{DType, Error, Result, Shape, Tensor};
use std::collections::HashMap;
use std::sync::Arc;

const SCALING_FACTOR: f32 = 0.3611;
const SHIFT_FACTOR: f32 = 0.1159;

// ---------------------------------------------------------------------------
// Layout helpers — GroupNorm wants NHWC, Conv2d wants NCHW
// ---------------------------------------------------------------------------

/// NCHW → NHWC. `Tensor::permute` returns a strided VIEW, not a
/// materialized tensor — feeding that view to `group_norm`/`Conv2d`
/// gives silent garbage because those readers walk storage as if
/// contiguous. The fix (cribbed from inference-flame's working
/// `LdmVAEDecoder`) is the dedicated GPU kernel that scatters to a
/// real contiguous NHWC buffer. Without it the VAE turned even pure
/// noise into the gray-stripe pattern seen 2026-05-04 — same bug
/// class as flame Conv3d needing `.contiguous()` after `Tensor::cat`.
fn to_nhwc(x: &Tensor) -> Result<Tensor> {
    flame_core::cuda_ops::GpuOps::permute_nchw_to_nhwc(x)
}

/// NHWC → NCHW. Same rationale as `to_nhwc`.
fn to_nchw(x: &Tensor) -> Result<Tensor> {
    flame_core::cuda_ops::GpuOps::permute_nhwc_to_nchw(x)
}

/// GroupNorm on NCHW tensor (converts to NHWC internally, converts back)
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

// ---------------------------------------------------------------------------
// ResBlock
// ---------------------------------------------------------------------------

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
        device: &Arc<cudarc::driver::CudaDevice>,
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
            norm1_w: get(&format!("{prefix}.norm1.weight"))?.to_dtype(flame_core::DType::BF16)?,
            norm1_b: get(&format!("{prefix}.norm1.bias"))?.to_dtype(flame_core::DType::BF16)?,
            conv1,
            norm2_w: get(&format!("{prefix}.norm2.weight"))?.to_dtype(flame_core::DType::BF16)?,
            norm2_b: get(&format!("{prefix}.norm2.bias"))?.to_dtype(flame_core::DType::BF16)?,
            conv2,
            shortcut,
        })
    }

    /// Forward: GroupNorm→SiLU→Conv→GroupNorm→SiLU→Conv + residual
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

// ---------------------------------------------------------------------------
// Attention block (Conv2d 1x1 self-attention)
// ---------------------------------------------------------------------------

struct AttnBlock {
    norm_w: Tensor,
    norm_b: Tensor,
    /// Q/K/V/proj_out weights squeezed from [C,C,1,1] to [C,C] for matmul
    q_w: Tensor,
    q_b: Tensor,
    k_w: Tensor,
    k_b: Tensor,
    v_w: Tensor,
    v_b: Tensor,
    proj_out_w: Tensor,
    proj_out_b: Tensor,
    channels: usize,
}

/// Squeeze Conv2d 1x1 weight [out, in, 1, 1] → [out, in]
fn squeeze_1x1(t: &Tensor) -> Result<Tensor> {
    let dims = t.shape().dims();
    if dims.len() == 4 && dims[2] == 1 && dims[3] == 1 {
        t.reshape(&[dims[0], dims[1]])
    } else {
        t.clone_result()
    }
}

/// 3D linear: [B, N, C] @ W^T + bias → [B, N, out]
fn linear_3d(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    let (b, n, c) = (dims[0], dims[1], dims[2]);
    let out_features = weight.shape().dims()[0];
    let x_2d = x.reshape(&[b * n, c])?;
    let wt = weight.permute(&[1, 0])?; // transpose
    let out_2d = x_2d.matmul(&wt)?;
    // broadcast add bias
    let bias_row = bias.reshape(&[1, out_features])?;
    let out_2d = out_2d.add(&bias_row)?;
    out_2d.reshape(&[b, n, out_features])
}

impl AttnBlock {
    fn from_weights(
        w: &HashMap<String, Tensor>,
        prefix: &str,
        channels: usize,
    ) -> Result<Self> {
        let get = |key: &str| -> Result<&Tensor> {
            w.get(key).ok_or_else(|| Error::InvalidInput(format!("Missing key: {key}")))
        };

        Ok(Self {
            norm_w: get(&format!("{prefix}.norm.weight"))?.to_dtype(flame_core::DType::BF16)?,
            norm_b: get(&format!("{prefix}.norm.bias"))?.to_dtype(flame_core::DType::BF16)?,
            q_w: squeeze_1x1(get(&format!("{prefix}.q.weight"))?)?.to_dtype(flame_core::DType::BF16)?,
            q_b: get(&format!("{prefix}.q.bias"))?.to_dtype(flame_core::DType::BF16)?,
            k_w: squeeze_1x1(get(&format!("{prefix}.k.weight"))?)?.to_dtype(flame_core::DType::BF16)?,
            k_b: get(&format!("{prefix}.k.bias"))?.to_dtype(flame_core::DType::BF16)?,
            v_w: squeeze_1x1(get(&format!("{prefix}.v.weight"))?)?.to_dtype(flame_core::DType::BF16)?,
            v_b: get(&format!("{prefix}.v.bias"))?.to_dtype(flame_core::DType::BF16)?,
            proj_out_w: squeeze_1x1(get(&format!("{prefix}.proj_out.weight"))?)?.to_dtype(flame_core::DType::BF16)?,
            proj_out_b: get(&format!("{prefix}.proj_out.bias"))?.to_dtype(flame_core::DType::BF16)?,
            channels,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let (b, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);
        let n = h * w;

        // GroupNorm
        let h_norm = group_norm_nchw(x, 32, Some(&self.norm_w), Some(&self.norm_b), 1e-6)?;

        // [B, C, H, W] → [B, H*W, C]
        let h_flat = h_norm.permute(&[0, 2, 3, 1])?.reshape(&[b, n, c])?;

        // Q, K, V projections
        let q = linear_3d(&h_flat, &self.q_w, &self.q_b)?;
        let k = linear_3d(&h_flat, &self.k_w, &self.k_b)?;
        let v = linear_3d(&h_flat, &self.v_w, &self.v_b)?;

        // SDPA expects [B, H, N, D] — use 1 head with D=C
        let q = q.unsqueeze(1)?; // [B, 1, N, C]
        let k = k.unsqueeze(1)?;
        let v = v.unsqueeze(1)?;

        let out = sdpa_forward(&q, &k, &v, None)?; // [B, 1, N, C]
        let out = out.squeeze(Some(1))?; // [B, N, C]

        // Output projection
        let out = linear_3d(&out, &self.proj_out_w, &self.proj_out_b)?;

        // [B, N, C] → [B, C, H, W]. permute() returns a strided view; use the
        // dedicated NHWC→NCHW kernel so the residual `x + out` feeds `add`
        // contiguous memory (same fix as `to_nhwc`/`to_nchw`).
        let out = out.reshape(&[b, h, w, c])?;
        let out = flame_core::cuda_ops::GpuOps::permute_nhwc_to_nchw(&out)?;

        // Residual
        x.add(&out)
    }
}

// ---------------------------------------------------------------------------
// Mid block
// ---------------------------------------------------------------------------

struct MidBlock {
    resnet0: ResBlock,
    attn: AttnBlock,
    resnet1: ResBlock,
}

impl MidBlock {
    fn from_weights(
        w: &HashMap<String, Tensor>,
        prefix: &str,
        channels: usize,
        device: &Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        Ok(Self {
            resnet0: ResBlock::from_weights(w, &format!("{prefix}.block_1"), channels, channels, device)?,
            attn: AttnBlock::from_weights(w, &format!("{prefix}.attn_1"), channels)?,
            resnet1: ResBlock::from_weights(w, &format!("{prefix}.block_2"), channels, channels, device)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.resnet0.forward(x)?;
        let x = self.attn.forward(&x)?;
        self.resnet1.forward(&x)
    }
}

// ---------------------------------------------------------------------------
// Up block
// ---------------------------------------------------------------------------

struct UpBlock {
    resnets: Vec<ResBlock>,
    upsample_conv: Option<Conv2d>,
}

impl UpBlock {
    fn from_weights(
        w: &HashMap<String, Tensor>,
        prefix: &str,
        in_ch: usize,
        out_ch: usize,
        num_resnets: usize,
        has_upsample: bool,
        device: &Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        let get = |key: &str| -> Result<&Tensor> {
            w.get(key).ok_or_else(|| Error::InvalidInput(format!("Missing key: {key}")))
        };

        let mut resnets = Vec::new();
        let mut ch = in_ch;
        for m in 0..num_resnets {
            resnets.push(ResBlock::from_weights(
                w,
                &format!("{prefix}.block.{m}"),
                ch,
                out_ch,
                device,
            )?);
            ch = out_ch;
        }

        let upsample_conv = if has_upsample {
            let mut conv = Conv2d::new_with_bias(out_ch, out_ch, 3, 1, 1, device.clone(), true)?;
            conv.copy_weight_from(get(&format!("{prefix}.upsample.conv.weight"))?)?;
            conv.copy_bias_from(get(&format!("{prefix}.upsample.conv.bias"))?)?;
            Some(conv)
        } else {
            None
        };

        Ok(Self {
            resnets,
            upsample_conv,
        })
    }

    fn forward(&self, x: &Tensor, kernels: &CudaKernels) -> Result<Tensor> {
        let mut x = x.clone_result()?;
        for resnet in &self.resnets {
            x = resnet.forward(&x)?;
        }
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

// ---------------------------------------------------------------------------
// Full Z-Image VAE Decoder
// ---------------------------------------------------------------------------

pub struct ZImageVAEDecoder {
    conv_in: Conv2d,
    mid_block: MidBlock,
    up_blocks: Vec<UpBlock>, // in processing order: up.3, up.2, up.1, up.0
    norm_out_w: Tensor,
    norm_out_b: Tensor,
    conv_out: Conv2d,
    kernels: CudaKernels,
}

/// Remap diffusers-format VAE keys to the LDM-format that this decoder
/// expects. Z-Image's published VAE
/// (`zimage_base/vae/diffusion_pytorch_model.safetensors`) uses the diffusers
/// AutoencoderKL layout (`mid_block.resnets.N`, `up_blocks.N`, `conv_norm_out`,
/// etc.); the legacy LDM `ae.safetensors` uses the historical layout
/// (`mid.block_{1,2}`, `up.{0..3}`, `norm_out`). Without this remap, passing
/// the published Z-Image VAE silently fails (no key matches the loader's
/// `decoder.norm_out`-style lookups, falling through to whatever was there last
/// — i.e. uninitialized weights → gray-stripe garbage).
fn remap_diffusers_to_ldm(w: HashMap<String, Tensor>) -> HashMap<String, Tensor> {
    let is_diffusers = w.keys().any(|k| k.contains("mid_block") || k.contains("up_blocks"));
    if !is_diffusers {
        return w;
    }
    println!("[ZImageVAE] Remapping diffusers keys to LDM format");
    let mut out = HashMap::with_capacity(w.len());
    for (key, val) in w {
        out.insert(remap_one_key(&key), val);
    }
    out
}

fn remap_one_key(key: &str) -> String {
    let k = key.to_string();

    if k.starts_with("decoder.conv_norm_out.") {
        return k.replace("decoder.conv_norm_out.", "decoder.norm_out.");
    }
    if k.starts_with("decoder.mid_block.resnets.") {
        let rest = &k["decoder.mid_block.resnets.".len()..];
        if let Some(dot) = rest.find('.') {
            let idx: usize = rest[..dot].parse().unwrap_or(0);
            let suffix = &rest[dot + 1..];
            return format!("decoder.mid.block_{}.{suffix}", idx + 1);
        }
    }
    if k.starts_with("decoder.mid_block.attentions.0.group_norm.") {
        let suffix = &k["decoder.mid_block.attentions.0.group_norm.".len()..];
        return format!("decoder.mid.attn_1.norm.{suffix}");
    }
    if k.starts_with("decoder.mid_block.attentions.0.to_q.") {
        let suffix = &k["decoder.mid_block.attentions.0.to_q.".len()..];
        return format!("decoder.mid.attn_1.q.{suffix}");
    }
    if k.starts_with("decoder.mid_block.attentions.0.to_k.") {
        let suffix = &k["decoder.mid_block.attentions.0.to_k.".len()..];
        return format!("decoder.mid.attn_1.k.{suffix}");
    }
    if k.starts_with("decoder.mid_block.attentions.0.to_v.") {
        let suffix = &k["decoder.mid_block.attentions.0.to_v.".len()..];
        return format!("decoder.mid.attn_1.v.{suffix}");
    }
    if k.starts_with("decoder.mid_block.attentions.0.to_out.0.") {
        let suffix = &k["decoder.mid_block.attentions.0.to_out.0.".len()..];
        return format!("decoder.mid.attn_1.proj_out.{suffix}");
    }
    if k.starts_with("decoder.up_blocks.") {
        let rest = &k["decoder.up_blocks.".len()..];
        if let Some(dot) = rest.find('.') {
            let diff_idx: usize = rest[..dot].parse().unwrap_or(0);
            let ldm_idx = 3 - diff_idx;
            let block_idx = ldm_idx.to_string();
            let inner = &rest[dot + 1..];
            if inner.starts_with("resnets.") {
                let rr = &inner["resnets.".len()..];
                if let Some(dot2) = rr.find('.') {
                    let resnet_idx = &rr[..dot2];
                    let suffix = &rr[dot2 + 1..];
                    let suffix = suffix.replace("conv_shortcut.", "nin_shortcut.");
                    return format!("decoder.up.{block_idx}.block.{resnet_idx}.{suffix}");
                }
            }
            if inner.starts_with("upsamplers.0.conv.") {
                let suffix = &inner["upsamplers.0.conv.".len()..];
                return format!("decoder.up.{block_idx}.upsample.conv.{suffix}");
            }
        }
    }
    k
}

impl ZImageVAEDecoder {
    /// Load decoder from safetensors file (mmap, decoder keys only).
    ///
    /// Accepts both the legacy LDM-format keys (`decoder.mid.block_1.*`,
    /// `decoder.up.0.*`, `decoder.norm_out.*`) and the diffusers-format keys
    /// shipped with the official Z-Image VAE
    /// (`decoder.mid_block.resnets.0.*`, `decoder.up_blocks.0.*`,
    /// `decoder.conv_norm_out.*`).
    pub fn from_safetensors(
        path: &str,
        device: &Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        let w = load_file_filtered(path, device, |key| {
            key.starts_with("decoder.") || key == "decoder.conv_in.weight"
        })?;
        println!("[ZImageVAE] Loaded {} decoder weight tensors", w.len());
        let w = remap_diffusers_to_ldm(w);
        Self::from_weights(w, device)
    }

    /// Build from a pre-loaded weight HashMap.
    pub fn from_weights(
        w: HashMap<String, Tensor>,
        device: &Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        let get = |key: &str| -> Result<&Tensor> {
            w.get(key).ok_or_else(|| Error::InvalidInput(format!("Missing key: {key}")))
        };

        let ch: usize = 128;
        let ch_mult: [usize; 4] = [1, 2, 4, 4];
        let num_resnets: usize = 3; // layers_per_block + 1

        let top_ch = ch * ch_mult[3]; // 512

        // conv_in: 16ch → 512ch
        let mut conv_in = Conv2d::new_with_bias(16, top_ch, 3, 1, 1, device.clone(), true)?;
        conv_in.copy_weight_from(get("decoder.conv_in.weight")?)?;
        conv_in.copy_bias_from(get("decoder.conv_in.bias")?)?;

        // mid block
        let mid_block = MidBlock::from_weights(&w, "decoder.mid", top_ch, device)?;

        // Up blocks — process 3→2→1→0
        let mut up_blocks = Vec::new();
        let mut prev_ch = top_ch;
        for ldm_idx in [3usize, 2, 1, 0] {
            let out_ch = ch * ch_mult[ldm_idx];
            let has_up = ldm_idx > 0;
            up_blocks.push(UpBlock::from_weights(
                &w,
                &format!("decoder.up.{ldm_idx}"),
                prev_ch,
                out_ch,
                num_resnets,
                has_up,
                device,
            )?);
            prev_ch = out_ch;
        }

        // norm_out + conv_out
        let mut conv_out = Conv2d::new_with_bias(ch, 3, 3, 1, 1, device.clone(), true)?;
        conv_out.copy_weight_from(get("decoder.conv_out.weight")?)?;
        conv_out.copy_bias_from(get("decoder.conv_out.bias")?)?;

        let kernels = CudaKernels::new(device.clone())?;

        Ok(Self {
            conv_in,
            mid_block,
            up_blocks,
            norm_out_w: get("decoder.norm_out.weight")?.to_dtype(flame_core::DType::BF16)?,
            norm_out_b: get("decoder.norm_out.bias")?.to_dtype(flame_core::DType::BF16)?,
            conv_out,
            kernels,
        })
    }

    /// Decode latents to RGB.
    ///
    /// Input: `[B, 16, H, W]` SCALED latent tensor (BF16). The decoder undoes
    /// the encoder's `(z - shift) * scale` normalization internally, so callers
    /// should pass the trainer/sampler's scaled latent directly. (Inference-flame
    /// LdmVAEDecoder behaves the same way.)
    /// Output: `[B, 3, H*8, W*8]` RGB tensor.
    pub fn decode(&self, z: &Tensor) -> Result<Tensor> {
        // Undo VAE encode-time normalization: encode is z_scaled = (z_raw - shift) * scale,
        // so decode inverts to z_raw = z_scaled / scale + shift.
        let z = z.mul_scalar(1.0 / SCALING_FACTOR)?.add_scalar(SHIFT_FACTOR)?;

        // decoder.conv_in
        let mut h = self.conv_in.forward(&z)?;

        // mid block
        h = self.mid_block.forward(&h)?;

        // up blocks (processed in order: up.3 → up.2 → up.1 → up.0)
        for block in &self.up_blocks {
            h = block.forward(&h, &self.kernels)?;
        }

        // final norm + silu + conv
        h = group_norm_nchw(&h, 32, Some(&self.norm_out_w), Some(&self.norm_out_b), 1e-6)?;
        h = h.silu()?;
        h = self.conv_out.forward(&h)?;

        Ok(h)
    }
}
