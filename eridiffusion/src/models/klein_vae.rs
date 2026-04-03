//! Flux 2 / Klein VAE decoder — correct architecture from Python+Flame blueprint.
//!
//! This is a clean rewrite fixing 10 critical bugs in the old Rust VAE:
//!   1. Wrong latent channels (was 16, should be 32)
//!   2. Wrong post_quant_conv channels (was 16→16, should be 32→32)
//!   3. Phantom attention in up_blocks (only mid_block has attention)
//!   4. Only 2 ResBlocks per up_block (should be 3 = num_res_blocks + 1)
//!   5. Mid-block attention used Conv2d Q/K/V (should be Linear — diffusers style)
//!   6. Wrong up_block channel progression
//!   7. Missing conv_shortcut on channel-changing ResBlocks
//!   8. Wrong GroupNorm eps (was 1e-5, should be 1e-6)
//!   9. Missing unpatchify (128ch → 32ch reverse pixel shuffle)
//!  10. Wrong upsample ordering
//!
//! Architecture (from diffusers Flux 2 VAE, ch=128, ch_mult=(1,2,4,4)):
//!   post_quant_conv: Conv2d(32, 32, 1)
//!   conv_in: Conv2d(32, 512, 3, pad=1)
//!   mid_block: ResBlock(512) + Attention(512) + ResBlock(512)
//!   up_blocks.0: 3x ResBlock(512→512) + Upsample
//!   up_blocks.1: 3x ResBlock(512→512) + Upsample
//!   up_blocks.2: 3x ResBlock(512→256) + Upsample
//!   up_blocks.3: 3x ResBlock(256→128) — NO upsample
//!   conv_norm_out: GroupNorm(32, 128)
//!   conv_out: Conv2d(128, 3, 3, pad=1)
//!
//! Weight key format: diffusers-style (decoder.up_blocks.{n}.resnets.{m}.*, etc.)

use crate::ops::{Conv2d, GroupNorm, Linear};
use flame_core::device::Device;
use flame_core::{CudaDevice, DType, Error, Result, Tensor};
#[cfg(test)]
use flame_core::Shape;
use std::collections::HashMap;
use std::sync::Arc;

type CudaArc = Arc<CudaDevice>;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const CH: usize = 128;
const CH_MULT: [usize; 4] = [1, 2, 4, 4];
const NUM_RES_BLOCKS: usize = 2;
const LATENT_CH: usize = 32;
const NORM_EPS: f32 = 1e-6;
const NORM_GROUPS: usize = 32;

// ---------------------------------------------------------------------------
// Weight loading helpers
// ---------------------------------------------------------------------------

fn get_weight(weights: &HashMap<String, Tensor>, key: &str) -> Result<Tensor> {
    weights
        .get(key)
        .cloned()
        .ok_or_else(|| Error::InvalidOperation(format!("Missing weight: {key}")))
}

fn load_conv(
    weights: &HashMap<String, Tensor>,
    prefix: &str,
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    device: &CudaArc,
) -> Result<Conv2d> {
    let mut conv =
        Conv2d::new_with_bias_zeroed(in_ch, out_ch, kernel, stride, padding, device.clone(), true)?;
    let w = get_weight(weights, &format!("{prefix}.weight"))?.to_dtype(DType::BF16)?;
    let b = get_weight(weights, &format!("{prefix}.bias"))?.to_dtype(DType::BF16)?;
    conv.copy_weight_from(&w)?;
    conv.copy_bias_from(&b)?;
    Ok(conv)
}

fn load_gn(
    weights: &HashMap<String, Tensor>,
    prefix: &str,
    channels: usize,
    device: &CudaArc,
) -> Result<GroupNorm> {
    let mut gn = GroupNorm::new(NORM_GROUPS, channels, NORM_EPS, true, DType::BF16, device.clone())?;
    let w = get_weight(weights, &format!("{prefix}.weight"))?.to_dtype(DType::BF16)?;
    let b = get_weight(weights, &format!("{prefix}.bias"))?.to_dtype(DType::BF16)?;
    gn.weight = Some(w);
    gn.bias = Some(b);
    Ok(gn)
}

fn load_linear(
    weights: &HashMap<String, Tensor>,
    prefix: &str,
    in_features: usize,
    out_features: usize,
    device: &CudaArc,
) -> Result<Linear> {
    let mut lin = Linear::new_zeroed(in_features, out_features, true, device)?;
    let w = get_weight(weights, &format!("{prefix}.weight"))?.to_dtype(DType::BF16)?;
    let b = get_weight(weights, &format!("{prefix}.bias"))?.to_dtype(DType::BF16)?;
    lin.copy_weight_from(&w)?;
    lin.copy_bias_from(&b)?;
    Ok(lin)
}

// ---------------------------------------------------------------------------
// ResBlock
// ---------------------------------------------------------------------------

/// Residual block: GroupNorm → SiLU → Conv3x3 → GroupNorm → SiLU → Conv3x3 + skip.
///
/// Weight keys: `{prefix}.norm1/conv1/norm2/conv2.*`
/// Optional: `{prefix}.conv_shortcut.*` when in_ch != out_ch.
struct ResBlock {
    norm1: GroupNorm,
    conv1: Conv2d,
    norm2: GroupNorm,
    conv2: Conv2d,
    conv_shortcut: Option<Conv2d>,
}

impl ResBlock {
    fn load(
        weights: &HashMap<String, Tensor>,
        prefix: &str,
        in_ch: usize,
        out_ch: usize,
        device: &CudaArc,
    ) -> Result<Self> {
        let norm1 = load_gn(weights, &format!("{prefix}.norm1"), in_ch, device)?;
        let conv1 = load_conv(weights, &format!("{prefix}.conv1"), in_ch, out_ch, 3, 1, 1, device)?;
        let norm2 = load_gn(weights, &format!("{prefix}.norm2"), out_ch, device)?;
        let conv2 = load_conv(weights, &format!("{prefix}.conv2"), out_ch, out_ch, 3, 1, 1, device)?;

        let shortcut_key = format!("{prefix}.conv_shortcut.weight");
        let conv_shortcut = if weights.contains_key(&shortcut_key) {
            Some(load_conv(
                weights,
                &format!("{prefix}.conv_shortcut"),
                in_ch,
                out_ch,
                1,
                1,
                0,
                device,
            )?)
        } else {
            None
        };

        Ok(Self { norm1, conv1, norm2, conv2, conv_shortcut })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.norm1.forward_nchw(x)?;
        let h = h.silu()?;
        let h = self.conv1.forward(&h)?;
        let h = self.norm2.forward_nchw(&h)?;
        let h = h.silu()?;
        let h = self.conv2.forward(&h)?;

        let skip = if let Some(cs) = &self.conv_shortcut {
            cs.forward(x)?
        } else {
            x.clone()
        };
        h.add(&skip)
    }
}

// ---------------------------------------------------------------------------
// Attention block (mid-block only — Linear Q/K/V, diffusers style)
// ---------------------------------------------------------------------------

/// Self-attention with Linear projections (NOT Conv2d).
///
/// Weight keys: `{prefix}.group_norm.*`, `{prefix}.to_q/k/v.*`, `{prefix}.to_out.0.*`
struct AttnBlock {
    group_norm: GroupNorm,
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    channels: usize,
}

impl AttnBlock {
    fn load(
        weights: &HashMap<String, Tensor>,
        prefix: &str,
        channels: usize,
        device: &CudaArc,
    ) -> Result<Self> {
        let group_norm = load_gn(weights, &format!("{prefix}.group_norm"), channels, device)?;
        let to_q = load_linear(weights, &format!("{prefix}.to_q"), channels, channels, device)?;
        let to_k = load_linear(weights, &format!("{prefix}.to_k"), channels, channels, device)?;
        let to_v = load_linear(weights, &format!("{prefix}.to_v"), channels, channels, device)?;
        let to_out = load_linear(weights, &format!("{prefix}.to_out.0"), channels, channels, device)?;

        Ok(Self { group_norm, to_q, to_k, to_v, to_out, channels })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let c = self.channels;
        let h = self.group_norm.forward_nchw(x)?;
        let dims = h.shape().dims();
        let (b, _c, height, width) = (dims[0], dims[1], dims[2], dims[3]);
        let n = height * width;

        // [B, C, H, W] → [B, H, W, C] → [B, H*W, C]
        let h_flat = h.permute(&[0, 2, 3, 1])?.reshape(&[b, n, c])?;

        let q = self.to_q.forward(&h_flat)?; // [B, N, C]
        let k = self.to_k.forward(&h_flat)?;
        let v = self.to_v.forward(&h_flat)?;

        // SDPA expects [B, heads, N, D] — single head with D=C
        let q = q.reshape(&[b, 1, n, c])?;
        let k = k.reshape(&[b, 1, n, c])?;
        let v = v.reshape(&[b, 1, n, c])?;

        let out = flame_core::attention::sdpa(&q, &k, &v, None)?; // [B, 1, N, C]
        let out = out.reshape(&[b, n, c])?; // squeeze head dim

        let out = self.to_out.forward(&out)?; // [B, N, C]

        // [B, N, C] → [B, H, W, C] → [B, C, H, W]
        let out = out.reshape(&[b, height, width, c])?.permute(&[0, 3, 1, 2])?;

        x.add(&out)
    }
}

// ---------------------------------------------------------------------------
// Mid block
// ---------------------------------------------------------------------------

/// Mid block: ResBlock(ch) → Attention(ch) → ResBlock(ch).
struct MidBlock {
    resnet0: ResBlock,
    attn: AttnBlock,
    resnet1: ResBlock,
}

impl MidBlock {
    fn load(
        weights: &HashMap<String, Tensor>,
        prefix: &str,
        channels: usize,
        device: &CudaArc,
    ) -> Result<Self> {
        let resnet0 = ResBlock::load(weights, &format!("{prefix}.resnets.0"), channels, channels, device)?;
        let attn = AttnBlock::load(weights, &format!("{prefix}.attentions.0"), channels, device)?;
        let resnet1 = ResBlock::load(weights, &format!("{prefix}.resnets.1"), channels, channels, device)?;
        Ok(Self { resnet0, attn, resnet1 })
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

/// Decoder up block: N ResBlocks + optional nearest-neighbor upsample + conv.
///
/// Weight keys: `{prefix}.resnets.{m}.*`, `{prefix}.upsamplers.0.conv.*`
struct UpBlock {
    resnets: Vec<ResBlock>,
    upsample_conv: Option<Conv2d>,
}

impl UpBlock {
    fn load(
        weights: &HashMap<String, Tensor>,
        prefix: &str,
        in_ch: usize,
        out_ch: usize,
        num_resnets: usize,
        has_upsample: bool,
        device: &CudaArc,
    ) -> Result<Self> {
        let mut resnets = Vec::with_capacity(num_resnets);
        let mut ch = in_ch;
        for m in 0..num_resnets {
            resnets.push(ResBlock::load(
                weights,
                &format!("{prefix}.resnets.{m}"),
                ch,
                out_ch,
                device,
            )?);
            ch = out_ch;
        }

        let upsample_conv = if has_upsample {
            Some(load_conv(
                weights,
                &format!("{prefix}.upsamplers.0.conv"),
                out_ch,
                out_ch,
                3,
                1,
                1,
                device,
            )?)
        } else {
            None
        };

        Ok(Self { resnets, upsample_conv })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = x.clone();
        for resnet in &self.resnets {
            h = resnet.forward(&h)?;
        }
        if let Some(conv) = &self.upsample_conv {
            let dims = h.shape().dims();
            let (h_out, w_out) = (dims[2] * 2, dims[3] * 2);
            h = flame_core::cuda_ops::GpuOps::upsample2d_nearest(&h, (h_out, w_out))?;
            h = conv.forward(&h)?;
        }
        Ok(h)
    }
}

// ---------------------------------------------------------------------------
// Unpatchify helper
// ---------------------------------------------------------------------------

/// Reverse pixel-shuffle: `[B, 128, H/2, W/2]` → `[B, 32, H, W]`.
///
/// If input already has 32 channels, returns as-is.
pub fn unpatchify_latents(latents: &Tensor) -> Result<Tensor> {
    let dims = latents.shape().dims();
    let (b, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);

    if c == LATENT_CH {
        return Ok(latents.clone());
    }
    if c != 128 {
        return Err(Error::InvalidOperation(format!(
            "unpatchify expects 128 channels, got {c}"
        )));
    }

    // [B, 128, H, W] → [B, 32, 2, 2, H, W] → [B, 32, H, 2, W, 2] → [B, 32, H*2, W*2]
    let x = latents.reshape(&[b, 32, 2, 2, h, w])?;
    let x = x.permute(&[0, 1, 4, 2, 5, 3])?;
    x.reshape(&[b, 32, h * 2, w * 2])
}

// ---------------------------------------------------------------------------
// Full decoder
// ---------------------------------------------------------------------------

/// Flux 2 / Klein VAE decoder.
///
/// Loads from a `HashMap<String, Tensor>` with diffusers-style weight keys.
/// Input: `[B, 32, H, W]` latents (or `[B, 128, H/2, W/2]` — auto-unpatchified).
/// Output: `[B, 3, H*8, W*8]` RGB in `[-1, 1]`.
pub struct KleinVaeDecoder {
    post_quant_conv: Conv2d,
    conv_in: Conv2d,
    mid_block: MidBlock,
    up_blocks: Vec<UpBlock>,
    conv_norm_out: GroupNorm,
    conv_out: Conv2d,
}

impl KleinVaeDecoder {
    /// Load decoder weights from a HashMap (diffusers-format keys).
    ///
    /// Required keys: `post_quant_conv.*`, `decoder.conv_in.*`, `decoder.mid_block.*`,
    /// `decoder.up_blocks.*`, `decoder.conv_norm_out.*`, `decoder.conv_out.*`.
    pub fn load(weights: &HashMap<String, Tensor>, device: &Device) -> Result<Self> {
        let cuda = device.cuda_device_arc();
        let top_ch = CH * CH_MULT[CH_MULT.len() - 1]; // 512

        // post_quant_conv: Conv2d(32, 32, 1) — 1x1 on latent space
        let post_quant_conv = load_conv(weights, "post_quant_conv", LATENT_CH, LATENT_CH, 1, 1, 0, &cuda)?;

        // conv_in: Conv2d(32, 512, 3, pad=1)
        let conv_in = load_conv(weights, "decoder.conv_in", LATENT_CH, top_ch, 3, 1, 1, &cuda)?;

        // mid block: ResBlock(512) + Attention(512) + ResBlock(512)
        let mid_block = MidBlock::load(weights, "decoder.mid_block", top_ch, &cuda)?;

        // up blocks: reversed ch_mult = [4, 4, 2, 1]
        //   block 0: 512→512, upsample
        //   block 1: 512→512, upsample
        //   block 2: 512→256, upsample
        //   block 3: 256→128, NO upsample
        let rev_mult: Vec<usize> = CH_MULT.iter().rev().copied().collect();
        let num_resnets = NUM_RES_BLOCKS + 1; // decoder has 3 resnets per block

        let mut up_blocks = Vec::with_capacity(4);
        for i in 0..CH_MULT.len() {
            let out_ch = CH * rev_mult[i];
            let in_ch = if i == 0 { top_ch } else { CH * rev_mult[i - 1] };
            let has_up = i < CH_MULT.len() - 1;

            up_blocks.push(UpBlock::load(
                weights,
                &format!("decoder.up_blocks.{i}"),
                in_ch,
                out_ch,
                num_resnets,
                has_up,
                &cuda,
            )?);
        }

        // conv_norm_out: GroupNorm(32, 128)
        let conv_norm_out = load_gn(weights, "decoder.conv_norm_out", CH, &cuda)?;

        // conv_out: Conv2d(128, 3, 3, pad=1)
        let conv_out = load_conv(weights, "decoder.conv_out", CH, 3, 3, 1, 1, &cuda)?;

        Ok(Self {
            post_quant_conv,
            conv_in,
            mid_block,
            up_blocks,
            conv_norm_out,
            conv_out,
        })
    }

    /// Decode latents to RGB.
    ///
    /// Accepts either `[B, 32, H, W]` or `[B, 128, H/2, W/2]` (auto-unpatchified).
    /// Returns `[B, 3, H*8, W*8]` in `[-1, 1]`.
    pub fn decode(&self, z: &Tensor) -> Result<Tensor> {
        // Auto-unpatchify if 128 channels
        let z = unpatchify_latents(z)?;

        // post_quant_conv: 1x1 on latent
        let z = self.post_quant_conv.forward(&z)?;

        // conv_in
        let mut h = self.conv_in.forward(&z)?;

        // mid block
        h = self.mid_block.forward(&h)?;

        // up blocks (0 → 1 → 2 → 3)
        for block in &self.up_blocks {
            h = block.forward(&h)?;
        }

        // final norm + silu + conv
        h = self.conv_norm_out.forward_nchw(&h)?;
        h = h.silu()?;
        self.conv_out.forward(&h)
    }

    /// Number of weight keys expected for the decoder (for validation).
    pub fn expected_key_count() -> usize {
        // post_quant_conv: 2 (weight + bias)
        // conv_in: 2
        // mid_block: 2 resnets * 8 + 1 attn * 10 = 26
        //   resnet: norm1(w,b) + conv1(w,b) + norm2(w,b) + conv2(w,b) = 8
        //   attn: group_norm(w,b) + to_q(w,b) + to_k(w,b) + to_v(w,b) + to_out.0(w,b) = 10
        // up_blocks:
        //   block 0: 3 resnets * 8 = 24
        //   block 1: 3 resnets * 8 = 24
        //   block 2: 3 resnets * 8 + conv_shortcut(w,b) = 26 (first resnet: 512→256)
        //   block 3: 3 resnets * 8 + conv_shortcut(w,b) = 26 (first resnet: 256→128)
        //   upsamplers: 3 * 2 = 6
        // conv_norm_out: 2
        // conv_out: 2
        2 + 2 + 26 + 24 + 24 + 26 + 26 + 6 + 2 + 2 // = 140
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unpatchify_passthrough_32ch() {
        // 32-channel input should pass through unchanged
        let device = CudaDevice::new(0).unwrap();
        let t = Tensor::zeros_dtype(
            Shape::from_dims(&[1, 32, 8, 8]),
            DType::BF16,
            device,
        )
        .unwrap();
        let out = unpatchify_latents(&t).unwrap();
        assert_eq!(out.shape().dims(), &[1, 32, 8, 8]);
    }

    #[test]
    fn test_unpatchify_128ch() {
        // 128-channel [1, 128, 4, 4] → [1, 32, 8, 8]
        let device = CudaDevice::new(0).unwrap();
        let t = Tensor::zeros_dtype(
            Shape::from_dims(&[1, 128, 4, 4]),
            DType::BF16,
            device,
        )
        .unwrap();
        let out = unpatchify_latents(&t).unwrap();
        assert_eq!(out.shape().dims(), &[1, 32, 8, 8]);
    }

    #[test]
    fn test_unpatchify_rejects_64ch() {
        let device = CudaDevice::new(0).unwrap();
        let t = Tensor::zeros_dtype(
            Shape::from_dims(&[1, 64, 4, 4]),
            DType::BF16,
            device,
        )
        .unwrap();
        assert!(unpatchify_latents(&t).is_err());
    }

    #[test]
    fn test_channel_progression() {
        // Verify the channel math matches the Python blueprint
        let rev_mult: Vec<usize> = CH_MULT.iter().rev().copied().collect();
        assert_eq!(rev_mult, vec![4, 4, 2, 1]);

        let top_ch = CH * CH_MULT[CH_MULT.len() - 1];
        assert_eq!(top_ch, 512);

        // Block 0: in=512, out=512
        assert_eq!(CH * rev_mult[0], 512);
        // Block 1: in=512, out=512
        assert_eq!(CH * rev_mult[1], 512);
        // Block 2: in=512, out=256
        assert_eq!(CH * rev_mult[2], 256);
        // Block 3: in=256, out=128
        assert_eq!(CH * rev_mult[3], 128);
    }

    #[test]
    fn test_num_resnets_per_block() {
        // Decoder has num_res_blocks + 1 = 3 resnets per up_block
        assert_eq!(NUM_RES_BLOCKS + 1, 3);
    }

    #[test]
    fn test_constants() {
        assert_eq!(LATENT_CH, 32);
        assert_eq!(CH, 128);
        assert_eq!(NORM_GROUPS, 32);
    }
}
