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

use flame_core::conv::Conv2d;
use flame_core::group_norm::GroupNorm;
use flame_core::linear::Linear;
use flame_core::device::Device;
use flame_core::{CudaDevice, DType, Error, Result, Shape, Tensor};
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
    /// BatchNorm running stats for inverse normalization (BFL latent space).
    /// `inv_normalize(z) = z * sqrt(running_var + eps) + running_mean`
    bn_scale: Tensor,  // sqrt(running_var + eps), [128] BF16
    bn_bias: Tensor,   // running_mean, [128] BF16
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

        // BatchNorm inverse normalization: z * sqrt(var + eps) + mean
        // BFL stores bn.running_mean [128] and bn.running_var [128] as F32.
        // eps must match the encoder side AND diffusers
        // `autoencoder_kl_flux2.py:104` (`batch_norm_eps: float = 1e-4`).
        // KLEIN_VERIFY.md §H4: prior 1e-5 here gave silent inference parity drift.
        let bn_eps = 1e-4f32;
        let bn_scale = if let Some(var) = weights.get("bn.running_var") {
            let var_f32 = var.to_dtype(DType::F32)?;
            let scale_f32 = var_f32.add_scalar(bn_eps)?.sqrt()?;
            scale_f32.to_dtype(DType::BF16)?
        } else {
            // Fallback: identity scale (no BN in older VAEs)
            Tensor::from_f32_to_bf16(vec![1.0f32; 128], Shape::from_dims(&[128]), cuda.clone())?
        };
        let bn_bias = if let Some(mean) = weights.get("bn.running_mean") {
            mean.to_dtype(DType::BF16)?
        } else {
            Tensor::zeros_dtype(Shape::from_dims(&[128]), DType::BF16, cuda.clone())?
        };

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
            bn_scale,
            bn_bias,
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
    /// Input: `[B, 128, H, W]` (packed 128-channel latents from Klein).
    /// Applies inverse BatchNorm → unpatchify → decoder → `[B, 3, H*16, W*16]`.
    pub fn decode(&self, z: &Tensor) -> Result<Tensor> {
        let dims = z.shape().dims();
        let (b, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);

        // Step 1: Inverse BatchNorm — z * scale + bias per channel
        // bn_scale, bn_bias are [128], broadcast over [B, 128, H, W]
        let z = if c == 128 {
            let scale = self.bn_scale.reshape(&[1, 128, 1, 1])?;
            let bias = self.bn_bias.reshape(&[1, 128, 1, 1])?;
            z.mul(&scale)?.add(&bias)?
        } else {
            z.clone()
        };

        // Step 2: Unpatchify [B, 128, H, W] → [B, 32, 2H, 2W]
        let z = unpatchify_latents(&z)?;

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
// Encoder — mirror of the decoder for offline latent caching.
// ---------------------------------------------------------------------------
//
// Architecture (Flux 2 / Klein VAE, ch=128, ch_mult=(1,2,4,4), 2 layers/block):
//   conv_in:        Conv2d(3, 128, 3, pad=1)
//   down_blocks.0:  2x ResBlock(128→128) + downsample (asymmetric pad)
//   down_blocks.1:  2x ResBlock(128→256) + downsample
//   down_blocks.2:  2x ResBlock(256→512) + downsample
//   down_blocks.3:  2x ResBlock(512→512)  — NO downsample
//   mid_block:      ResBlock(512) + Attn(512) + ResBlock(512)
//   conv_norm_out:  GroupNorm(32, 512)
//   conv_out:       Conv2d(512, 64, 3, pad=1)   (64 = 2 * latent_ch)
//   quant_conv:     Conv2d(64, 64, 1)
//
// After encoder we take the deterministic mean (first 32 channels of the
// 64-channel quantized output), patchify with 2x2 pixel-unshuffle to get
// 128 channels, then apply the BatchNorm normalization
// `(z - running_mean) / sqrt(running_var + eps)`. The result matches what
// `KleinVaeDecoder::decode` consumes.

const ENCODER_BLOCK_CHANNELS: [usize; 4] = [128, 256, 512, 512];
const ENCODER_LAYERS_PER_BLOCK: usize = 2;

/// Pad an NCHW tensor with zeros: `(left, right, top, bottom)`.
fn pad2d_zeros(
    x: &Tensor,
    left: usize,
    right: usize,
    top: usize,
    bottom: usize,
) -> Result<Tensor> {
    if left == 0 && right == 0 && top == 0 && bottom == 0 {
        return Ok(x.clone());
    }
    let dims = x.shape().dims();
    if dims.len() != 4 {
        return Err(Error::InvalidOperation(format!(
            "pad2d_zeros: expected NCHW, got rank {}",
            dims.len()
        )));
    }
    let (b, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);
    let device = x.device().clone();
    let dtype = x.dtype();

    // Pad width first (left/right) along axis 3.
    let mut current = x.clone();
    if left > 0 {
        let zeros = Tensor::zeros_dtype(
            Shape::from_dims(&[b, c, h, left]),
            dtype,
            device.clone(),
        )?;
        current = Tensor::cat(&[&zeros, &current], 3)?;
    }
    if right > 0 {
        let zeros = Tensor::zeros_dtype(
            Shape::from_dims(&[b, c, h, right]),
            dtype,
            device.clone(),
        )?;
        current = Tensor::cat(&[&current, &zeros], 3)?;
    }

    // Then pad height (top/bottom) along axis 2.
    let new_w = w + left + right;
    if top > 0 {
        let zeros = Tensor::zeros_dtype(
            Shape::from_dims(&[b, c, top, new_w]),
            dtype,
            device.clone(),
        )?;
        current = Tensor::cat(&[&zeros, &current], 2)?;
    }
    if bottom > 0 {
        let zeros = Tensor::zeros_dtype(
            Shape::from_dims(&[b, c, bottom, new_w]),
            dtype,
            device,
        )?;
        current = Tensor::cat(&[&current, &zeros], 2)?;
    }
    Ok(current)
}

/// Forward pixel-unshuffle: `[B, 32, 2H, 2W]` → `[B, 128, H, W]`.
///
/// This is the inverse of [`unpatchify_latents`].
pub fn patchify_latents(latents: &Tensor) -> Result<Tensor> {
    let dims = latents.shape().dims();
    let (b, c, h2, w2) = (dims[0], dims[1], dims[2], dims[3]);

    if c == 128 {
        return Ok(latents.clone());
    }
    if c != LATENT_CH {
        return Err(Error::InvalidOperation(format!(
            "patchify expects {LATENT_CH} channels, got {c}"
        )));
    }
    if h2 % 2 != 0 || w2 % 2 != 0 {
        return Err(Error::InvalidOperation(format!(
            "patchify expects even spatial dims, got {h2}x{w2}"
        )));
    }
    let (h, w) = (h2 / 2, w2 / 2);

    // Inverse of unpatchify_latents permutation:
    //   unpatchify: [B, 32, 2, 2, H, W] -> permute(0,1,4,2,5,3) -> [B, 32, H, 2, W, 2]
    //   patchify:   [B, 32, H, 2, W, 2] -> permute(0,1,3,5,2,4) -> [B, 32, 2, 2, H, W]
    let x = latents.reshape(&[b, LATENT_CH, h, 2, w, 2])?;
    let x = x.permute(&[0, 1, 3, 5, 2, 4])?;
    x.reshape(&[b, 128, h, w])
}

/// One encoder down block: `num_resnets` resnets in series, optional
/// asymmetric-pad downsample conv at the end.
struct DownBlock {
    resnets: Vec<ResBlock>,
    downsample_conv: Option<Conv2d>,
}

impl DownBlock {
    fn load(
        weights: &HashMap<String, Tensor>,
        prefix: &str,
        in_ch: usize,
        out_ch: usize,
        num_resnets: usize,
        has_downsample: bool,
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
        let downsample_conv = if has_downsample {
            Some(load_conv(
                weights,
                &format!("{prefix}.downsamplers.0.conv"),
                out_ch,
                out_ch,
                3,
                2,
                0, // padding handled manually (asymmetric (0,1,0,1))
                device,
            )?)
        } else {
            None
        };
        Ok(Self {
            resnets,
            downsample_conv,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = x.clone();
        for resnet in &self.resnets {
            h = resnet.forward(&h)?;
        }
        if let Some(conv) = &self.downsample_conv {
            // Asymmetric pad (right + bottom only) to match diffusers Encoder.
            h = pad2d_zeros(&h, 0, 1, 0, 1)?;
            h = conv.forward(&h)?;
        }
        Ok(h)
    }
}

/// Klein / Flux 2 VAE encoder.
///
/// Input:  `[B, 3, H, W]` RGB in `[-1, 1]`.
/// Output: `[B, 128, H/16, W/16]` packed latents (post-BN, post-patchify),
/// matching the format that [`KleinVaeDecoder::decode`] consumes.
pub struct KleinVaeEncoder {
    conv_in: Conv2d,
    down_blocks: Vec<DownBlock>,
    mid_block: MidBlock,
    conv_norm_out: GroupNorm,
    conv_out: Conv2d,
    quant_conv: Option<Conv2d>,
    bn_inv_scale: Tensor, // 1 / sqrt(running_var + eps), [128] BF16
    bn_neg_mean: Tensor,  // -running_mean, [128] BF16
}

impl KleinVaeEncoder {
    /// Load encoder weights from a HashMap (diffusers-format keys).
    ///
    /// Required keys: `encoder.conv_in.*`, `encoder.down_blocks.*`,
    /// `encoder.mid_block.*`, `encoder.conv_norm_out.*`, `encoder.conv_out.*`,
    /// `quant_conv.*` (optional), `bn.running_mean`, `bn.running_var`.
    pub fn load(weights: &HashMap<String, Tensor>, device: &Device) -> Result<Self> {
        let cuda = device.cuda_device_arc();
        let top_ch = ENCODER_BLOCK_CHANNELS[ENCODER_BLOCK_CHANNELS.len() - 1];

        // conv_in: Conv2d(3, 128, 3, pad=1)
        let conv_in = load_conv(
            weights,
            "encoder.conv_in",
            3,
            ENCODER_BLOCK_CHANNELS[0],
            3,
            1,
            1,
            &cuda,
        )?;

        // 4 down blocks; final block has no downsample.
        let mut down_blocks = Vec::with_capacity(ENCODER_BLOCK_CHANNELS.len());
        for i in 0..ENCODER_BLOCK_CHANNELS.len() {
            let in_ch = if i == 0 {
                ENCODER_BLOCK_CHANNELS[0]
            } else {
                ENCODER_BLOCK_CHANNELS[i - 1]
            };
            let out_ch = ENCODER_BLOCK_CHANNELS[i];
            let has_down = i < ENCODER_BLOCK_CHANNELS.len() - 1;
            down_blocks.push(DownBlock::load(
                weights,
                &format!("encoder.down_blocks.{i}"),
                in_ch,
                out_ch,
                ENCODER_LAYERS_PER_BLOCK,
                has_down,
                &cuda,
            )?);
        }

        // mid_block: shares MidBlock loader with the decoder side.
        let mid_block = MidBlock::load(weights, "encoder.mid_block", top_ch, &cuda)?;

        // conv_norm_out: GroupNorm(32, 512)
        let conv_norm_out = load_gn(weights, "encoder.conv_norm_out", top_ch, &cuda)?;

        // conv_out: Conv2d(512, 64, 3, pad=1) — outputs 2 * latent_ch
        let conv_out = load_conv(
            weights,
            "encoder.conv_out",
            top_ch,
            2 * LATENT_CH,
            3,
            1,
            1,
            &cuda,
        )?;

        // quant_conv: Conv2d(64, 64, 1) — optional in diffusers; Klein has it.
        let quant_conv = if weights.contains_key("quant_conv.weight") {
            Some(load_conv(
                weights,
                "quant_conv",
                2 * LATENT_CH,
                2 * LATENT_CH,
                1,
                1,
                0,
                &cuda,
            )?)
        } else {
            None
        };

        // BN normalization is applied AFTER patchify, so it sees 128 channels.
        let bn_eps = 1e-4f32;
        let var = weights
            .get("bn.running_var")
            .ok_or_else(|| Error::InvalidOperation("missing bn.running_var".into()))?
            .to_dtype(DType::F32)?;
        let mean = weights
            .get("bn.running_mean")
            .ok_or_else(|| Error::InvalidOperation("missing bn.running_mean".into()))?
            .to_dtype(DType::F32)?;
        // inv_scale = 1 / sqrt(var + eps); compute via host-side reciprocal.
        let scale_f32 = var.add_scalar(bn_eps)?.sqrt()?;
        let scale_vec: Vec<f32> = scale_f32.to_vec()?;
        let inv_scale_vec: Vec<f32> = scale_vec.into_iter().map(|s| 1.0 / s).collect();
        let inv_scale_f32 = Tensor::from_vec(
            inv_scale_vec,
            scale_f32.shape().clone(),
            cuda.clone(),
        )?;
        let neg_mean_f32 = mean.mul_scalar(-1.0f32)?;
        let bn_inv_scale = inv_scale_f32.to_dtype(DType::BF16)?;
        let bn_neg_mean = neg_mean_f32.to_dtype(DType::BF16)?;

        Ok(Self {
            conv_in,
            down_blocks,
            mid_block,
            conv_norm_out,
            conv_out,
            quant_conv,
            bn_inv_scale,
            bn_neg_mean,
        })
    }

    /// Encode an image batch `[B, 3, H, W]` (BF16 in `[-1, 1]`) to packed
    /// latents `[B, 128, H/16, W/16]`.
    ///
    /// Uses the deterministic mean of the diagonal-Gaussian posterior
    /// (no sampling) — the right choice for caching training latents.
    pub fn encode(&self, image: &Tensor) -> Result<Tensor> {
        let mut h = self.conv_in.forward(image)?;
        for block in &self.down_blocks {
            h = block.forward(&h)?;
        }
        h = self.mid_block.forward(&h)?;
        h = self.conv_norm_out.forward_nchw(&h)?;
        h = h.silu()?;
        h = self.conv_out.forward(&h)?;
        if let Some(qc) = &self.quant_conv {
            h = qc.forward(&h)?;
        }

        // h is [B, 64, h, w] = [mu (32) | logvar (32)]. Take deterministic mean.
        let dims = h.shape().dims();
        let (b, c, h_, w_) = (dims[0], dims[1], dims[2], dims[3]);
        if c != 2 * LATENT_CH {
            return Err(Error::InvalidOperation(format!(
                "encoder conv_out produced {c} channels, expected {}",
                2 * LATENT_CH
            )));
        }
        let mu = h.narrow(1, 0, LATENT_CH)?;

        // Patchify: [B, 32, h, w] -> [B, 128, h/2, w/2]
        let z = patchify_latents(&mu)?;

        // BatchNorm forward: (z + neg_mean) * inv_scale, broadcast over [B, 128, H, W].
        let scale = self.bn_inv_scale.reshape(&[1, 128, 1, 1])?;
        let bias = self.bn_neg_mean.reshape(&[1, 128, 1, 1])?;
        let centered = z.add(&bias)?;
        let _ = (b, h_, w_); // suppress unused
        centered.mul(&scale)
    }

    /// Same as `encode()` but dumps every intermediate activation to a safetensors
    /// file in `dump_dir`. Used for VAE-parity bisection vs diffusers.
    pub fn encode_with_dump(
        &self,
        image: &Tensor,
        dump_dir: &std::path::Path,
    ) -> Result<Tensor> {
        std::fs::create_dir_all(dump_dir).ok();
        let dump = |name: &str, t: &Tensor| {
            let mut h = HashMap::new();
            h.insert(name.to_string(), t.clone());
            let p = dump_dir.join(format!("{name}.safetensors"));
            let _ = flame_core::serialization::save_file(&h, &p);
        };
        dump("00_input", image);
        let h = self.conv_in.forward(image)?;
        dump("01_after_conv_in", &h);
        let mut h = h;
        for (i, block) in self.down_blocks.iter().enumerate() {
            h = block.forward(&h)?;
            dump(&format!("02_after_down_block_{i}"), &h);
        }
        let h = self.mid_block.forward(&h)?;
        dump("03_after_mid_block", &h);
        let h = self.conv_norm_out.forward_nchw(&h)?;
        dump("04_after_conv_norm_out", &h);
        let h = h.silu()?;
        dump("05_after_silu", &h);
        let h = self.conv_out.forward(&h)?;
        dump("06_after_conv_out", &h);
        let h = if let Some(qc) = &self.quant_conv {
            let qh = qc.forward(&h)?;
            dump("07_after_quant_conv", &qh);
            qh
        } else { h };
        let mu = h.narrow(1, 0, LATENT_CH)?;
        dump("08_mu", &mu);
        let z = patchify_latents(&mu)?;
        dump("09_after_patchify", &z);
        let scale = self.bn_inv_scale.reshape(&[1, 128, 1, 1])?;
        let bias = self.bn_neg_mean.reshape(&[1, 128, 1, 1])?;
        let centered = z.add(&bias)?;
        dump("10_after_bn_center", &centered);
        let final_out = centered.mul(&scale)?;
        dump("11_after_bn_scale_FINAL", &final_out);
        Ok(final_out)
    }

    /// Encode WITHOUT BatchNorm — returns raw patchified latents `[B, 128, H/16, W/16]`.
    ///
    /// Use this for edit/reference conditioning where the model expects
    /// un-normalized latents (matching the Python `vae.encode()` + patchify path).
    /// The standard `encode()` applies BN which is correct for training cache
    /// but wrong for edit reference tokens.
    pub fn encode_raw(&self, image: &Tensor) -> Result<Tensor> {
        let mut h = self.conv_in.forward(image)?;
        for block in &self.down_blocks {
            h = block.forward(&h)?;
        }
        h = self.mid_block.forward(&h)?;
        h = self.conv_norm_out.forward_nchw(&h)?;
        h = h.silu()?;
        h = self.conv_out.forward(&h)?;
        if let Some(qc) = &self.quant_conv {
            h = qc.forward(&h)?;
        }

        let c = h.shape().dims()[1];
        if c != 2 * LATENT_CH {
            return Err(Error::InvalidOperation(format!(
                "encoder conv_out produced {c} channels, expected {}",
                2 * LATENT_CH
            )));
        }
        let mu = h.narrow(1, 0, LATENT_CH)?;
        patchify_latents(&mu)
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
