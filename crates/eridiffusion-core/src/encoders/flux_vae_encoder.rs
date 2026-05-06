//! Generic LDM VAE encoder — pure Rust, diffusers-format weight keys.
//!
//! Architecture: Standard LDM AutoencoderKL encoder with configurable latent channels.
//! - block_out_channels = (128, 256, 512, 512)
//! - 2 resnets per down block (layers_per_block)
//! - Mid block: ResBlock + SelfAttention + ResBlock
//! - Optional quant_conv (some models have it, some don't)
//! - Scaling: z = (z - shift_factor) * scaling_factor after encode
//!
//! Input:  `[B, 3, H, W]` float pixels normalized to `[-1, 1]`
//! Output: `[B, latent_ch, H/8, W/8]` latent tensor (deterministic mean)
//!
//! Diffusers key format:
//!   encoder.conv_in.weight/bias
//!   encoder.down_blocks.{i}.resnets.{j}.norm1/conv1/norm2/conv2.weight/bias
//!   encoder.down_blocks.{i}.resnets.{j}.conv_shortcut.weight/bias (when in_ch != out_ch)
//!   encoder.down_blocks.{i}.downsamplers.0.conv.weight/bias
//!   encoder.mid_block.resnets.{0,1}.norm1/conv1/norm2/conv2.weight/bias
//!   encoder.mid_block.attentions.0.group_norm/to_q/to_k/to_v/to_out.0.weight/bias
//!   encoder.conv_norm_out.weight/bias
//!   encoder.conv_out.weight/bias
//!   quant_conv.weight/bias (top-level, NOT under encoder prefix)

use flame_core::conv::Conv2d;
use flame_core::group_norm::GroupNorm;
use flame_core::sdpa::forward as sdpa_forward;
use flame_core::serialization::load_file_filtered;
use flame_core::{DType, Error, Result, Shape, Tensor};
use std::collections::HashMap;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// GroupNorm helper — NCHW path for encoder parity with PyTorch.
//
// Uses GroupNorm::forward_nchw which dispatches directly to the NCHW kernel,
// avoiding the NCHW->NHWC->(group_norm internal NCHW->kernel->NHWC)->NCHW
// permutation chain the old wrapper used.  This alone reduces mean-abs-diff
// from ~0.20 to ~0.012 (16x) against PyTorch, likely because the extra
// permutations in the old path accumulated contiguity/rounding artifacts.
// ---------------------------------------------------------------------------

fn group_norm_nchw(
    x: &Tensor,
    num_groups: usize,
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    eps: f32,
) -> Result<Tensor> {
    let num_channels = x.shape().dims()[1];
    let gn = GroupNorm {
        num_groups,
        num_channels,
        eps,
        affine: weight.is_some(),
        weight: weight.cloned(),
        bias: bias.cloned(),
    };
    gn.forward_nchw(x)
}

// ---------------------------------------------------------------------------
// ResBlock (diffusers key format)
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
            w.get(key)
                .ok_or_else(|| Error::InvalidInput(format!("Missing key: {key}")))
        };

        let mut conv1 = Conv2d::new_with_bias(in_ch, out_ch, 3, 1, 1, device.clone(), true)?;
        conv1.copy_weight_from(get(&format!("{prefix}.conv1.weight"))?)?;
        conv1.copy_bias_from(get(&format!("{prefix}.conv1.bias"))?)?;

        let mut conv2 = Conv2d::new_with_bias(out_ch, out_ch, 3, 1, 1, device.clone(), true)?;
        conv2.copy_weight_from(get(&format!("{prefix}.conv2.weight"))?)?;
        conv2.copy_bias_from(get(&format!("{prefix}.conv2.bias"))?)?;

        // conv_shortcut when channels change (diffusers uses "conv_shortcut", not "nin_shortcut")
        let shortcut_key = format!("{prefix}.conv_shortcut.weight");
        let shortcut = if w.contains_key(&shortcut_key) {
            let mut s = Conv2d::new_with_bias(in_ch, out_ch, 1, 1, 0, device.clone(), true)?;
            s.copy_weight_from(get(&format!("{prefix}.conv_shortcut.weight"))?)?;
            s.copy_bias_from(get(&format!("{prefix}.conv_shortcut.bias"))?)?;
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

// ---------------------------------------------------------------------------
// Attention block (Conv2d 1x1 self-attention, same as decoder)
// ---------------------------------------------------------------------------

struct AttnBlock {
    norm_w: Tensor,
    norm_b: Tensor,
    q_w: Tensor,
    q_b: Tensor,
    k_w: Tensor,
    k_b: Tensor,
    v_w: Tensor,
    v_b: Tensor,
    proj_out_w: Tensor,
    proj_out_b: Tensor,
    #[allow(dead_code)]
    channels: usize,
}

/// Squeeze Conv2d 1x1 weight [out, in, 1, 1] -> [out, in]
fn squeeze_1x1(t: &Tensor) -> Result<Tensor> {
    let dims = t.shape().dims();
    if dims.len() == 4 && dims[2] == 1 && dims[3] == 1 {
        t.reshape(&[dims[0], dims[1]])
    } else if dims.len() == 2 {
        // Already 2D (Linear weight format from diffusers)
        t.clone_result()
    } else {
        t.clone_result()
    }
}

/// 3D linear: [B, N, C] @ W^T + bias -> [B, N, out]
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

impl AttnBlock {
    /// Load from diffusers-format keys: `{prefix}.group_norm.*`, `{prefix}.to_q/k/v.*`,
    /// `{prefix}.to_out.0.*`
    fn from_weights_diffusers(
        w: &HashMap<String, Tensor>,
        prefix: &str,
        channels: usize,
    ) -> Result<Self> {
        let get = |key: &str| -> Result<&Tensor> {
            w.get(key)
                .ok_or_else(|| Error::InvalidInput(format!("Missing key: {key}")))
        };

        Ok(Self {
            norm_w: get(&format!("{prefix}.group_norm.weight"))?.clone_result()?,
            norm_b: get(&format!("{prefix}.group_norm.bias"))?.clone_result()?,
            q_w: squeeze_1x1(get(&format!("{prefix}.to_q.weight"))?)?,
            q_b: get(&format!("{prefix}.to_q.bias"))?.clone_result()?,
            k_w: squeeze_1x1(get(&format!("{prefix}.to_k.weight"))?)?,
            k_b: get(&format!("{prefix}.to_k.bias"))?.clone_result()?,
            v_w: squeeze_1x1(get(&format!("{prefix}.to_v.weight"))?)?,
            v_b: get(&format!("{prefix}.to_v.bias"))?.clone_result()?,
            proj_out_w: squeeze_1x1(get(&format!("{prefix}.to_out.0.weight"))?)?,
            proj_out_b: get(&format!("{prefix}.to_out.0.bias"))?.clone_result()?,
            channels,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let (b, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);
        let n = h * w;

        let h_norm = group_norm_nchw(x, 32, Some(&self.norm_w), Some(&self.norm_b), 1e-6)?;

        // [B, C, H, W] -> [B, H*W, C]
        let h_flat = h_norm.permute(&[0, 2, 3, 1])?.reshape(&[b, n, c])?;

        let q = linear_3d(&h_flat, &self.q_w, &self.q_b)?;
        let k = linear_3d(&h_flat, &self.k_w, &self.k_b)?;
        let v = linear_3d(&h_flat, &self.v_w, &self.v_b)?;

        let q = q.unsqueeze(1)?; // [B, 1, N, C]
        let k = k.unsqueeze(1)?;
        let v = v.unsqueeze(1)?;

        // Tiled attention for large spatial dims (same as decoder)
        const ATTN_TILE: usize = 1024;
        let out = if n <= ATTN_TILE {
            sdpa_forward(&q, &k, &v, None)?
        } else {
            let mut tiles: Vec<Tensor> = Vec::with_capacity(n.div_ceil(ATTN_TILE));
            let mut start = 0;
            while start < n {
                let len = (n - start).min(ATTN_TILE);
                let q_tile = q.narrow(2, start, len)?;
                let out_tile = sdpa_forward(&q_tile, &k, &v, None)?;
                tiles.push(out_tile);
                start += len;
            }
            let tile_refs: Vec<&Tensor> = tiles.iter().collect();
            Tensor::cat(&tile_refs, 2)?
        };
        let out = out.squeeze(Some(1))?; // [B, N, C]

        let out = linear_3d(&out, &self.proj_out_w, &self.proj_out_b)?;

        // [B, N, C] -> [B, C, H, W]
        let out = out.reshape(&[b, h, w, c])?.permute(&[0, 3, 1, 2])?;

        x.add(&out)
    }
}

// ---------------------------------------------------------------------------
// Mid block (diffusers key format)
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
            resnet0: ResBlock::from_weights(
                w,
                &format!("{prefix}.resnets.0"),
                channels,
                channels,
                device,
            )?,
            attn: AttnBlock::from_weights_diffusers(
                w,
                &format!("{prefix}.attentions.0"),
                channels,
            )?,
            resnet1: ResBlock::from_weights(
                w,
                &format!("{prefix}.resnets.1"),
                channels,
                channels,
                device,
            )?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.resnet0.forward(x)?;
        let x = self.attn.forward(&x)?;
        self.resnet1.forward(&x)
    }
}

// ---------------------------------------------------------------------------
// Down block
// ---------------------------------------------------------------------------

struct DownBlock {
    resnets: Vec<ResBlock>,
    downsample_conv: Option<Conv2d>,
}

/// Pad an NCHW tensor with zeros: `(left, right, top, bottom)`.
fn pad2d_zeros(
    x: &Tensor,
    left: usize,
    right: usize,
    top: usize,
    bottom: usize,
) -> Result<Tensor> {
    if left == 0 && right == 0 && top == 0 && bottom == 0 {
        return Ok(x.clone_result()?);
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

    let mut current = x.clone_result()?;
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

impl DownBlock {
    fn from_weights(
        w: &HashMap<String, Tensor>,
        prefix: &str,
        in_ch: usize,
        out_ch: usize,
        num_resnets: usize,
        has_downsample: bool,
        device: &Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        let get = |key: &str| -> Result<&Tensor> {
            w.get(key)
                .ok_or_else(|| Error::InvalidInput(format!("Missing key: {key}")))
        };

        let mut resnets = Vec::with_capacity(num_resnets);
        let mut ch = in_ch;
        for m in 0..num_resnets {
            resnets.push(ResBlock::from_weights(
                w,
                &format!("{prefix}.resnets.{m}"),
                ch,
                out_ch,
                device,
            )?);
            ch = out_ch;
        }

        let downsample_conv = if has_downsample {
            // Stride-2 conv with NO padding — asymmetric (0,1,0,1) pad applied manually
            let mut conv =
                Conv2d::new_with_bias(out_ch, out_ch, 3, 2, 0, device.clone(), true)?;
            conv.copy_weight_from(get(&format!("{prefix}.downsamplers.0.conv.weight"))?)?;
            conv.copy_bias_from(get(&format!("{prefix}.downsamplers.0.conv.bias"))?)?;
            Some(conv)
        } else {
            None
        };

        Ok(Self {
            resnets,
            downsample_conv,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = x.clone_result()?;
        for resnet in &self.resnets {
            h = resnet.forward(&h)?;
        }
        if let Some(ref conv) = self.downsample_conv {
            // Asymmetric pad: right + bottom only (0,1,0,1) — matches diffusers encoder
            h = pad2d_zeros(&h, 0, 1, 0, 1)?;
            h = conv.forward(&h)?;
        }
        Ok(h)
    }
}

// ---------------------------------------------------------------------------
// Full LDM VAE Encoder (generic, configurable latent channels)
// ---------------------------------------------------------------------------

/// LDM-format VAE encoder with configurable latent channels.
///
/// Works for any model using the standard LDM AutoencoderKL encoder layout:
/// SDXL (4ch), SD 1.5 (4ch), SD3 (16ch), Z-Image (16ch), QwenImage (16ch), etc.
///
/// Input:  `[B, 3, H, W]` pixels in `[-1, 1]` (BF16).
/// Output: `[B, latent_ch, H/8, W/8]` deterministic mean latents.
pub struct LdmVAEEncoder {
    conv_in: Conv2d,
    down_blocks: Vec<DownBlock>,
    mid_block: MidBlock,
    norm_out_w: Tensor,
    norm_out_b: Tensor,
    conv_out: Conv2d,
    quant_conv: Option<Conv2d>,
    latent_channels: usize,
}

/// Remap LDM-format encoder keys to diffusers format.
///
/// LDM format:
///   encoder.down.{i}.block.{j}.*         → encoder.down_blocks.{i}.resnets.{j}.*
///   encoder.down.{i}.downsample.conv.*   → encoder.down_blocks.{i}.downsamplers.0.conv.*
///   encoder.mid.block_{1,2}.*            → encoder.mid_block.resnets.{0,1}.*
///   encoder.mid.attn_1.{norm,q,k,v,proj_out}.* → encoder.mid_block.attentions.0.{group_norm,to_q,to_k,to_v,to_out.0}.*
///   encoder.norm_out.*                   → encoder.conv_norm_out.*
///   *.nin_shortcut.*                     → *.conv_shortcut.*
fn remap_ldm_to_diffusers(w: HashMap<String, Tensor>) -> HashMap<String, Tensor> {
    let is_ldm = w.keys().any(|k| k.contains("encoder.down.") || k.contains("encoder.mid.block_"));
    if !is_ldm {
        return w;
    }

    println!("[LdmVAEEncoder] Remapping LDM keys to diffusers format");
    let mut out = HashMap::with_capacity(w.len());
    for (key, val) in w {
        let new_key = remap_enc_ldm_key(&key);
        out.insert(new_key, val);
    }
    out
}

fn remap_enc_ldm_key(key: &str) -> String {
    let k = key.to_string();

    // encoder.norm_out.* → encoder.conv_norm_out.*
    if k.starts_with("encoder.norm_out.") {
        return k.replace("encoder.norm_out.", "encoder.conv_norm_out.");
    }

    // encoder.mid.block_1.* → encoder.mid_block.resnets.0.*
    // encoder.mid.block_2.* → encoder.mid_block.resnets.1.*
    if k.starts_with("encoder.mid.block_") {
        let rest = &k["encoder.mid.block_".len()..];
        if let Some(dot) = rest.find('.') {
            let idx: usize = rest[..dot].parse().unwrap_or(1);
            let suffix = &rest[dot + 1..];
            let suffix = suffix.replace("nin_shortcut.", "conv_shortcut.");
            return format!("encoder.mid_block.resnets.{}.{suffix}", idx - 1);
        }
    }

    // encoder.mid.attn_1.norm.* → encoder.mid_block.attentions.0.group_norm.*
    if k.starts_with("encoder.mid.attn_1.norm.") {
        let suffix = &k["encoder.mid.attn_1.norm.".len()..];
        return format!("encoder.mid_block.attentions.0.group_norm.{suffix}");
    }
    // encoder.mid.attn_1.q.* → encoder.mid_block.attentions.0.to_q.*
    if k.starts_with("encoder.mid.attn_1.q.") {
        let suffix = &k["encoder.mid.attn_1.q.".len()..];
        return format!("encoder.mid_block.attentions.0.to_q.{suffix}");
    }
    if k.starts_with("encoder.mid.attn_1.k.") {
        let suffix = &k["encoder.mid.attn_1.k.".len()..];
        return format!("encoder.mid_block.attentions.0.to_k.{suffix}");
    }
    if k.starts_with("encoder.mid.attn_1.v.") {
        let suffix = &k["encoder.mid.attn_1.v.".len()..];
        return format!("encoder.mid_block.attentions.0.to_v.{suffix}");
    }
    // encoder.mid.attn_1.proj_out.* → encoder.mid_block.attentions.0.to_out.0.*
    if k.starts_with("encoder.mid.attn_1.proj_out.") {
        let suffix = &k["encoder.mid.attn_1.proj_out.".len()..];
        return format!("encoder.mid_block.attentions.0.to_out.0.{suffix}");
    }

    // encoder.down.{i}.block.{j}.* → encoder.down_blocks.{i}.resnets.{j}.*
    if k.starts_with("encoder.down.") {
        let rest = &k["encoder.down.".len()..];
        if let Some(dot) = rest.find('.') {
            let block_idx = &rest[..dot];
            let inner = &rest[dot + 1..];

            if inner.starts_with("block.") {
                let rr = &inner["block.".len()..];
                if let Some(dot2) = rr.find('.') {
                    let resnet_idx = &rr[..dot2];
                    let suffix = &rr[dot2 + 1..];
                    let suffix = suffix.replace("nin_shortcut.", "conv_shortcut.");
                    return format!("encoder.down_blocks.{block_idx}.resnets.{resnet_idx}.{suffix}");
                }
            }

            // encoder.down.{i}.downsample.conv.* → encoder.down_blocks.{i}.downsamplers.0.conv.*
            if inner.starts_with("downsample.conv.") {
                let suffix = &inner["downsample.conv.".len()..];
                return format!("encoder.down_blocks.{block_idx}.downsamplers.0.conv.{suffix}");
            }
        }
    }

    // Unchanged (conv_in, conv_out, quant_conv, etc.)
    k
}

impl LdmVAEEncoder {
    /// Load encoder from safetensors file (mmap, encoder keys + quant_conv only).
    ///
    /// `latent_channels`: number of latent channels (4 for SDXL/SD1.5, 16 for ZImage/QwenImage/SD3).
    pub fn from_safetensors(
        path: &str,
        latent_channels: usize,
        device: &Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        let raw = load_file_filtered(path, device, |key| {
            key.starts_with("encoder.")
                || key.starts_with("first_stage_model.encoder.")
                || key == "quant_conv.weight"
                || key == "quant_conv.bias"
                || key == "first_stage_model.quant_conv.weight"
                || key == "first_stage_model.quant_conv.bias"
        })?;

        // Strip "first_stage_model." prefix and cast to BF16.
        let fsm = "first_stage_model.";
        let mut w = HashMap::with_capacity(raw.len());
        for (key, val) in raw {
            let k = key.strip_prefix(fsm).unwrap_or(&key).to_string();
            let val_bf16 = if val.dtype() == DType::BF16 {
                val
            } else {
                val.to_dtype(DType::BF16)?
            };
            w.insert(k, val_bf16);
        }
        println!(
            "[LdmVAEEncoder] Loaded {} encoder weight tensors (cast to BF16)",
            w.len()
        );

        let w = remap_ldm_to_diffusers(w);
        Self::from_weights(w, latent_channels, device)
    }

    /// Build from a pre-loaded weight HashMap (diffusers-format keys).
    pub fn from_weights(
        w: HashMap<String, Tensor>,
        latent_channels: usize,
        device: &Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        let get = |key: &str| -> Result<&Tensor> {
            w.get(key)
                .ok_or_else(|| Error::InvalidInput(format!("Missing key: {key}")))
        };

        let ch: usize = 128;
        let ch_mult: [usize; 4] = [1, 2, 4, 4];
        let layers_per_block: usize = 2;
        let top_ch = ch * ch_mult[3]; // 512

        // conv_in: Conv2d(3, 128, 3, pad=1)
        let mut conv_in = Conv2d::new_with_bias(3, ch, 3, 1, 1, device.clone(), true)?;
        conv_in.copy_weight_from(get("encoder.conv_in.weight")?)?;
        conv_in.copy_bias_from(get("encoder.conv_in.bias")?)?;

        // 4 down blocks with channel progression [128, 256, 512, 512]
        // Blocks 0-2 have downsample, block 3 does not
        let mut down_blocks = Vec::with_capacity(4);
        let mut prev_ch = ch;
        for i in 0..4usize {
            let out_ch = ch * ch_mult[i];
            let has_down = i < 3;
            down_blocks.push(DownBlock::from_weights(
                &w,
                &format!("encoder.down_blocks.{i}"),
                prev_ch,
                out_ch,
                layers_per_block,
                has_down,
                device,
            )?);
            prev_ch = out_ch;
        }

        // mid block
        let mid_block = MidBlock::from_weights(&w, "encoder.mid_block", top_ch, device)?;

        // conv_norm_out + conv_out
        let norm_out_w = get("encoder.conv_norm_out.weight")?.clone_result()?;
        let norm_out_b = get("encoder.conv_norm_out.bias")?.clone_result()?;

        let out_channels = 2 * latent_channels;
        let mut conv_out = Conv2d::new_with_bias(top_ch, out_channels, 3, 1, 1, device.clone(), true)?;
        conv_out.copy_weight_from(get("encoder.conv_out.weight")?)?;
        conv_out.copy_bias_from(get("encoder.conv_out.bias")?)?;

        // quant_conv: Conv2d(2*latent_ch, 2*latent_ch, 1) — optional, top-level key
        let quant_conv = if w.contains_key("quant_conv.weight") {
            let mut qc = Conv2d::new_with_bias(
                out_channels,
                out_channels,
                1,
                1,
                0,
                device.clone(),
                true,
            )?;
            qc.copy_weight_from(get("quant_conv.weight")?)?;
            qc.copy_bias_from(get("quant_conv.bias")?)?;
            println!("[LdmVAEEncoder] quant_conv loaded ({out_channels} -> {out_channels})");
            Some(qc)
        } else {
            println!("[LdmVAEEncoder] No quant_conv found (disabled)");
            None
        };

        Ok(Self {
            conv_in,
            down_blocks,
            mid_block,
            norm_out_w,
            norm_out_b,
            conv_out,
            quant_conv,
            latent_channels,
        })
    }

    /// Encode image to latent space (deterministic mean, no sampling).
    ///
    /// Input:  `[B, 3, H, W]` pixels in `[-1, 1]` (BF16).
    /// Output: `[B, latent_ch, H/8, W/8]` latent tensor.
    pub fn encode(&self, image: &Tensor) -> Result<Tensor> {
        // conv_in
        let mut h = self.conv_in.forward(image)?;

        // down blocks
        for block in &self.down_blocks {
            h = block.forward(&h)?;
        }

        // mid block
        h = self.mid_block.forward(&h)?;

        // norm_out + silu + conv_out
        h = group_norm_nchw(&h, 32, Some(&self.norm_out_w), Some(&self.norm_out_b), 1e-6)?;
        h = h.silu()?;
        h = self.conv_out.forward(&h)?;

        // quant_conv BEFORE channel split (operates on 2*latent_ch)
        if let Some(ref qc) = self.quant_conv {
            h = qc.forward(&h)?;
        }

        // Take first latent_ch channels (deterministic mean, discard logvar)
        let c = h.shape().dims()[1];
        if c != 2 * self.latent_channels {
            return Err(Error::InvalidOperation(format!(
                "encoder output has {c} channels, expected {}",
                2 * self.latent_channels
            )));
        }
        h.narrow(1, 0, self.latent_channels)
    }

    /// Encode and apply VAE scaling: `z = (z - shift_factor) * scaling_factor`.
    ///
    /// This matches the BFL/diffusers encode convention:
    /// ```python
    /// z = self.scale_factor * (z - self.shift_factor)
    /// ```
    pub fn encode_scaled(
        &self,
        image: &Tensor,
        scaling_factor: f32,
        shift_factor: f32,
    ) -> Result<Tensor> {
        let z = self.encode(image)?;
        z.add_scalar(-shift_factor)?.mul_scalar(scaling_factor)
    }

    /// Return the configured latent channel count.
    pub fn latent_channels(&self) -> usize {
        self.latent_channels
    }
}
