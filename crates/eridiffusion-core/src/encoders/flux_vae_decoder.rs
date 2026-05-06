//! Generic LDM VAE decoder — pure Rust, LDM-format weight keys.
//!
//! Architecture: Standard LDM AutoencoderKL decoder with configurable latent channels.
//! - block_out_channels = (128, 256, 512, 512)
//! - 3 resnets per up block (layers_per_block + 1)
//! - Mid block: ResBlock + SelfAttention + ResBlock
//! - No post_quant_conv (Z-Image disables it)
//! - Scaling: z = (z - shift_factor) / scaling_factor before decode
//!
//! Adapted from flame-core's ZImageVAEDecoder with configurable scaling/shift factors
//! so it can handle any LDM-format VAE (Z-Image, SD 1.5, SDXL, etc.).
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

use flame_core::conv::Conv2d;
use flame_core::cuda_kernels::CudaKernels;
use flame_core::group_norm::group_norm;
use flame_core::sdpa::forward as sdpa_forward;
use flame_core::serialization::load_file_filtered;
use flame_core::{DType, Error, Result, Shape, Tensor};
use std::collections::HashMap;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Layout helpers -- GroupNorm wants NHWC, Conv2d wants NCHW
// ---------------------------------------------------------------------------
//
// The naive 4D permutes `[0,2,3,1]` and `[0,3,1,2]` fall through to
// flame-core's "general permutation fallback" in tensor.rs, which runs a
// GPU→CPU copy + 4D scalar Rust loop + CPU→GPU copy for every call and
// downcasts BF16 to F32 in the process. For `[1,128,1024,1024]` that's
// ~8 s per call; the VAE decoder triggered dozens of them, adding up to
// the 188 s decode. flame-core's other 4D "hot path" `permute_0132` is
// *also* a CPU loop.
//
// The only genuine GPU permute primitive available is `permute_021`
// (via `GpuOps::permute_021`), which swaps the last two axes of a 3D
// BF16-preserving tensor. We reach NCHW↔NHWC with a single GPU permute
// by reshaping to 3D first:
//
//   NCHW [N,C,H,W] → reshape [N,C,H*W] → permute[0,2,1] → [N,H*W,C]
//                  → reshape [N,H,W,C] = NHWC
//   NHWC [N,H,W,C] → reshape [N,H*W,C] → permute[0,2,1] → [N,C,H*W]
//                  → reshape [N,C,H,W] = NCHW
//
// All reshapes are views (input tensors are already contiguous in the
// expected layout). The single `permute_021` call is real GPU work and
// preserves BF16 storage.

/// NCHW -> NHWC. `Tensor::permute` returns a strided VIEW, not a
/// materialized tensor — feeding that view to `group_norm`/`Conv2d`
/// gives garbage because they read storage as if contiguous. Use the
/// dedicated GPU kernel that actually scatters to a contiguous NHWC
/// buffer.
fn to_nhwc(x: &Tensor) -> Result<Tensor> {
    flame_core::cuda_ops::GpuOps::permute_nchw_to_nhwc(x)
}

/// NHWC -> NCHW. Same rationale as `to_nhwc`.
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

/// VAE per-stage telemetry, gated on SDXL_STAGE_DEBUG (same env var as unet
/// so a single run shows both halves). Capped at 20 total calls so the log
/// is readable. When VAE_SAVE_STAGES is set, also dumps each tensor to
/// safetensors for layer-by-layer parity vs diffusers.
fn vae_log_stage(name: &str, t: &Tensor) {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let stats_on = std::env::var("SDXL_STAGE_DEBUG").is_ok();
    let save_on  = std::env::var("VAE_SAVE_STAGES").is_ok();
    if !stats_on && !save_on {
        return;
    }
    static CALL_COUNT: AtomicUsize = AtomicUsize::new(0);
    const MAX_CALLS: usize = 20;
    let idx = CALL_COUNT.fetch_add(1, Ordering::Relaxed);
    if idx >= MAX_CALLS {
        return;
    }
    if save_on {
        use std::collections::HashMap;
        use std::path::Path;
        let safe_name = name.replace(':', "_").replace('.', "_");
        let p_str = format!(
            "/home/alex/EriDiffusion/inference-flame/output/vae_stage_{idx:02}_{safe_name}.safetensors"
        );
        let p = Path::new(&p_str);
        let mut m: HashMap<String, Tensor> = HashMap::new();
        if let Ok(c) = t.clone_result() {
            m.insert("value".to_string(), c);
            let _ = flame_core::serialization::save_file(&m, p);
            eprintln!("[vae-save] {idx:02} {name}");
        }
    }
    if !stats_on {
        return;
    }
    let f32t = match t.to_dtype(DType::F32) {
        Ok(x) => x,
        Err(_) => return,
    };
    let data = match f32t.to_vec() {
        Ok(v) => v,
        Err(_) => return,
    };
    let n = data.len();
    let mut mn = f32::INFINITY;
    let mut mx = f32::NEG_INFINITY;
    let mut abs_sum = 0f64;
    let mut nan_cnt = 0usize;
    let mut inf_cnt = 0usize;
    for v in &data {
        if v.is_nan() {
            nan_cnt += 1;
            continue;
        }
        if v.is_infinite() {
            inf_cnt += 1;
            continue;
        }
        if *v < mn {
            mn = *v;
        }
        if *v > mx {
            mx = *v;
        }
        abs_sum += v.abs() as f64;
    }
    let finite = n - nan_cnt - inf_cnt;
    let abs_mean = if finite > 0 {
        abs_sum / finite as f64
    } else {
        0.0
    };
    let dims = t.shape().dims();
    eprintln!(
        "[stage] {name:<36} shape={dims:?} min={mn:+.4e} max={mx:+.4e} abs_mean={abs_mean:+.4e} nan={nan_cnt} inf={inf_cnt}"
    );
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
            w.get(key)
                .ok_or_else(|| Error::InvalidInput(format!("Missing key: {key}")))
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

    /// Forward: GroupNorm->SiLU->Conv->GroupNorm->SiLU->Conv + residual
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

/// Squeeze Conv2d 1x1 weight [out, in, 1, 1] -> [out, in]
fn squeeze_1x1(t: &Tensor) -> Result<Tensor> {
    let dims = t.shape().dims();
    if dims.len() == 4 && dims[2] == 1 && dims[3] == 1 {
        t.reshape(&[dims[0], dims[1]])
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
            w.get(key)
                .ok_or_else(|| Error::InvalidInput(format!("Missing key: {key}")))
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
            channels,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let (b, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);
        let n = h * w;

        // GroupNorm
        let h_norm = group_norm_nchw(x, 32, Some(&self.norm_w), Some(&self.norm_b), 1e-6)?;

        // [B, C, H, W] -> [B, H*W, C]
        let h_flat = h_norm.permute(&[0, 2, 3, 1])?.reshape(&[b, n, c])?;

        // Q, K, V projections
        let q = linear_3d(&h_flat, &self.q_w, &self.q_b)?;
        let k = linear_3d(&h_flat, &self.k_w, &self.k_b)?;
        let v = linear_3d(&h_flat, &self.v_w, &self.v_b)?;

        // SDPA expects [B, H, N, D] with head_dim in {64, 96, 128} to hit
        // flash attention. The LDM VAE uses single-head attention with
        // head_dim = C = 512, which drops to `forward_bf16_fallback` and
        // materializes the full `[B, 1, N, N]` scores matrix — 1 GB for
        // `N = 16384` at 1024² latent. Multi-head reshape would change
        // the math (softmax is nonlinear over channels); the correct fix
        // is to tile the Q dimension and run the fallback per q-chunk.
        //
        // Per-tile peak: `[B, 1, TILE, N]` F32 scores ≈ `TILE * N * 4` bytes.
        //   N=16384, TILE=1024 → 64 MB per tile (was 1 GB full).
        // The `cuda_ops_bf16::sdpa_stream_bf16` fallback eats the streamed
        // path for true flash on head_dim=512; we tile at the Rust level
        // instead so we stay in the standard materialized path (which is
        // fast once the scores tensor is small enough to fit sensibly).
        let q = q.unsqueeze(1)?; // [B, 1, N, C]
        let k = k.unsqueeze(1)?;
        let v = v.unsqueeze(1)?;

        const ATTN_TILE: usize = 1024;
        let out = if n <= ATTN_TILE {
            // Small enough to run in one shot.
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

        // Output projection
        let out = linear_3d(&out, &self.proj_out_w, &self.proj_out_b)?;

        // [B, N, C] -> [B, C, H, W]. permute() returns a strided view; call
        // the dedicated NHWC->NCHW kernel so the residual `x + out` feeds
        // `add` contiguous memory (else same class of bug as `to_nhwc`).
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
            resnet0: ResBlock::from_weights(
                w,
                &format!("{prefix}.block_1"),
                channels,
                channels,
                device,
            )?,
            attn: AttnBlock::from_weights(w, &format!("{prefix}.attn_1"), channels)?,
            resnet1: ResBlock::from_weights(
                w,
                &format!("{prefix}.block_2"),
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
            w.get(key)
                .ok_or_else(|| Error::InvalidInput(format!("Missing key: {key}")))
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
// Full LDM VAE Decoder (generic, configurable scaling)
// ---------------------------------------------------------------------------

/// LDM-format VAE decoder with configurable scaling and shift factors.
///
/// Works for any model that uses the standard LDM AutoencoderKL decoder layout:
/// Z-Image (16ch, scale=0.3611, shift=0.1159), SD 1.5 (4ch, scale=0.18215),
/// SDXL (4ch, scale=0.13025), etc.
pub struct LdmVAEDecoder {
    /// SDXL / SD 1.5 VAE has a 1x1 post_quant_conv(in_ch, in_ch) applied
    /// to the (un-rescaled) latent before the main decoder. Z-Image disables
    /// it. We construct it when weights are present in the checkpoint.
    post_quant_conv: Option<Conv2d>,
    conv_in: Conv2d,
    mid_block: MidBlock,
    up_blocks: Vec<UpBlock>, // in processing order: up.3, up.2, up.1, up.0
    norm_out_w: Tensor,
    norm_out_b: Tensor,
    conv_out: Conv2d,
    kernels: CudaKernels,
    scaling_factor: f32,
    shift_factor: f32,
}

/// Remap diffusers-format VAE keys to LDM-format.
///
/// Diffusers:  decoder.mid_block.resnets.0.*      → LDM: decoder.mid.block_1.*
/// Diffusers:  decoder.mid_block.attentions.0.*    → LDM: decoder.mid.attn_1.*
/// Diffusers:  decoder.up_blocks.N.resnets.M.*     → LDM: decoder.up.N.block.M.*
/// Diffusers:  decoder.up_blocks.N.upsamplers.0.*  → LDM: decoder.up.N.upsample.*
/// Diffusers:  decoder.conv_norm_out.*              → LDM: decoder.norm_out.*
fn remap_diffusers_to_ldm(w: HashMap<String, Tensor>) -> HashMap<String, Tensor> {
    // Check if keys are diffusers-format (presence of "mid_block" or "up_blocks")
    let is_diffusers = w.keys().any(|k| k.contains("mid_block") || k.contains("up_blocks"));
    if !is_diffusers {
        return w;
    }

    println!("[LdmVAE] Remapping diffusers keys to LDM format");
    let mut out = HashMap::with_capacity(w.len());
    for (key, val) in w {
        let new_key = remap_one_key(&key);
        out.insert(new_key, val);
    }
    out
}

fn remap_one_key(key: &str) -> String {
    let k = key.to_string();

    // decoder.conv_norm_out.* → decoder.norm_out.*
    if k.starts_with("decoder.conv_norm_out.") {
        return k.replace("decoder.conv_norm_out.", "decoder.norm_out.");
    }

    // decoder.mid_block.resnets.0.* → decoder.mid.block_1.*
    // decoder.mid_block.resnets.1.* → decoder.mid.block_2.*
    if k.starts_with("decoder.mid_block.resnets.") {
        let rest = &k["decoder.mid_block.resnets.".len()..];
        if let Some(dot) = rest.find('.') {
            let idx: usize = rest[..dot].parse().unwrap_or(0);
            let suffix = &rest[dot + 1..];
            return format!("decoder.mid.block_{}.{suffix}", idx + 1);
        }
    }

    // decoder.mid_block.attentions.0.group_norm.* → decoder.mid.attn_1.norm.*
    if k.starts_with("decoder.mid_block.attentions.0.group_norm.") {
        let suffix = &k["decoder.mid_block.attentions.0.group_norm.".len()..];
        return format!("decoder.mid.attn_1.norm.{suffix}");
    }
    // decoder.mid_block.attentions.0.to_q.* → decoder.mid.attn_1.q.*
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
    // decoder.mid_block.attentions.0.to_out.0.* → decoder.mid.attn_1.proj_out.*
    if k.starts_with("decoder.mid_block.attentions.0.to_out.0.") {
        let suffix = &k["decoder.mid_block.attentions.0.to_out.0.".len()..];
        return format!("decoder.mid.attn_1.proj_out.{suffix}");
    }

    // decoder.up_blocks.N.resnets.M.* → decoder.up.(3-N).block.M.*
    // Diffusers up_blocks are in reverse order vs LDM up blocks
    if k.starts_with("decoder.up_blocks.") {
        let rest = &k["decoder.up_blocks.".len()..];
        if let Some(dot) = rest.find('.') {
            let diff_idx: usize = rest[..dot].parse().unwrap_or(0);
            let ldm_idx = 3 - diff_idx; // reverse: 0→3, 1→2, 2→1, 3→0
            let block_idx = ldm_idx.to_string();
            let inner = &rest[dot + 1..];

            if inner.starts_with("resnets.") {
                let rr = &inner["resnets.".len()..];
                if let Some(dot2) = rr.find('.') {
                    let resnet_idx = &rr[..dot2];
                    let suffix = &rr[dot2 + 1..];
                    // conv_shortcut → nin_shortcut
                    let suffix = suffix.replace("conv_shortcut.", "nin_shortcut.");
                    return format!("decoder.up.{block_idx}.block.{resnet_idx}.{suffix}");
                }
            }

            // decoder.up_blocks.N.upsamplers.0.conv.* → decoder.up.N.upsample.conv.*
            if inner.starts_with("upsamplers.0.conv.") {
                let suffix = &inner["upsamplers.0.conv.".len()..];
                return format!("decoder.up.{block_idx}.upsample.conv.{suffix}");
            }
        }
    }

    // Unchanged (conv_in, conv_out, etc.)
    k
}

impl LdmVAEDecoder {
    /// Load decoder from safetensors file (mmap, decoder keys only).
    ///
    /// `in_channels` is the latent channel count (16 for Z-Image, 4 for SD/SDXL).
    /// `scaling_factor` and `shift_factor` control the latent normalization:
    ///   z = (z - shift_factor) / scaling_factor
    pub fn from_safetensors(
        path: &str,
        in_channels: usize,
        scaling_factor: f32,
        shift_factor: f32,
        device: &Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        // Support both standalone VAE files (decoder.*) and combined checkpoints
        // (first_stage_model.decoder.*). Strip prefix if present.
        let raw = load_file_filtered(path, device, |key| {
            key.starts_with("decoder.")
                || key.starts_with("first_stage_model.decoder.")
                || key == "post_quant_conv.weight"
                || key == "post_quant_conv.bias"
                || key == "first_stage_model.post_quant_conv.weight"
                || key == "first_stage_model.post_quant_conv.bias"
        })?;
        // Strip "first_stage_model." prefix, and cast every tensor to BF16.
        // Some VAE checkpoints (FLUX 1 ae.safetensors) are F32 on disk and the
        // loader preserves that; `group_norm_bf16` and other BF16 kernels
        // require BF16 storage, so cast here. Same class of cast as the
        // CLIP-L / T5-XXL FP16 → BF16 fix in their respective loaders.
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
        println!("[LdmVAE] Loaded {} decoder weight tensors (cast to BF16)", w.len());
        // Remap diffusers-format keys to LDM-format if needed
        let w = remap_diffusers_to_ldm(w);
        Self::from_weights(w, in_channels, scaling_factor, shift_factor, device)
    }

    /// Build from a pre-loaded weight HashMap.
    pub fn from_weights(
        w: HashMap<String, Tensor>,
        in_channels: usize,
        scaling_factor: f32,
        shift_factor: f32,
        device: &Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        let get = |key: &str| -> Result<&Tensor> {
            w.get(key)
                .ok_or_else(|| Error::InvalidInput(format!("Missing key: {key}")))
        };

        let ch: usize = 128;
        let ch_mult: [usize; 4] = [1, 2, 4, 4];
        let num_resnets: usize = 3; // layers_per_block + 1

        let top_ch = ch * ch_mult[3]; // 512

        // Optional post_quant_conv: 1x1 Conv(in_channels, in_channels) applied
        // pre-decoder. Present in SDXL/SD 1.5 VAE, absent in Z-Image VAE.
        let post_quant_conv = if w.contains_key("post_quant_conv.weight") {
            let mut c = Conv2d::new_with_bias(
                in_channels,
                in_channels,
                1,
                1,
                0,
                device.clone(),
                true,
            )?;
            c.copy_weight_from(get("post_quant_conv.weight")?)?;
            c.copy_bias_from(get("post_quant_conv.bias")?)?;
            println!("[LdmVAE] post_quant_conv loaded");
            Some(c)
        } else {
            None
        };

        // conv_in: in_channels -> 512ch
        let mut conv_in = Conv2d::new_with_bias(in_channels, top_ch, 3, 1, 1, device.clone(), true)?;
        conv_in.copy_weight_from(get("decoder.conv_in.weight")?)?;
        conv_in.copy_bias_from(get("decoder.conv_in.bias")?)?;

        // mid block
        let mid_block = MidBlock::from_weights(&w, "decoder.mid", top_ch, device)?;

        // Up blocks -- process 3->2->1->0
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
            post_quant_conv,
            conv_in,
            mid_block,
            up_blocks,
            norm_out_w: get("decoder.norm_out.weight")?.clone_result()?,
            norm_out_b: get("decoder.norm_out.bias")?.clone_result()?,
            conv_out,
            kernels,
            scaling_factor,
            shift_factor,
        })
    }

    /// Decode latents to RGB.
    ///
    /// Input: `[B, C, H, W]` latent tensor (BF16).
    /// Output: `[B, 3, H*8, W*8]` RGB tensor.
    ///
    /// ## BFL reference — src/flux/modules/autoencoder.py:308-315
    /// ```python
    /// def encode(self, x: Tensor) -> Tensor:
    ///     z = self.reg(self.encoder(x))
    ///     z = self.scale_factor * (z - self.shift_factor)
    ///     return z
    ///
    /// def decode(self, z: Tensor) -> Tensor:
    ///     z = z / self.scale_factor + self.shift_factor
    ///     return self.decoder(z)
    /// ```
    /// Inverse of `encode`: divide by scale first, then add shift.
    pub fn decode(&self, z: &Tensor) -> Result<Tensor> {
        vae_log_stage("vae:0:z_input", z);

        // Undo VAE encode-time normalization: z = z / scale_factor + shift_factor
        let z = z.mul_scalar(1.0 / self.scaling_factor)?
            .add_scalar(self.shift_factor)?;
        vae_log_stage("vae:0:z_rescaled", &z);

        // SDXL / SD 1.5: post_quant_conv (1x1 Conv) applied pre-decoder.
        // Matches diffusers AutoencoderKL._decode which runs post_quant_conv
        // on the already-rescaled latent before passing it to self.decoder.
        let z = if let Some(ref pqc) = self.post_quant_conv {
            let h = pqc.forward(&z)?;
            vae_log_stage("vae:0:post_quant_conv", &h);
            h
        } else {
            z
        };

        // decoder.conv_in
        let mut h = self.conv_in.forward(&z)?;
        vae_log_stage("vae:1:conv_in", &h);

        // mid block
        h = self.mid_block.forward(&h)?;
        vae_log_stage("vae:2:mid_block", &h);

        // up blocks (processed in order: up.3 -> up.2 -> up.1 -> up.0)
        for (i, block) in self.up_blocks.iter().enumerate() {
            h = block.forward(&h, &self.kernels)?;
            vae_log_stage(&format!("vae:3:up_block.{i}"), &h);
        }

        // final norm + silu + conv
        h = group_norm_nchw(&h, 32, Some(&self.norm_out_w), Some(&self.norm_out_b), 1e-6)?;
        vae_log_stage("vae:4:final_gn", &h);
        h = h.silu()?;
        vae_log_stage("vae:4:final_silu", &h);
        h = self.conv_out.forward(&h)?;
        vae_log_stage("vae:4:final_conv", &h);

        Ok(h)
    }
}
