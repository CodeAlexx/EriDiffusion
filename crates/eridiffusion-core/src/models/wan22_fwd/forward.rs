//! Top-level training forward for Wan 2.2.
//!
//! Port of `wan-trainer/src/forward_impl/forward.rs`. The archive depended
//! on `inference_flame::models::wan22_dit::Wan22Dit` for helpers
//! (`compute_embeddings`, `patchify_public`, `linear_bias_pub`,
//! `shared_weight`); here those helpers are re-implemented as free
//! functions that read directly from a `&HashMap<String, Tensor>` of
//! shared (non-block) weights.
//!
//! Inputs:
//! - `x`         — `[C, F, H, W]` BF16 noised latent (no_grad)
//! - `timestep`  — scalar f32 in `[0, 1000]`
//! - `context`   — `[1, text_len, 4096]` BF16 UMT5 embedding
//! - `seq_len`   — must equal `(F/p_t) * (H/p_h) * (W/p_w)` (training contract)
//!
//! Returns predicted velocity `[C_out, F, H, W]` BF16.

use flame_core::{DType, Error, Result, Shape, Tensor};
use std::collections::HashMap;

use super::super::wan22::{Wan22Config, Wan22LoraBundle};
use super::{block, head, rope::WanRope};

/// Sinusoidal timestep embedding matching Wan's `sinusoidal_embedding_1d`.
/// Output: `[N, freq_dim]` F32.
fn sinusoidal_embedding(
    timesteps: &[f32],
    freq_dim: usize,
    device: std::sync::Arc<cudarc::driver::CudaDevice>,
) -> Result<Tensor> {
    let half = freq_dim / 2;
    let n = timesteps.len();

    let mut data = vec![0.0f32; n * freq_dim];
    for (t_idx, &pos) in timesteps.iter().enumerate() {
        let pos = pos as f64;
        for i in 0..half {
            let freq = 10000.0f64.powf(-(i as f64) / half as f64);
            let angle = pos * freq;
            data[t_idx * freq_dim + i] = angle.cos() as f32;
            data[t_idx * freq_dim + half + i] = angle.sin() as f32;
        }
    }

    Tensor::from_vec(data, Shape::from_dims(&[n, freq_dim]), device)
}

/// Wan linear with bias: `y = x @ W^T + b` via autograd-aware matmul.
/// Same convention as `block::linear_bias` but free here so `forward` can
/// call it without a back-import; weights are `[Cout, Cin]` PyTorch-format.
fn linear_bias(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    if dims.len() != 3 {
        return Err(Error::InvalidInput(format!(
            "forward::linear_bias expects [B, N, C], got {dims:?}"
        )));
    }
    let (b, n, c) = (dims[0], dims[1], dims[2]);
    let weight_t = weight.transpose()?;
    let x_2d = x.reshape(&[b * n, c])?;
    let out_2d = x_2d.matmul(&weight_t)?;
    let out_cout = out_2d.shape().dims()[1];
    let out = out_2d.reshape(&[b, n, out_cout])?;
    out.add(bias)
}

/// Compute time + text embeddings.
/// Returns (e, e0, txt):
/// - e:  `[1, 1, dim]` BF16 — time embedding (per-batch; broadcasts in head)
/// - e0: `[1, 1, 6, dim]` F32 — time projection (per-batch modulation table;
///       broadcasts in block::build_modulation)
/// - txt:`[1, text_len, dim]` BF16 — text embedding
///
/// Audit H3 (2026-05-09): the prior version produced `[1, seq_len, ...]`
/// time tensors, which at video seq lengths (~130K tokens for 480p)
/// allocated ~16 GB of F32 e0 per forward — the ENTIRE timestep is the
/// SAME scalar across all tokens here (no image-conditioning path), so
/// computing the MLP once and broadcasting saves the seq_len factor with
/// zero precision impact. Diffusers does the same except for TI2V-5B
/// with image conditioning (which we don't have wired). Block- and
/// head-level modulation already use broadcast-compatible adds, so the
/// downstream code path is unchanged.
fn compute_embeddings(
    cfg: &Wan22Config,
    weights: &HashMap<String, Tensor>,
    timestep: f32,
    context: &Tensor,
    _seq_len: usize, // kept for ABI; per-batch path doesn't need it
    device: std::sync::Arc<cudarc::driver::CudaDevice>,
) -> Result<(Tensor, Tensor, Tensor)> {
    // Time embedding — single scalar timestep, single sinusoid row.
    let t_vals = vec![timestep];
    let sin_emb = sinusoidal_embedding(&t_vals, cfg.freq_dim, device.clone())?;
    // [1, freq_dim] → [1, 1, freq_dim] BF16
    let sin_emb_bf16 = sin_emb.to_dtype(DType::BF16)?.unsqueeze(0)?;

    let get = |k: &str| -> Result<&Tensor> {
        weights
            .get(k)
            .ok_or_else(|| Error::InvalidInput(format!("compute_embeddings: missing weight '{k}'")))
    };
    let te_w0 = get("time_embedding.0.weight")?;
    let te_b0 = get("time_embedding.0.bias")?;
    let te_w2 = get("time_embedding.2.weight")?;
    let te_b2 = get("time_embedding.2.bias")?;
    let e = linear_bias(&sin_emb_bf16, te_w0, te_b0)?; // [1, 1, dim]
    let e = e.silu()?;
    let e = linear_bias(&e, te_w2, te_b2)?; // [1, 1, dim]

    let tp_w = get("time_projection.1.weight")?;
    let tp_b = get("time_projection.1.bias")?;
    let e_silu = e.silu()?;
    let e0_flat = linear_bias(&e_silu, tp_w, tp_b)?; // [1, 1, 6*dim]
    let e0 = e0_flat
        .reshape(&[1, 1, 6, cfg.dim])?
        .to_dtype(DType::F32)?;

    // Text embedding
    let txt_w0 = get("text_embedding.0.weight")?;
    let txt_b0 = get("text_embedding.0.bias")?;
    let txt_w2 = get("text_embedding.2.weight")?;
    let txt_b2 = get("text_embedding.2.bias")?;
    let ctx_len = context.shape().dims()[1];
    let ctx_padded = if ctx_len < cfg.text_len {
        let pad = Tensor::zeros_dtype(
            Shape::from_dims(&[1, cfg.text_len - ctx_len, cfg.text_dim]),
            context.dtype(),
            device.clone(),
        )?;
        Tensor::cat(&[context, &pad], 1)?
    } else {
        context.narrow(1, 0, cfg.text_len)?
    };
    let txt = linear_bias(&ctx_padded, txt_w0, txt_b0)?;
    let txt = txt.gelu()?;
    let txt = linear_bias(&txt, txt_w2, txt_b2)?;

    Ok((e, e0, txt))
}

/// Patchify `[C, F, H, W]` BF16 latent into `[n_patches, C * p_t * p_h * p_w]` BF16.
/// Host-loop port — input is a no_grad VAE latent so the host round-trip is
/// harmless. Mirrors `Wan22Dit::patchify` (inference-flame).
fn patchify(
    x: &Tensor,
    f: usize,
    h: usize,
    w: usize,
    patch_size: [usize; 3],
    device: std::sync::Arc<cudarc::driver::CudaDevice>,
) -> Result<Tensor> {
    let c = x.shape().dims()[0];
    let (pf, ph, pw) = (patch_size[0], patch_size[1], patch_size[2]);
    let fo = f / pf;
    let ho = h / ph;
    let wo = w / pw;
    let patch_dim = c * pf * ph * pw;
    let n_patches = fo * ho * wo;

    let x_data = x.to_dtype(DType::F32)?.to_vec()?;
    let mut out = vec![0.0f32; n_patches * patch_dim];

    for fi in 0..fo {
        for hi in 0..ho {
            for wi in 0..wo {
                let patch_idx = fi * ho * wo + hi * wo + wi;
                for pfi in 0..pf {
                    for phi in 0..ph {
                        for pwi in 0..pw {
                            for ci in 0..c {
                                let src_f = fi * pf + pfi;
                                let src_h = hi * ph + phi;
                                let src_w = wi * pw + pwi;
                                let src_idx =
                                    ci * f * h * w + src_f * h * w + src_h * w + src_w;
                                let dst_ch = ci * pf * ph * pw + pfi * ph * pw + phi * pw + pwi;
                                out[patch_idx * patch_dim + dst_ch] = x_data[src_idx];
                            }
                        }
                    }
                }
            }
        }
    }

    let out_f32 = Tensor::from_vec(
        out,
        Shape::from_dims(&[n_patches, patch_dim]),
        device,
    )?;
    out_f32.to_dtype(DType::BF16)
}

/// Top-level training forward.
///
/// Required: `seq_len == n_patches` (no inference-style padding at training).
///
/// `text_mask`: optional `[1, text_len]` F32 (1=real, 0=pad). When set,
/// builds the cross-attention `[1, 1, n_patches, text_len]` BF16 mask and
/// threads it into every block — fixes audit H1 (padded text positions
/// dilute real-text conditioning when None).
pub fn forward_with_lora(
    cfg: &Wan22Config,
    weights: &HashMap<String, Tensor>,
    bundle: &Wan22LoraBundle,
    x: &Tensor,
    timestep: f32,
    context: &Tensor,
    seq_len: usize,
    text_mask: Option<&Tensor>,
    offloader: Option<&std::sync::Arc<std::sync::Mutex<crate::training::block_offload::BlockOffloader>>>,
) -> Result<Tensor> {
    let dim = cfg.dim;
    let num_heads = cfg.num_heads;
    let head_dim = cfg.head_dim;
    let eps = cfg.eps;
    let device = x.device().clone();

    // 1) Patchify
    let x_dims = x.shape().dims().to_vec();
    if x_dims.len() != 4 {
        return Err(Error::InvalidInput(format!(
            "forward_with_lora: x must be [C, F, H, W], got {x_dims:?}"
        )));
    }
    let (c_in, f_in, h_in, w_in) = (x_dims[0], x_dims[1], x_dims[2], x_dims[3]);
    let (pt, ph, pw) = (cfg.patch_size[0], cfg.patch_size[1], cfg.patch_size[2]);
    let fo = f_in / pt;
    let ho = h_in / ph;
    let wo = w_in / pw;
    let n_patches = fo * ho * wo;
    let grid_sizes = (fo, ho, wo);
    let patch_dim = c_in * pt * ph * pw;

    if seq_len != n_patches {
        return Err(Error::InvalidInput(format!(
            "forward_with_lora (training): expected seq_len == n_patches ({n_patches}), got seq_len = {seq_len}. \
             Pass the exact patch count — training doesn't need inference-style padding."
        )));
    }

    let patched = patchify(x, f_in, h_in, w_in, cfg.patch_size, device.clone())?;

    // Conv3d-as-linear: [1, n_patches, patch_dim] → [1, n_patches, dim]
    let pe_w = weights
        .get("patch_embedding.weight")
        .ok_or_else(|| Error::InvalidInput("missing patch_embedding.weight".into()))?;
    let pe_b = weights
        .get("patch_embedding.bias")
        .ok_or_else(|| Error::InvalidInput("missing patch_embedding.bias".into()))?;
    let pe_w_flat = pe_w.reshape(&[dim, patch_dim])?;
    let patched_3d = patched.unsqueeze(0)?; // [1, n_patches, patch_dim]
    let img = linear_bias(&patched_3d, &pe_w_flat, pe_b)?; // [1, n_patches, dim]

    // 2) Time + text embeddings
    let (e, e0, txt) =
        compute_embeddings(cfg, weights, timestep, context, seq_len, device.clone())?;

    // 3) Precompute RoPE table
    let rope = WanRope::new(grid_sizes, head_dim, cfg.rope_theta, device.clone())?;

    // 3.5) Build cross-attention text padding mask once per forward (audit H1).
    // text_mask is [1, text_len] F32 (1=real, 0=pad). SDPA needs a 4D mask
    // shape `[B, 1, Sq, Sk]`; we broadcast on the Q (n_patches) dim by
    // shaping `[1, 1, 1, text_len]` BF16 — the SDPA path expands it
    // implicitly. When text_mask is absent, fall back to the prior None
    // behavior (full padded text contributes; convergence-noisy but works).
    let cross_attn_mask: Option<Tensor> = match text_mask {
        Some(m) => {
            let dims = m.shape().dims();
            if dims.len() != 2 || dims[1] != cfg.text_len {
                return Err(Error::InvalidInput(format!(
                    "text_mask must be [1, text_len={}], got {:?}",
                    cfg.text_len, dims
                )));
            }
            let m_bf16 = m.to_dtype(DType::BF16)?;
            Some(m_bf16.reshape(&[1, 1, 1, cfg.text_len])?)
        }
        None => None,
    };
    let cross_attn_mask_ref = cross_attn_mask.as_ref();

    // 4) Transformer blocks
    let n_blocks = std::env::var("WAN_MAX_BLOCKS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .map(|n| n.min(cfg.num_layers))
        .unwrap_or(cfg.num_layers);
    if n_blocks != cfg.num_layers {
        log::warn!("[wan22] WAN_MAX_BLOCKS={n_blocks} (debug — forward stops early)");
    }

    let mut img = img;
    if let Some(off) = offloader {
        // BlockOffloader path: per-block fetch from pinned CPU + run inside
        // checkpoint_offload so backward can refetch and the GPU slot reuse
        // doesn't dangle the autograd tape's saved-tensor references.
        let bundle_arc = std::sync::Arc::new(bundle.clone());
        for i in 0..n_blocks {
            let img_c = img.clone();
            let e0_c = e0.clone();
            let txt_c = txt.clone();
            let rope_c = rope.clone();
            let mask_c: Option<Tensor> = cross_attn_mask.clone();
            let off_c = off.clone();
            let bundle_c = bundle_arc.clone();
            let dev_c = device.clone();
            let bi = i;
            img = flame_core::autograd::AutogradContext::checkpoint_offload(
                &[img_c.clone()],
                move || {
                    let w = off_c.lock().unwrap().ensure_block(bi)
                        .map_err(|e| flame_core::Error::InvalidInput(format!("ensure_block({bi}): {e}")))?;
                    block::block_forward_with_lora(
                        &img_c, &e0_c, &txt_c, &rope_c, eps, num_heads, head_dim,
                        &w, &bundle_c, bi,
                        mask_c.as_ref(), &dev_c,
                    )
                },
            )?;
        }
    } else {
        // Resident path: weights map already holds all block weights.
        for i in 0..n_blocks {
            img = block::block_forward_with_lora(
                &img, &e0, &txt, &rope, eps, num_heads, head_dim, weights, bundle, i,
                cross_attn_mask_ref, &device,
            )?;
        }
    }

    // 5) Head
    let head_mod = weights
        .get("head.modulation")
        .ok_or_else(|| Error::InvalidInput("missing head.modulation".into()))?;
    let head_w = weights
        .get("head.head.weight")
        .ok_or_else(|| Error::InvalidInput("missing head.head.weight".into()))?;
    let head_b = weights
        .get("head.head.bias")
        .ok_or_else(|| Error::InvalidInput("missing head.head.bias".into()))?;
    let head_out = head::head_forward(&img, &e, head_mod, head_w, head_b, eps)?;

    // 6) Unpatchify
    head::unpatchify(
        &head_out,
        n_patches,
        grid_sizes,
        (pt, ph, pw),
        cfg.out_channels,
    )
}
