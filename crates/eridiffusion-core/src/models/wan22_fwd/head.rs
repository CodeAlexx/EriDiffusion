//! Pure-GPU head forward and unpatchify for Wan 2.2.
//!
//! Verbatim port of `wan-trainer/src/forward_impl/head.rs`. The head:
//! `LayerNorm → modulate(shift = head_mod[0]+e, scale = head_mod[1]+e) → linear`,
//! followed by patch unembedding.

use flame_core::{DType, Error, Result, Tensor};

/// Head forward: layer-norm → modulate → linear.
///
/// * `x`         — `[1, seq_len, dim]` BF16 (final block output)
/// * `e`         — `[1, seq_len, dim]` BF16 (time embedding, pre-projection)
/// * `head_mod`  — `[1, 2, dim]` learnable additive table
/// * `head_w`    — `[out_features, dim]` BF16 proj weight
/// * `head_b`    — `[out_features]` BF16 proj bias
/// * `eps`       — LN epsilon
///
/// Returns `[1, seq_len, out_features]` BF16.
pub fn head_forward(
    x: &Tensor,
    e: &Tensor,
    head_mod: &Tensor,
    head_w: &Tensor,
    head_b: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    let dims = x.shape().dims();
    if dims.len() != 3 {
        return Err(Error::InvalidInput(format!(
            "head_forward expects [1, seq, dim], got {dims:?}"
        )));
    }
    let sl = dims[1];
    let dim = dims[2];

    let head_mod_dims = head_mod.shape().dims();
    if head_mod_dims.len() != 3 || head_mod_dims[1] != 2 || head_mod_dims[2] != dim {
        return Err(Error::InvalidInput(format!(
            "head_forward: head_mod must be [1, 2, dim={dim}], got {head_mod_dims:?}"
        )));
    }
    let shift_base = head_mod.narrow(1, 0, 1)?; // [1, 1, dim]
    let scale_base = head_mod.narrow(1, 1, 1)?; // [1, 1, dim]

    let shift = shift_base.add(e)?;
    let scale = scale_base.add(e)?;

    let normed = flame_core::layer_norm::layer_norm(x, &[dim], None, None, eps)?;

    let one_plus = scale.add_scalar(1.0)?;
    let head_in = normed.mul(&one_plus)?.add(&shift)?;

    let head_w_t = head_w.to_dtype(DType::BF16)?.transpose()?;
    let head_in_2d = head_in.reshape(&[sl, dim])?;
    let out_2d = head_in_2d.matmul(&head_w_t)?;
    let out_dim = out_2d.shape().dims()[1];
    let out_3d = out_2d.reshape(&[1, sl, out_dim])?;
    out_3d.add(head_b)
}

/// Trim padding tokens and unpatchify.
///
/// Input: `[1, seq_len, c_out * p_t * p_h * p_w]` (padded)
/// Output: `[c_out, F * p_t, H * p_h, W * p_w]`
pub fn unpatchify(
    x: &Tensor,
    n_patches: usize,
    grid: (usize, usize, usize),
    patch: (usize, usize, usize),
    out_channels: usize,
) -> Result<Tensor> {
    let dims = x.shape().dims();
    if dims.len() != 3 {
        return Err(Error::InvalidInput(format!(
            "unpatchify expects [1, seq, dim_flat], got {dims:?}"
        )));
    }
    if dims[1] < n_patches {
        return Err(Error::InvalidInput(format!(
            "unpatchify: seq_len {} < n_patches {n_patches}",
            dims[1]
        )));
    }
    let (fo, ho, wo) = grid;
    let (pt, ph, pw) = patch;
    let patch_dim = out_channels * pt * ph * pw;
    if dims[2] != patch_dim {
        return Err(Error::InvalidInput(format!(
            "unpatchify: expected last dim = out_ch * prod(patch) = {patch_dim}, got {}",
            dims[2]
        )));
    }
    if fo * ho * wo != n_patches {
        return Err(Error::InvalidInput(format!(
            "unpatchify: grid {fo}×{ho}×{wo} != n_patches {n_patches}"
        )));
    }

    // Trim padding first.
    let trimmed = x.narrow(1, 0, n_patches)?; // [1, n_patches, patch_dim]

    // View as [fo, ho, wo, pt, ph, pw, c_out] then einsum 'fhwpqrc -> cfphqwr'.
    let as_7d = trimmed.reshape(&[fo, ho, wo, pt, ph, pw, out_channels])?;
    let permuted = as_7d.permute(&[6, 0, 3, 1, 4, 2, 5])?;
    permuted.reshape(&[out_channels, fo * pt, ho * ph, wo * pw])
}
