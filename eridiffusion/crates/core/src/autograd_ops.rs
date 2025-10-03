//! Autograd helpers and reference backward implementations

use crate::{Error, Result};
use flame_core::Tensor;

/// Scaled dot-product attention backward via recomputation path.
/// Shapes: Q[B,H,Sq,D], K[B,H,Sk,D], V[B,H,Sk,D], dO[B,H,Sq,D]
/// Returns (dQ[B,H,Sq,D], dK[B,H,Sk,D], dV[B,H,Sk,D]).
/// mask is an optional additive mask broadcastable to [B,H,Sq,Sk].
pub fn attention_backward_recompute(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    dout: &Tensor,
    mask: Option<&Tensor>,
    scale: f32,
) -> Result<(Tensor, Tensor, Tensor)> {
    // 1) logits = (Q K^T) * scale [+ mask]
    let kt = k.transpose_dims(2, 3).map_err(|e| Error::TensorOp(e.to_string()))?; // [B,H,D,Sk]
    let mut logits = q
        .bmm(&kt)
        .map_err(|e| Error::TensorOp(e.to_string()))?; // [B,H,Sq,Sk]
    logits = logits
        .mul_scalar(scale)
        .map_err(|e| Error::TensorOp(e.to_string()))?;
    if let Some(m) = mask {
        logits = logits.add(m).map_err(|e| Error::TensorOp(e.to_string()))?;
    }

    // 2) attn = softmax(logits)
    let attn = logits.softmax(-1).map_err(|e| Error::TensorOp(e.to_string()))?; // [B,H,Sq,Sk]

    // 3) dV = attn^T @ dO
    let attn_t = attn
        .transpose_dims(2, 3)
        .map_err(|e| Error::TensorOp(e.to_string()))?; // [B,H,Sk,Sq]
    let d_v = attn_t
        .bmm(dout)
        .map_err(|e| Error::TensorOp(e.to_string()))?; // [B,H,Sk,D]

    // 4) dAttn = dO @ V^T
    let vt = v
        .transpose_dims(2, 3)
        .map_err(|e| Error::TensorOp(e.to_string()))?; // [B,H,D,Sk]
    let d_attn = dout
        .bmm(&vt)
        .map_err(|e| Error::TensorOp(e.to_string()))?; // [B,H,Sq,Sk]

    // 5) Softmax backward: dLogits = (dAttn - sum(dAttn*attn, -1, keepdim)) * attn
    let dattn_times_attn = d_attn
        .mul(&attn)
        .map_err(|e| Error::TensorOp(e.to_string()))?; // [B,H,Sq,Sk]
    let sum_term = dattn_times_attn
        .sum_dim_keepdim(3)
        .map_err(|e| Error::TensorOp(e.to_string()))?; // [B,H,Sq,1]
    let d_logits = d_attn
        .sub(&sum_term)
        .and_then(|x| x.mul(&attn))
        .map_err(|e| Error::TensorOp(e.to_string()))?; // [B,H,Sq,Sk]

    // 6) dQ = dLogits @ K
    let d_q = d_logits
        .bmm(k)
        .map_err(|e| Error::TensorOp(e.to_string()))?; // [B,H,Sq,D]

    // 7) dK = dLogits^T @ Q
    let d_logits_t = d_logits
        .transpose_dims(2, 3)
        .map_err(|e| Error::TensorOp(e.to_string()))?; // [B,H,Sk,Sq]
    let d_k = d_logits_t
        .bmm(q)
        .map_err(|e| Error::TensorOp(e.to_string()))?; // [B,H,Sk,D]

    Ok((d_q, d_k, d_v))
}

