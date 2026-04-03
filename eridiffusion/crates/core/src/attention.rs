//! FlashAttention-like packed attention with SDPA fallback.
//!
//! API:
//!   fn flash_attention_packed(
//!       qkv: &Tensor, q_lens: Option<&[i32]>, k_lens: Option<&[i32]>,
//!       num_heads: i32, scale: f32, attn_mask: Option<&Tensor>
//!   ) -> Result<Tensor>
//!
//! Layout expectation:
//! - qkv shape: [S, 3, H, D]
//!   * If q_lens and k_lens provided and sum(q_lens)+sum(k_lens) == S,
//!     then rows [0..sum(q_lens)) hold Q, and rows [sum(q_lens)..S) hold K and V (stacked mode, ragged supported).
//!   * Otherwise, treat as shared mode (self-attn): Q, K, V share the same S tokens.
//! - Returns packed output concatenated for all Q tokens: [sum(q_lens) or S, H, D] with input dtype.
//! - Supports BF16 inputs; reductions run in FP32.

use crate::{Result, Error};
use flame_core::{Tensor, DType};
// FlashAttention is optional; import only when feature is enabled
#[cfg(feature = "flash_attn")]
use flame_core::flash_attention::flash_attention_forward;

fn sum_i32(xs: &[i32]) -> usize { xs.iter().map(|&v| v as usize).sum::<usize>() }

fn sdpa_attention_packed_impl(
    qkv: &Tensor,
    q_lens: Option<&[i32]>,
    k_lens: Option<&[i32]>,
    num_heads: i32,
    scale: f32,
    attn_mask: Option<&Tensor>,
) -> Result<Tensor> {
    let h = num_heads as usize;
    let dims = qkv.shape().dims().to_vec();
    if dims.len() != 4 || dims[1] != 3 || dims[2] != h {
        return Err(Error::TensorOp(format!(
            "flash_attention_packed: expected qkv [S,3,H,D]; got shape {:?}", dims
        )));
    }
    let s = dims[0];

    let stacked = match (q_lens, k_lens) {
        (Some(qs), Some(ks)) if sum_i32(qs) + sum_i32(ks) == s => true,
        _ => false,
    };

    let (q_rows_total, k_rows_total) = if stacked {
        let (qs, ks) = match (q_lens, k_lens) {
            (Some(qs), Some(ks)) => (qs, ks),
            _ => return Err(Error::InvalidInput("stacked=true but missing lens".into())),
        };
        (sum_i32(qs), sum_i32(ks))
    } else {
        (s, s)
    };

    // Extract packed Q/K/V views according to mode.
    let q_pack = if stacked {
        qkv.narrow(0, 0, q_rows_total)?
    } else { qkv.clone() };
    let kv_pack = if stacked {
        qkv.narrow(0, q_rows_total, k_rows_total)?
    } else { qkv.clone() };

    let q_all = q_pack.narrow(1, 0, 1)?.squeeze(Some(1))?; // [Sq, H, D]
    let k_all = kv_pack.narrow(1, 1, 1)?.squeeze(Some(1))?; // [Sk, H, D]
    let v_all = kv_pack.narrow(1, 2, 1)?.squeeze(Some(1))?; // [Sk, H, D]

    // Helper: compute for a single sequence segment
    let compute_seq = |q_seq: Tensor, k_seq: Tensor, v_seq: Tensor| -> Result<Tensor> {
        // q_seq: [Q, H, D], k_seq/v_seq: [K, H, D]
        let q32 = q_seq.to_dtype(DType::F32)?;
        let k32 = k_seq.to_dtype(DType::F32)?;
        let v32 = v_seq.to_dtype(DType::F32)?;
        let q_bhd = q32.permute(&[1, 0, 2])?; // [H,Q,D]
        let k_bhd = k32.permute(&[1, 0, 2])?; // [H,K,D]
        let v_bhd = v32.permute(&[1, 0, 2])?; // [H,K,D]
        let kt = k_bhd.transpose_dims(1, 2)?;  // [H,D,K]
        let mut scores = q_bhd.bmm(&kt)?;      // [H,Q,K]
        if scale != 0.0 { scores = scores.mul_scalar(scale)?; }
        // Apply additive mask if provided; broadcast if needed
        let scores = if let Some(m) = attn_mask {
            // Try to broadcast mask to [H,Q,K]
            let md = m.shape().dims().to_vec();
            match md.len() {
                2 => {
                    // [Q,K]
                    let m_b = m.reshape(&[1, md[0], md[1]])?;
                    scores.add(&m_b)?
                }
                3 => {
                    scores.add(m)?
                }
                _ => scores, // ignore incompatible mask
            }
        } else { scores };
        let attn = scores.softmax(-1)?;        // [H,Q,K]
        let ctx = attn.bmm(&v_bhd)?;           // [H,Q,D]
        let out = ctx.permute(&[1, 0, 2])?;    // [Q,H,D]
        Ok(out)
    };

    // If ragged, iterate segments
    let out = if stacked {
        let mut q_start = 0usize;
        let mut k_start = 0usize;
        let (qs, ks) = match (q_lens, k_lens) {
            (Some(qs), Some(ks)) => (qs, ks),
            _ => return Err(Error::InvalidInput("stacked=true but missing lens".into())),
        };
        let mut parts: Vec<Tensor> = Vec::with_capacity(qs.len());
        for i in 0..qs.len() {
            let qlen = qs[i] as usize;
            let klen = ks[i] as usize;
            let q_seq = q_all.narrow(0, q_start, qlen)?; // [qlen,H,D]
            let k_seq = k_all.narrow(0, k_start, klen)?; // [klen,H,D]
            let v_seq = v_all.narrow(0, k_start, klen)?; // [klen,H,D]
            let part = compute_seq(q_seq, k_seq, v_seq)?;
            parts.push(part);
            q_start += qlen; k_start += klen;
        }
        // Concatenate along the sequence dimension
        let refs: Vec<&Tensor> = parts.iter().collect();
        Tensor::cat(&refs, 0)?
    } else {
        compute_seq(q_all, k_all, v_all)?
    };

    // Cast back to input dtype if needed
    let out = if qkv.dtype() != DType::F32 { out.to_dtype(qkv.dtype())? } else { out };
    Ok(out)
}

/// Public SDPA packed attention (identical API but fallible)
pub fn sdpa_attention_packed(
    qkv: &Tensor, q_lens: Option<&[i32]>, k_lens: Option<&[i32]>,
    num_heads: i32, scale: f32, attn_mask: Option<&Tensor>
) -> Result<Tensor> {
    let out = sdpa_attention_packed_impl(qkv, q_lens, k_lens, num_heads, scale, attn_mask)?;
    Ok(out)
}

/// FlashAttention (packed) — currently routes to SDPA for correctness.
pub fn flash_attention_packed(
    qkv: &Tensor, q_lens: Option<&[i32]>, k_lens: Option<&[i32]>,
    num_heads: i32, scale: f32, attn_mask: Option<&Tensor>
) -> Result<Tensor> {
    // Config knob via env var (ATTENTION_IMPL=flash|sdpa). If flash not compiled or fails, fallback to SDPA.
    let want_flash = matches!(std::env::var("ATTENTION_IMPL").ok().as_deref(), Some("flash"));
    if want_flash {
        // Only attempt flash when feature is enabled
        #[cfg(feature = "flash_attn")]
        {
            let h = num_heads as usize;
            let dims = qkv.shape().dims().to_vec();
            if dims.len() == 4 && dims[1] == 3 && dims[2] == h {
                let d = dims[3];
                // Unpack
                let q_all = match qkv.narrow(1, 0, 1).and_then(|t| t.squeeze(Some(1))) { Ok(t) => t, Err(_) => return sdpa_attention_packed(qkv, q_lens, k_lens, num_heads, scale, attn_mask) };
                let k_all = match qkv.narrow(1, 1, 1).and_then(|t| t.squeeze(Some(1))) { Ok(t) => t, Err(_) => return sdpa_attention_packed(qkv, q_lens, k_lens, num_heads, scale, attn_mask) };
                let v_all = match qkv.narrow(1, 2, 1).and_then(|t| t.squeeze(Some(1))) { Ok(t) => t, Err(_) => return sdpa_attention_packed(qkv, q_lens, k_lens, num_heads, scale, attn_mask) };

                let out_res: Result<Tensor> = (|| {
                    if let (Some(qs), Some(ks)) = (q_lens, k_lens) {
                        // Ragged segments
                        let mut q_start = 0usize; let mut k_start = 0usize;
                        let mut outs: Vec<Tensor> = Vec::with_capacity(qs.len());
                        for i in 0..qs.len() {
                            let (qlen, klen) = (qs[i] as usize, ks[i] as usize);
                            let q_seq = q_all.narrow(0, q_start, qlen)?; // [Q,H,D]
                            let k_seq = k_all.narrow(0, k_start, klen)?; // [K,H,D]
                            let v_seq = v_all.narrow(0, k_start, klen)?; // [K,H,D]
                            let q_b = q_seq.permute(&[1,0,2])?.reshape(&[1, h, qlen, d])?;
                            let k_b = k_seq.permute(&[1,0,2])?.reshape(&[1, h, klen, d])?;
                            let v_b = v_seq.permute(&[1,0,2])?.reshape(&[1, h, klen, d])?;
                            let o_b = flash_attention_forward(&q_b, &k_b, &v_b, None, Some(scale), false)
                                .map_err(|e| Error::Training(e.to_string()))?;
                            let o_qhd = o_b.reshape(&[h, qlen, d])?.permute(&[1,0,2])?;
                            outs.push(o_qhd);
                            q_start += qlen; k_start += klen;
                        }
                        let refs: Vec<&Tensor> = outs.iter().collect();
                        Tensor::cat(&refs, 0)
                    } else {
                        // Shared self-attn
                        let qlen = q_all.shape().dims()[0]; let klen = k_all.shape().dims()[0];
                        let q_b = q_all.permute(&[1,0,2])?.reshape(&[1, h, qlen, d])?;
                        let k_b = k_all.permute(&[1,0,2])?.reshape(&[1, h, klen, d])?;
                        let v_b = v_all.permute(&[1,0,2])?.reshape(&[1, h, klen, d])?;
                        let o_b = flash_attention_forward(&q_b, &k_b, &v_b, None, Some(scale), false)
                            .map_err(|e| Error::Training(e.to_string()))?;
                        o_b.reshape(&[h, qlen, d])?.permute(&[1,0,2])
                    }
                })();
                if let Ok(out) = out_res { return Ok(out); }
            }
        }
        // if feature not enabled or failed → fall through
    }
    sdpa_attention_packed(qkv, q_lens, k_lens, num_heads, scale, attn_mask)
}
