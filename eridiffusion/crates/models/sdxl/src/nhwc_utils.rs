use anyhow::Result;
use flame_core::{Tensor, DType};

pub fn nhwc_to_nchw(x: &Tensor) -> Result<Tensor> { Ok(x.permute(&[0, 3, 1, 2])?) }
pub fn nchw_to_nhwc(x: &Tensor) -> Result<Tensor> { Ok(x.permute(&[0, 2, 3, 1])?) }
pub fn silu(x: &Tensor) -> Result<Tensor> { Ok(x.sigmoid()?.mul(x)?) }
/// [KH,KW,IC,OC] → [OC,IC,KH,KW]
pub fn khw_kicoc_to_oihw(w: &Tensor) -> Result<Tensor> { Ok(w.permute(&[3, 2, 0, 1])?) }
/// Broadcast [B,1,1,S] → [B,Hd,HW,S]
pub fn broadcast_mask(mask: &Tensor, heads: usize, hw: usize) -> Result<Tensor> {
    let m = mask
        .repeat(&[1, heads, 1, 1])?
        .repeat(&[1, 1, hw, 1])?;
    Ok(m)
}

/// Concatenate two [B, S, D] tensors along last dimension -> [B, S, D1+D2] (GPU path; no host copy)
pub fn concat_last_dim(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let da = a.shape().dims().to_vec();
    let db = b.shape().dims().to_vec();
    anyhow::ensure!(da.len()==3 && db.len()==3 && da[0]==db[0] && da[1]==db[1],
        "concat_last_dim: expected [B,S,Da]/[B,S,Db], got {:?}/{:?}", da, db);
    let a32 = a.to_dtype(DType::F32)?;
    let b32 = b.to_dtype(DType::F32)?;
    Tensor::cat(&[&a32, &b32], 2)
}
