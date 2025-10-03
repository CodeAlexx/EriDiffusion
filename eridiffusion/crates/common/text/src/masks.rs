use anyhow::Result;
use flame_core::{Tensor, Shape, DType};

/// lengths [B] -> padding mask [B,1,1,seq] with 0 for tokens < len, and large negative for pad
pub fn build_padding_mask(lengths: &Tensor, seq: usize) -> Result<Tensor> {
    let mask_bool = build_padding_mask_bool(lengths, seq)?;
    // cast Bool -> F32 and scale pads by -1e4
    let mask_f = mask_bool.to_dtype(DType::F32)?;
    Ok(mask_f.mul_scalar(-1.0e4)?)
}

/// Replace -inf with large negative; clamp extreme values
pub fn sanitize_mask(mask: &Tensor) -> Result<Tensor> {
    let v = mask.to_vec()?;
    let fixed: Vec<f32> = v.into_iter().map(|x| if !x.is_finite() { -1.0e4 } else { x.max(-1.0e6).min(1.0e6) }).collect();
    Ok(Tensor::from_vec(fixed, mask.shape().clone(), mask.device().clone())?)
}

// no duplicate DType import

/// lengths I32[B] -> Bool [B,1,1,seq] where true marks pad positions
pub fn build_padding_mask_bool(lengths: &Tensor, seq: usize) -> Result<Tensor> {
    let b = lengths.shape().dims()[0];
    let lens = lengths.to_dtype(DType::F32)?.to_vec()?;
    let mut buf = Vec::with_capacity(b * seq);
    for i in 0..b {
        let l = lens[i] as usize;
        for j in 0..seq { buf.push(if j < l { 0.0 } else { 1.0 }); }
    }
    let t = Tensor::from_vec(buf, Shape::from_dims(&[b,1,1,seq]), lengths.device().clone())?;
    Ok(t.to_dtype(DType::Bool)?)
}
