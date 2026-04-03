use anyhow::Result;
use flame_core::{Device, DType, Tensor, Shape};
use crate::xattn::CrossAttn;

#[test]
fn flash_vs_sdpa_close() -> Result<()> {
    let dev = Device::cuda(0)?;
    let b=1usize; let h=2usize; let w=2usize; let c=64usize; let s=8usize; let hd=16usize; let nh=c/hd;
    // Random projections
    let to = |sh:&[usize]| Tensor::randn(Shape::from_dims(sh), 0.0, 0.02, dev.cuda_device().clone());
    let qw = to(&[c,c])?; let qb = to(&[c])?;
    let kw = to(&[c,c])?; let kb = to(&[c])?;
    let vw = to(&[c,c])?; let vb = to(&[c])?;
    let ow = to(&[c,c])?; let ob = to(&[c])?;
    let att = CrossAttn { to_q_w: qw.clone(), to_q_b: qb.clone(), to_k_w: kw.clone(), to_k_b: kb.clone(), to_v_w: vw.clone(), to_v_b: vb.clone(), to_out_w: ow.clone(), to_out_b: ob.clone(), n_heads: nh, head_dim: hd };
    let x = Tensor::randn(Shape::from_dims(&[b,h,w,c]), 0.0, 0.9, dev.cuda_device().clone())?;
    let ctx = Tensor::randn(Shape::from_dims(&[b,s,c]), 0.0, 0.9, dev.cuda_device().clone())?;
    let mask = Tensor::ones(Shape::from_dims(&[b,1,1,s]), dev.cuda_device().clone())?;
    // SDPA path
    std::env::set_var("ATTENTION_IMPL", "sdpa");
    let y_sdpa = att.forward(&x, &ctx, &mask)?;
    // Flash path
    std::env::set_var("ATTENTION_IMPL", "flash");
    let y_flash = att.forward(&x, &ctx, &mask)?;
    let diff = y_sdpa.to_dtype(DType::F32)?.sub(&y_flash.to_dtype(DType::F32)?)?.abs()?.mean()?.to_scalar::<f32>()?;
    assert!(diff < 1e-2, "Flash vs SDPA diff too large: {}", diff);
    Ok(())
}

