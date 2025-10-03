use anyhow::Result;
use flame_core::{Tensor, DType};
use crate::xattn::CrossAttn;
use adapters::adapter::AdapterSet;
use crate::nhwc_utils::concat_last_dim;

pub struct STBlock {
    pub self_attn: Option<CrossAttn>,
    pub xattn1: CrossAttn,
    pub xattn2: CrossAttn,
    pub ff_fc1_w: Tensor, pub ff_fc1_b: Tensor,
    pub ff_fc2_w: Tensor, pub ff_fc2_b: Tensor,
    pub key_prefix: String,
}

pub struct SpatialTransformer {
    pub proj_in_w: Tensor, pub proj_in_b: Tensor,
    pub blocks: Vec<STBlock>,
    pub proj_out_w: Tensor, pub proj_out_b: Tensor,
}

impl SpatialTransformer {
    pub fn forward(&self, x_nhwc:&Tensor, ctx1:&Tensor, ctx2:&Tensor, mask_f32:&Tensor, adapters: Option<&AdapterSet>) -> Result<Tensor> {
        let sh = x_nhwc.shape().dims().to_vec();
        let (b,h,w,c) = (sh[0], sh[1], sh[2], sh[3]);
        // proj_in
        let mut y = x_nhwc.to_dtype(DType::F32)?.reshape(&[b, (h*w), c])?
                    .matmul(&self.proj_in_w)?.add(&self.proj_in_b)?; // [B,HW,C]
        // blocks
        for bl in &self.blocks {
            let y_nhwc = y.reshape(&[b,h,w,c])?;
            // SDXL: concatenate contexts along last dim to 2048 once per block
            let ctx_cat = concat_last_dim(ctx1, ctx2)?; // [B,S,2048]
            // Single cross-attn over concatenated context
            let y1 = bl.xattn1.forward(&y_nhwc, &ctx_cat, mask_f32, adapters, &bl.key_prefix)?;
            let z  = y1.to_dtype(DType::F32)?.reshape(&[b,(h*w),c])?;
            let z1 = z.matmul(&bl.ff_fc1_w)?.add(&bl.ff_fc1_b)?.silu()?;
            y = z1.matmul(&bl.ff_fc2_w)?.add(&bl.ff_fc2_b)?; // [B,HW,C]
        }
        // proj_out → NHWC bf16
        y = y.matmul(&self.proj_out_w)?.add(&self.proj_out_b)?;
        Ok(y.to_dtype(DType::BF16)?.reshape(&[b, h, w, c])?)
    }
}
