use anyhow::Result;
use flame_core::{Tensor, DType, Shape};
use crate::nhwc_utils::{nhwc_to_nchw, nchw_to_nhwc, khw_kicoc_to_oihw, silu};

pub struct ResBlk {
    pub in_gn_w: Tensor, pub in_gn_b: Tensor,
    pub in_conv_w: Tensor, pub in_conv_b: Tensor,
    pub out_gn_w: Tensor, pub out_gn_b: Tensor,
    pub out_conv_w: Tensor, pub out_conv_b: Tensor,
    pub skip_w: Option<Tensor>, pub skip_b: Option<Tensor>,
    pub temb2_w: Option<Tensor>, pub temb2_b: Option<Tensor>,
    pub groups: usize,
}

impl ResBlk {
    fn group_norm(&self, x: &Tensor, gamma: &Tensor, beta: &Tensor, groups: usize) -> Result<Tensor> {
        // x: NHWC; normalize per group across C
        let x32 = x.to_dtype(DType::F32)?;
        let dims = x32.shape().dims().to_vec();
        let (b,h,w,c) = (dims[0], dims[1], dims[2], dims[3]);
        anyhow::ensure!(c % groups == 0, "group_norm: C={} not divisible by groups={}", c, groups);
        let cg = c / groups;
        let xg = x32.reshape(&[b,h,w,groups,cg])?;
        let mean = xg.sum_dim_keepdim(4)?.div_scalar(cg as f32)?; // [B,H,W,G,1]
        let xc = xg.sub(&mean)?;
        let var = xc.square()?.sum_dim_keepdim(4)?.div_scalar(cg as f32)?; // [B,H,W,G,1]
        let eps = Tensor::from_scalar(1e-5f32, x.device().clone())?;
        let inv = var.add(&eps)?.rsqrt()?;
        let xn = xc.mul(&inv)?;
        let xn = xn.reshape(&[b,h,w,c])?;
        // affine: gamma/beta are [C]
        let g = gamma.to_dtype(DType::F32)?.reshape(&[1,1,1,c])?;
        let be = beta.to_dtype(DType::F32)?.reshape(&[1,1,1,c])?;
        Ok(xn.mul(&g)?.add(&be)?)
    }

    fn conv3x3(&self, x: &Tensor, w_khwicoc: &Tensor, b: &Tensor, stride: usize, pad: usize) -> Result<Tensor> {
        let w = khw_kicoc_to_oihw(w_khwicoc)?;
        let x_nc = nhwc_to_nchw(&x.to_dtype(DType::F32)?)?;
        let y_nc = x_nc.conv2d(&w, Some(b), stride, pad)?;
        nchw_to_nhwc(&y_nc)
    }

    pub fn forward(&self, x_nhwc: &Tensor, temb: Option<&Tensor>) -> Result<Tensor> {
        let x0 = x_nhwc;
        // In GN → SiLU → Conv3x3
        let h1 = self.group_norm(x0, &self.in_gn_w, &self.in_gn_b, self.groups)?;
        let a1 = silu(&h1)?;
        let y1 = self.conv3x3(&a1, &self.in_conv_w, &self.in_conv_b, 1, 1)?;

        // Out GN → SiLU → Conv3x3
        let mut h2 = self.group_norm(&y1, &self.out_gn_w, &self.out_gn_b, self.groups)?;
        // time embedding addition before second conv: (silu(temb) @ temb2_w + temb2_b)
        if let (Some(emb), Some(w), Some(b)) = (temb, self.temb2_w.as_ref(), self.temb2_b.as_ref()) {
            let add = silu(emb)?.matmul(&w.to_dtype(DType::F32)?)?.add(&b.to_dtype(DType::F32)?)?; // [B,C]
            let [bch, _c] = [add.shape().dims()[0], add.shape().dims()[1]];
            let add_b = add.reshape(&[bch,1,1,_c])?;
            h2 = h2.add(&add_b)?;
        }
        let a2 = silu(&h2)?;
        let y2 = self.conv3x3(&a2, &self.out_conv_w, &self.out_conv_b, 1, 1)?;

        // Skip path (1x1 conv) if provided, else identity
        let skip = if let (Some(w), Some(b)) = (&self.skip_w, &self.skip_b) {
            // Emulate 1x1 via matmul over C at each spatial location
            let dims = x0.shape().dims().to_vec();
            let (bch, hh, ww, cin) = (dims[0], dims[1], dims[2], dims[3]);
            let cout = b.shape().dims()[0];
            let x_flat = x0.to_dtype(DType::F32)?.reshape(&[bch*hh*ww, cin])?;
            let w2 = w.to_dtype(DType::F32)?; // [1,1,cin,cout] treated as [cin,cout]
            let w_mat = w2.reshape(&[cin, cout])?;
            let b_vec = b.to_dtype(DType::F32)?;
            let y = x_flat.matmul(&w_mat)?.add(&b_vec)?;
            y.reshape(&[bch, hh, ww, cout])?
        } else { x0.to_dtype(DType::F32)? };

        Ok(y2.add(&skip)?.to_dtype(DType::BF16)?)
    }
}
