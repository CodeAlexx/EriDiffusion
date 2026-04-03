use anyhow::Result;
use flame_core::{Tensor, DType};
use std::env;
use crate::nhwc_utils::broadcast_mask;
use adapters::adapter::AdapterSet;

pub struct CrossAttn {
    pub to_q_w: Tensor, pub to_q_b: Tensor,
    pub to_k_w: Tensor, pub to_k_b: Tensor,
    pub to_v_w: Tensor, pub to_v_b: Tensor,
    pub to_out_w: Tensor, pub to_out_b: Tensor,
    pub n_heads: usize, pub head_dim: usize,
}

impl CrossAttn {
    pub fn forward(&self, x_bhwc:&Tensor, ctx_bsd:&Tensor, mask_f32:&Tensor, adapters: Option<&AdapterSet>, key_prefix:&str) -> Result<Tensor> {
        let sh = x_bhwc.shape().dims().to_vec();
        let (b,h,w,c) = (sh[0], sh[1], sh[2], sh[3]);
        let hw = h*w;
        // FP32 compute
        let x   = x_bhwc.to_dtype(DType::F32)?.reshape(&[b, hw, c])?;
        let ctx = ctx_bsd.to_dtype(DType::F32)?; // [B,S,2048]
        // projections
        let mut q = x.matmul(&self.to_q_w)?.add(&self.to_q_b)?;       // [B,HW,C]
        let mut k = ctx.matmul(&self.to_k_w)?.add(&self.to_k_b)?;     // [B,S,C]
        let mut v = ctx.matmul(&self.to_v_w)?.add(&self.to_v_b)?;     // [B,S,C]
        if let Some(ad) = adapters {
            // Δy = X @ ΔW; flatten X accordingly
            let xf = x.reshape(&[b*hw, c])?;
            let sf = ctx.reshape(&[ctx.shape().dims()[0]*ctx.shape().dims()[1], ctx.shape().dims()[2]])?;
            let kq = format!("{}.to_q", key_prefix);
            if let Some(a) = ad.get(&kq) { let dq = a.delta_out_linear(&xf)?; q = q.add(&dq.reshape(&[b,hw,c])?)?; }
            let kk = format!("{}.to_k", key_prefix);
            if let Some(a) = ad.get(&kk) { let dk = a.delta_out_linear(&sf)?; k = k.add(&dk.reshape(&[ctx.shape().dims()[0],ctx.shape().dims()[1],c])?)?; }
            let kv = format!("{}.to_v", key_prefix);
            if let Some(a) = ad.get(&kv) { let dv = a.delta_out_linear(&sf)?; v = v.add(&dv.reshape(&[ctx.shape().dims()[0],ctx.shape().dims()[1],c])?)?; }
        }
        // heads
        let hs = self.n_heads; let hd = self.head_dim;
        let q = q.reshape(&[b, hw, hs*hd])?.permute(&[0,2,1])?.reshape(&[b, hs, hw, hd])?; // [B,h,HW,hd]
        let s = ctx.shape().dims()[1];
        let k = k.reshape(&[b, s, hs*hd])?.permute(&[0,2,1])?.reshape(&[b, hs, s, hd])?;   // [B,h,S,hd]
        let v = v.reshape(&[b, s, hs*hd])?.permute(&[0,2,1])?.reshape(&[b, hs, s, hd])?;   // [B,h,S,hd]
        // Choose attention backend: flash or sdpa (default)
        let impl_sel = env::var("ATTENTION_IMPL").unwrap_or_else(|_| "sdpa".into());
        let out_bt = if impl_sel == "flash" {
            // FlashAttention expects [B,h,T,hd]. Provide mask as FP32 [B,1,1,S] broadcast to [B,h,T,S] if supported by kernel
            let q4 = q.clone();
            let k4 = k.clone();
            let v4 = v.clone();
            let scale = (self.head_dim as f32).sqrt().recip();
            let fa = flame_core::flash_attention::FlashAttention::new().with_scale(scale);
            // Attempt flash; on error, fallback to sdpa
            match fa.forward(&q4, &k4, &v4, Some(mask_f32)) {
                Ok(ctx4) => ctx4.permute(&[0,2,1,3])?.reshape(&[b, hw, hs*hd])?,
                Err(_) => {
                    // fallback to SDPA path
                    let scale = (self.head_dim as f32).sqrt().recip();
                    let mut logits = q.matmul(&k.permute(&[0,1,3,2])?)?; // [B,h,HW,S]
                    logits = logits.mul_scalar(scale)?;
                    let m = broadcast_mask(mask_f32, self.n_heads, hw)?;
                    let logits = logits.add(&m)?;
                    let attn = logits.softmax(-1)?;
                    attn.matmul(&v)? // [B,h,HW,hd]
                        .permute(&[0,2,1,3])?.reshape(&[b, hw, hs*hd])?
                }
            }
        } else {
            let scale = (self.head_dim as f32).sqrt().recip();
            let mut logits = q.matmul(&k.permute(&[0,1,3,2])?)?; // [B,h,HW,S]
            logits = logits.mul_scalar(scale)?;
            let m = broadcast_mask(mask_f32, self.n_heads, hw)?;
            let logits = logits.add(&m)?;
            let attn = logits.softmax(-1)?; // [B,h,HW,S]
            attn.matmul(&v)? // [B,h,HW,hd]
                .permute(&[0,2,1,3])?.reshape(&[b, hw, hs*hd])?
        };
        // out proj (+ optional adapter delta on out)
        let mut out = out_bt.matmul(&self.to_out_w)?.add(&self.to_out_b)?;      // [B,HW,C]
        if let Some(ad) = adapters {
            let ko = format!("{}.to_out", key_prefix);
            if let Some(a) = ad.get(&ko) {
                let zf = out_bt.reshape(&[b*hw, c])?;
                let d = a.delta_out_linear(&zf)?;
                out = out.add(&d.reshape(&[b,hw,c])?)?;
            }
        }
        Ok(out.to_dtype(DType::BF16)?.reshape(&[b, h, w, c])?)
    }
}
