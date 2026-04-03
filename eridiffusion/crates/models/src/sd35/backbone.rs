use anyhow::Result;
use flame_core::{Tensor, DType, Device, Shape};

/// Per-layer weight pack for SD3.5 (DiT-like).
pub struct Sd35WeightPack { pub wq: Tensor, pub wk: Tensor, pub wv: Tensor, pub wo: Tensor, pub fc1: Tensor, pub fc2: Tensor }

/// SD-3.5 Backbone: DiT-like stack.
pub struct Sd35Backbone { pub device: Device, pub dtype: DType, pub hidden: usize, pub heads: usize, pub packs: Vec<Sd35WeightPack>, pub base_frozen: bool }

impl Sd35Backbone {
    pub fn new(device: Device, dtype: DType, hidden: usize, heads: usize, layers: usize) -> Result<Self> {
        let mut packs = Vec::with_capacity(layers);
        for _ in 0..layers {
            packs.push(Sd35WeightPack {
                wq: Tensor::zeros_dtype(Shape::from_dims(&[hidden, hidden]), dtype, device.cuda_device().clone())?,
                wk: Tensor::zeros_dtype(Shape::from_dims(&[hidden, hidden]), dtype, device.cuda_device().clone())?,
                wv: Tensor::zeros_dtype(Shape::from_dims(&[hidden, hidden]), dtype, device.cuda_device().clone())?,
                wo: Tensor::zeros_dtype(Shape::from_dims(&[hidden, hidden]), dtype, device.cuda_device().clone())?,
                fc1: Tensor::zeros_dtype(Shape::from_dims(&[hidden, hidden*4]), dtype, device.cuda_device().clone())?,
                fc2: Tensor::zeros_dtype(Shape::from_dims(&[hidden*4, hidden]), dtype, device.cuda_device().clone())?,
            });
        }
        Ok(Self { device, dtype, hidden, heads, packs, base_frozen: false })
    }

    pub fn from_packs(device: Device, dtype: DType, hidden: usize, heads: usize, packs: Vec<Sd35WeightPack>) -> Result<Self> {
        Ok(Self { device, dtype, hidden, heads, packs, base_frozen: false })
    }

    /// Forward with conditioning and time embedding (placeholders in math for now — cond/t_emb fuse points are model-specific).
    pub fn forward(&self, x_latent: &Tensor, _cond: &crate::sd35::Sd35Cond, _t_emb: &Tensor) -> Result<Tensor> {
        let dims = x_latent.shape().dims().to_vec();
        anyhow::ensure!(dims.len()==3 && dims[2]==self.hidden, "SD3.5 expects [B,Seq,{}]", self.hidden);
        let mut x = x_latent.clone();
        for p in &self.packs {
            x = self.block_forward(&x, p)?;
        }
        Ok(x)
    }

    fn block_forward(&self, x: &Tensor, p: &Sd35WeightPack) -> Result<Tensor> {
        let h = self.hidden; let (b,s,_) = (x.shape().dims()[0], x.shape().dims()[1], x.shape().dims()[2]);
        // RMSNorm FP32
        let x32 = x.to_dtype(DType::F32)?;
        let last = h as f32;
        let mean = x32.sum_dim_keepdim(2)?.div_scalar(last)?;
        let xc = x32.sub(&mean)?;
        let var = xc.square()?.sum_dim_keepdim(2)?.div_scalar(last)?;
        let inv = var.add_scalar(1e-5)?.rsqrt()?;
        let xn = xc.mul(&inv)?.to_dtype(x.dtype())?;
        // linear [B,S,D]x[D,Out]
        let linear_bt = |bt:&Tensor, w:&Tensor| -> Result<Tensor> { let flat=bt.reshape(&[b*s, h])?; let out= if flat.dtype()==DType::BF16 && w.dtype()==DType::BF16 { flat.matmul_bf16(w)? } else { flat.matmul(w)? }; let od=out.shape().dims()[1]; out.reshape(&[b,s,od]) };
        let q = linear_bt(&xn, &p.wq)?; let k = linear_bt(&xn, &p.wk)?; let v = linear_bt(&xn, &p.wv)?;
        let kt = k.transpose_dims(1,2)?; let mut scores = q.to_dtype(DType::F32)?.bmm(&kt.to_dtype(DType::F32)?)?;
        scores = scores.mul_scalar((h as f32).sqrt().recip())?;
        let cap = Tensor::from_scalar(30.0f32, scores.device().clone())?; let ncap = Tensor::from_scalar(-30.0f32, scores.device().clone())?;
        scores = scores.minimum(&cap)?.maximum(&ncap)?;
        let probs = eridiffusion_training::tensor_utils::softmax_stable(&scores, -1).map_err(|e| anyhow::anyhow!(e.to_string()))?;
        let ctx = probs.to_dtype(DType::F32)?.bmm(&v.to_dtype(DType::F32)?)?;
        let attn_out = linear_bt(&ctx, &p.wo)?;
        let x1 = x.add(&attn_out)?;
        let h1 = linear_bt(&x1, &p.fc1)?; let a1 = h1.gelu()?; let out0 = linear_bt(&a1, &p.fc2)?;
        Ok(x1.add(&out0)?)
    }

    pub fn freeze_base(&mut self) { self.base_frozen = true; }
    pub fn is_base_frozen(&self) -> bool { self.base_frozen }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn forward_shape_ok() -> anyhow::Result<()> {
        let dev = Device::cuda(0)?; let dtype = DType::BF16; let hidden=64; let heads=4; let layers=2;
        let bb = Sd35Backbone::new(dev.clone(), dtype, hidden, heads, layers)?;
        let x = Tensor::randn(Shape::from_dims(&[2, 128, hidden]), 0.0, 1.0, dev.clone())?.to_dtype(dtype)?;
        let cond = crate::sd35::Sd35Cond { text_hidden: Tensor::zeros_dtype(Shape::from_dims(&[2,77,hidden]), dtype, dev.clone())? };
        let t = Tensor::from_vec(vec![0.0f32;2], Shape::from_dims(&[2]), dev.clone())?;
        let y = bb.forward(&x, &cond, &t)?; assert_eq!(y.shape().dims(), &[2,128,hidden]); Ok(())
    }
}
