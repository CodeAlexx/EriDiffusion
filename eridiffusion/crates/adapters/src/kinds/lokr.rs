use eridiffusion_core::{Tensor, DType, Result};
use crate::adapter::Adapter;

/// LoKr: Kronecker low-rank approximation (simplified)
pub struct LoKr {
    name: String,
    p: Tensor, q: Tensor, r: Tensor, // store Kron factors; MVP uses r as flat delta
    alpha: f32,
}

impl LoKr {
    pub fn new(name:String, p:Tensor, q:Tensor, r:Tensor, alpha:f32) -> Self { Self { name, p, q, r, alpha } }
    fn delta_flat(&self, dtype: DType) -> Result<Tensor> {
        Ok(self.r.to_dtype(DType::F32)?.mul_scalar(self.alpha)?.to_dtype(dtype)?)
    }
}

impl Adapter for LoKr {
    fn name(&self) -> &str { &self.name }
    fn scale(&self) -> f32 { self.alpha }
    fn params(&self) -> Vec<Tensor> { vec![self.p.clone(), self.q.clone(), self.r.clone()] }
    fn apply_linear(&self, base_w:&Tensor, x:&Tensor) -> Result<Tensor> {
        let dw = self.delta_flat(base_w.dtype())?; // [IN,OUT]
        let w = base_w.add(&dw)?;
        if x.shape().dims().len()==2 { x.matmul(&w) } else { let (b,t,in_)=(x.shape().dims()[0],x.shape().dims()[1],x.shape().dims()[2]); x.reshape(&[b*t,in_])?.matmul(&w)?.reshape(&[b,t,w.shape().dims()[1]]) }
    }
    fn apply_conv2d(&self, base_w:&Tensor, x:&Tensor, stride:(usize,usize), pad:(usize,usize)) -> Result<Tensor> {
        let dims = base_w.shape().dims().to_vec();
        let (kh,kw,ic,oc) = (dims[0],dims[1],dims[2],dims[3]);
        let dw_flat = self.delta_flat(base_w.dtype())?; // assume [IC*KH*KW, OC] or [OC, IC*KH*KW]
        let dw = if dw_flat.shape().dims()==&[ic*kh*kw, oc] { dw_flat.reshape(&[kh,kw,ic,oc])? } else { dw_flat.reshape(&[oc, ic*kh*kw])?.reshape(&[kh,kw,ic,oc])? };
        let w = base_w.add(&dw)?;
        let x_nc = x.permute(&[0,3,1,2])?;
        let y_nc = x_nc.conv2d(&w.permute(&[3,2,0,1])?, None, stride.0, pad.0)?;
        y_nc.permute(&[0,2,3,1])
    }
    fn delta_weight_linear(&self, dtype: DType) -> Result<Tensor> { self.delta_flat(dtype) }
    fn delta_out_linear(&self, x:&Tensor) -> Result<Tensor> { Ok(x.to_dtype(DType::F32)?.matmul(&self.r.to_dtype(DType::F32)?)?.to_dtype(x.dtype())?) }
}

