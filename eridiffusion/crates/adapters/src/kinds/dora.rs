use eridiffusion_core::{Tensor, DType, Result};
use crate::adapter::Adapter;

pub struct DoRA {
    name: String,
    a: Tensor, b: Tensor, alpha: f32, r: usize,
    gain: Option<Tensor>,
}

impl DoRA {
    pub fn new(name:String, a:Tensor, b:Tensor, alpha:f32, r:usize, gain:Option<Tensor>) -> Self { Self { name, a, b, alpha, r, gain } }
    fn lora_delta(&self, dtype: DType) -> Result<Tensor> {
        let a32 = self.a.to_dtype(DType::F32)?;
        let b32 = self.b.to_dtype(DType::F32)?;
        a32.matmul(&b32)?.mul_scalar(self.alpha / (self.r as f32))?.to_dtype(dtype)
    }
}

impl Adapter for DoRA {
    fn name(&self) -> &str { &self.name }
    fn scale(&self) -> f32 { self.alpha }
    fn params(&self) -> Vec<Tensor> { let mut v=vec![self.a.clone(), self.b.clone()]; if let Some(g)=&self.gain { v.push(g.clone()); } v }
    fn apply_linear(&self, base_w:&Tensor, x:&Tensor) -> Result<Tensor> {
        let mut w = base_w.clone();
        if let Some(g) = &self.gain { let g32 = g.to_dtype(DType::F32)?; w = w.to_dtype(DType::F32)?.mul(&g32.reshape(&[1, g32.shape().dims()[0]])?)?.to_dtype(base_w.dtype())?; }
        let dw = self.lora_delta(base_w.dtype())?; let w = w.add(&dw)?;
        if x.shape().dims().len()==2 { x.matmul(&w) } else { let (b,t,in_)=(x.shape().dims()[0],x.shape().dims()[1],x.shape().dims()[2]); x.reshape(&[b*t,in_])?.matmul(&w)?.reshape(&[b,t,w.shape().dims()[1]]) }
    }
    fn apply_conv2d(&self, base_w:&Tensor, x:&Tensor, stride:(usize,usize), pad:(usize,usize)) -> Result<Tensor> {
        let mut w = base_w.clone();
        if let Some(g)=&self.gain { let g32 = g.to_dtype(DType::F32)?; w = w.to_dtype(DType::F32)?.mul(&g32.reshape(&[1,1,1,g32.shape().dims()[0]])?)?.to_dtype(base_w.dtype())?; }
        let dw = self.lora_delta(base_w.dtype())?; let dw = dw.reshape(&[base_w.shape().dims()[0], base_w.shape().dims()[1], base_w.shape().dims()[2], base_w.shape().dims()[3]])?; let w = w.add(&dw)?;
        let x_nc = x.permute(&[0,3,1,2])?; let y_nc = x_nc.conv2d(&w.permute(&[3,2,0,1])?, None, stride.0, pad.0)?; y_nc.permute(&[0,2,3,1])
    }
    fn delta_weight_linear(&self, dtype:DType) -> Result<Tensor> { self.lora_delta(dtype) }
    fn delta_out_linear(&self, x:&Tensor) -> Result<Tensor> { Ok(x.to_dtype(DType::F32)?.matmul(&self.lora_delta(DType::F32)?)?.to_dtype(x.dtype())?) }
}

