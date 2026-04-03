use flame_core::{Tensor, DType};
use eridiffusion_core::{Error, Result};
use crate::adapter::Adapter;

pub struct LoCon2d {
    name: String,
    a: Tensor,   // [OC, R] bf16
    b: Tensor,   // [R, IC*KH*KW] bf16
    alpha: f32,
    r: usize,
    kh: usize, kw: usize, ic: usize, oc: usize,
}

impl LoCon2d {
    pub fn new(name:String, a:Tensor, b:Tensor, alpha:f32, r:usize,
               kh:usize, kw:usize, ic:usize, oc:usize) -> Self {
        Self { name, a, b, alpha, r, kh, kw, ic, oc }
    }
    fn delta_kernel(&self, dtype: DType) -> Result<Tensor> {
        let a32 = self.a.to_dtype(DType::F32)?;
        let b32 = self.b.to_dtype(DType::F32)?;
        let flat = a32.matmul(&b32)?.mul_scalar(self.alpha / (self.r as f32))?; // [OC, IC*KH*KW]
        let t = flat.reshape(&[self.oc, self.ic * self.kh * self.kw])?.to_dtype(dtype)?;
        Ok(t.reshape(&[self.kh, self.kw, self.ic, self.oc])?)
    }
}

impl Adapter for LoCon2d {
    fn name(&self) -> &str { &self.name }
    fn scale(&self) -> f32 { self.alpha / (self.r as f32) }
    fn params(&self) -> Vec<Tensor> { vec![self.a.clone(), self.b.clone()] }
    fn apply_linear(&self, _bw:&Tensor, _x:&Tensor) -> Result<Tensor> {
        Err(Error::Unsupported("LoCon2d is for conv2d; use LoRA for linear".into()))
    }
    fn apply_conv2d(&self, base_w:&Tensor, x:&Tensor, stride:(usize,usize), pad:(usize,usize)) -> Result<Tensor> {
        // x is NHWC; convert to NCHW → conv2d → back to NHWC
        let dw = self.delta_kernel(base_w.dtype())?;
        let w = base_w.add(&dw)?; // [KH,KW,IC,OC]
        let x_nc = x.permute(&[0,3,1,2])?;
        let w_oihw = w.permute(&[3,2,0,1])?;
        let y_nc = x_nc.conv2d(&w_oihw, None, stride.0, pad.0)?;
        Ok(y_nc.permute(&[0,2,3,1])?)
    }
    fn delta_weight_linear(&self, _dtype: DType) -> Result<Tensor> {
        Err(Error::Unsupported("LoCon2d: linear delta not supported".into()))
    }
    fn delta_out_linear(&self, _x: &Tensor) -> Result<Tensor> {
        Err(Error::Unsupported("LoCon2d: linear delta not supported".into()))
    }
}
