use flame_core::{Tensor, DType};
use eridiffusion_core::{Device, Error, Result};
use crate::adapter::Adapter;

pub struct IA3 { name:String, v_in:Option<Tensor>, v_out:Option<Tensor> }
impl IA3 { pub fn new(name:String, v_in:Option<Tensor>, v_out:Option<Tensor>) -> Self { Self { name, v_in, v_out } } }

impl Adapter for IA3 {
    fn name(&self) -> &str { &self.name }
    fn scale(&self) -> f32 { 1.0 }
    fn params(&self) -> Vec<Tensor> { let mut v=Vec::new(); if let Some(x)=&self.v_in { v.push(x.clone()); } if let Some(x)=&self.v_out { v.push(x.clone()); } v }
    fn apply_linear(&self, base_w:&Tensor, x:&Tensor) -> Result<Tensor> {
        let mut xx = x.clone();
        if let Some(v) = &self.v_in { let v32 = v.to_dtype(DType::F32)?; xx = xx.to_dtype(DType::F32)?.mul(&v32.reshape(&[1, v32.shape().dims()[0]])?)?.to_dtype(x.dtype())?; }
        let mut y = xx.matmul(base_w)?;
        if let Some(v) = &self.v_out { let v32 = v.to_dtype(DType::F32)?; y = y.to_dtype(DType::F32)?.mul(&v32.reshape(&[1, v32.shape().dims()[0]])?)?.to_dtype(y.dtype())?; }
        Ok(y)
    }
    fn apply_conv2d(&self, base_w:&Tensor, x:&Tensor, stride:(usize,usize), pad:(usize,usize)) -> Result<Tensor> {
        let mut xx = x.clone();
        if let Some(v) = &self.v_in { let v32 = v.to_dtype(DType::F32)?; xx = xx.to_dtype(DType::F32)?.mul(&v32.reshape(&[1,1,1,v32.shape().dims()[0]])?)?.to_dtype(x.dtype())?; }
        let x_nc = xx.permute(&[0,3,1,2])?; let y_nc = x_nc.conv2d(&base_w.permute(&[3,2,0,1])?, None, stride.0, pad.0)?; let mut y = y_nc.permute(&[0,2,3,1])?;
        if let Some(v) = &self.v_out { let v32 = v.to_dtype(DType::F32)?; y = y.to_dtype(DType::F32)?.mul(&v32.reshape(&[1,1,1,v32.shape().dims()[0]])?)?.to_dtype(y.dtype())?; }
        Ok(y)
    }
    fn delta_weight_linear(&self, _dtype:DType) -> Result<Tensor> { Err(Error::Unsupported("IA3 has no delta weight".into())) }
    fn delta_out_linear(&self, _x:&Tensor) -> Result<Tensor> { Err(Error::Unsupported("IA3 has no delta output".into())) }
}
