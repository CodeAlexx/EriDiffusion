use flame_core::{Tensor, DType};
use eridiffusion_core::{Device, Error, Result};
use crate::adapter::Adapter;

/// LoHa: Hadamard product of two low-rank terms for linear weights only
/// ΔW = s * (A @ B) ⊙ (C @ D)
pub struct LoHa {
    name: String,
    a: Tensor, b: Tensor, // A:[IN,R1], B:[R1,OUT]
    c: Tensor, d: Tensor, // C:[IN,R2], D:[R2,OUT]
    alpha: f32,
    r1: usize, r2: usize,
}

impl LoHa {
    pub fn new(name:String, a:Tensor, b:Tensor, c:Tensor, d:Tensor, alpha:f32, r1:usize, r2:usize) -> Self {
        Self { name, a, b, c, d, alpha, r1, r2 }
    }
    fn delta_linear(&self, dtype: DType) -> Result<Tensor> {
        let a1 = self.a.to_dtype(DType::F32)?;
        let b1 = self.b.to_dtype(DType::F32)?;
        let a2 = self.c.to_dtype(DType::F32)?;
        let b2 = self.d.to_dtype(DType::F32)?;
        let t1 = a1.matmul(&b1)?; // [IN,OUT]
        let t2 = a2.matmul(&b2)?; // [IN,OUT]
        let had = t1.mul(&t2)?.mul_scalar(self.alpha / ((self.r1 as f32)+(self.r2 as f32)).max(1.0))?;
        had.to_dtype(dtype)
    }
}

impl Adapter for LoHa {
    fn name(&self) -> &str { &self.name }
    fn scale(&self) -> f32 { self.alpha }
    fn params(&self) -> Vec<Tensor> { vec![self.a.clone(), self.b.clone(), self.c.clone(), self.d.clone()] }
    fn apply_linear(&self, base_w: &Tensor, x: &Tensor) -> Result<Tensor> {
        let dw = self.delta_linear(base_w.dtype())?;
        let w = base_w.add(&dw)?;
        let dims = x.shape().dims().to_vec();
        if dims.len()==2 { x.matmul(&w) } else { let (b,t,_)= (dims[0],dims[1],dims[2]); x.reshape(&[b*t, dims[2]])?.matmul(&w)?.reshape(&[b,t,w.shape().dims()[1]]) }
    }
    fn apply_conv2d(&self, _bw:&Tensor, _x:&Tensor, _s:(usize,usize), _p:(usize,usize)) -> Result<Tensor> {
        Err(Error::Unsupported("LoHa conv2d not implemented".into()))
    }
    fn delta_weight_linear(&self, dtype: DType) -> Result<Tensor> { self.delta_linear(dtype) }
    fn delta_out_linear(&self, x: &Tensor) -> Result<Tensor> {
        let dw = self.delta_linear(DType::F32)?;
        Ok(x.to_dtype(DType::F32)?.matmul(&dw)?.to_dtype(x.dtype())?)
    }
}
