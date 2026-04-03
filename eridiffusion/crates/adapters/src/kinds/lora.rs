use flame_core::{Tensor, DType};
use eridiffusion_core::{Error, Result};
use crate::adapter::Adapter;

pub struct LoRA {
    name: String,
    // Expect base linear weight to be [IN, OUT]; factors match delta shape [IN, OUT]
    a: Tensor,   // [IN, R] bf16
    b: Tensor,   // [R, OUT] bf16
    alpha: f32,
    r: usize,
}

impl LoRA {
    pub fn new(name: String, a: Tensor, b: Tensor, alpha: f32, r: usize) -> Self { Self { name, a, b, alpha, r } }
    fn delta_linear(&self, dtype: DType) -> Result<Tensor> {
        let a32 = self.a.to_dtype(DType::F32)?;
        let b32 = self.b.to_dtype(DType::F32)?;
        let dw32 = a32.matmul(&b32)?.mul_scalar(self.alpha / (self.r as f32))?; // [IN, OUT]
        Ok(dw32.to_dtype(dtype)?)
    }
}

impl Adapter for LoRA {
    fn name(&self) -> &str { &self.name }
    fn scale(&self) -> f32 { self.alpha / (self.r as f32) }
    fn params(&self) -> Vec<Tensor> { vec![self.a.clone(), self.b.clone()] }
    fn apply_linear(&self, base_w: &Tensor, x: &Tensor) -> Result<Tensor> {
        // Support x rank 2 or 3; treat last dim as IN; base_w [IN, OUT]
        let dtype = base_w.dtype();
        let dw = self.delta_linear(dtype)?; // [IN, OUT]
        let w = base_w.add(&dw)?;
        let dims = x.shape().dims().to_vec();
        if dims.len()==2 {
            // [N, IN] x [IN, OUT] -> [N, OUT]
            Ok(x.matmul(&w)?)
        } else if dims.len()==3 {
            let (b,t,in_) = (dims[0], dims[1], dims[2]);
            let flat = x.reshape(&[b*t, in_])?;
            let out = flat.matmul(&w)?;
            let od = out.shape().dims()[1];
            Ok(out.reshape(&[b, t, od])?)
        } else {
            Err(Error::InvalidInput(format!("LoRA.apply_linear: unsupported x rank {}", dims.len())))
        }
    }
    fn apply_conv2d(&self, _base_w:&Tensor, _x:&Tensor, _s:(usize,usize), _p:(usize,usize)) -> Result<Tensor> {
        Err(Error::Unsupported("LoRA is for linear weights; use LoCon2d for conv".into()))
    }
    fn delta_weight_linear(&self, dtype: DType) -> Result<Tensor> { self.delta_linear(dtype) }
    fn delta_out_linear(&self, x: &Tensor) -> Result<Tensor> {
        // x [*, IN], A[IN,R], B[R,OUT] => (x @ A) @ B
        let a32 = self.a.to_dtype(DType::F32)?;
        let b32 = self.b.to_dtype(DType::F32)?;
        let xa = x.to_dtype(DType::F32)?.matmul(&a32)?;
        let y = xa.matmul(&b32)?.mul_scalar(self.alpha / (self.r as f32))?;
        Ok(y.to_dtype(x.dtype())?)
    }
}
