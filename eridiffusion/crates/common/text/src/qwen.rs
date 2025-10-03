use anyhow::Result;
use flame_core::{Tensor, Shape, Device, DType};

pub struct QwenTE { pub ctx_dim: usize, pub seq: usize, pub device: Device }

impl QwenTE {
    pub fn new(device: Device, seq: usize, ctx_dim: usize) -> Result<Self> { Ok(Self { ctx_dim, seq, device }) }
    pub fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        let b = ids.shape().dims()[0];
        Ok(Tensor::zeros_dtype(Shape::from_dims(&[b, self.seq, self.ctx_dim]), DType::BF16, self.device.cuda_device().clone())?)
    }
}
