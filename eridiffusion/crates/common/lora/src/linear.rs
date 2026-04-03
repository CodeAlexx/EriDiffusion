use anyhow::Result;
use flame_core::{Tensor, DType};
use eridiffusion_core::Device;

/// Simple LoRA linear adapter (GPU BF16).
pub struct LoraLinear {
    pub in_dim: usize,
    pub out_dim: usize,
    pub rank: usize,
    pub alpha: f32,
    pub a: Tensor,
    pub b: Tensor,
    scale: f32,
    device: Device,
    dtype: DType,
}

impl LoraLinear {
    pub fn new(in_dim: usize, out_dim: usize, rank: usize, alpha: f32, device: &Device, dtype: DType) -> Result<Self> {
        if rank == 0 { return Err(flame_core::Error::InvalidInput("rank must be >0".into()).into()); }
        let cuda = eridiffusion_core::device::shared_cuda_device()?;
        let a = Tensor::zeros_device(flame_core::Shape::from_dims(&[in_dim, rank]), cuda.clone(), dtype)?;
        let b = Tensor::zeros_device(flame_core::Shape::from_dims(&[rank, out_dim]), cuda.clone(), dtype)?;
        Ok(Self { in_dim, out_dim, rank, alpha, scale: alpha / rank as f32, a, b, device: device.clone(), dtype })
    }

    pub fn delta_weight(&self) -> Result<Tensor> {
        let dw = self.a.matmul(&self.b)?;
        dw.mul_scalar(self.scale).map_err(|e| e.into())
    }

    pub fn apply_delta(&self, base: &Tensor) -> Result<Tensor> {
        let base = if base.dtype() == self.dtype { base.clone() } else { base.to_dtype(self.dtype)? };
        let dw = self.delta_weight()?;
        base.add(&dw).map_err(|e| e.into())
    }

    pub fn forward_delta(&self, x: &Tensor) -> Result<Tensor> {
        let x = if x.dtype() == self.dtype { x.clone() } else { x.to_dtype(self.dtype)? };
        let xa = x.matmul(&self.a)?;
        let out = xa.matmul(&self.b)?;
        out.mul_scalar(self.scale).map_err(|e| e.into())
    }
}
