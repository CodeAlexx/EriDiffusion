//! SDXL Time IDs MLP (6 → 256 → 256), BF16 params / FP32 math.
//! Place at: crates/models/src/sdxl/blocks/timeids_mlp.rs
//!
//! This matches the common SDXL pattern of projecting the 6-element time_ids vector
//! into a richer embedding used in conditioning paths. We keep it 256-D here.
//!
//! Storage: Linear weights are **[IN, OUT]** per project policy.
//! Init: zero bias, Kaiming-like randn for weights is OK (loader will overwrite for real runs).

use anyhow::{bail, Result};
use eridiffusion_core::Device;
use flame_core::{Tensor, DType, Shape};

pub struct TimeIdsMLP {
    w0: Tensor, // [6, 256]
    b0: Tensor, // [256]
    w1: Tensor, // [256, 256]
    b1: Tensor, // [256]
}

impl TimeIdsMLP {
    pub fn new(dev: Device) -> Result<Self> {
        let w0 = Tensor::zeros_dtype(Shape::from_dims(&[6, 256]), DType::BF16, dev.clone())?;
        let b0 = Tensor::zeros_dtype(Shape::from_dims(&[256]), DType::BF16, dev.clone())?;
        let w1 = Tensor::zeros_dtype(Shape::from_dims(&[256, 256]), DType::BF16, dev.clone())?;
        let b1 = Tensor::zeros_dtype(Shape::from_dims(&[256]), DType::BF16, dev)?;
        Ok(Self { w0, b0, w1, b1 })
    }

    pub fn from_tensors(w0: Tensor, b0: Tensor, w1: Tensor, b1: Tensor) -> Result<Self> {
        // quick shape guards
        if w0.shape().dims() != [6, 256] {
            bail!("timeids w0 expected [6,256], got {:?}", w0.shape().dims());
        }
        if b0.shape().dims() != [256] {
            bail!("timeids b0 expected [256], got {:?}", b0.shape().dims());
        }
        if w1.shape().dims() != [256, 256] {
            bail!("timeids w1 expected [256,256], got {:?}", w1.shape().dims());
        }
        if b1.shape().dims() != [256] {
            bail!("timeids b1 expected [256], got {:?}", b1.shape().dims());
        }
        Ok(Self { w0, b0, w1, b1 })
    }

    /// Forward: x [N,6] → [N,256]
    pub fn forward(&self, time_ids: &Tensor) -> Result<Tensor> {
        let dims = time_ids.shape().dims();
        if dims.len() != 2 || dims[1] != 6 {
            bail!("TimeIdsMLP: expected time_ids [N,6], got {:?}", dims);
        }
        let x = time_ids.matmul(&self.w0)?.add(&self.b0)?.silu()?; // [N,256]
        let x = x.matmul(&self.w1)?.add(&self.b1)?.silu()?;        // [N,256]
        Ok(x)
    }
}
