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
use flame_core::{DType, Shape, Tensor};

pub struct TimeIdsMLP {
    w0: Tensor, // [6, hidden_dim]
    b0: Tensor, // [hidden_dim]
    w1: Tensor, // [hidden_dim, hidden_dim]
    b1: Tensor, // [hidden_dim]
    hidden_dim: usize,
}

impl TimeIdsMLP {
    pub fn new(dev: Device) -> Result<Self> {
        let hidden_dim = 256;
        let w0 =
            Tensor::zeros_dtype(Shape::from_dims(&[6, hidden_dim]), DType::BF16, dev.clone())?;
        let b0 = Tensor::zeros_dtype(Shape::from_dims(&[hidden_dim]), DType::BF16, dev.clone())?;
        let w1 =
            Tensor::zeros_dtype(Shape::from_dims(&[hidden_dim, hidden_dim]), DType::BF16, dev.clone())?;
        let b1 = Tensor::zeros_dtype(Shape::from_dims(&[hidden_dim]), DType::BF16, dev)?;
        Ok(Self { w0, b0, w1, b1, hidden_dim })
    }

    pub fn from_tensors(w0: Tensor, b0: Tensor, w1: Tensor, b1: Tensor) -> Result<Self> {
        let w0_dims = w0.shape().dims();
        if w0_dims.len() != 2 || w0_dims[0] != 6 {
            bail!("timeids w0 expected [6,hidden], got {:?}", w0_dims);
        }
        let hidden_dim = w0_dims[1];
        if b0.shape().dims() != [hidden_dim] {
            bail!("timeids b0 expected [{hidden_dim}], got {:?}", b0.shape().dims());
        }
        if w1.shape().dims() != [hidden_dim, hidden_dim] {
            bail!(
                "timeids w1 expected [{},{}], got {:?}",
                hidden_dim,
                hidden_dim,
                w1.shape().dims()
            );
        }
        if b1.shape().dims() != [hidden_dim] {
            bail!("timeids b1 expected [{hidden_dim}], got {:?}", b1.shape().dims());
        }
        Ok(Self { w0, b0, w1, b1, hidden_dim })
    }

    /// Forward: x [N,6] → [N,256]
    pub fn forward(&self, time_ids: &Tensor) -> Result<Tensor> {
        let dims = time_ids.shape().dims();
        if dims.len() != 2 || dims[1] != 6 {
            bail!("TimeIdsMLP: expected time_ids [N,6], got {:?}", dims);
        }
        let x = time_ids.matmul(&self.w0)?.add(&self.b0)?.silu()?; // [N,hidden]
        let x = x.matmul(&self.w1)?.add(&self.b1)?.silu()?; // [N,hidden]
        Ok(x)
    }
}
