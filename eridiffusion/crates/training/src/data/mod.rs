use anyhow::Result;
use eridiffusion_core::device::shared_cuda_device;
use flame_core::{DType, Shape, Tensor};

pub struct Batch {
    pub latents: Tensor,        // NHWC
    pub timesteps: Tensor,      // [B]
    pub embeds: Option<Tensor>, // [B, T, C]
}

/// Create a small synthetic batch on the shared CUDA device.
pub fn synthetic_batch(
    batch: usize,
    h: usize,
    w: usize,
    c: usize,
    embed_dim: Option<usize>,
) -> Result<Batch> {
    let dev = shared_cuda_device()?;
    let latents =
        Tensor::zeros_dtype(Shape::from_dims(&[batch, h, w, c]), DType::F32, dev.clone())?;
    let timesteps = Tensor::zeros_dtype(Shape::from_dims(&[batch]), DType::F32, dev.clone())?;
    let embeds = if let Some(ed) = embed_dim {
        Some(Tensor::zeros_dtype(Shape::from_dims(&[batch, 4, ed]), DType::F32, dev.clone())?)
    } else {
        None
    };
    Ok(Batch { latents, timesteps, embeds })
}
