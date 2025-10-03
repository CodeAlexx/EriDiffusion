use crate::devtensor::tensor_from_vec_on;
use anyhow::Result;
use eridiffusion_core::Device;
use flame_core::{DType, Shape, Tensor};

/// Simple sinusoidal timestep embedding: [B] -> [B,H]
pub fn timestep_embedding(
    t: &Tensor,
    hidden: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    // t expected [B]
    let b = t.shape().dims()[0];
    let half = hidden / 2;
    let inv_vec: Vec<f32> = (0..half)
        .map(|i| (10000f32).powf(-(i as f32) / (half as f32)))
        .collect();
    let t_vec = t.to_dtype(DType::F32)?.to_vec_f32()?;
    let mut emb_host = Vec::with_capacity(b * hidden);
    let scale = 1.0f32 / (hidden as f32).sqrt();
    for &time in &t_vec {
        for &freq in &inv_vec {
            let arg = time * freq;
            emb_host.push((arg.sin() * scale).clamp(-1e3f32, 1e3f32));
        }
        for &freq in &inv_vec {
            let arg = time * freq;
            emb_host.push((arg.cos() * scale).clamp(-1e3f32, 1e3f32));
        }
    }
    let emb_f32 = tensor_from_vec_on(emb_host, Shape::from_dims(&[b, hidden]), device, DType::F32)?;
    if dtype == DType::F32 {
        Ok(emb_f32)
    } else {
        Ok(emb_f32.to_dtype(dtype)?)
    }
}
