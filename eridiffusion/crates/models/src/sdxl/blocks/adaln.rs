//! SDXL AdaLayerNorm helpers (BF16 params, FP32 math).

use eridiffusion_core::{Device, Error, Result};
use flame_core::{DType, Tensor};

const EPS: f32 = 1e-5;

fn full(dev: &Device, val: f32, shape: &[i64]) -> Result<Tensor> {
    Tensor::full(val, shape, dev.clone().into(), DType::BF16)
        .map_err(|e| Error::from_msg(format!("tensor::full: {e}")))
}

fn linear(cond: &Tensor, w: &Tensor, b: &Tensor) -> Result<Tensor> {
    cond.matmul(w)?.add(b)
}

fn layer_norm_affine(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    let last = *x.shape().last().ok_or_else(|| Error::from_msg("layer_norm: empty shape"))?;
    let mean = x.mean_keepdim(-1)?;
    let var = (x - &mean)?.powf(2.0)?.mean_keepdim(-1)?;
    let denom = (var + EPS)?.sqrt()?;
    let xhat = (x - &mean)? / &denom?;
    let reshape = |t: &Tensor| t.reshape(&[1, 1, 1, last])?;
    Ok(xhat * &reshape(weight)? + &reshape(bias)?)
}

pub struct AdaLayerNorm {
    ln_weight: Tensor,
    ln_bias: Tensor,
    proj_w: Tensor,
    proj_b: Tensor,
}

impl AdaLayerNorm {
    pub fn new(dev: Device, channels: usize, cond_dim: usize) -> Result<Self> {
        Ok(Self {
            ln_weight: full(&dev, 1.0, &[channels as i64])?,
            ln_bias: full(&dev, 0.0, &[channels as i64])?,
            proj_w: Tensor::randn(&[cond_dim as i64, (2 * channels) as i64], &dev, DType::BF16)?,
            proj_b: full(&dev, 0.0, &[(2 * channels) as i64])?,
        })
    }

    pub fn forward(&self, x: &Tensor, cond: &Tensor) -> Result<Tensor> {
        let ln = layer_norm_affine(x, &self.ln_weight, &self.ln_bias)?;
        let aff = linear(cond, &self.proj_w, &self.proj_b)?;
        let c = self.ln_weight.shape()[0];
        let shift = aff.narrow_last(0, c)?;
        let scale = aff.narrow_last(c as i64, c as i64)?;
        let reshape = |t: Tensor| t.reshape(&[t.shape()[0], 1, 1, c])?.to_dtype(DType::F32);
        let shift4 = reshape(shift)?;
        let scale4 = reshape(scale)?;
        Ok((ln * (&scale4 + 1.0)?)? + &shift4)
    }
}

pub struct AdaLayerNormZero {
    ln_weight: Tensor,
    ln_bias: Tensor,
    proj_w: Tensor,
    proj_b: Tensor,
}

impl AdaLayerNormZero {
    pub fn new(dev: Device, channels: usize, cond_dim: usize) -> Result<Self> {
        Ok(Self {
            ln_weight: full(&dev, 1.0, &[channels as i64])?,
            ln_bias: full(&dev, 0.0, &[channels as i64])?,
            proj_w: full(&dev, 0.0, &[cond_dim as i64, (3 * channels) as i64])?,
            proj_b: full(&dev, 0.0, &[(3 * channels) as i64])?,
        })
    }

    pub fn forward(&self, x: &Tensor, cond: &Tensor) -> Result<(Tensor, Tensor)> {
        let ln = layer_norm_affine(x, &self.ln_weight, &self.ln_bias)?;
        let aff = linear(cond, &self.proj_w, &self.proj_b)?;
        let c = self.ln_weight.shape()[0];
        let shift = aff.narrow_last(0, c)?;
        let scale = aff.narrow_last(c as i64, c as i64)?;
        let gate = aff.narrow_last((2 * c) as i64, c as i64)?;
        let reshape = |t: Tensor| t.reshape(&[t.shape()[0], 1, 1, c])?.to_dtype(DType::F32);
        let shift4 = reshape(shift)?;
        let scale4 = reshape(scale)?;
        let gate4 = reshape(gate)?;
        let y = (ln * (&scale4 + 1.0)?)? + &shift4;
        Ok((y, gate4))
    }
}
