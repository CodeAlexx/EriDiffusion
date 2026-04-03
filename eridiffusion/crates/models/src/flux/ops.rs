use anyhow::Result;
use flame_core::{DType, Tensor};

use crate::devtensor::to_dtype;

/// Cached F32 weights/bias for local compute islands that require F32 storage.
pub struct Affine32 {
    w_bf16: Tensor,
    b_bf16: Option<Tensor>,
    w_f32: Option<Tensor>,
    b_f32: Option<Tensor>,
}

impl Affine32 {
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Self {
        Self {
            w_bf16: weight,
            b_bf16: bias,
            w_f32: None,
            b_f32: None,
        }
    }

    fn weight_f32(&mut self) -> Result<Tensor> {
        if let Some(w) = &self.w_f32 {
            return Ok(w.clone());
        }
        let w32 = self.w_bf16.to_dtype(DType::F32)?;
        self.w_f32 = Some(w32.clone());
        Ok(w32)
    }

    fn bias_f32(&mut self) -> Result<Option<Tensor>> {
        match (&self.b_f32, &self.b_bf16) {
            (Some(b), _) => Ok(Some(b.clone())),
            (None, Some(bf)) => {
                let b32 = bf.to_dtype(DType::F32)?;
                self.b_f32 = Some(b32.clone());
                Ok(Some(b32))
            }
            _ => Ok(None),
        }
    }

    pub fn forward(&mut self, x_any: &Tensor) -> Result<Tensor> {
        let x = if x_any.dtype() != DType::F32 {
            to_dtype(x_any, DType::F32)?
        } else {
            x_any.clone()
        };
        let w = self.weight_f32()?;
        debug_assert_eq!(x.dtype(), DType::F32, "Affine32 expects F32 input");
        debug_assert_eq!(w.dtype(), DType::F32, "Affine32 expects F32 weights");
        let mut y = x.matmul(&w)?;
        if let Some(b) = self.bias_f32()? {
            y = y.add(&b)?;
        }
        Ok(y)
    }
}

/// F32 positional embedding helper (sin/cos + optional projection).
pub struct PositionalEmbed32 {
    affine: Option<Affine32>,
}

impl PositionalEmbed32 {
    pub fn new(proj_weight: Option<Tensor>, proj_bias: Option<Tensor>) -> Self {
        let affine = proj_weight.map(|w| Affine32::new(w, proj_bias));
        Self { affine }
    }

    pub fn forward(&mut self, t_any: &Tensor, hidden: usize, device: &crate::models::devtensor::Device) -> Result<Tensor> {
        let t32 = if t_any.dtype() != DType::F32 {
            to_dtype(t_any, DType::F32)?
        } else {
            t_any.clone()
        };
        let base = crate::models::flux::timestep_internal::timestep_embedding_f32(&t32, hidden, device)?;
        if let Some(aff) = &mut self.affine {
            aff.forward(&base)
        } else {
            Ok(base)
        }
    }
}
