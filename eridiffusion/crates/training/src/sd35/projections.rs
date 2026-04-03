use eridiffusion_core::{Error, Result};
use flame_core::{DType, Device as FlameDevice, Parameter, Shape, Tensor};

use crate::utils::dtype::cast_preserve_grad;

const TARGET_TOKENS: usize = 77;
const TEXT_INPUT_DIM: usize = 6144;
const LATENT_IN_CHANNELS: usize = 16;
const LATENT_SPATIAL_DIMS: usize = 4;
const VELOCITY_OUT_CHANNELS: usize = 16;

fn kaiming_scale(rank: usize) -> f32 {
    (1.0f32 / rank.max(1) as f32).sqrt()
}

pub struct Latent1x1Conv {
    weight: Parameter,
    bias: Parameter,
    hidden: usize,
}

impl Latent1x1Conv {
    pub fn new(device: FlameDevice, hidden: usize) -> Result<Self> {
        let scale = kaiming_scale(LATENT_IN_CHANNELS);
        let w = Tensor::randn(
            Shape::from_dims(&[LATENT_IN_CHANNELS, hidden]),
            0.0,
            scale,
            device.cuda_device_arc(),
        )
        .map_err(Error::from)?
        .to_dtype(DType::BF16)
        .map_err(Error::from)?
        .requires_grad_(true);
        let b =
            Tensor::zeros_dtype(Shape::from_dims(&[hidden]), DType::BF16, device.cuda_device_arc())
                .map_err(Error::from)?
                .requires_grad_(true);
        Ok(Self { weight: Parameter::new(w), bias: Parameter::new(b), hidden })
    }

    pub fn forward(&self, latents: &Tensor) -> Result<Tensor> {
        let dims = latents.shape().dims().to_vec();
        if dims.len() != LATENT_SPATIAL_DIMS {
            return Err(Error::Training(format!(
                "Latent1x1Conv expects 4D tensor, got {:?}",
                dims
            )));
        }
        let (b, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);
        if c != LATENT_IN_CHANNELS {
            return Err(Error::Training(format!(
                "Latent channels mismatch: expected {}, got {}",
                LATENT_IN_CHANNELS, c
            )));
        }

        let x_nhwc = latents.permute(&[0, 2, 3, 1])?;
        let x32 = cast_preserve_grad(&x_nhwc, DType::F32)?;
        let flat = x32.reshape(&[b * h * w, c])?;

        let w_t = self.weight.tensor().map_err(Error::from)?;
        let w32 = cast_preserve_grad(&w_t, DType::F32)?;
        let mut out = flat.matmul(&w32)?; // [BHW, hidden]

        let bias = self.bias.tensor().map_err(Error::from)?;
        let bias32 = cast_preserve_grad(&bias, DType::F32)?;
        out = out.add(
            &bias32.unsqueeze(0)?.broadcast_to(&Shape::from_dims(&[b * h * w, self.hidden]))?,
        )?;

        let out = out.reshape(&[b, h, w, self.hidden])?;
        let out_nchw = out.permute(&[0, 3, 1, 2])?;
        cast_preserve_grad(&out_nchw, latents.dtype())
    }

    pub fn parameters(&self) -> Vec<Parameter> {
        vec![self.weight.clone(), self.bias.clone()]
    }
}

pub struct TextLinearPad {
    weight: Parameter,
    bias: Parameter,
    hidden: usize,
}

impl TextLinearPad {
    pub fn new(device: FlameDevice, hidden: usize) -> Result<Self> {
        let scale = kaiming_scale(TEXT_INPUT_DIM);
        let w = Tensor::randn(
            Shape::from_dims(&[TEXT_INPUT_DIM, hidden]),
            0.0,
            scale,
            device.cuda_device_arc(),
        )
        .map_err(Error::from)?
        .to_dtype(DType::BF16)
        .map_err(Error::from)?
        .requires_grad_(true);
        let b =
            Tensor::zeros_dtype(Shape::from_dims(&[hidden]), DType::BF16, device.cuda_device_arc())
                .map_err(Error::from)?
                .requires_grad_(true);
        Ok(Self { weight: Parameter::new(w), bias: Parameter::new(b), hidden })
    }

    pub fn forward(&self, text: &Tensor) -> Result<Tensor> {
        let dims = text.shape().dims().to_vec();
        if dims.len() != 3 {
            return Err(Error::Training(format!(
                "TextLinearPad expects [B,Seq,{}], got {:?}",
                TEXT_INPUT_DIM, dims
            )));
        }
        let (b, seq, dim) = (dims[0], dims[1], dims[2]);
        if dim != TEXT_INPUT_DIM {
            return Err(Error::Training(format!(
                "Text embedding dim mismatch: expected {}, got {}",
                TEXT_INPUT_DIM, dim
            )));
        }

        let mut work =
            if seq > TARGET_TOKENS { text.narrow(1, 0, TARGET_TOKENS)? } else { text.clone() };
        if seq < TARGET_TOKENS {
            let pad = Tensor::zeros_dtype(
                Shape::from_dims(&[b, TARGET_TOKENS - seq, TEXT_INPUT_DIM]),
                text.dtype(),
                text.device().clone(),
            )
            .map_err(Error::from)?;
            let refs: [&Tensor; 2] = [&work, &pad];
            work = Tensor::cat(&refs, 1).map_err(Error::from)?;
        }

        let work32 = cast_preserve_grad(&work, DType::F32)?;
        let flat = work32.reshape(&[b * TARGET_TOKENS, TEXT_INPUT_DIM])?;
        let w_t = self.weight.tensor().map_err(Error::from)?;
        let w32 = cast_preserve_grad(&w_t, DType::F32)?;
        let mut out = flat.matmul(&w32)?;
        let bias = self.bias.tensor().map_err(Error::from)?;
        let bias32 = cast_preserve_grad(&bias, DType::F32)?;
        out = out.add(
            &bias32
                .unsqueeze(0)?
                .broadcast_to(&Shape::from_dims(&[b * TARGET_TOKENS, self.hidden]))?,
        )?;
        let out = out.reshape(&[b, TARGET_TOKENS, self.hidden])?;
        cast_preserve_grad(&out, text.dtype())
    }

    pub fn parameters(&self) -> Vec<Parameter> {
        vec![self.weight.clone(), self.bias.clone()]
    }
}

pub struct VelocityHead {
    weight: Parameter,
    bias: Parameter,
}

impl VelocityHead {
    pub fn new(device: FlameDevice, hidden: usize) -> Result<Self> {
        let scale = kaiming_scale(hidden);
        let w = Tensor::randn(
            Shape::from_dims(&[hidden, VELOCITY_OUT_CHANNELS]),
            0.0,
            scale,
            device.cuda_device_arc(),
        )
        .map_err(Error::from)?
        .to_dtype(DType::BF16)
        .map_err(Error::from)?
        .requires_grad_(true);
        let b = Tensor::zeros_dtype(
            Shape::from_dims(&[VELOCITY_OUT_CHANNELS]),
            DType::BF16,
            device.cuda_device_arc(),
        )
        .map_err(Error::from)?
        .requires_grad_(true);
        Ok(Self { weight: Parameter::new(w), bias: Parameter::new(b) })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let dims = hidden_states.shape().dims().to_vec();
        if dims.len() != LATENT_SPATIAL_DIMS {
            return Err(Error::Training(format!("VelocityHead expects 4D tensor, got {:?}", dims)));
        }
        let (b, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);

        let x_nhwc = hidden_states.permute(&[0, 2, 3, 1])?;
        let x32 = cast_preserve_grad(&x_nhwc, DType::F32)?;
        let flat = x32.reshape(&[b * h * w, c])?;
        let w_t = self.weight.tensor().map_err(Error::from)?;
        let w32 = cast_preserve_grad(&w_t, DType::F32)?;
        let mut out = flat.matmul(&w32)?;
        let bias = self.bias.tensor().map_err(Error::from)?;
        let bias32 = cast_preserve_grad(&bias, DType::F32)?;
        out = out.add(
            &bias32
                .unsqueeze(0)?
                .broadcast_to(&Shape::from_dims(&[b * h * w, VELOCITY_OUT_CHANNELS]))?,
        )?;
        let out = out.reshape(&[b, h, w, VELOCITY_OUT_CHANNELS])?;
        let out_nchw = out.permute(&[0, 3, 1, 2])?;
        cast_preserve_grad(&out_nchw, hidden_states.dtype())
    }

    pub fn parameters(&self) -> Vec<Parameter> {
        vec![self.weight.clone(), self.bias.clone()]
    }
}
