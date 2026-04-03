use anyhow::Result;
use flame_core::{Tensor, DType, Shape};
use eridiffusion_core::Device;
use crate::devtensor::{randn_on, tensor_from_vec_on};

#[derive(Clone, Debug)]
pub struct SdxlTEConfig {
    pub ctx_tokens: usize,
    pub use_pooled: bool,
    pub use_time_ids: bool,
}

impl Default for SdxlTEConfig {
    fn default() -> Self {
        Self { ctx_tokens: 77, use_pooled: true, use_time_ids: true }
    }
}

/// Dual text encoders (TE1: CLIP-L, TE2: OpenCLIP-G) — stub wiring with correct shapes.
pub struct DualTextEncoders {
    device: Device,
    dtype: DType,
    cfg: SdxlTEConfig,
}

impl DualTextEncoders {
    pub fn new(device: Device, dtype: DType, cfg: SdxlTEConfig) -> Self { Self { device, dtype, cfg } }

    /// Tokenize+encode prompts; returns (seq features [B,77,Dim], pooled [B,1280], time_ids [B,6])
    pub fn encode(&self, prompts: &[String]) -> Result<(Tensor, Option<Tensor>, Option<Tensor>)> {
        let b = prompts.len();
        let ctx = self.cfg.ctx_tokens;
        let dim = 768usize;     // CLIP-L hidden size
        let pooled_dim = 1280;  // OpenCLIP-G pooled
        let seq = randn_on(Shape::from_dims(&[b, ctx, dim]), DType::F32, self.device.clone(), 0)?.to_dtype(self.dtype)?;
        let pooled = if self.cfg.use_pooled {
            Some(randn_on(Shape::from_dims(&[b, pooled_dim]), DType::F32, self.device.clone(), 0)?.to_dtype(self.dtype)?)
        } else { None };
        let time_ids = if self.cfg.use_time_ids {
            Some(tensor_from_vec_on(vec![0.0f32; b * 6], Shape::from_dims(&[b, 6]), DType::F32, self.device.clone())?.to_dtype(self.dtype)?)
        } else { None };
        Ok((seq, pooled, time_ids))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn encode_shapes_ok() -> anyhow::Result<()> {
        let dev = Device::cuda(0)?; let dtype = DType::BF16;
        let enc = DualTextEncoders::new(dev.clone(), dtype, SdxlTEConfig::default());
        let (seq, pooled, time_ids) = enc.encode(&["a cat".into(), "a dog".into()])?;
        assert_eq!(seq.shape().dims(), &[2,77,768]);
        assert_eq!(pooled.as_ref().unwrap().shape().dims(), &[2,1280]);
        assert_eq!(time_ids.as_ref().unwrap().shape().dims(), &[2,6]);
        Ok(())
    }
}
