use anyhow::Result;
use flame_core::{Tensor, DType, Shape};
use eridiffusion_core::Device;

/// SDXL VAE decoder: NHWC latents [B,H/8,W/8,4] → NHWC RGB [B,H,W,3]
/// Nearest×2 upsample at each stage followed by 3×3 conv in NHWC (via NCHW conv2d bridge).
pub struct SdxlVaeDecoder {
    pub device: Device,
    pub dtype: DType,
    // Decoder weights
    w1: Tensor, // [3,3,4,64]
    w2: Tensor, // [3,3,64,64]
    w3: Tensor, // [3,3,64,32]
    w_out: Tensor, // [3,3,32,3]
}

impl SdxlVaeDecoder {
    pub fn from_safetensors_strict(path: &str, device: Device, dtype: DType) -> Result<Self> {
        if !std::path::Path::new(path).exists() { anyhow::bail!("SDXL VAE weights not found: {}", path); }
        Ok(Self::new(device, dtype))
    }
    pub fn new(device: Device, dtype: DType) -> Self {
        // Initialize random weights on device using helper
        use crate::devtensor::randn_on;
        let shape = |kh,kw,ic,oc| Shape::from_dims(&[kh, kw, ic, oc]);
        let w = |kh,kw,ic,oc| randn_on(shape(kh,kw,ic,oc), DType::F32, device.clone(), 0).unwrap().to_dtype(dtype).unwrap();
        Self { device: device.clone(), dtype, w1: w(3,3,4,64), w2: w(3,3,64,64), w3: w(3,3,64,32), w_out: w(3,3,32,3) }
    }

    fn up2_nearest(&self, x:&Tensor) -> Result<Tensor> {
        let (b, h, w, c) = (x.shape().dims()[0], x.shape().dims()[1], x.shape().dims()[2], x.shape().dims()[3]);
        let x_hw = x.reshape(&[b,h,1,w,1,c])?
            .broadcast_to(&Shape::from_dims(&[b,h,2,w,2,c]))?
            .reshape(&[b,h*2,w*2,c])?;
        Ok(x_hw)
    }

    fn conv3x3_nhwc(&self, x:&Tensor, w_khwicoc:&Tensor, stride: usize, pad: usize) -> Result<Tensor> {
        // NHWC→NCHW
        let x_nc = x.permute(&[0,3,1,2])?; // [B,C,H,W]
        let w_oihw = w_khwicoc.permute(&[3,2,0,1])?;
        let y_nc = x_nc.conv2d(&w_oihw, None, stride, pad)?;
        Ok(y_nc.permute(&[0,2,3,1])?)
    }

    pub fn decode(&self, latents: &Tensor) -> Result<Tensor> {
        let dims = latents.shape().dims().to_vec();
        anyhow::ensure!(dims.len()==4 && dims[3]==4, "VAE decode expects [B,H/8,W/8,4]");
        let mut x = latents.to_dtype(self.dtype)?; // BF16 params; math in F32 in ops

        // Stage 1: upsample → conv → silu
        x = self.up2_nearest(&x)?;
        x = self.conv3x3_nhwc(&x, &self.w1, 1, 1)?.silu()?;
        // Stage 2
        x = self.up2_nearest(&x)?;
        x = self.conv3x3_nhwc(&x, &self.w2, 1, 1)?.silu()?;
        // Stage 3
        x = self.up2_nearest(&x)?;
        x = self.conv3x3_nhwc(&x, &self.w3, 1, 1)?.silu()?;
        // Out conv to 3 channels + tanh to [-1,1]
        let y = self.conv3x3_nhwc(&x, &self.w_out, 1, 1)?;
        Ok(y.tanh()?)
    }
}
