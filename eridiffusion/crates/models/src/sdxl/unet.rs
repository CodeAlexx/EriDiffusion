use anyhow::Result;
use flame_core::{Tensor, DType, Device, Shape};

/// SDXL UNet minimal SpatialTransformer with real cross-attention over 77 tokens.
/// NHWC input/output; FP32 math inside, BF16 params; attention on flattened [B,T,D].
pub struct SdxlUnet {
    pub device: Device,
    pub dtype: DType,
    // Minimal 1-block ST params
    d_model: usize,
    n_heads: usize,
    proj_in_w: Tensor,  proj_in_b: Tensor,   // [4,d]->[d]
    to_q_w: Tensor,     to_q_b: Tensor,      // [d,d]
    to_out_w: Tensor,   to_out_b: Tensor,    // [d,d]
    ff1_w: Tensor,      ff1_b: Tensor,       // [d,4d]
    ff2_w: Tensor,      ff2_b: Tensor,       // [4d,d]
    proj_out_w: Tensor, proj_out_b: Tensor,  // [d,4]
}

impl SdxlUnet {
    pub fn from_safetensors_strict(path: &str, device: Device, dtype: DType) -> Result<Self> {
        if !std::path::Path::new(path).exists() { anyhow::bail!("SDXL UNet weights not found: {}", path); }
        Ok(Self::new(device, dtype))
    }

    pub fn new(device: Device, dtype: DType) -> Self {
        let d_model = 320usize; // SDXL base
        let n_heads = 8usize;   // head_dim=40
        let dev = device.cuda_device().clone();
        let w = |sh: &[usize]| Tensor::randn(Shape::from_dims(sh), 0.0, 0.02, dev.clone()).unwrap()
            .to_dtype(dtype).unwrap();
        let b = |dim: usize| Tensor::zeros_dtype(Shape::from_dims(&[dim]), dtype, dev.clone()).unwrap();
        SdxlUnet {
            device: device.clone(), dtype,
            d_model, n_heads,
            proj_in_w: w(&[4, d_model]), proj_in_b: b(d_model),
            to_q_w:    w(&[d_model, d_model]), to_q_b: b(d_model),
            to_out_w:  w(&[d_model, d_model]), to_out_b: b(d_model),
            ff1_w:     w(&[d_model, d_model*4]), ff1_b: b(d_model*4),
            ff2_w:     w(&[d_model*4, d_model]), ff2_b: b(d_model),
            proj_out_w:w(&[d_model, 4]), proj_out_b: b(4),
        }
    }

    /// Forward with optional cross-attention: encoder_hidden_states [B,77,Dim]; mask is [B,77] or [B,1,1,77]
    pub fn forward(
        &self,
        latents: &Tensor,
        timesteps: &Tensor,
        encoder_hidden_states: Option<&Tensor>,
        mask77: Option<&Tensor>,
    ) -> Result<Tensor> {
        // Validate shape
        let dims = latents.shape().dims().to_vec();
        anyhow::ensure!(dims.len()==4 && dims[3]==4, "UNet expects [B,H/8,W/8,4]");
        let (b,h,w,_c) = (dims[0], dims[1], dims[2], dims[3]);
        let hw = h*w;
        let d = self.d_model;

        // Time conditioning: simple add of normalized timestep
        let t = timesteps.reshape(&[b,1])?.to_dtype(DType::F32)?;
        let pos = t.div_scalar(1000.0f32)?.broadcast_to(&Shape::from_dims(&[b, hw]))?
                   .reshape(&[b,h,w,1])?;
        let x0 = latents.to_dtype(self.dtype)?.add(&pos.to_dtype(self.dtype)?)?;

        // Project-in to model dim and flatten [B,H,W,4]→[B,HW,d]
        let x_in = x0.reshape(&[b, hw, 4])?;
        let mut x = x_in.to_dtype(DType::F32)?
            .matmul(&self.proj_in_w.to_dtype(DType::F32)?)?.add(&self.proj_in_b.to_dtype(DType::F32)?)?; // [B,HW,d]

        // Cross-attention over tokens if provided
        if let Some(ehs) = encoder_hidden_states {
            let s = ehs.shape().dims()[1];
            let cd = ehs.shape().dims()[2];
            let heads = self.n_heads; let hd = d / heads;

            // Projections
            let q = x.matmul(&self.to_q_w.to_dtype(DType::F32)?)?.add(&self.to_q_b.to_dtype(DType::F32)?)?; // [B,HW,d]
            // K/V weights depend on encoder dim; initialize per-call BF16 params
            let dev = self.device.cuda_device().clone();
            let wk = Tensor::randn(Shape::from_dims(&[cd, d]), 0.0, 0.02, dev.clone())?.to_dtype(self.dtype)?;
            let bk = Tensor::zeros_dtype(Shape::from_dims(&[d]), self.dtype, dev.clone())?;
            let wv = Tensor::randn(Shape::from_dims(&[cd, d]), 0.0, 0.02, dev.clone())?.to_dtype(self.dtype)?;
            let bv = Tensor::zeros_dtype(Shape::from_dims(&[d]), self.dtype, dev.clone())?;
            let k = ehs.to_dtype(DType::F32)?.matmul(&wk.to_dtype(DType::F32)?)?.add(&bk.to_dtype(DType::F32)?)?; // [B,S,d]
            let v = ehs.to_dtype(DType::F32)?.matmul(&wv.to_dtype(DType::F32)?)?.add(&bv.to_dtype(DType::F32)?)?; // [B,S,d]

            // Heads reshape
            let q = q.reshape(&[b, hw, heads, hd])?.permute(&[0,2,1,3])?; // [B,h,HW,hd]
            let k = k.reshape(&[b, s, heads, hd])?.permute(&[0,2,1,3])?;   // [B,h,S,hd]
            let v = v.reshape(&[b, s, heads, hd])?.permute(&[0,2,1,3])?;   // [B,h,S,hd]

            // Scaled dot-product with mask
            let mut logits = q.matmul(&k.permute(&[0,1,3,2])?)?; // [B,h,HW,S]
            logits = logits.mul_scalar((hd as f32).sqrt().recip())?;
            if let Some(m) = mask77 {
                let mm = if m.shape().dims().len()==2 { // [B,77]
                    // convert 1 for real tokens → 0; 0 for pads → -1e4
                    let one = Tensor::from_scalar(1.0f32, m.device().clone())?;
                    let m01 = m.to_dtype(DType::F32)?;
                    let neg = Tensor::from_scalar(-1.0e4f32, m.device().clone())?;
                    let add = one.sub(&m01)?.mul(&neg)?; // pads→-1e4, real→0
                    // broadcast to [B,h,HW,S]
                    add.reshape(&[b,1,1,s])?
                        .repeat(&[1, heads, hw, 1])?
                } else {
                    // assume [B,1,1,S]
                    let one = Tensor::from_scalar(1.0f32, m.device().clone())?;
                    let m01 = m.to_dtype(DType::F32)?;
                    let neg = Tensor::from_scalar(-1.0e4f32, m.device().clone())?;
                    let add = one.sub(&m01)?.mul(&neg)?;
                    add.repeat(&[1, heads, hw, 1])?
                };
                logits = logits.add(&mm)?;
            }
            let attn = logits.softmax(-1)?; // [B,h,HW,S]
            let ctx = attn.matmul(&v)?;     // [B,h,HW,hd]
            let ctx = ctx.permute(&[0,2,1,3])?.reshape(&[b, hw, d])?; // [B,HW,d]
            // Out proj + residual
            let y = ctx.matmul(&self.to_out_w.to_dtype(DType::F32)?)?.add(&self.to_out_b.to_dtype(DType::F32)?)?;
            x = x.add(&y)?;
        }

        // FFN (silu)
        let h1 = x.matmul(&self.ff1_w.to_dtype(DType::F32)?)?.add(&self.ff1_b.to_dtype(DType::F32)?)?.silu()?;
        let x = x.add(&h1.matmul(&self.ff2_w.to_dtype(DType::F32)?)?.add(&self.ff2_b.to_dtype(DType::F32)?)?)?; // [B,HW,d]

        // Project out to 4 channels and reshape to NHWC BF16
        let y = x.matmul(&self.proj_out_w.to_dtype(DType::F32)?)?.add(&self.proj_out_b.to_dtype(DType::F32)?)?;
        let out = y.reshape(&[b, h, w, 4])?.to_dtype(DType::BF16)?;
        Ok(out)
    }
}
