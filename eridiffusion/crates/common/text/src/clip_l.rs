use crate::gpu_utils::{ensure_cuda, to_bf16_owned};
use anyhow::{anyhow, Context, Result};
use eridiffusion_common_weights as cw;
use flame_core::{kernels::adaln::layernorm_affine_bf16_inplace, DType, Device, Tensor};

pub struct ClipL {
    pub ctx_dim: usize,
    pub seq: usize,
    _device: Device,
    _emb_key: String,
    emb: Tensor, // [vocab, ctx_dim] BF16 on device
}

impl ClipL {
    /// Strict load and auto-detect embedding dims via largest 2D weight [V, D]
    pub fn from_weights_auto(path: &str, device: &Device, seq: usize) -> Result<Self> {
        let mut ld = cw::SafeLoader::open(path)?;
        let mut pick: Option<(String, Vec<usize>)> = None;
        for k in ld.list_keys()? {
            if let Ok(shape) = ld.shape_of(&k) {
                if shape.len() == 2 {
                    let score = shape[0] * shape[1];
                    if pick.as_ref().map(|(_, s): &(String, Vec<usize>)| s[0] * s[1]).unwrap_or(0)
                        < score
                    {
                        pick = Some((k, shape));
                    }
                }
            }
        }
        let (emb_key, shape) = pick.context("no 2D weight found in text encoder checkpoint")?;
        let ctx_dim = shape[1];
        tracing::info!("ClipL: detected embedding '{}' with ctx_dim={}", emb_key, ctx_dim);
        let emb = ld.get_bf16(&emb_key)?; // [V, D]
        Ok(Self { ctx_dim, seq, _device: device.clone(), _emb_key: emb_key, emb })
    }

    /// ids [B,seq] as I32; returns [B,seq,ctx_dim] BF16
    pub fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        let dims = ids.shape().dims().to_vec();
        if dims.len() != 2 {
            return Err(anyhow!("ClipL.forward: ids must be [B, T], got {:?}", dims));
        }
        if ids.dtype() != DType::I32 {
            return Err(anyhow!("ClipL.forward: ids must be I32"));
        }

        // Gather embeddings on device: [B, T, D]
        let gathered = self.emb.index_select0(ids)?;
        ensure_cuda("clip_l.gather", &gathered)?;

        let b = dims[0] as i32;
        let t = dims[1] as i32;
        let mut nhwc = gathered.reshape(&[dims[0], dims[1], 1, self.ctx_dim])?.clone_result()?;
        layernorm_affine_bf16_inplace(&mut nhwc, None, None, b, t, 1, self.ctx_dim as i32, 1e-5)
            .map_err(|e| anyhow!("ClipL.layernorm failed: {e}"))?;
        let norm = nhwc.reshape(&[dims[0], dims[1], self.ctx_dim])?;
        Ok(norm)
    }

    pub fn pooled(&self, hidden: &Tensor) -> Result<Tensor> {
        ensure_cuda("clip_l.pooled.in", hidden)?;
        let dims = hidden.shape().dims().to_vec();
        if dims.len() != 3 {
            return Err(anyhow!("ClipL.pooled: expected [B,T,D], got {:?}", dims));
        }
        let seq = dims[1] as f32;
        let sum = hidden.sum_dim_keepdim(1)?; // [B,1,D]
        let mean = sum.div_scalar(seq)?;
        let pooled = mean.reshape(&[dims[0], dims[2]])?;
        to_bf16_owned("clip_l.pooled.out", &pooled)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use flame_core::{shape::Shape, Device as FDevice, Tensor};

    fn build_test_clip(device: &FDevice, vocab: usize, dim: usize, seq: usize) -> Result<ClipL> {
        let emb = Tensor::zeros_dtype(
            Shape::from_dims(&[vocab, dim]),
            DType::BF16,
            device.cuda_device_arc(),
        )?;
        Ok(ClipL { ctx_dim: dim, seq, _device: device.clone(), _emb_key: "test".into(), emb })
    }

    fn make_ids(device: &FDevice, b: usize, t: usize, vocab: usize) -> Result<Tensor> {
        let mut data = Vec::with_capacity(b * t);
        for batch in 0..b {
            for pos in 0..t {
                let id = ((batch * t + pos) % vocab) as f32;
                data.push(id);
            }
        }
        let ids = Tensor::from_vec(data, Shape::from_dims(&[b, t]), device.cuda_device_arc())?;
        Ok(ids.to_dtype(DType::I32)?)
    }

    #[test]
    fn forward_returns_bf16_on_cuda() -> Result<()> {
        let device = match FDevice::cuda(0) {
            Ok(dev) => dev,
            Err(_) => return Ok(()),
        };
        let clip = build_test_clip(&device, 128, 32, 77)?;
        let ids = make_ids(&device, 2, 16, 128)?;
        let out = clip.forward(&ids)?;
        assert_eq!(out.dtype(), DType::BF16);
        assert_eq!(out.device().ordinal(), ids.device().ordinal());
        assert_eq!(out.shape().dims(), &[2, 16, 32]);
        Ok(())
    }

    #[test]
    fn pooled_returns_bf16_on_cuda() -> Result<()> {
        let device = match FDevice::cuda(0) {
            Ok(dev) => dev,
            Err(_) => return Ok(()),
        };
        let clip = build_test_clip(&device, 128, 32, 77)?;
        let ids = make_ids(&device, 2, 16, 128)?;
        let hidden = clip.forward(&ids)?;
        let pooled = clip.pooled(&hidden)?;
        assert_eq!(pooled.dtype(), DType::BF16);
        assert_eq!(pooled.device().ordinal(), hidden.device().ordinal());
        assert_eq!(pooled.shape().dims(), &[2, 32]);
        Ok(())
    }
}
