use anyhow::{Context, Result, bail};
use safetensors::{SafeTensors, tensor::TensorView};
use std::{fs, path::PathBuf};
use half::{f16, bf16};
use flame_core::{Tensor, Shape, CudaDevice};

/// Lightweight safetensors shard wrapper for fast tensor reads
pub struct SafeTensorFile {
    path: PathBuf,
    st: SafeTensors<'static>,
}

impl SafeTensorFile {
    pub fn open(path: &std::path::Path) -> Result<Self> {
        let pb = PathBuf::from(path);
        let bytes = fs::read(&pb).with_context(|| format!("reading shard: {}", pb.display()))?;
        let boxed = bytes.into_boxed_slice();
        let static_ref: &'static [u8] = Box::leak(boxed);
        let st = SafeTensors::deserialize(static_ref)?;
        Ok(Self { path: pb, st })
    }

    /// Get tensor by key as BF16 on cuda:0 device in NHWC or [seq,dim] layouts as stored
    pub fn get(&mut self, key: &str) -> Result<Tensor> {
        let tv: TensorView<'_> = self.st.tensor(key)
            .with_context(|| format!("missing key '{}' in {}", key, self.path.display()))?;
        let shape = Shape::from_dims(tv.shape());
        let device = CudaDevice::new(0)?;
        // Convert to f32 vec first (no bytemuck bound on half types)
        let f32_data: Vec<f32> = match tv.dtype() {
            safetensors::Dtype::F32 => {
                let bytes = tv.data();
                let mut out = Vec::with_capacity(bytes.len() / 4);
                for chunk in bytes.chunks_exact(4) {
                    out.push(f32::from_le_bytes([chunk[0],chunk[1],chunk[2],chunk[3]]));
                }
                out
            }
            safetensors::Dtype::F16 => {
                let bytes = tv.data();
                let mut out = Vec::with_capacity(bytes.len() / 2);
                for chunk in bytes.chunks_exact(2) {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    out.push(f16::from_bits(bits).to_f32());
                }
                out
            }
            safetensors::Dtype::BF16 => {
                let bytes = tv.data();
                let mut out = Vec::with_capacity(bytes.len() / 2);
                for chunk in bytes.chunks_exact(2) {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    out.push(f32::from(bf16::from_bits(bits)));
                }
                out
            }
            other => bail!("unsupported dtype {:?} for key {}", other, key),
        };
        // Create BF16 tensor on device (consistent with latents/ctx usage)
        let t = Tensor::from_f32_to_bf16(f32_data, shape, device)?;
        Ok(t)
    }
}
