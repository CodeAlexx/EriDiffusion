//! GPU-only LoRA helpers for EriDiffusion pipelines.
//! No CPU fallbacks, no stubs. Storage defaults to BF16; math accumulates in FP32.

use std::sync::Arc;

use eridiffusion_core::Device;
use flame_core::{CudaDevice, DType, Error as CoreError, Result as CoreResult, Shape, Tensor, rng};

/// Configuration for a LoRA site.
#[derive(Clone, Copy, Debug)]
pub struct LoRAConfig {
    pub rank: i64,
    pub alpha: f32,
    pub dtype: DType,
    pub seed: Option<u64>,
}

impl Default for LoRAConfig {
    fn default() -> Self {
        Self { rank: 16, alpha: 16.0, dtype: DType::BF16, seed: None }
    }
}

/// Models implement this to advertise LoRA sites.
pub trait HasLoRA {
    fn enumerate_lora_sites(&self) -> Vec<String>;
}

#[derive(Debug)]
pub struct LoRALinear {
    pub site: String,
    pub in_dim: i64,
    pub out_dim: i64,
    pub r: i64,
    pub alpha: f32,
    pub scale: f32,
    pub a: Tensor, // [IN,R]
    pub b: Tensor, // [R,OUT]
    pub dtype: DType,
    device: Device,
}

impl LoRALinear {
    pub fn new(site: impl Into<String>, in_dim: i64, out_dim: i64, cfg: LoRAConfig, device: &Device) -> CoreResult<Self> {
        if cfg.rank <= 0 {
            return Err(CoreError::InvalidInput("LoRA rank must be > 0".into()));
        }
        let cuda = cuda_for(device)?;
        if let Some(seed) = cfg.seed { rng::set_seed(seed)?; }
    let shape_a = Shape::from_dims(&[in_dim as usize, cfg.rank as usize]);
    let mut a = Tensor::randn(shape_a, 0.0, 1.0, cuda.clone())?;
        a = a.to_dtype(cfg.dtype)?;
        a = a.mul_scalar(0.01f32)?;
        if let Some(seed) = cfg.seed { rng::set_seed(seed ^ 0x9E37_79B1_85EB_CA87)?; }
    let shape_b = Shape::from_dims(&[cfg.rank as usize, out_dim as usize]);
    let mut b = Tensor::randn(shape_b, 0.0, 1.0, cuda.clone())?;
        b = b.to_dtype(cfg.dtype)?;
        b = b.mul_scalar(0.01f32)?;
        let scale = cfg.alpha / cfg.rank as f32;
        Ok(Self { site: site.into(), in_dim, out_dim, r: cfg.rank, alpha: cfg.alpha, scale, a, b, dtype: cfg.dtype, device: device.clone() })
    }

    pub fn new_zero(site: impl Into<String>, in_dim: i64, out_dim: i64, cfg: LoRAConfig, device: &Device) -> CoreResult<Self> {
        if cfg.rank <= 0 {
            return Err(CoreError::InvalidInput("LoRA rank must be > 0".into()));
        }
        let cuda = cuda_for(device)?;
        let a = Tensor::zeros_dtype(Shape::from_dims(&[in_dim as usize, cfg.rank as usize]), cfg.dtype, cuda.clone())?;
        let b = Tensor::zeros_dtype(Shape::from_dims(&[cfg.rank as usize, out_dim as usize]), cfg.dtype, cuda.clone())?;
        let scale = cfg.alpha / cfg.rank as f32;
        Ok(Self { site: site.into(), in_dim, out_dim, r: cfg.rank, alpha: cfg.alpha, scale, a, b, dtype: cfg.dtype, device: device.clone() })
    }

    pub fn delta_weight(&self) -> CoreResult<Tensor> {
        // BF16 storage, FP32 accumulate handled inside matmul
        let dw = self.a.matmul(&self.b)?;
        dw.mul_scalar(self.scale)
    }

    pub fn apply_delta_to(&self, base: &Tensor) -> CoreResult<Tensor> {
        ensure_same_device(base, &self.device)?;
        let base = to_dtype(base, self.dtype)?;
        let dw = self.delta_weight()?;
        base.add(&dw)
    }

    pub fn forward_delta(&self, x: &Tensor) -> CoreResult<Tensor> {
        ensure_same_device(x, &self.device)?;
        let x = to_dtype(x, self.dtype)?;
        let xa = x.matmul(&self.a)?;
        let out = xa.matmul(&self.b)?;
        out.mul_scalar(self.scale)
    }

    pub fn keys(&self) -> (String, String) {
        (format!("{}.lora_A", self.site), format!("{}.lora_B", self.site))
    }

    pub fn to_f32(&self) -> CoreResult<(Vec<f32>, Vec<f32>)> {
        let a32 = self.a.to_dtype(DType::F32)?;
        let b32 = self.b.to_dtype(DType::F32)?;
        let av = a32.to_vec_f32().map_err(|e| CoreError::InvalidInput(format!("to_vec_f32(A): {e}")))?;
        let bv = b32.to_vec_f32().map_err(|e| CoreError::InvalidInput(format!("to_vec_f32(B): {e}")))?;
        Ok((av, bv))
    }
}

#[derive(Default)]
pub struct LoRASites {
    pub sites: Vec<String>,
}

impl LoRASites {
    pub fn push(&mut self, site: impl Into<String>) { self.sites.push(site.into()); }
}

impl HasLoRA for LoRASites {
    fn enumerate_lora_sites(&self) -> Vec<String> { self.sites.clone() }
}

#[inline]
pub fn lora_key(block: usize, suffix: &str, part: char) -> String {
    format!("blocks.{:02}.{}.lora_{}", block, suffix, part)
}

#[inline]
pub fn lora_param_ids(site: &str) -> (String, String) {
    (format!("{site}.lora_A"), format!("{site}.lora_B"))
}

fn cuda_for(device: &Device) -> CoreResult<Arc<CudaDevice>> {
    match device {
        Device::Cpu => Err(CoreError::Unsupported("CPU device not supported".into())),
        Device::Cuda(idx) => CudaDevice::new(*idx)
            .map_err(|e| CoreError::InvalidInput(format!("cuda {idx}: {e}")))
    }
}

fn ensure_same_device(t: &Tensor, device: &Device) -> CoreResult<()> {
    let expected = cuda_for(device)?;
    let actual = t.device();
    if actual.ordinal() != expected.ordinal() {
        Err(CoreError::Unsupported(format!(
            "tensor on cuda:{} expected cuda:{}",
            actual.ordinal(),
            expected.ordinal()
        )))
    } else { Ok(()) }
}

fn to_dtype(t: &Tensor, dtype: DType) -> CoreResult<Tensor> {
    if t.dtype() == dtype { Ok(t.clone()) } else { t.to_dtype(dtype) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lora_delta_shapes() {
        let dev = Device::Cuda { index: 0 };
        let cfg = LoRAConfig { rank: 8, alpha: 8.0, dtype: DType::BF16, seed: Some(123) };
        let l = LoRALinear::new("blocks.00.attn.q", 64, 32, cfg, &dev).unwrap();
        let cuda = cuda_for(&dev).unwrap();
        let base = Tensor::zeros_dtype(Shape::from(&[64, 32]), DType::BF16, cuda.clone()).unwrap();
        let fused = l.apply_delta_to(&base).unwrap();
        assert_eq!(fused.shape().as_slice(), &[64, 32]);
        let x = Tensor::zeros_dtype(Shape::from(&[4, 64]), DType::BF16, cuda).unwrap();
        let y = l.forward_delta(&x).unwrap();
        assert_eq!(y.shape().as_slice(), &[4, 32]);
    }
}
