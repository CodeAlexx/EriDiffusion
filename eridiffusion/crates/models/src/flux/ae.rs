use anyhow::{bail, Context, Result};
use eridiffusion_common_vae as common_vae;
use eridiffusion_common_vae::{VaeKind, VaePolicy, VaeSpec};
use eridiffusion_core::Device;
use flame_core::{DType, Tensor};
use std::path::Path;

/// Lightweight Flux auto-encoder wrapper.
///
/// The historic Flux stacks used a 4-channel latent with an 8× spatial downscale
/// and a fixed decode scale (~0.13025).  For Phase 4 recovery we only need a
/// deterministic GPU path that mirrors those conventions so the trainer can
/// obtain latents without touching CPU fallbacks.
pub struct FluxAE {
    spec: VaeSpec,
    device: Device,
    policy: VaePolicy,
    storage: DType,
}

impl FluxAE {
    /// Construct a Flux VAE descriptor.  The actual math is delegated to the
    /// shared `common_vae` helpers which provide GPU-only encode/decode shims.
    pub fn load(path: &str, device: Device, storage: DType) -> Result<Self> {
        if !matches!(device, Device::Cuda(_)) {
            bail!("FluxAE requires a CUDA device");
        }
        if !Path::new(path).exists() {
            bail!("FluxAE weights not found at {}", path);
        }
        let spec = VaeSpec {
            kind: VaeKind::Flux,
            path: path.to_string(),
            latent_div: 8,
            latent_channels: 4,
            latent_scale: 0.13025,
        };
        Ok(Self { spec, device, policy: VaePolicy::GpuFirst, storage })
    }

    /// Encode NHWC images (BF16/F32) into NHWC latents (BF16/F32).
    pub fn encode_latents(&self, images: &Tensor) -> Result<Tensor> {
        let latents = common_vae::encode(&self.spec, images, self.policy)
            .with_context(|| "FluxAE encode failed")?;
        self.ensure_dtype(latents)
    }

    /// Decode NHWC latents back to images.  Handy for parity checks.
    #[allow(dead_code)]
    pub fn decode_latents(&self, latents: &Tensor) -> Result<Tensor> {
        let images = common_vae::decode(&self.spec, latents, self.policy)
            .with_context(|| "FluxAE decode failed")?;
        self.ensure_dtype(images)
    }

    pub fn latent_channels(&self) -> usize { self.spec.latent_channels }
    pub fn latent_div(&self) -> usize { self.spec.latent_div }
    pub fn storage_dtype(&self) -> DType { self.storage }
    pub fn device(&self) -> &Device { &self.device }

    fn ensure_dtype(&self, tensor: Tensor) -> Result<Tensor> {
        if tensor.dtype() == self.storage { Ok(tensor) } else { Ok(tensor.to_dtype(self.storage)?) }
    }
}
