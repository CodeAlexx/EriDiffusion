use anyhow::Result;
use flame_core::Tensor;
use crate::spec::{VaeSpec, VaePolicy};

pub fn encode(spec: &VaeSpec, images: &Tensor, policy: VaePolicy) -> Result<Tensor> {
    // For now, same path as SDXL; real FLUX VAE can diverge later
    crate::sdxl::encode(spec, images, policy)
}

pub fn decode(spec: &VaeSpec, latents: &Tensor, policy: VaePolicy) -> Result<Tensor> {
    crate::sdxl::decode(spec, latents, policy)
}
