use anyhow::{Result};
use crate::spec::{VaeSpec, VaePolicy};

pub fn encode(spec: &VaeSpec, images: &flame_core::Tensor, policy: VaePolicy) -> Result<flame_core::Tensor> {
    // Same simple implementation as SDXL for now
    crate::sdxl::encode(spec, images, policy)
}

pub fn decode(spec: &VaeSpec, latents: &flame_core::Tensor, policy: VaePolicy) -> Result<flame_core::Tensor> {
    crate::sdxl::decode(spec, latents, policy)
}

