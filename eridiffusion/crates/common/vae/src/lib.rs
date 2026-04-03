use anyhow::Result;
use flame_core::Tensor;

pub mod spec;
pub mod blocks;
pub mod sdxl;
pub mod flux;
pub mod sd35;
pub mod scale;

pub use spec::{VaeSpec, VaeKind, VaePolicy};
pub use scale::{read_vae_scaling, apply_encode_scale, apply_decode_scale};

/// images NHWC bf16 [B,H,W,3] -> latents NHWC bf16 [B,H/ld,W/ld,C]
pub fn encode(spec: &VaeSpec, images: &Tensor, policy: VaePolicy) -> Result<Tensor> {
    match spec.kind {
        VaeKind::Sdxl => sdxl::encode(spec, images, policy),
        VaeKind::Flux => flux::encode(spec, images, policy),
        VaeKind::Sd35 => sd35::encode(spec, images, policy),
    }
}

/// latents NHWC bf16 [B,H/ld,W/ld,C] -> images NHWC bf16 [B,H,W,3]
pub fn decode(spec: &VaeSpec, latents: &Tensor, policy: VaePolicy) -> Result<Tensor> {
    match spec.kind {
        VaeKind::Sdxl => sdxl::decode(spec, latents, policy),
        VaeKind::Flux => flux::decode(spec, latents, policy),
        VaeKind::Sd35 => sd35::decode(spec, latents, policy),
    }
}
