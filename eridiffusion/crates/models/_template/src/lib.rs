use anyhow::Result;
use flame_core::Tensor;

pub mod config;
pub mod weight_load;
pub mod blocks { pub mod attn; pub mod mlp; }
pub mod lora;
pub mod forward;

/// Core interface implemented by diffusion backbones.
pub trait DiffusionModule {
    /// latents: NHWC [B,H/8,W/8,4], t: [B] i32, optional ctx: [B,seq,ctx_dim]
    fn forward(&self, latents: &Tensor, t: &Tensor, ctx: Option<&Tensor>) -> Result<Tensor>;
}

/// Minimal placeholder model that only validates the contract.
pub struct PlaceholderModel { pub ctx_dim: usize }

impl PlaceholderModel { pub fn new(ctx_dim: usize) -> Self { Self { ctx_dim } } }

impl DiffusionModule for PlaceholderModel {
    fn forward(&self, latents: &Tensor, _t: &Tensor, _ctx: Option<&Tensor>) -> Result<Tensor> {
        forward::forward_contract_passthrough(latents)
    }
}

