use anyhow::Result;
use flame_core::Tensor;

use crate::forward::{ChromaModule, forward_contract_passthrough};

/// Minimal Chroma model placeholder. Real blocks will be added incrementally.
pub struct ChromaModel {
    pub hidden_dim: usize,
    pub text_dim: usize,
}

impl ChromaModel {
    pub fn new(hidden_dim: usize, text_dim: usize) -> Self {
        Self { hidden_dim, text_dim }
    }
}

impl ChromaModule for ChromaModel {
    fn forward(&self, latents: &Tensor, t: &Tensor, ctx: &Tensor) -> Result<Tensor> {
        // For now, enforce shape contract and return zeros; later wire UNet-like blocks
        forward_contract_passthrough(latents, t, ctx)
    }
}
