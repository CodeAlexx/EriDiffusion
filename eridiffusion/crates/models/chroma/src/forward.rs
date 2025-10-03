use anyhow::Result;
use flame_core::{Tensor, Shape};

/// Contract for Chroma forward pass.
/// Inputs:
/// - latents: NHWC [B, H/8, W/8, 4] bf16
/// - t: [B] i32 timesteps
/// - ctx: [B, seq, dim] bf16 text context
/// Output:
/// - epsilon: same shape as latents
pub trait ChromaModule {
    fn forward(&self, latents: &Tensor, t: &Tensor, ctx: &Tensor) -> Result<Tensor>;
}

/// Validate basic NHWC latent contract and produce a zero output with same shape.
/// This is a placeholder implementation until blocks are wired.
pub fn forward_contract_passthrough(latents: &Tensor, _t: &Tensor, _ctx: &Tensor) -> Result<Tensor> {
    // Assert NHWC with channels=4
    let shape = latents.shape();
    let dims = shape.dims();
    anyhow::ensure!(dims.len() == 4, "latents must be NHWC rank-4, got {:?}", dims);
    anyhow::ensure!(dims[3] == 4, "latents NHWC expects C=4, got C={}", dims[3]);
    // Produce zeros with same shape on same device
    let out = Tensor::zeros(shape.clone(), latents.device().clone())?;
    Ok(out)
}
