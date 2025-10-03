use anyhow::Result;
use flame_core::{Tensor, Shape};

/// Validate NHWC latent contract and return zeros with same shape (placeholder).
pub fn forward_contract_passthrough(latents: &Tensor) -> Result<Tensor> {
    let sh = latents.shape();
    let d = sh.dims();
    anyhow::ensure!(d.len() == 4, "latents rank-4 NHWC required, got {:?}", d);
    anyhow::ensure!(d[3] == 4, "expected C=4 for latents, got {}", d[3]);
    Tensor::zeros(sh.clone(), latents.device().clone())
}

