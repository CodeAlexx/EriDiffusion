//! Flux training loaders (strict packs + streaming façade).
//! Wraps the models::flux strict pack loaders so training crate has a stable API.

use std::{path::Path, sync::Arc};

use eridiffusion_core::Device;
use eridiffusion_models::flux::{keys::KeyConv, load_flux_packs, load_flux_packs_with, FluxPacks};
use flame_core::Result as CoreResult;

/// Strict loader: load full pack into GPU (BF16) using default schema.
pub fn load_flux_strict(
    weights_path: impl AsRef<Path>,
    device: &Device,
) -> CoreResult<Arc<FluxPacks>> {
    Ok(Arc::new(load_flux_packs(weights_path, device)?))
}

/// Strict loader with explicit key schema/num_blocks.
pub fn load_flux_strict_with(
    weights_path: impl AsRef<Path>,
    device: &Device,
    kc: KeyConv,
    num_blocks: Option<usize>,
) -> CoreResult<Arc<FluxPacks>> {
    Ok(Arc::new(load_flux_packs_with(weights_path, device, kc, num_blocks)?))
}
