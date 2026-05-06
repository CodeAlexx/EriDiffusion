//! Flux 1 VAE — 16-channel SDXL-style LDM autoencoder used by FLUX.1-dev/schnell.
//!
//! Re-exports the LDM encoder/decoder modules with Flux-specific defaults:
//!   - `LATENT_CHANNELS = 16`
//!   - `SHIFT = 0.1159`, `SCALE = 0.3611` (Flux 1 latent normalization)
//!
//! Reference: `inference-flame/src/vae/ldm_{encoder,decoder}.rs` (verified
//! pure-Rust forward), `/home/alex/upstream Python/modules/model/FluxModel.py:VAE`.

pub use crate::encoders::flux_vae_encoder::LdmVAEEncoder as FluxVaeEncoder;
pub use crate::encoders::flux_vae_decoder::LdmVAEDecoder as FluxVaeDecoder;

/// Flux 1 latent channels.
pub const LATENT_CHANNELS: usize = 16;
/// Flux 1 latent shift (subtract before scaling for `encode_scaled`).
pub const SHIFT: f32 = 0.1159;
/// Flux 1 latent scale (divide encoded latents by this; multiply back to decode).
pub const SCALE: f32 = 0.3611;
