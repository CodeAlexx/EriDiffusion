use flame_core::device::Device;
use flame_core::optimizers::{Adam, SGD};
use flame_core::{DType, Result, Shape, Tensor};

// Compatibility module for autoencoder_kl imports
// Re-exports types from existing VAE implementations

pub use crate::models::flux_vae::{AutoencoderKL, AutoencoderKLConfig};
