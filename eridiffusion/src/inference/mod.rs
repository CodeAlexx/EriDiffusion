//! Inference utilities for diffusion models

pub mod vae_inference;

// Re-export key types
pub use vae_inference::{VAEInference, VAEType, VAEInfo};