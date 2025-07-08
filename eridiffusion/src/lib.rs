//! EriDiffusion - Pure Rust diffusion model trainer and inference
//!
//! A comprehensive library for training and running modern diffusion models
//! including Stable Diffusion, SDXL, SD3, SD3.5, and Flux.

pub mod models;
pub mod ops;
pub mod trainers;
pub mod memory;
pub mod networks;
pub mod loaders;
pub mod inference;

#[cfg(feature = "cuda")]
pub mod kernels;

pub use eridiffusion_core as core;

// Re-export commonly used types
pub use crate::models::flux_lora::{FluxModelWithLoRA, FluxConfig};
pub use crate::trainers::train_from_config;
pub use crate::trainers::text_encoders;

// Re-export memory management
pub use crate::memory::{MemoryPool, MemoryPoolConfig, DiffusionConfig};