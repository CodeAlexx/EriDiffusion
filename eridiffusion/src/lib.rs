//! EriDiffusion - Pure Rust diffusion model trainer and inference
//!
//! A comprehensive library for training and running modern diffusion models
//! including Stable Diffusion, SDXL, SD3, SD3.5, and Flux.

pub use flame_core::adam::{Adam, AdamW};
use flame_core::{DType, Result, Shape, Tensor};
pub use trainers::adam8bit::{Adam8bit, Adam8bitState};

// Configuration
pub mod config;

// Device abstraction
pub mod device;
pub use device::{cuda_device, Device};

// cuDNN backend integration
pub mod data;
pub mod flame_training;
pub mod flame_vision;
pub mod inference;
pub mod loaders;
pub mod logging;
pub mod memory;
pub mod models;
pub mod networks;
pub mod ops;
pub mod samplers;
pub mod schedulers;
pub mod trainers;
#[cfg(feature = "cudnn")]
// pub mod cudnn_backend; // Removed - cuDNN now integrated directly in FLAME

// FLAME-only modules - FLAME has been completely removed
pub mod weight_loader;

// Compatibility layer for FLAME -> FLAME migration

// Compatibility layer for stable_diffusion imports
pub mod stable_diffusion_compat;

// Compatibility module for disabled eridiffusion_core crate
pub mod eridiffusion_core;

// Tokenizer utilities
pub mod tokenizers;

// Tensor backend compatibility
pub mod tensor_backend;

// Optimizers
pub mod optimizers;

#[cfg(feature = "cuda")]
pub mod kernels;

// Training verification
pub mod training_verifier;

// Re-export commonly used types
// pub use crate::models::flux_lora::{FluxModelWithLoRA, FluxConfig};
pub use crate::trainers::text_encoders;
pub use crate::trainers::train_from_config;

// Re-export memory management
pub use crate::memory::{DiffusionConfig, MemoryPool, MemoryPoolConfig};
