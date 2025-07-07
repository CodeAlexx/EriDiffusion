//! GPU-accelerated operations for Flux
//! 
//! This module provides CUDA implementations of critical operations
//! that would otherwise run on CPU and bottleneck training.

pub mod group_norm;
pub mod rope;
pub mod attention;

pub use group_norm::{GroupNorm, group_norm};
pub use rope::{RotaryEmbedding, get_1d_positions, get_2d_positions, apply_rotary_emb};
pub use attention::attention;