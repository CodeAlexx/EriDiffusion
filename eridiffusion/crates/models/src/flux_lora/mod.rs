//! Flux model with built-in LoRA support

pub mod double_block;
pub mod model;

pub use double_block::{FluxDoubleBlockWithLoRA, FluxSingleBlockWithLoRA, DoubleBlockConfig, SingleBlockConfig};
pub use model::{FluxModelWithLoRA, FluxConfig};