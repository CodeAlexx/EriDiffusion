//! Temporary LoRA configuration until networks crate is fixed

use serde::{Serialize, Deserialize};

/// Configuration for LoRA layers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoRALayerConfig {
    pub rank: usize,
    pub alpha: f32,
    pub dropout: f32,
    pub use_bias: bool,
}

impl Default for LoRALayerConfig {
    fn default() -> Self {
        Self {
            rank: 32,
            alpha: 32.0,
            dropout: 0.0,
            use_bias: false,
        }
    }
}