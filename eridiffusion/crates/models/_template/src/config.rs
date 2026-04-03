use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelConfig {
    pub name: String,
    pub backbone: String, // "dit" | "unet" | "auto"
    pub dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub ctx_dim: usize,
}

