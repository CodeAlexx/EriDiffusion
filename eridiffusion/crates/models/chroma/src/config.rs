use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChromaConfig {
    pub device: Option<String>,          // e.g., cuda:0
    pub dtype: Option<String>,           // bf16
    pub freeze_base: bool,               // true -> LoRA only
    pub gradient_checkpointing: Option<String>, // balanced
    pub grad_accum: Option<usize>,
    pub lr_sched: Option<String>,        // cosine
    pub optimizer: Option<String>,       // adamw
    pub weights: Option<String>,         // .safetensors
    pub vae: Option<String>,             // optional VAE path
}

