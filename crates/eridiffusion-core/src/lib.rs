pub mod adapter;
pub mod config;
pub mod data;
pub mod debug;
pub mod lora;
pub mod lycoris;
pub mod models;
pub mod pipeline;
pub mod training;
pub mod utils;
pub mod encoders;
pub mod sampler;

use thiserror::Error;

pub use flame_core;

#[derive(Error, Debug)]
pub enum EriDiffusionError {
    #[error("Config: {0}")]
    Config(String),
    #[error("Data: {0}")]
    Data(String),
    #[error("Training: {0}")]
    Training(String),
    #[error("Model: {0}")]
    Model(String),
    #[error("LoRA: {0}")]
    Lora(String),
    #[error("Pipeline: {0}")]
    Pipeline(String),
    #[error("IO: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("Flame: {0}")]
    Flame(flame_core::FlameError),
    #[error("Safetensors: {0}")]
    Safetensors(String),
    #[error("{0}")]
    Other(String),
}

impl From<flame_core::FlameError> for EriDiffusionError {
    fn from(e: flame_core::FlameError) -> Self { EriDiffusionError::Flame(e) }
}

/// Reverse direction — needed for ported model code that returns
/// `flame_core::Result` but calls into EDv2-side helpers (e.g. LoRALinear
/// methods that return `crate::Result`). `?` then converts back to a
/// flame-core error for the caller.
impl From<EriDiffusionError> for flame_core::FlameError {
    fn from(e: EriDiffusionError) -> Self {
        match e {
            EriDiffusionError::Flame(inner) => inner,
            other => flame_core::FlameError::InvalidOperation(other.to_string()),
        }
    }
}

pub type Result<T> = std::result::Result<T, EriDiffusionError>;
