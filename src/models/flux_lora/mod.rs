//! Flux model with built-in LoRA support

pub mod double_block;
pub mod model;
pub mod modulation;
pub mod norm_wrapper;
pub mod attention_rope;
pub mod attention_flux;
pub mod tensor_mapping;
pub mod lora_config;
pub mod lora_layers;
pub mod save_lora;
pub mod weight_translator;

// Re-export constants and types
pub use double_block::{FLUX_NORM_GROUPS, FLUX_NORM_EPS, FluxDoubleBlockWithLoRA, FluxSingleBlockWithLoRA, DoubleBlockConfig, SingleBlockConfig};

// Re-export main types
pub use model::{FluxModelWithLoRA, FluxConfig};
pub use modulation::{Modulation, Modulation1, Modulation2, ModulationOut, ModulationParams, apply_modulation};
pub use save_lora::save_flux_lora;
pub use weight_translator::FluxWeightTranslator;
pub use lora_config::LoRALayerConfig;
pub use lora_layers::{AttentionWithLoRA, LinearWithLoRA};