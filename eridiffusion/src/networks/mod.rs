use flame_core::device::Device;
use flame_core::GradientMap;
use flame_core::{DType, Shape, Tensor};
// backward is a method on Tensor, not a standalone function
use flame_core::optimizers::{Adam, SGD};

// Network adapters for parameter-efficient fine-tuning

pub mod lora;
pub mod sd_lora;

pub use lora::{LoRABuilder, LoRACollection, LoRAConfig, LoRAModule};

// Compatibility alias
pub type LoRAModel = LoRAModule;
pub use sd_lora::{
    SDAdapterCollection, SDAdapterConfig, SDAdapterType, SDLoKrModule, SDLoRAModule,
};
