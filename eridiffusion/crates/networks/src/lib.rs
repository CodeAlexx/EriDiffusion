//! Network adapters for AI-Toolkit

pub mod lora;
pub mod dora;
pub mod locon;
pub mod lokr;
pub mod glora;
pub mod controlnet;
pub mod ip_adapter;
pub mod t2i_adapter;
pub mod utils;

// Re-exports
pub use lora::{LoRAAdapter, LoRAConfig, LoRALayer};
pub use dora::{DoRAAdapter, DoRAConfig};
pub use locon::{LoConAdapter, LoConConfig};
pub use lokr::{LoKrAdapter, LoKrConfig, LoKrLayer};
pub use glora::{GLoRAAdapter, GLoRAConfig};
pub use controlnet::{ControlNetAdapter, ControlNetConfig};
pub use ip_adapter::{IPAdapter, IPAdapterConfig};
pub use t2i_adapter::{T2IAdapter, T2IAdapterConfig};

use eridiffusion_core::Result;

/// Initialize the networks module
pub fn initialize() -> Result<()> {
    // Register network types
    Ok(())
}