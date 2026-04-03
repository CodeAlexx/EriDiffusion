#[cfg(feature = "flux")]
pub mod flux;
#[cfg(feature = "flux")]
pub mod flux_names;
#[cfg(feature = "sdxl")]
pub mod sdxl;
#[cfg(feature = "sd35")]
pub mod sd35;
#[cfg(feature = "chroma")]
pub mod chroma;
pub mod common;
pub mod util;
pub mod devtensor;

// Re-export common helper crates so downstream modules can call them via eridiffusion_models::common_*
pub use eridiffusion_common_text    as common_text;
pub use eridiffusion_common_io      as common_io;
pub use eridiffusion_common_lora    as common_lora;
pub use eridiffusion_common_telemetry as common_telemetry;
pub use eridiffusion_common_vae     as common_vae;
pub use eridiffusion_common_weights as common_weights;

// Minimal model-agnostic traits to satisfy data/training crates without pulling heavy deps.
pub mod traits {
    use anyhow::Result;
    use flame_core::Tensor;

    pub trait VAE: Send + Sync {
        fn encode(&self, images: &Tensor) -> Result<Tensor>;
    }

    pub trait TextEncoder: Send + Sync {
        fn encode(&self, prompts: &[String]) -> Result<(Tensor, Option<Tensor>)>;
    }
}

pub use traits::{VAE, TextEncoder};

// Re-exports per feature for convenience (optional)
#[cfg(feature = "flux")]
pub use flux::*;
#[cfg(feature = "flux")]
pub use flux_names::*;
#[cfg(feature = "sdxl")]
pub use sdxl::*;
#[cfg(feature = "sd35")]
pub use sd35::*;
#[cfg(feature = "chroma")]
pub use chroma::*;
