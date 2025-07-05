//! Core functionality for AI-Toolkit
//! 
//! This crate provides the fundamental traits and types used throughout the eridiffusion ecosystem.

pub mod error;
pub mod model;
pub mod network;
pub mod plugin;
pub mod tensor;
pub mod config;
pub mod device;
pub mod dtype;
pub mod memory;
pub mod validation;
pub mod cuda;
pub mod candle_utils;
pub mod async_utils;

// Re-exports
pub use error::{Error, Result, ErrorContext};
pub use model::{DiffusionModel, ModelArchitecture, ModelInputs, ModelOutput, ModelMetadata, ModelLoadConfig, FluxVariant};
pub use network::{NetworkAdapter, NetworkType, NetworkMetadata};
pub use plugin::{Plugin, PluginRegistry, PluginContext};
pub use tensor::{TensorOps, TensorView, TensorExt};
pub use config::{Config, ConfigLoader, DatasetConfig};
pub use device::{Device, DeviceManager};
pub use dtype::DType;
pub use candle_utils::{VarExt, VarMapExt, randint, vec_to_shape};

// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const MIN_SUPPORTED_VERSION: &str = "0.1.0";

/// Initialize the eridiffusion runtime
pub fn initialize() -> Result<()> {
    // Initialize tracing
    tracing::info!("Initializing eridiffusion v{}", VERSION);
    
    // Initialize device manager
    device::initialize_devices()?;
    
    // Initialize plugin system
    plugin::initialize_plugin_system()?;
    
    Ok(())
}