#![deny(rust_2018_idioms)]

pub mod loader;
pub mod layout;
pub mod prefix_guard;
pub mod registry;
pub mod metrics;
pub mod writer;
pub mod file;
pub mod dtype;
pub mod read;
pub mod strict_loader;

pub use loader::SafeLoader;
pub use layout::{normalize_linear, normalize_conv2d};
pub use prefix_guard::assert_not_text_encoder;
pub use registry::{ParamRegistry, ParamId, vram_used_gb, assert_vram_below};
pub use metrics::log_step_metrics;
pub use writer::write_safetensors;
pub use file::SafeTensorFile;
pub use strict_loader::{StrictMmapLoader, TensorInfo, tensor_from_bytes};
