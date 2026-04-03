pub mod data_loader;
pub mod dtype_contracts;
pub mod host_tensor;
pub mod infer;
pub mod inference_runtime;
pub mod keymap;
pub mod label_emb;
pub mod registry;
pub mod runtime;
pub mod scheduler;
pub(crate) mod trace;
pub mod weights;

#[cfg(feature = "sdxl")]
pub use infer::SamplerMode;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RuntimeMode {
    Resident,
    Streamed,
}
