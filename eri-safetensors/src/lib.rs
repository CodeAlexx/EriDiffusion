//! Pure-Rust safetensors mmap loader.
//!
//! Vendored from `serenity-safetensors/src/` with all pyo3 bindings stripped
//! so flame-core (and the rest of the EriDiffusion stack) can depend on it
//! without pulling in the Python ABI. The serenity crate keeps its Python
//! bindings; this crate is the pure-Rust subset.
//!
//! Today: just the mmap path (`MmapFile`, `MmapRegion`, `TensorRef`).
//! Follow-on work to vendor the full read/save surface (GGUF, pickle .bin,
//! O_DIRECT writes, format auto-detect, diffusers layouts, probe).

pub mod mmap;

pub use mmap::{MmapError, MmapFile, MmapRegion, TensorRef};
