//! Video utilities — cross-model helpers used by Wan 2.2, LTX-2, Anima,
//! and any future video DiT trainer.
//!
//! ## Modules
//! - [`decode`]  — read mp4 / webm / mov / image files into BF16 latents
//!   `[B=1, 3, F, H, W]` in `[-1, 1]`. Uses `ffmpeg` out-of-process.
//! - [`shape`]   — `[B, C, F, H, W]` reshape utilities (squeeze/unsqueeze
//!   batch dim, frame strider, latent ↔ pixel size converters).
//! - [`prep`]    — file → tensor pipelines combining decode + resize +
//!   normalization. The single entry point for `prepare_*` binaries to
//!   share image-and-video reading without each binary re-implementing
//!   `image::open` + ffmpeg shell-out.
//!
//! ## Why a shared module
//! Audit and Phase 6 user direction (2026-05-09) — Wan 2.2's prepare path
//! handles only images today; LTX-2's prepare predates this and uses
//! its own host-loop frame extraction; Anima image-only. Centralizing
//! decode + shape + prep lets future trainers consume one well-tested
//! API and lets bug fixes (precision, codec quirks) land once.
//!
//! ## What lives here vs in `encoders/`
//! - `encoders/wan22_vae.rs`, `encoders/wan21_encoder.rs`, `encoders/umt5.rs`
//!   are MODEL-specific. They stay in their existing modules.
//! - `video/` holds the I/O and shape glue. It must NOT depend on any
//!   specific model's encoder (only on flame_core types).

pub mod decode;
pub mod prep;
pub mod shape;
