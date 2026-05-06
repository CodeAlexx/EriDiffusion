//! OneTrainer-parity training features.
//!
//! Shared building blocks reused across trainers (zimage, klein, sd3, sdxl,
//! ernie). Each submodule is self-contained:
//!
//! - [`sampler`] — generic in-training sampling hook (every-N-steps callback)
//! - [`optimizers`] — Adafactor, AdamW8bit (partial), Prodigy, Lion + dispatch enum
//! - [`timestep_dist`] — OneTrainer's 6 distributions with `noising_weight/bias`
//! - [`ema`] — alias re-export of [`crate::ema::ParameterEma`]
//!
//! All of these are wired into trainers via their config files. AdamW8bit is
//! a host-RAM-frugal port (state quantized between steps, F32 math on-device,
//! per-step PCIe round-trips). It is **slower** than [`flame_core::adam::AdamW`]
//! today and only worth using under host-RAM pressure. See its docstring.

pub mod ema;
pub mod optimizers;
pub mod sampler;
pub mod timestep_dist;

pub use ema::EmaShadow;
pub use optimizers::{Adafactor, AdamW8bit, Lion, Optimizer, OptimizerKind, Prodigy};
#[allow(deprecated)]
pub use sampler::{InTrainingSampler, SamplePromptSpec};
pub use timestep_dist::{TimestepConfig, TimestepDistribution};
