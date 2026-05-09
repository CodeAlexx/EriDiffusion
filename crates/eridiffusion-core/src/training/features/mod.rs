//! EriDiffusion-v2 modular training feature scaffolding.
//!
//! Status: SKELETON — Phase 0 of the multi-feature rollout
//! (see `/home/alex/.claude/plans/crispy-sauteeing-sparkle.md`).
//!
//! Each submodule is a default-off, behavior-neutral stub. Real implementations
//! land in Phase 1+. Trainers MUST NOT call into these yet — Phase 0 is
//! schema + plumbing only.
//!
//! Per-module phase targets:
//!   - caption_dropout      : Phase 1
//!   - loss_weight          : Phase 1
//!   - noise_modifiers      : Phase 1
//!   - lr_schedule          : Phase 1 (Constant/CosineWithWarmup) + Phase 5 (rest)
//!   - validation           : Phase 2
//!   - multi_backend        : Phase 2
//!   - masked_loss          : Phase 3
//!   - ema_advanced         : Phase 3
//!   - tread                : Phase 4

pub mod caption_aug;
pub mod caption_dropout;
pub mod disk_check;
pub mod ema_advanced;
pub mod health;
pub mod image_aug;
pub mod loss_weight;
pub mod lr_schedule;
pub mod masked_loss;
pub mod multi_backend;
pub mod noise_modifiers;
pub mod sample_library;
pub mod slider;
pub mod timestep_bias;
pub mod tread;
pub mod validation;
pub mod webhook;
