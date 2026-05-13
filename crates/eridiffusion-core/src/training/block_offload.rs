//! Re-export shim: BlockOffloader moved to flame_core::offload.
//!
//! Per tenets §1 (flame-core is one framework; per-call inefficiency in any
//! primitive multiplies across every model that uses it): the block-weight
//! offloader is a framework memory/IO primitive used by 13+ model families
//! across DiT, MMDiT, MoE, video DiT, and multimodal architectures. It
//! belongs in flame-core, not in a trainer-side crate.
//!
//! This file is the migration shim. Existing callers via
//! `eridiffusion_core::training::block_offload::*` keep working unchanged.
//! New code should import directly from `flame_core::offload` (or via
//! `flame_core::*` if the symbol is re-exported at the crate root).
//!
//! Future work — once all callers have migrated to the flame-core path,
//! this shim can be deleted.
//!
//! The re-export is unconditional: flame-core's default features include
//! `cuda` + `bf16_u16`, so `flame_core::offload` is always present in this
//! workspace.

pub use flame_core::offload::*;
