//! Training-side forward pass for Wan 2.2 (TI2V-5B + T2V/I2V-A14B).
//!
//! Direct port of `flame-diffusion-archive/wan-trainer/src/forward_impl/`,
//! adapted to EDv2 conventions:
//!
//! - The archive depended on `inference_flame::models::wan22_dit::Wan22Dit`
//!   for helper methods (`compute_embeddings`, `patchify_public`,
//!   `linear_bias_pub`, `shared_weight`). EDv2's `Wan22Model` holds the
//!   weights flat in `pub weights: HashMap<String, Tensor>`, so those
//!   helpers are re-implemented here as free functions reading directly
//!   from a passed-in weight map.
//! - The optional `FLAME_ACTIVATION_OFFLOAD` checkpointing path from the
//!   archive is skipped — it requires `Wan22LoraBundle: Clone`, which we
//!   don't derive yet. Standard non-checkpointed forward only.
//! - Profile sections (`profile::section`) are no-ops in this port —
//!   delete entirely; perf instrumentation can be re-added if needed.
//!
//! ## Modules
//!
//! - [`rope`]: 3-axis RoPE table + autograd-clean apply (uses
//!   `flame_core::bf16_ops::rope_fused_bf16`)
//! - [`head`]: final layer-norm-modulate-linear + unpatchify
//! - [`block`]: one Wan transformer block (self_attn + cross_attn + ffn)
//!   with LoRA injection
//! - [`forward`]: top-level forward — patchify → embeddings → blocks → head

pub mod block;
pub mod forward;
pub mod head;
pub mod rope;

pub use forward::forward_with_lora;
pub use rope::WanRope;
