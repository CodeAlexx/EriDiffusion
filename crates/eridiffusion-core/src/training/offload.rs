//! Activation offload pool setup helper.
//!
//! Provides a single `setup_activation_offload()` function that trainers
//! call after model load to construct and install the global pool.
//! Model-agnostic — sizing is computed from block count and activation size.

use flame_core::activation_offload::{ActivationOffloadPool, OffloadCompression};
use flame_core::autograd::set_activation_offload_pool;
use flame_core::Result;
use std::sync::{Arc, Mutex};

/// Parameters for activation offload pool construction.
pub struct OffloadConfig {
    /// Number of transformer blocks that will be checkpointed.
    pub num_blocks: usize,
    /// Maximum activation size in bytes (per-block input tensor, BF16).
    /// For a single block: `batch * seq_len * inner_dim * 2`.
    pub max_activation_bytes: usize,
    /// Whether to use FP8 compression (halves pinned memory + PCIe).
    pub compression: OffloadCompression,
    /// Extra slot headroom beyond num_blocks (default: 8).
    pub extra_slots: usize,
}

impl OffloadConfig {
    /// Compute pool sizing from model parameters.
    ///
    /// `seq_len` = MAXIMUM tokens per block across the entire dataset
    /// (use the largest bucket's token count, not a single sample).
    /// `inner_dim` = transformer hidden dimension.
    pub fn from_model(
        num_blocks: usize,
        seq_len: usize,
        inner_dim: usize,
        compression: OffloadCompression,
    ) -> Self {
        // BF16: 2 bytes per element. Per-block activation = batch(1) × seq × dim × 2.
        let max_activation_bytes = seq_len * inner_dim * 2;
        Self {
            num_blocks,
            max_activation_bytes,
            compression,
            extra_slots: 8,
        }
    }
}

/// Construct and install the global activation offload pool.
///
/// Call once after model load, before the training loop. Returns the pool
/// stats (num_slots, pinned_bytes) for logging.
pub fn setup_activation_offload(
    device: &Arc<cudarc::driver::CudaDevice>,
    config: &OffloadConfig,
) -> Result<(usize, usize)> {
    let num_slots = config.num_blocks + config.extra_slots;
    let pool = ActivationOffloadPool::new(
        device,
        num_slots,
        config.max_activation_bytes,
        config.compression,
    )?;
    let pinned_bytes = pool.host_bytes();
    let pool_arc = Arc::new(Mutex::new(pool));
    set_activation_offload_pool(pool_arc)?;
    Ok((num_slots, pinned_bytes))
}
