//! Streaming interfaces (traits only) for sequential weight loading.

use anyhow::Result;
use flame_core::Tensor;

/// Base weights staged on device for a block/head.
pub struct DeviceWeights {
    pub tensors: Vec<Tensor>,
}

/// Provides block/head weights with prefetch and release hints.
pub trait WeightProvider: Send + Sync {
    fn load_block_to_gpu(&self, block_id: usize) -> Result<DeviceWeights>;
    fn load_head_to_gpu(&self) -> Result<DeviceWeights>;
    fn prefetch_block(&self, block_id: usize) -> Result<()>;
    fn release_block(&self, block_id: isize) -> Result<()>;
}

/// Static key map for strict weight validation.
pub trait KeyMap {
    fn block_count() -> usize;
    fn keys_for_block(i: usize) -> &'static [&'static str];
    fn keys_for_head() -> &'static [&'static str];
}

/// Extend KeyMap with an owned-keys API without forcing static slices.
/// Models can either return static slices via `keys_for_block` or
/// generate owned Strings dynamically.
pub trait KeyMapOwned: KeyMap {
    /// Default: if the static slice is non-empty, return it as owned Strings.
    /// Otherwise, allow impls to generate keys dynamically.
    fn owned_keys_for_block(i: usize) -> Vec<String> {
        let s = Self::keys_for_block(i);
        if !s.is_empty() {
            return s.iter().map(|k| k.to_string()).collect();
        }
        Self::gen_keys_for_block(i)
    }
    /// Dynamic generator (default empty).
    fn gen_keys_for_block(_i: usize) -> Vec<String> {
        Vec::new()
    }
}

// Note: Implementation of concrete block registries and adapter handling
// lives in model-specific code to avoid coupling.
