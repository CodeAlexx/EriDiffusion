//! Temporary stub for SDXL training entry points.
//! Full LoRA/fine-tune loop not yet ported; attempting to use this path will return an error.

use anyhow::{bail, Result};

#[derive(Clone, Debug)]
pub struct Config {
    pub shard_path: String,
    pub device_ordinal: usize,
    pub probe_blocks: usize,
}

pub fn train_loop(_cfg: &Config) -> Result<()> {
    bail!("SDXL training loop is not yet implemented in Phase-4")
}
