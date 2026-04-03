use eridiffusion_core::{Device as EriDevice, Result};
use flame_core::Device as FlameDevice;

use crate::flux::{registry::build_layer_registry, weights::FluxWeightProvider};

#[derive(Debug, Clone)]
pub struct Config {
    pub shard_path: String,
    pub device_ordinal: usize,
    /// How many blocks to probe (0 = all)
    pub probe_blocks: usize,
}

/// Minimal Flux trainer entry to validate streaming registry and weights.
pub fn train_loop(cfg: &Config) -> Result<()> {
    println!("[flux] opening shard: {}", cfg.shard_path);
    let dev = FlameDevice::cuda(cfg.device_ordinal as usize)?;
    let provider = FluxWeightProvider::from_path(
        &cfg.shard_path,
        EriDevice::from_flame_cuda(dev.cuda_device().as_ref()),
    )?;
    let registry = build_layer_registry(&provider)?;

    println!("[flux] blocks discovered: {}", registry.blocks.len());
    let limit = if cfg.probe_blocks == 0 {
        registry.blocks.len()
    } else {
        cfg.probe_blocks.min(registry.blocks.len())
    };
    println!("[flux] probing {} blocks", limit);

    for idx in 0..limit {
        let keys = provider.block_keys(idx);
        println!("  block {} → {} tensors", idx, keys.len());
        for key in keys {
            println!("    - {}", key);
        }
    }

    println!("[flux] streaming validation complete");
    Ok(())
}
