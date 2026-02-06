use crate::sd35::keymap::Sd35KeyMap;
use crate::sd35::{registry::LayerRegistry, weights::Sd35WeightProvider};
use crate::streaming::{KeyMap, WeightProvider};
use eridiffusion_common_weights::strict_loader::StrictMmapLoader;
use eridiffusion_core::Result;
use flame_core::Device as FlameDevice;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct Config {
    pub shard_path: String,
    pub device_ordinal: usize,
    pub probe_blocks: usize,
}

/// Minimal SD3.5 trainer entry to validate streaming registry and weights.
pub fn train_loop(cfg: &Config) -> Result<()> {
    println!("[sd3.5] opening shard: {}", cfg.shard_path);
    let ld = Arc::new(StrictMmapLoader::open(std::path::Path::new(&cfg.shard_path))?);
    let dev = FlameDevice::cuda(cfg.device_ordinal)?;
    let wp: Sd35WeightProvider = Sd35WeightProvider::new(ld.clone(), dev.clone());

    let reg = LayerRegistry::new();
    let total = <Sd35KeyMap as KeyMap>::block_count();
    let limit = if cfg.probe_blocks == 0 { total } else { cfg.probe_blocks.min(total) };
    println!("[sd3.5] blocks: {} (probing {})", total, limit);

    for i in reg.forward_ids().take(limit) {
        let w = wp.load_block_to_gpu(i)?;
        println!("[sd3.5] block {}: {} tensors", i, w.tensors.len());
        for (ti, t) in w.tensors.iter().enumerate() {
            println!("  - t{} shape={:?} dtype={:?}", ti, t.shape().dims().to_vec(), t.dtype());
        }
        wp.release_block(i as isize)?;
    }
    println!("[sd3.5] streaming validation complete");
    Ok(())
}
