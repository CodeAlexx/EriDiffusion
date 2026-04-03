use anyhow::Result;
use tracing::info;
use eridiffusion_common_blockswap::{BlockSwapCfg, BlockSwapManager};
use eridiffusion_common_weights::{ParamRegistry, SafeLoader};

/// Derive layer key prefixes in compute order by filtering and sorting.
pub fn plan_layer_keys(keys: &[String], prefix: &str) -> Vec<String> {
    let mut layers: Vec<String> = keys
        .iter()
        .filter(|k| k.starts_with(prefix))
        .map(|k| {
            // Trim to layer group prefix: up to last '.' before parameter name
            if let Some(pos) = k.rfind('.') { k[..pos].to_string() } else { k.clone() }
        })
        .collect();
    layers.sort();
    layers.dedup();
    layers
}

pub fn forward_with_swap<F>(
    mut mgr: BlockSwapManager,
    ld: &mut SafeLoader,
    reg: &mut ParamRegistry,
    layer_keys: Vec<String>,
    mut run_layer: F,
) -> Result<()>
where
    F: FnMut(&str) -> Result<()>,
{
    for k in layer_keys {
        let _ = mgr.prefetch_next(ld, reg)?;
        run_layer(&k)?;
        let _ = mgr.release_previous(reg)?;
    }
    Ok(())
}
