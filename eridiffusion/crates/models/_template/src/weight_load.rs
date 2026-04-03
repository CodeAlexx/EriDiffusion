use anyhow::{Result, Context};
use eridiffusion_common_weights as cw;

/// Open weights, run prefix guard, and return loader + keys.
pub fn open_and_guard(path: &str) -> Result<(cw::SafeLoader, Vec<String>)> {
    let ld = cw::SafeLoader::open(path)?;
    let keys = ld.list_keys()?;
    cw::assert_not_text_encoder(&keys)?;
    Ok((ld, keys))
}

