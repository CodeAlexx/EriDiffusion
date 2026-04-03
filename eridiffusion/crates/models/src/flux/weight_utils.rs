use anyhow::{Context, Result};
use eridiffusion_common_io::STFile;

/// Infer the per-token hidden size from the fused qkv weights.
pub fn infer_hidden_dim(st: &STFile) -> Result<usize> {
    for key in st.keys() {
        if key.contains("img_attn.qkv.weight") {
            let view = st.tensor(&key).context("tensor missing despite key listing")?;
            let shape = view.shape();
            if shape.len() == 2 && shape[0] % 3 == 0 {
                return Ok((shape[0] / 3) as usize);
            }
        }
    }
    anyhow::bail!("unable to infer hidden size: no qkv weight found")
}

/// Count transformer blocks by scanning canonical key prefixes.
pub fn infer_block_count(st: &STFile) -> usize {
    let mut max_idx = 0usize;
    for key in st.keys() {
        if let Some(rest) = key.strip_prefix("double_blocks.")
            .or_else(|| key.strip_prefix("single_blocks."))
            .or_else(|| key.strip_prefix("blocks."))
        {
            if let Some((idx, _)) = rest.split_once('.') {
                if let Ok(parsed) = idx.parse::<usize>() {
                    max_idx = max_idx.max(parsed + 1);
                }
            }
        }
    }
    max_idx
}

/// Gather a small sample of key/shape pairs for debugging or telemetry.
pub fn snapshot_shapes(st: &STFile, limit: usize) -> Vec<(String, Vec<usize>)> {
    st.keys()
        .into_iter()
        .take(limit)
        .filter_map(|k| st.tensor(&k).map(|tv| (k, tv.shape().iter().map(|d| *d as usize).collect::<Vec<_>>())))
        .collect()
}
