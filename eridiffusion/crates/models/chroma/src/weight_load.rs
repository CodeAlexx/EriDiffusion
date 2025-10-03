use anyhow::{Result, bail, Context};
use safetensors::{SafeTensors, tensor::TensorView};
use std::fs;
use std::collections::HashSet;
use flame_core::{Tensor, Shape};

/// Strict safetensors loader skeleton. Performs key accounting; actual tensor placement TBD.
pub fn load_safetensors_strict(path: &str) -> Result<()> {
    let data = fs::read(path).with_context(|| format!("reading weights file: {}", path))?;
    let st = SafeTensors::deserialize(&data)?;
    let mut seen = HashSet::new();
    for name in st.names() {
        let tv: TensorView = st.tensor(name)?;
        // Record name as used
        seen.insert(name.to_string());
        // Validate dtype BF16
        let dt = tv.dtype();
        anyhow::ensure!(matches!(dt, safetensors::Dtype::BF16 | safetensors::Dtype::F16 | safetensors::Dtype::F32),
            "unexpected dtype {:?} for {}", dt, name);
        // We defer actual GPU tensor creation/transposes until wiring real modules.
    }
    // In strict mode, error if nothing was loaded
    if seen.is_empty() {
        bail!("no tensors found in safetensors file {}", path);
    }
    Ok(())
}
