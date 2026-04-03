use anyhow::{bail, Result};
use flame_core::DType;

/// Map a safetensors dtype to our internal DType
pub fn map_st_dtype(dt: &safetensors::tensor::Dtype) -> Result<DType> {
    use safetensors::tensor::Dtype as ST;
    Ok(match dt {
        ST::F32 => DType::F32,
        ST::F16 => DType::F16,
        ST::BF16 => DType::BF16,
        ST::I32 => DType::I32,
        ST::BOOL => DType::Bool,
        // Extend here as needed; explicitly reject unsupported for now
        other => bail!("unsupported dtype from safetensors: {:?}", other),
    })
}

