use anyhow::{Context, Result};
use safetensors::tensor::{serialize, Dtype, TensorView};
use std::{collections::BTreeMap, path::Path, fs};
use flame_core::{Tensor, DType};
use half::{bf16, f16};

fn map_dtype(dt: DType) -> Option<Dtype> {
    match dt {
        DType::BF16 => Some(Dtype::BF16),
        DType::F16 => Some(Dtype::F16),
        DType::F32 => Some(Dtype::F32),
        // Integer dtypes are not currently representable by flame_core::Tensor storage
        // (tensors are backed by f32 even when dtype flag differs). Don't silently mislabel.
        DType::I32 | DType::I64 | DType::I8 | DType::U8 | DType::U32 | DType::F64 | DType::Bool => None,
    }
}

/// Write a set of tensors to a safetensors file.
/// Assumes tensors are small enough to collect on CPU; converts BF16 as needed.
pub fn write_safetensors(path: &Path, items: &[(String, Tensor)]) -> Result<()> {
    let mut map: BTreeMap<String, TensorView<'_>> = BTreeMap::new();
    // We need owned buffers to outlive TensorView; keep them in a Vec so their lifetimes persist.
    let mut owned: Vec<(Vec<u8>, Dtype, Vec<usize>)> = Vec::with_capacity(items.len());

    for (name, t) in items {
        let shape = t.shape().dims().to_vec();
        let dtype = t.dtype();
        map_dtype(dtype)
            .ok_or_else(|| anyhow::anyhow!("unsupported dtype {:?} for key '{}' when writing safetensors", dtype, name))?;

        match dtype {
            DType::BF16 => {
                // BF16 stored as f32 internally; convert f32->bf16 bytes
                let f = t.to_vec()?; // Vec<f32>
                let mut bytes = Vec::with_capacity(f.len() * 2);
                for x in f {
                    let v = bf16::from_f32(x);
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
                owned.push((bytes, Dtype::BF16, shape));
            }
            DType::F16 => {
                // F16 down-conversion
                let f = t.to_vec()?; // Vec<f32>
                let mut bytes = Vec::with_capacity(f.len() * 2);
                for x in f {
                    let v = f16::from_f32(x);
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
                owned.push((bytes, Dtype::F16, shape));
            }
            DType::F32 => {
                let f = t.to_vec()?; // Vec<f32>
                let mut bytes = Vec::with_capacity(f.len() * 4);
                for x in f { bytes.extend_from_slice(&x.to_le_bytes()); }
                owned.push((bytes, Dtype::F32, shape));
            }
            // All others are rejected above
            _ => unreachable!(),
        }
    }

    for ((name, _), (bytes, dt, shape)) in items.iter().zip(owned.iter()) {
        let tv = TensorView::new(*dt, shape.clone(), bytes)
            .with_context(|| format!("building TensorView for key '{}': {:?} {:?}", name, dt, shape))?;
        map.insert(name.clone(), tv);
    }

    let metadata: Option<std::collections::HashMap<String, String>> = None;
    let bytes = serialize(&map, &metadata)?;
    fs::write(path, bytes).with_context(|| format!("writing safetensors {:?}", path))?;
    Ok(())
}
