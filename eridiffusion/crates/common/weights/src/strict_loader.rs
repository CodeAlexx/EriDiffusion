//! Strict SafeTensors loader (mmap-style API surface) for streaming base weights.
//! Minimal, compile-safe implementation using safetensors. Validates dtype/shape
//! when materializing tensors via `tensor_from_bytes`.

use anyhow::{bail, Context, Result};
use safetensors::{tensor::TensorView, Dtype as SafeDtype, SafeTensors};
use std::{collections::BTreeSet, path::Path};

use flame_core::{Device as FlameDevice, DType, Shape, Tensor};

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub key: String,
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub off: u64,
    pub nbytes: u64,
}

/// Loader that owns the underlying bytes and exposes a strict index API.
pub struct StrictMmapLoader {
    bytes: Vec<u8>,
    used: BTreeSet<String>,
}

impl StrictMmapLoader {
    pub fn open(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path)
            .with_context(|| format!("read safetensors: {}", path.display()))?;
        // Validate it parses
        let _ = SafeTensors::deserialize(&bytes)
            .context("parse safetensors header")?;
        Ok(Self { bytes, used: BTreeSet::new() })
    }

    pub fn info(&self, key: &str) -> Result<TensorInfo> {
        let st = SafeTensors::deserialize(&self.bytes)
            .context("parse safetensors header")?;
        let tv: TensorView<'_> = st.tensor(key)
            .with_context(|| format!("tensor '{}' not found", key))?;
        let dtype = map_dtype(tv.dtype())?;
        let shape = tv.shape().to_vec();
        let nbytes = tv.data().len() as u64;
        Ok(TensorInfo { key: key.to_string(), dtype, shape, off: 0, nbytes })
    }

    pub fn bytes(&self, key: &str) -> Result<&[u8]> {
        let st = SafeTensors::deserialize(&self.bytes)
            .context("parse safetensors header")?;
        let tv: TensorView<'_> = st.tensor(key)
            .with_context(|| format!("tensor '{}' not found", key))?;
        Ok(tv.data())
    }

    pub fn mark_used(&mut self, key: &str) { let _ = self.used.insert(key.to_string()); }

    /// Ensure the set of marked keys matches exactly the file contents.
    pub fn validate_used_exactly(&self) -> Result<()> {
        let st = SafeTensors::deserialize(&self.bytes)
            .context("parse safetensors header")?;
        let mut actual: BTreeSet<String> = BTreeSet::new();
        for name in st.names() { actual.insert(name.to_string()); }
        if self.used != actual {
            // Report a short diff
            let missing: Vec<_> = actual.difference(&self.used).take(5).cloned().collect();
            let unused: Vec<_> = self.used.difference(&actual).take(5).cloned().collect();
            bail!(
                "Strict loader key mismatch. missing={:?} unused={:?}",
                missing, unused
            );
        }
        Ok(())
    }
}

fn map_dtype(d: SafeDtype) -> Result<DType> {
    Ok(match d {
        SafeDtype::F32 => DType::F32,
        SafeDtype::F16 => DType::F16,
        SafeDtype::BF16 => DType::BF16,
        SafeDtype::I32 => DType::I32,
        SafeDtype::I64 => DType::I64,
        SafeDtype::U32 => DType::U32,
        SafeDtype::U8 => DType::U8,
        SafeDtype::BOOL => DType::Bool,
        other => bail!("unsupported dtype in strict loader: {:?}", other),
    })
}

/// Convert raw bytes to a BF16/F16/F32 tensor on the given device. Checks size.
/// Base weights are returned detached (no-grad) to enforce adapters-only grads.
pub fn tensor_from_bytes(dev: FlameDevice, ti: &TensorInfo, bytes: &[u8]) -> Result<Tensor> {
    // Validate size
    let elem_size = match ti.dtype { DType::F32 => 4, DType::F16 | DType::BF16 => 2, _ => 4 };
    let n_elems: usize = ti.shape.iter().product();
    let expect = n_elems * elem_size;
    if expect != bytes.len() { bail!("tensor '{}' size mismatch: expect {}B got {}B", ti.key, expect, bytes.len()); }

    // Materialize as F32 then cast if requested (safer path in Flame)
    let f32_data: Vec<f32> = match ti.dtype {
        DType::F32 => bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect(),
        DType::BF16 => {
            use half::bf16;
            bytes.chunks_exact(2)
                .map(|b| bf16::from_le_bytes([b[0], b[1]]).to_f32())
                .collect()
        }
        DType::F16 => {
            use half::f16;
            bytes.chunks_exact(2)
                .map(|b| f16::from_le_bytes([b[0], b[1]]).to_f32())
                .collect()
        }
        DType::I32 | DType::U32 => bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect(),
        DType::I64 | DType::F64 => {
            // Downcast to f32 (best-effort)
            bytes.chunks_exact(8)
                .map(|b| f64::from_le_bytes([b[0],b[1],b[2],b[3],b[4],b[5],b[6],b[7]]) as f32)
                .collect()
        }
        DType::U8 | DType::I8 | DType::Bool => bytes.iter().map(|&u| u as f32).collect(),
    };

    let shape = Shape::from_dims(&ti.shape);
    let t = Tensor::from_vec(f32_data, shape, dev.cuda_device_arc())?;
    let t = if matches!(ti.dtype, DType::F16 | DType::BF16) { t.to_dtype(ti.dtype)? } else { t };
    Ok(t) // Detach: Flame tensors are not requires_grad by default
}
