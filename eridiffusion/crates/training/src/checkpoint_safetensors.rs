use std::{borrow::Cow, collections::HashMap, path::Path};

use anyhow::{bail, Result};
use eridiffusion_core::Device;
use eridiffusion_models::devtensor::tensor_from_slice_on;
use flame_core::{DType, Shape, Tensor};
use half::bf16;
use safetensors::{serialize_to_file, tensor::SafeTensors, Dtype as SafeDtype, View};

#[derive(Clone)]
pub struct OwnedTensorView {
    dtype: SafeDtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}

impl OwnedTensorView {
    pub fn new(dtype: SafeDtype, shape: Vec<usize>, data: Vec<u8>) -> Self {
        Self { dtype, shape, data }
    }
}

impl View for OwnedTensorView {
    fn dtype(&self) -> SafeDtype {
        self.dtype
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> Cow<'_, [u8]> {
        Cow::Borrowed(&self.data)
    }

    fn data_len(&self) -> usize {
        self.data.len()
    }
}

pub fn tensor_to_bf16_bytes(tensor: &Tensor) -> Result<(Vec<usize>, Vec<u8>)> {
    let f32_tensor = tensor.to_dtype(DType::F32)?;
    let shape = f32_tensor.shape().dims().to_vec();
    let values = f32_tensor.to_vec()?;
    let mut bytes = Vec::with_capacity(values.len() * 2);
    for v in values {
        bytes.extend_from_slice(&bf16::from_f32(v).to_le_bytes());
    }
    Ok((shape, bytes))
}

pub fn tensor_to_f32_bytes(tensor: &Tensor) -> Result<(Vec<usize>, Vec<u8>)> {
    let f32_tensor = tensor.to_dtype(DType::F32)?;
    let shape = f32_tensor.shape().dims().to_vec();
    let values = f32_tensor.to_vec()?;
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    Ok((shape, bytes))
}

pub fn bf16_bytes_to_f32_vec(bytes: &[u8]) -> Result<Vec<f32>> {
    if bytes.len() % 2 != 0 {
        bail!("BF16 byte length must be even, got {}", bytes.len());
    }
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for chunk in bytes.chunks_exact(2) {
        out.push(bf16::from_le_bytes([chunk[0], chunk[1]]).to_f32());
    }
    Ok(out)
}

pub fn f32_bytes_to_vec(bytes: &[u8]) -> Result<Vec<f32>> {
    if bytes.len() % 4 != 0 {
        bail!("F32 byte length must be multiple of 4, got {}", bytes.len());
    }
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        let mut arr = [0u8; 4];
        arr.copy_from_slice(chunk);
        out.push(f32::from_le_bytes(arr));
    }
    Ok(out)
}

pub fn serialize_tensors(
    path: &Path,
    metadata: HashMap<String, String>,
    mut tensors: Vec<(String, OwnedTensorView)>,
) -> Result<()> {
    tensors.sort_by(|a, b| a.0.cmp(&b.0));
    let views: Vec<(&str, OwnedTensorView)> =
        tensors.iter().map(|(name, view)| (name.as_str(), view.clone())).collect();
    serialize_to_file(views, &Some(metadata), path)?;
    Ok(())
}

pub struct LoadedTensor {
    pub dtype: SafeDtype,
    pub shape: Vec<usize>,
    pub bytes: Vec<u8>,
}

pub fn deserialize_tensors(path: &Path) -> Result<HashMap<String, LoadedTensor>> {
    let data = std::fs::read(path)?;
    let tensors = SafeTensors::deserialize(&data)
        .map_err(|e| anyhow::anyhow!("failed to deserialize {}: {e}", path.display()))?;
    let mut out = HashMap::new();
    for name in tensors.names() {
        let view = tensors.tensor(name)?;
        out.insert(
            name.to_string(),
            LoadedTensor {
                dtype: view.dtype(),
                shape: view.shape().to_vec(),
                bytes: view.data().to_vec(),
            },
        );
    }
    Ok(out)
}

pub fn loaded_to_tensor(
    device: &Device,
    target_dtype: DType,
    shape: &[usize],
    src_dtype: SafeDtype,
    bytes: &[u8],
) -> Result<Tensor> {
    let values_f32 = match src_dtype {
        SafeDtype::BF16 => bf16_bytes_to_f32_vec(bytes)?,
        SafeDtype::F32 => f32_bytes_to_vec(bytes)?,
        other => bail!("unsupported safetensor dtype {:?}", other),
    };
    let tensor = tensor_from_slice_on(&values_f32, Shape::from_dims(shape), device, DType::F32)?;
    if target_dtype == DType::F32 {
        Ok(tensor)
    } else {
        Ok(tensor.to_dtype(target_dtype)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::SystemTime;

    #[test]
    fn bf16_roundtrip() {
        let device = match Device::cuda(0) {
            Ok(dev) => dev,
            Err(_) => {
                eprintln!("skipping bf16_roundtrip (CUDA unavailable)");
                return;
            }
        };
        let values = vec![0.0f32, 1.0, -2.5, 123.25];
        let tensor =
            tensor_from_slice_on(&values, Shape::from_dims(&[values.len()]), &device, DType::F32)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap();
        let (_shape, bytes) = tensor_to_bf16_bytes(&tensor).unwrap();
        let decoded = bf16_bytes_to_f32_vec(&bytes).unwrap();
        assert_eq!(decoded.len(), values.len());
        for (orig, round) in values.iter().zip(decoded.iter()) {
            assert!((orig - round).abs() < 1e-3);
        }
    }

    #[test]
    fn serialize_deserialize_roundtrip() {
        let device = match Device::cuda(0) {
            Ok(dev) => dev,
            Err(_) => {
                eprintln!("skipping serialize_deserialize_roundtrip (CUDA unavailable)");
                return;
            }
        };
        let names = vec!["layer1.weight".to_string(), "layer1.bias".to_string()];
        let mut tensors = Vec::new();
        for (idx, name) in names.iter().enumerate() {
            let vals: Vec<f32> = (0..6).map(|j| idx as f32 + j as f32 * 0.1).collect();
            let tensor =
                tensor_from_slice_on(&vals, Shape::from_dims(&[2, 3]), &device, DType::F32)
                    .unwrap()
                    .to_dtype(DType::BF16)
                    .unwrap();
            let (shape, data) = tensor_to_bf16_bytes(&tensor).unwrap();
            tensors.push((name.clone(), tensor, shape, data));
        }
        let entries: Vec<(String, OwnedTensorView)> = tensors
            .iter()
            .map(|(name, _, shape, data)| {
                (
                    format!("lora.{name}"),
                    OwnedTensorView::new(SafeDtype::BF16, shape.clone(), data.clone()),
                )
            })
            .collect();
        let mut metadata = HashMap::new();
        metadata.insert("step".to_string(), "5".to_string());
        let path = std::env::temp_dir().join(format!(
            "checkpoint_test_{}.safetensors",
            SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_nanos()
        ));
        serialize_tensors(&path, metadata, entries).unwrap();
        let loaded = deserialize_tensors(&path).unwrap();
        for (name, original, _shape, _bytes) in tensors.iter() {
            let key = format!("lora.{name}");
            let loaded_entry = loaded.get(&key).unwrap();
            let restored = loaded_to_tensor(
                &device,
                DType::BF16,
                &loaded_entry.shape,
                loaded_entry.dtype,
                &loaded_entry.bytes,
            )
            .unwrap();
            let orig_vec = original.to_dtype(DType::F32).unwrap().to_vec().unwrap();
            let restored_vec = restored.to_dtype(DType::F32).unwrap().to_vec().unwrap();
            for (a, b) in orig_vec.iter().zip(restored_vec.iter()) {
                assert!((a - b).abs() < 1e-3);
            }
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn serialize_deserialize_ema_roundtrip() {
        let device = match Device::cuda(0) {
            Ok(dev) => dev,
            Err(_) => {
                eprintln!("skipping serialize_deserialize_ema_roundtrip (CUDA unavailable)");
                return;
            }
        };
        let ema_name = "layer".to_string();
        let vals: Vec<f32> = (0..4).map(|j| (j as f32) * 1.5).collect();
        let tensor =
            tensor_from_slice_on(&vals, Shape::from_dims(&[2, 2]), &device, DType::F32).unwrap();
        let (shape, data) = tensor_to_f32_bytes(&tensor).unwrap();
        let mut metadata = HashMap::new();
        metadata.insert("step".to_string(), "9".to_string());
        metadata.insert("ema_decay".to_string(), "0.9".to_string());
        let path = std::env::temp_dir().join(format!(
            "ema_checkpoint_test_{}.safetensors",
            SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_nanos()
        ));
        let entries = vec![(
            format!("ema.{ema_name}"),
            OwnedTensorView::new(SafeDtype::F32, shape.clone(), data.clone()),
        )];
        serialize_tensors(&path, metadata, entries).unwrap();
        let loaded = deserialize_tensors(&path).unwrap();
        let entry = loaded.get(&format!("ema.{ema_name}")).unwrap();
        let restored =
            loaded_to_tensor(&device, DType::F32, &entry.shape, entry.dtype, &entry.bytes).unwrap();
        let restored_vec = restored.to_vec().unwrap();
        for (a, b) in vals.iter().zip(restored_vec.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
        let _ = std::fs::remove_file(&path);
    }
}
