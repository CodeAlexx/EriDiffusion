#![cfg(feature = "sd35_strict_loader")]

use std::{
    collections::{HashMap, HashSet},
    path::Path,
    sync::{Mutex, OnceLock},
};

use anyhow::{bail, Context, Result};
use eridiffusion_common_weights::strict_loader::{tensor_from_bytes, StrictMmapLoader, TensorInfo};
use flame_core::{DType, Device, Tensor};
use super::keymap::{all_tensors, TensorLoad, TensorSpec};

static LAST_MESSAGE: OnceLock<Mutex<Option<String>>> = OnceLock::new();

#[derive(Clone, Debug)]
pub struct TensorMeta {
    pub logical_key: String,
    pub original_shape: Vec<usize>,
    pub optional: bool,
}

#[derive(Clone, Debug)]
pub struct Sd35StrictBundle {
    pub(crate) tensors: HashMap<String, Tensor>,
    pub(crate) meta: Vec<TensorMeta>,
}

pub fn load_sd35_strict(path: &str) -> Result<Sd35StrictBundle> {
    let device = Device::cuda(0).context("create CUDA device for SD3.5 strict loader")?;
    let mut loader = StrictMmapLoader::open(Path::new(path))?;
    let mut fused_cache: HashMap<String, Tensor> = HashMap::new();
    let mut marked: HashSet<String> = HashSet::new();
    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    let mut meta: Vec<TensorMeta> = Vec::new();

    for spec in all_tensors() {
        if let Some(prepared) =
            load_spec(&spec, &device, &mut loader, &mut fused_cache, &mut marked)?
        {
            tensors.insert(spec.logical_key.clone(), prepared);
        }
        meta.push(TensorMeta {
            logical_key: spec.logical_key.clone(),
            original_shape: spec.shape.clone(),
            optional: spec.optional,
        });
    }

    loader.validate_used_exactly()?;
    println!("Strict load complete.");
    record_message("Strict load complete.");

    Ok(Sd35StrictBundle { tensors, meta })
}

fn load_spec(
    spec: &TensorSpec,
    device: &Device,
    loader: &mut StrictMmapLoader,
    fused_cache: &mut HashMap<String, Tensor>,
    marked: &mut HashSet<String>,
) -> Result<Option<Tensor>> {
    match &spec.load {
        TensorLoad::Direct => {
            let info = match loader.info(&spec.tensor_key) {
                Ok(info) => info,
                Err(err) => {
                    if spec.optional {
                        return Ok(None);
                    }
                    return Err(err);
                }
            };
            let tensor = materialize_tensor(device, loader, &info, marked)?;
            ensure_shape(&spec.logical_key, &spec.shape, tensor.shape().dims())?;
            let prepared = prepare_tensor(spec, tensor)?;
            Ok(Some(prepared))
        }
        TensorLoad::Slice { axis, start, len } => {
            let fused = match fused_cache.get(&spec.tensor_key) {
                Some(t) => t.clone(),
                None => {
                    let info = match loader.info(&spec.tensor_key) {
                        Ok(info) => info,
                        Err(err) => {
                            if spec.optional {
                                return Ok(None);
                            }
                            return Err(err);
                        }
                    };
                    let tensor = materialize_tensor(device, loader, &info, marked)?;
                    fused_cache.insert(spec.tensor_key.clone(), tensor.clone());
                    tensor
                }
            };
            ensure_slice_bounds(spec, fused.shape().dims(), *axis, *start, *len)?;
            let slice = fused.narrow(*axis, *start, *len)?;
            ensure_shape(&spec.logical_key, &spec.shape, slice.shape().dims())?;
            let prepared = prepare_tensor(spec, slice)?;
            Ok(Some(prepared))
        }
    }
}

fn materialize_tensor(
    device: &Device,
    loader: &mut StrictMmapLoader,
    info: &TensorInfo,
    marked: &mut HashSet<String>,
) -> Result<Tensor> {
    let bytes = loader.bytes(&info.key)?;
    let tensor = tensor_from_bytes(device.clone(), info, bytes)?;
    if marked.insert(info.key.clone()) {
        loader.mark_used(&info.key);
    }
    Ok(tensor)
}

fn ensure_slice_bounds(
    spec: &TensorSpec,
    fused_shape: &[usize],
    axis: usize,
    start: usize,
    len: usize,
) -> Result<()> {
    if axis >= fused_shape.len() {
        bail!(
            "tensor {} slice axis {} out of bounds for fused shape {:?}",
            spec.tensor_key,
            axis,
            fused_shape
        );
    }
    let avail = fused_shape[axis];
    if start + len > avail {
        bail!(
            "tensor {} slice (start={}, len={}) exceeds dimension size {}",
            spec.tensor_key,
            start,
            len,
            avail
        );
    }
    Ok(())
}

fn ensure_shape(key: &str, expected: &[usize], actual: &[usize]) -> Result<()> {
    if expected != actual {
        bail!("tensor {} shape mismatch: expected {:?} got {:?}", key, expected, actual);
    }
    Ok(())
}

fn prepare_tensor(spec: &TensorSpec, tensor: Tensor) -> Result<Tensor> {
    let mut out = tensor;
    let rank = out.shape().dims().len();
    if rank == 2 && spec.logical_key.ends_with(".weight") {
        out = out.transpose()?;
    } else if rank == 4 && spec.logical_key.ends_with(".weight") {
        out = out.permute(&[2, 3, 1, 0])?;
    }
    cast_preserve_grad(&out, DType::BF16)
}

fn cast_preserve_grad(t: &Tensor, dtype: DType) -> Result<Tensor> {
    if t.dtype() == dtype {
        return Ok(t.clone());
    }
    let scaled = t.mul_scalar(1.0f32)?;
    Ok(scaled.to_dtype(dtype)?)
}

fn record_message(msg: &str) {
    let storage = LAST_MESSAGE.get_or_init(|| Mutex::new(None));
    if let Ok(mut guard) = storage.lock() {
        *guard = Some(msg.to_string());
    }
}

pub fn last_strict_loader_message() -> Option<String> {
    LAST_MESSAGE
        .get()
        .and_then(|storage| storage.lock().ok())
        .and_then(|guard| guard.clone())
}
