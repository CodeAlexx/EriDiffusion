use std::{fs::File, io::Read, mem, path::Path};

use anyhow::{bail, ensure, Context, Result};
use flame_core::{DType, Result as CoreResult, Tensor};
use safetensors::{tensor::Dtype as SafeDtype, tensor::TensorView, SafeTensors};

use eridiffusion_core::Device;
use crate::devtensor::tensor_from_vec_on;
use half::{bf16, f16};
use bytemuck::cast_slice;

pub struct STFile {
    _bytes: Box<[u8]>,
    tensors: SafeTensors<'static>,
}

impl STFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path_ref = path.as_ref();
        let mut file = File::open(path_ref)
            .with_context(|| format!("open {}", path_ref.display()))?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)
            .with_context(|| format!("read {}", path_ref.display()))?;
        let bytes = bytes.into_boxed_slice();
        let slice_static: &'static [u8] = unsafe { mem::transmute::<&[u8], &'static [u8]>(&*bytes) };
        let tensors = SafeTensors::deserialize(slice_static)
            .map_err(|e| anyhow::anyhow!("safetensors deserialize failed: {e}"))?;
        Ok(Self { _bytes: bytes, tensors })
    }

    pub fn keys(&self) -> Vec<String> {
        self.tensors
            .names()
            .into_iter()
            .map(|n| n.to_string())
            .collect()
    }

    pub fn tensor(&self, key: &str) -> Option<TensorView<'_>> {
        self.tensors.tensor(key).ok()
    }
}

fn core<T>(res: CoreResult<T>) -> Result<T> {
    res.map_err(|e| anyhow::anyhow!(e.to_string()))
}

fn map_dtype(dt: SafeDtype) -> Result<DType> {
    Ok(match dt {
        SafeDtype::F32 => DType::F32,
        SafeDtype::BF16 => DType::BF16,
        SafeDtype::F16 => DType::F16,
        SafeDtype::I32 => DType::I32,
        other => bail!("unsupported dtype: {:?}", other),
    })
}

fn tensor_view_to_f32(view: &TensorView<'_>) -> Result<Vec<f32>> {
    Ok(match view.dtype() {
        SafeDtype::F32 => cast_slice::<u8, f32>(view.data()).to_vec(),
        SafeDtype::BF16 => cast_slice::<u8, u16>(view.data())
            .iter()
            .map(|&u| bf16::from_bits(u).to_f32())
            .collect(),
        SafeDtype::F16 => cast_slice::<u8, u16>(view.data())
            .iter()
            .map(|&u| f16::from_bits(u).to_f32())
            .collect(),
        other => bail!("cannot widen dtype {:?} to f32", other),
    })
}

pub fn load_tensor_to_device(
    st: &STFile,
    key: &str,
    device: &Device,
    force_dtype: Option<DType>,
) -> Result<Tensor> {
    let view = st
        .tensor(key)
        .ok_or_else(|| anyhow::anyhow!("missing tensor '{key}'"))?;
    let target = force_dtype.unwrap_or(map_dtype(view.dtype())?);
    let data = tensor_view_to_f32(&view)?;
    let dims_vec = view.shape().to_vec();
    let shape = flame_core::Shape::from_dims(&dims_vec);
    let tensor = core(tensor_from_vec_on(data, shape, device, target))?;
    Ok(tensor)
}

pub fn split_qkv_weight(fused: Tensor) -> Result<(Tensor, Tensor, Tensor)> {
    let dims = fused.shape().dims().to_vec();
    ensure!(dims.len() == 2, "qkv weight must be 2D, got {:?}", dims);
    let rows = dims[0] as usize;
    ensure!(rows % 3 == 0, "qkv weight rows not divisible by 3: {}", rows);
    let chunk = rows / 3;
    let q = fused.narrow(0, 0, chunk)?.transpose_dims(0, 1)?;
    let k = fused.narrow(0, chunk, chunk)?.transpose_dims(0, 1)?;
    let v = fused
        .narrow(0, 2 * chunk, chunk)?
        .transpose_dims(0, 1)?;
    Ok((q, k, v))
}

pub fn split_qkv_bias(fused: Tensor) -> Result<(Tensor, Tensor, Tensor)> {
    let dims = fused.shape().dims().to_vec();
    ensure!(dims.len() == 1, "qkv bias must be 1D, got {:?}", dims);
    let len = dims[0] as usize;
    ensure!(len % 3 == 0, "qkv bias len not divisible by 3: {}", len);
    let chunk = len / 3;
    let q = fused.narrow(0, 0, chunk)?;
    let k = fused.narrow(0, chunk, chunk)?;
    let v = fused.narrow(0, 2 * chunk, chunk)?;
    Ok((q, k, v))
}

pub fn transpose_out_in(weight: Tensor) -> Result<Tensor> {
    let dims = weight.shape().dims().to_vec();
    ensure!(dims.len() == 2, "linear weight must be 2D, got {:?}", dims);
    Ok(weight.transpose_dims(0, 1)?)
}
