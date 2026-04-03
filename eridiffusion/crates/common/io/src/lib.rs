//! SafeTensors IO helpers for EriDiffusion (GPU-only, BF16 storage).
use std::path::{Path, PathBuf};
use std::sync::Arc;
use anyhow::Result;
use safetensors::{SafeTensors, tensor::{TensorView, Dtype}};
use bytemuck::cast_slice;
use half::{bf16, f16};
use flame_core::{Tensor, DType, Shape, Error as CoreError};
use eridiffusion_core::Device;
fn cuda_for(device: &Device) -> Result<Arc<flame_core::CudaDevice>> {
    match device {
        Device::Cuda(index) => Ok(flame_core::CudaDevice::new(*index)?),
        Device::Cpu => Err(CoreError::Unsupported("CPU device is not supported (GPU-only build)".into()).into()),
    }
}
fn map_dtype(dtype: Dtype) -> Result<DType> {
    match dtype {
        Dtype::BF16 => Ok(DType::BF16),
        Dtype::F16 => Ok(DType::F16),
        Dtype::F32 => Ok(DType::F32),
        Dtype::I32 => Ok(DType::I32),
        Dtype::I64 => Ok(DType::I64),
        other => Err(CoreError::Unsupported(format!("unsupported safetensors dtype: {:?}", other)).into()),
    }
}
#[derive(Clone)]
pub struct STFile {
    path: PathBuf,
    _bytes: Arc<Vec<u8>>,
    st: Arc<SafeTensors<'static>>,
}
impl STFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let bytes = Arc::new(std::fs::read(&path)?);
        let leaked: &'static [u8] = unsafe { std::mem::transmute::<&[u8], &'static [u8]>(bytes.as_slice()) };
        let st = Arc::new(SafeTensors::deserialize(leaked)?);
        Ok(Self { path, _bytes: bytes, st })
    }
    pub fn path(&self) -> &Path { &self.path }
    pub fn keys(&self) -> Vec<String> {
        self.st.names().iter().map(|s| s.to_string()).collect()
    }
    pub fn tensor(&self, key: &str) -> Option<TensorView<'_>> {
        self.st.tensor(key).ok()
    }
}
fn load_f32_view(tv: TensorView) -> Result<Vec<f32>> {
    Ok(match tv.dtype() {
        Dtype::F32 => cast_slice::<u8, f32>(tv.data()).to_vec(),
        Dtype::BF16 => cast_slice::<u8, u16>(tv.data()).iter().map(|&u| bf16::from_bits(u).to_f32()).collect(),
        Dtype::F16 => cast_slice::<u8, u16>(tv.data()).iter().map(|&u| f16::from_bits(u).to_f32()).collect(),
        Dtype::I32 => cast_slice::<u8, i32>(tv.data()).iter().map(|&v| v as f32).collect(),
        Dtype::I64 => cast_slice::<u8, i64>(tv.data()).iter().map(|&v| v as f32).collect(),
        other => return Err(CoreError::Unsupported(format!("cannot load dtype {:?}", other)).into()),
    })
}
pub fn load_tensor_to_device(st: &STFile, key: &str, device: &Device, force_dtype: Option<DType>) -> Result<Tensor> {
    let tv = st.tensor(key).ok_or_else(|| CoreError::InvalidInput(format!("missing tensor key {key}")))?;
    let target = force_dtype.unwrap_or(map_dtype(tv.dtype())?);
    let shape_vec: Vec<usize> = tv.shape().iter().map(|&d| d as usize).collect();
    let shape = Shape::from_dims(&shape_vec);
    let host = load_f32_view(tv)?;
    let cuda = cuda_for(device)?;
    let mut tensor = Tensor::from_vec_dtype(host, shape, cuda, DType::F32)?;
    if target != DType::F32 { tensor = tensor.to_dtype(target)?; }
    Ok(tensor)
}
pub fn load_bf16_with_shape(st: &STFile, key: &str, device: &Device, expected_shape: &[usize]) -> Result<Tensor> {
    let t = load_tensor_to_device(st, key, device, Some(DType::BF16))?;
    if t.shape().dims().to_vec().as_slice() != expected_shape {
        return Err(CoreError::InvalidInput(format!(
            "tensor {key} shape mismatch: expected {:?}, got {:?}", expected_shape, t.shape()
        )).into());
    }
    Ok(t)
}
